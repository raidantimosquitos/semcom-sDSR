"""
AnomalyEvaluator: DCASE2020 Task 2 metrics (AUC, pAUC).

Evaluates trained model on test data (normal + anomalous).
Model interface: forward(x) -> M_out (B, 2, H, W) segmentation logits.
Anomaly score per clip: mean over spatial dims of anomaly channel (index 1).
Supports optional machine-ID conditioned normalization using train-set stats.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

_EPS = 1e-8


def _partial_auc(y_true: list[float] | list[int], y_score: list[float], max_fpr: float) -> float:
    """Partial AUC with max_fpr (default 0.1). Uses sklearn if available."""
    if roc_auc_score is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


class AnomalyEvaluator:
    """
    Evaluator for anomaly detection: AUC and pAUC per machine ID.

    Model must implement forward(x) returning M_out (B, 2, H, W) where
    channel 1 is the anomaly logit.

    With ``report_recon_mse=True``, the model must accept
    ``forward(x, return_intermediates=True)`` and return
    ``(m_out, x_general, x_specific)`` (as in :class:`~src.models.sDSR.s_dsr.sDSR`).

    If ``subset_machine_id`` is set, only clips with that DCASE machine_id are
    scored; per-ID entries and ``average`` reflect that subset (used for Stage 2
    val-best when training on one ID).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_dataset: Any,
        device: str | torch.device = "cuda",
        pauc_max_fpr: float = 0.1,
        batch_size: int = 32,
        train_score_stats: dict[str, tuple[float, float]] | None = None,
        train_score_stats_fallback: tuple[float, float] | None = None,
        subset_machine_id: str | None = None,
        report_recon_mse: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pauc_max_fpr = pauc_max_fpr
        self.loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.machine_type = getattr(test_dataset, "machine_type", "unknown")
        self.train_score_stats = train_score_stats
        self.train_score_stats_fallback = train_score_stats_fallback
        self.subset_machine_id = subset_machine_id
        self.report_recon_mse = report_recon_mse

    def _anomaly_scores(self, m_out: torch.Tensor) -> torch.Tensor:
        """
        Aggregate M_out to per-clip anomaly score (mean over spatial dims).
        Returns (B,) tensor of mean anomaly probability per clip.
        """
        probs = torch.softmax(m_out, dim=1)
        anomaly_prob = probs[:, 1]  # (B, H, W)
        return anomaly_prob.view(m_out.shape[0], -1).mean(dim=1).cpu()

    def evaluate(self) -> dict[str, Any]:
        """
        Run evaluation. Returns:
            {machine_type: {id: {auc, pauc}, "average": {auc, pauc}}, ...}

        If ``report_recon_mse`` is True, also sets key ``_recon_mse`` with per-stratum
        mean squared errors (VQ-VAE general path vs object-specific decoder), using the
        same ``subset_machine_id`` filter as AUC. Callers should ``pop`` ``_recon_mse``
        before iterating machine-type entries as IDs.
        """
        self.model.eval()
        scores_by_id: dict[str, list[tuple[float, int]]] = defaultdict(list)

        recon_sums_vq: dict[str, float] = {"normal": 0.0, "anomalous": 0.0}
        recon_sums_sp: dict[str, float] = {"normal": 0.0, "anomalous": 0.0}
        recon_counts: dict[str, int] = {"normal": 0, "anomalous": 0}
        recon_failed = False

        with torch.no_grad():
            for batch in self.loader:
                if len(batch) == 3:
                    x, labels, machine_ids = batch
                else:
                    x, labels = batch
                    machine_ids = [""] * x.shape[0]
                x = x.to(self.device)
                if self.report_recon_mse and not recon_failed:
                    try:
                        out = self.model(x, return_intermediates=True)
                    except TypeError:
                        recon_failed = True
                        m_out = self.model(x)
                        mse_vq_b = mse_sp_b = None
                    else:
                        m_out, x_general, x_specific = out
                        mse_vq_b = (x_general - x) ** 2
                        mse_sp_b = (x_specific - x) ** 2
                        mse_vq_b = mse_vq_b.mean(dim=(1, 2, 3))
                        mse_sp_b = mse_sp_b.mean(dim=(1, 2, 3))
                else:
                    m_out = self.model(x)
                    mse_vq_b = mse_sp_b = None
                sc_mean = self._anomaly_scores(m_out)
                for i in range(x.shape[0]):
                    mid = machine_ids[i] if isinstance(machine_ids[i], str) else str(machine_ids[i])
                    if self.subset_machine_id is not None and mid != self.subset_machine_id:
                        continue
                    label = int(labels[i].item())
                    score = sc_mean[i].item()
                    if self.train_score_stats is not None:
                        stats = self.train_score_stats.get(mid, self.train_score_stats_fallback)
                        if stats is not None:
                            mean_val, std_val = stats
                            score = (score - mean_val) / (std_val + _EPS)
                    scores_by_id[mid].append((score, label))

                    if (
                        self.report_recon_mse
                        and not recon_failed
                        and mse_vq_b is not None
                        and mse_sp_b is not None
                    ):
                        bucket = "normal" if label == 0 else "anomalous"
                        recon_sums_vq[bucket] += float(mse_vq_b[i].item())
                        recon_sums_sp[bucket] += float(mse_sp_b[i].item())
                        recon_counts[bucket] += 1

        result: dict[str, Any] = {self.machine_type: {}}

        for mid in sorted(scores_by_id.keys()):
            pairs = scores_by_id[mid]
            y_true = [p[1] for p in pairs]
            y_score = [p[0] for p in pairs]

            auc = roc_auc_score(y_true, y_score) if roc_auc_score else float("nan")
            pauc = _partial_auc(y_true, y_score, self.pauc_max_fpr)

            result[self.machine_type][mid] = {
                "auc": auc,
                "pauc": pauc,
            }

        ids = [k for k in result[self.machine_type].keys() if k != "average"]
        n = len(ids)
        if self.subset_machine_id is not None and n == 1:
            # Single-ID run: "average" matches that ID (used for val-best checkpointing).
            only = ids[0]
            result[self.machine_type]["average"] = {
                "auc": result[self.machine_type][only]["auc"],
                "pauc": result[self.machine_type][only]["pauc"],
            }
        else:
            result[self.machine_type]["average"] = {
                "auc": sum(result[self.machine_type][mid]["auc"] for mid in ids) / n if n else float("nan"),
                "pauc": sum(result[self.machine_type][mid]["pauc"] for mid in ids) / n if n else float("nan"),
            }

        if self.report_recon_mse:
            total_n = recon_counts["normal"] + recon_counts["anomalous"]

            def _mean(sums: dict[str, float], key: str) -> float:
                c = recon_counts[key]
                return sums[key] / c if c else float("nan")

            result["_recon_mse"] = {
                "subset_machine_id": self.subset_machine_id,
                "vqvae_decode_general": {
                    "normal": _mean(recon_sums_vq, "normal"),
                    "anomalous": _mean(recon_sums_vq, "anomalous"),
                    "all": (
                        (recon_sums_vq["normal"] + recon_sums_vq["anomalous"]) / total_n
                        if total_n
                        else float("nan")
                    ),
                },
                "object_specific_decoder": {
                    "normal": _mean(recon_sums_sp, "normal"),
                    "anomalous": _mean(recon_sums_sp, "anomalous"),
                    "all": (
                        (recon_sums_sp["normal"] + recon_sums_sp["anomalous"]) / total_n
                        if total_n
                        else float("nan")
                    ),
                },
                "counts": dict(recon_counts),
                "skipped_due_to_model": recon_failed,
            }

        return result
