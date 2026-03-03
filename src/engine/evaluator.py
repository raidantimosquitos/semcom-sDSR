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

    def _anomaly_scores(self, m_out: torch.Tensor) -> torch.Tensor:
        """
        Aggregate M_out to per-clip anomaly score (mean over spatial dims).
        Returns (B,) tensor of mean anomaly logit per clip.
        """
        logits = m_out[:, 1]  # (B, H, W) — anomaly channel
        flat = logits.view(m_out.shape[0], -1)  # (B, H*W)
        return flat.mean(dim=1).cpu()

    def evaluate(self) -> dict[str, Any]:
        """
        Run evaluation. Returns:
            {machine_type: {id: {auc, pauc}, "average": {auc, pauc}}, ...}
        """
        self.model.eval()
        scores_by_id: dict[str, list[tuple[float, int]]] = defaultdict(list)

        with torch.no_grad():
            for batch in self.loader:
                if len(batch) == 3:
                    x, labels, machine_ids = batch
                else:
                    x, labels = batch
                    machine_ids = [""] * x.shape[0]
                x = x.to(self.device)
                m_out = self.model(x)
                sc_mean = self._anomaly_scores(m_out)
                for i in range(x.shape[0]):
                    mid = machine_ids[i] if isinstance(machine_ids[i], str) else str(machine_ids[i])
                    label = int(labels[i].item())
                    score = sc_mean[i].item()
                    if self.train_score_stats is not None:
                        stats = self.train_score_stats.get(mid, self.train_score_stats_fallback)
                        if stats is not None:
                            mean_val, std_val = stats
                            score = (score - mean_val) / (std_val + _EPS)
                    scores_by_id[mid].append((score, label))

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
        result[self.machine_type]["average"] = {
            "auc": sum(result[self.machine_type][mid]["auc"] for mid in ids) / n if n else float("nan"),
            "pauc": sum(result[self.machine_type][mid]["pauc"] for mid in ids) / n if n else float("nan"),
        }

        return result
