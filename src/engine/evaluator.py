"""
AnomalyEvaluator: DCASE2020 Task 2 metrics (AUC, pAUC).

Evaluates trained model on test data (normal + anomalous).
Model interface: forward(x) -> M_out (B, 2, H, W) segmentation logits.
Anomaly score per clip: max over spatial dims of anomaly channel (index 1).
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

    def _anomaly_scores(self, m_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate M_out to per-clip anomaly scores with three methods.
        Returns (scores_mean, scores_max, scores_p95), each (B,).
        Computed directly from the output mask (no smoothing).
        """
        logits = m_out[:, 1]  # (B, H, W) — anomaly channel
        flat = logits.view(m_out.shape[0], -1)  # (B, H*W)
        sc_mean = flat.mean(dim=1)
        sc_max  = flat.max(dim=1).values
        sc_p95  = flat.kthvalue(max(1, int(0.95 * flat.shape[1])), dim=1).values
        return sc_mean.cpu(), sc_max.cpu(), sc_p95.cpu()

    def evaluate(self) -> dict[str, Any]:
        """
        Run evaluation. Returns:
            {machine_type: {id: {auc, pauc}, "average": {auc, pauc}}, ...}
        """
        self.model.eval()
        # list of (score_mean, score_max, score_p95, label) per clip
        scores_by_id: dict[str, list[tuple[float, float, float, int]]] = defaultdict(list)

        with torch.no_grad():
            for batch in self.loader:
                if len(batch) == 3:
                    x, labels, machine_ids = batch
                else:
                    x, labels = batch
                    machine_ids = [""] * x.shape[0]
                x = x.to(self.device)
                m_out = self.model(x)
                sc_mean, sc_max, sc_p95 = self._anomaly_scores(m_out)
                for i in range(x.shape[0]):
                    mid = machine_ids[i] if isinstance(machine_ids[i], str) else str(machine_ids[i])
                    label = int(labels[i].item())
                    scores_by_id[mid].append((sc_mean[i].item(), sc_max[i].item(), sc_p95[i].item(), label))

        result: dict[str, Any] = {self.machine_type: {}}

        for mid in sorted(scores_by_id.keys()):
            pairs = scores_by_id[mid]
            y_true = [p[3] for p in pairs]
            y_mean = [p[0] for p in pairs]
            y_max = [p[1] for p in pairs]
            y_p95 = [p[2] for p in pairs]

            auc_mean = roc_auc_score(y_true, y_mean) if roc_auc_score else float("nan")
            pauc_mean = _partial_auc(y_true, y_mean, self.pauc_max_fpr)
            auc_max = roc_auc_score(y_true, y_max) if roc_auc_score else float("nan")
            pauc_max = _partial_auc(y_true, y_max, self.pauc_max_fpr)
            auc_p95 = roc_auc_score(y_true, y_p95) if roc_auc_score else float("nan")
            pauc_p95 = _partial_auc(y_true, y_p95, self.pauc_max_fpr)

            result[self.machine_type][mid] = {
                "auc": auc_mean,
                "pauc": pauc_mean,
                "auc_max": auc_max,
                "pauc_max": pauc_max,
                "auc_p95": auc_p95,
                "pauc_p95": pauc_p95,
            }

        # Average over machine IDs (paper: "average metrics over all machine IDs for a specific machine type")
        ids = list(result[self.machine_type].keys())
        n = len(ids)
        result[self.machine_type]["average"] = {
            "auc": sum(result[self.machine_type][mid]["auc"] for mid in ids) / n if n else float("nan"),
            "pauc": sum(result[self.machine_type][mid]["pauc"] for mid in ids) / n if n else float("nan"),
            "auc_max": sum(result[self.machine_type][mid]["auc_max"] for mid in ids) / n if n else float("nan"),
            "pauc_max": sum(result[self.machine_type][mid]["pauc_max"] for mid in ids) / n if n else float("nan"),
            "auc_p95": sum(result[self.machine_type][mid]["auc_p95"] for mid in ids) / n if n else float("nan"),
            "pauc_p95": sum(result[self.machine_type][mid]["pauc_p95"] for mid in ids) / n if n else float("nan"),
        }

        return result
