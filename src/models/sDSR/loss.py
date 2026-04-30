"""
Loss functions for sDSR anomaly detection.

- FocalLoss: for segmentation, reduces weight on easy pixels (paper Section).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary/multi-class segmentation.

    Reduces contribution of easy (well-classified) pixels; focuses on hard examples.
    Used for training the anomaly detection module against ground-truth anomaly map M.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
        smoothing: float = 1e-5,
        from_logits: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            gamma: focusing parameter; higher downweights easy examples
            alpha: class weight for positive class; None = equal
            reduction: 'mean', 'sum', or 'none'
            smoothing: label smoothing factor in [0, 1)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.smoothing = smoothing
        self.from_logits = from_logits
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, 2, H, W) or (B, 1, H, W) — class logits (from_logits=True)
                    or class probabilities (from_logits=False, e.g. softmax output)
            target: (B, 1, H, W) or (B, 2, H, W) — ground-truth anomaly map
                    values in {0, 1} for binary; if (B, 1, H, W), expanded to 2ch

        Returns:
            Focal loss scalar
        """
        B, C, H, W = inputs.shape
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        if target.shape[1] == 1 and C == 2:
            # Binary target: convert to 2-channel one-hot
            # target: 1 = anomaly, 0 = normal
            # one_hot: [normal, anomaly] -> index 0 = normal, 1 = anomaly
            t_normal = 1.0 - target  # (B, 1, H, W)
            t_anomaly = target
            target_2ch = torch.cat([t_normal, t_anomaly], dim=1)  # (B, 2, H, W)
        else:
            target_2ch = target

        if self.smoothing > 0:
            target_2ch = target_2ch * (1 - self.smoothing) + self.smoothing / C

        if self.from_logits:
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
        else:
            probs = inputs.clamp(min=self.eps, max=1.0 - self.eps)
            # When provided probabilities, compute log(p) directly.
            log_probs = probs.log()
        pt = (target_2ch * probs).sum(dim=1, keepdim=True).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        ce = -(target_2ch * log_probs).sum(dim=1)
        loss = focal_weight * ce
        if self.alpha is not None:
            alpha_t = self.alpha * target_2ch[:, 1] + (1 - self.alpha) * target_2ch[:, 0]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
