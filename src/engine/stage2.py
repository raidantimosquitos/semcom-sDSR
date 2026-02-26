"""
Stage 2 trainer: sDSR anomaly detection.

Model interface: forward_train(x, M_gt) -> dict with m_out, x_s, M, x
Requires pre-trained encoder (vq_vae) loaded from Stage 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from .base import BaseTrainer


class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage 2 (sDSR anomaly detection).

    Model must implement forward_train(x, M_gt) returning dict with:
        m_out, x_s, M, x
    Only trainable params (object_decoder, anomaly_detection) are optimized.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        machine_type: str = "unknown",
        lambda_recon: float = 10.0,
        lambda_focal: float = 1.0,
        lambda_sub: float = 1.0,
        lr: float = 2e-4,
        lr_warmup_iters: int = 200,
        lr_min: float = 1e-6,
        batch_size: int = 16,
        grad_clip: float | None = 1.0,
        log_every: int = 50,
        ckpt_every: int = 500,
        ckpt_dir: str | Path = "./checkpoints",
        use_amp: bool = True,
        device: str = "cuda",
    ) -> None:
        self.machine_type = machine_type
        self.lambda_recon = lambda_recon
        self.lambda_focal = lambda_focal
        self.lambda_sub = lambda_sub
        self.lr = lr
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_min = lr_min

        # Dataset is already for a single machine_type; no filtering needed
        if getattr(dataset, "machine_type", None) not in (None, machine_type):
            raise ValueError(f"Dataset machine_type '{getattr(dataset, 'machine_type')}' != trainer machine_type '{machine_type}'")

        ckpt_path = Path(ckpt_dir) / "stage2" / machine_type
        super().__init__(
            model=model,
            dataset=dataset,
            device=device,
            ckpt_dir=ckpt_path,
            log_every=log_every,
            ckpt_every=ckpt_every,
            batch_size=batch_size,
            grad_clip=grad_clip,
            use_amp=use_amp,
        )

        # Optimizer for trainable params only (object_decoder, anomaly_detection)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        from ..models.sDSR.loss import FocalLoss
        self.focal_loss = FocalLoss(gamma=2.0)

        self.best_total_loss = float("inf")
        self._last_ckpt_path: Path | None = None

        n_params = sum(p.numel() for p in trainable)
        self._tee(f"Stage2 | Device: {self.device} | AMP: {self.use_amp} | Trainable params: {n_params:,}")

    def _get_lr(self, step: int, total_steps: int) -> float:
        if step < self.lr_warmup_iters:
            return self.lr * (step + 1) / self.lr_warmup_iters
        progress = (step - self.lr_warmup_iters) / max(
            1, total_steps - self.lr_warmup_iters
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + cosine * (self.lr - self.lr_min)

    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        x = batch["image"].to(self.device, non_blocking=True)
        M_gt = batch["anomaly_mask"].to(self.device, non_blocking=True)

        lr = self._get_lr(step, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model.forward_train(x, M_gt=M_gt)
            m_out = out["m_out"]
            x_s = out["x_s"]
            M = out["M"]
            has_anom = (M_gt.sum(dim=(1, 2, 3)) > 0)
            if has_anom.any():
                loss_recon = F.mse_loss(out["x"][has_anom], out["x_s"][has_anom])
            else:
                loss_recon = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            loss_focal = self.focal_loss(m_out, M)

            total_loss = (
                self.lambda_recon * loss_recon
                + self.lambda_focal * loss_focal
            )
            if self.lambda_sub != 0 and "recon_feat_bot" in out and "recon_feat_top" in out and "q_bot" in out and "q_top" in out and has_anom.any():
                loss_sub_bot = F.mse_loss(out["recon_feat_bot"][has_anom], out["q_bot"].detach()[has_anom])
                loss_sub_top = F.mse_loss(out["recon_feat_top"][has_anom], out["q_top"].detach()[has_anom])
                loss_sub = 0.5 * loss_sub_bot + 0.5 * loss_sub_top
                total_loss = total_loss + self.lambda_sub * loss_sub
                sub_value = loss_sub.item()
            else:
                sub_value = 0.0

        self.scaler.scale(total_loss).backward()

        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total": total_loss.item(),
            "recon": loss_recon.item(),
            "focal": loss_focal.item(),
            "sub": sub_value,
            "lr": lr,
        }

    def _log(self, avg: dict[str, float], its_sec: float) -> None:
        total = avg["total"] if math.isfinite(avg.get("total", 0)) else 0.0
        recon = avg["recon"] if math.isfinite(avg.get("recon", 0)) else 0.0
        focal = avg["focal"] if math.isfinite(avg.get("focal", 0)) else 0.0
        sub = avg.get("sub", 0)
        sub = sub if math.isfinite(sub) else 0.0
        self._tee(
            f"[{self.global_step:>6d}] "
            f"loss={total:.4f}  recon={recon:.4f}  focal={focal:.4f}  sub={sub:.4f}  "
            f"lr={avg['lr']:.2e}  ({its_sec:.1f} it/s)"
        )

    def _save_checkpoint(
        self, tag: str | None = None, avg: dict[str, float] | None = None
    ) -> None:
        payload = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_total_loss": self.best_total_loss,
        }

        # 1. Delete previous latest checkpoint before saving new one
        tag = tag or f"iter_{self.global_step:06d}"
        latest_path = self.ckpt_dir / f"stage2_{self.machine_type}_{tag}.pt"
        if self._last_ckpt_path is not None and self._last_ckpt_path.exists():
            self._last_ckpt_path.unlink()
            self._tee(f"  Deleted previous checkpoint: {self._last_ckpt_path}")

        # 2. Save current model as latest
        torch.save(payload, latest_path)
        self._last_ckpt_path = latest_path
        self._tee(f"  Checkpoint saved: {latest_path}")

        # 3. Update best_model if we improved (skip when total is non-finite)
        total_loss = avg.get("total", float("inf")) if avg else float("inf")
        if math.isfinite(total_loss) and total_loss < self.best_total_loss:
            self.best_total_loss = total_loss
            best_path = self.ckpt_dir / f"stage2_{self.machine_type}_best.pt"
            torch.save(payload, best_path)
            self._tee(f"  New best model saved: {best_path} (loss={total_loss:.4f})")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if "best_total_loss" in ckpt:
            self.best_total_loss = ckpt["best_total_loss"]
        self._tee(f"Resumed from {path} at step {self.global_step}")
