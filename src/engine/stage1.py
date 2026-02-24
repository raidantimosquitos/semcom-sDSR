"""
Stage 1 trainer: VQ-VAE-2 reconstruction.

Model interface: forward(x) -> (loss_bot, loss_top, recon, q_top, q_bot, perplexity_top, perplexity_bot)
Any encoder that returns this tuple can be used.
"""

from __future__ import annotations

import math
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from .base import BaseTrainer


class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage 1 (VQ-VAE-2 style reconstruction).

    Model must implement forward(x) returning:
        (loss_bot, loss_top, recon, q_top, q_bot, perplexity_top, perplexity_bot)
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        machine_type: str = "unknown",
        lambda_recon: float = 1.0,
        lr: float = 1e-4,
        lr_warmup_iters: int = 500,
        lr_min: float = 1e-6,
        batch_size: int = 64,
        grad_clip: float | None = 1.0,
        log_every: int = 100,
        ckpt_every: int = 2000,
        ckpt_dir: str | Path = "./checkpoints",
        use_amp: bool = True,
        device: str = "cuda",
    ) -> None:
        self.machine_type = machine_type
        self.lambda_recon = lambda_recon
        self.lr = lr
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_min = lr_min

        # Dataset is already for a single machine_type; no filtering needed
        if getattr(dataset, "machine_type", None) not in (None, machine_type):
            raise ValueError(f"Dataset machine_type '{getattr(dataset, 'machine_type')}' != trainer machine_type '{machine_type}'")

        ckpt_path = Path(ckpt_dir) / "stage1" / machine_type
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.best_total_loss = float("inf")
        self._last_ckpt_path: Path | None = None

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Stage1 | Device: {self.device} | AMP: {self.use_amp} | Params: {n_params:,}")

    def _get_lr(self, step: int, total_steps: int) -> float:
        if step < self.lr_warmup_iters:
            return self.lr * (step + 1) / self.lr_warmup_iters
        progress = (step - self.lr_warmup_iters) / max(
            1, total_steps - self.lr_warmup_iters
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + cosine * (self.lr - self.lr_min)

    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(self.device, non_blocking=True)

        lr = self._get_lr(step, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model(x)
            loss_b, loss_t, recon, q_t, q_b, perp_t, perp_b = out
            recon_loss = F.mse_loss(recon, x)
            total_loss = loss_b + loss_t + self.lambda_recon * recon_loss

        self.scaler.scale(total_loss).backward()

        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "loss_b": loss_b.item(),
            "loss_t": loss_t.item(),
            "lr": lr,
            "perplexity_t": perp_t.item(),
            "perplexity_b": perp_b.item(),
        }

    def _log(self, avg: dict[str, float], its_sec: float) -> None:
        print(
            f"[{self.global_step:>6d}] "
            f"loss={avg['total']:.4f}  recon={avg['recon']:.4f}  "
            f"loss_b={avg['loss_b']:.4f}  loss_t={avg['loss_t']:.4f}  "
            f"lr={avg['lr']:.2e}  perp_t={avg['perplexity_t']:.2f}  "
            f"perp_b={avg['perplexity_b']:.2f}  ({its_sec:.1f} it/s)"
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
        latest_path = self.ckpt_dir / f"stage1_{self.machine_type}_{tag}.pt"
        if self._last_ckpt_path is not None and self._last_ckpt_path.exists():
            self._last_ckpt_path.unlink()
            print(f"  Deleted previous checkpoint: {self._last_ckpt_path}")

        # 2. Save current model as latest
        torch.save(payload, latest_path)
        self._last_ckpt_path = latest_path
        print(f"  Checkpoint saved: {latest_path}")

        # 3. Update best_model if we improved
        total_loss = avg.get("total", float("inf")) if avg else float("inf")
        if total_loss < self.best_total_loss:
            self.best_total_loss = total_loss
            best_path = self.ckpt_dir / f"stage1_{self.machine_type}_best.pt"
            torch.save(payload, best_path)
            print(f"  New best model saved: {best_path} (loss={total_loss:.4f})")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        if "best_total_loss" in ckpt:
            self.best_total_loss = ckpt["best_total_loss"]
        print(f"Resumed from {path} at step {self.global_step}")


# Backward compatibility alias
VQ_VAE_2LayerTrainer = Stage1Trainer
