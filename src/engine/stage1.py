"""
Stage 1 trainer: VQ-VAE-2 reconstruction.

Model interface: forward(x) -> (loss_fine, loss_coarse, recon, q_coarse, q_fine, perplexity_coarse, perplexity_fine)
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

from ..models.vq_vae.autoencoders import VQ_VAE_2Layer
from ..models.vq_vae.kmeans_init import init_vqvae_codebooks_from_loader

from .base import BaseTrainer


class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage 1 (VQ-VAE-2 style reconstruction).

    Model must implement forward(x) returning:
        (loss_fine, loss_coarse, recon, q_coarse, q_fine, perplexity_coarse, perplexity_fine)
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        machine_type: str = "unknown",
        lambda_recon: float = 0.25,
        lr: float = 2e-4,
        lr_warmup_iters: int = 1000,
        lr_min: float = 2e-4,
        batch_size: int = 64,
        grad_clip: float | None = 1.0,
        log_every: int = 100,
        ckpt_every: int = 2000,
        ckpt_dir: str | Path = "./checkpoints",
        use_amp: bool = True,
        device: str = "cuda",
        total_steps: int = 20000,
        num_workers: int = 0,
        *,
        kmeans_init_after_warmup: bool = False,
        kmeans_max_samples: int = 500_000,
        kmeans_iters: int = 15,
        kmeans_seed: int = 0,
        kmeans_max_batches: int | None = None,
        kmeans_reset_adam: bool = False,
    ) -> None:
        self.machine_type = machine_type
        self.lambda_recon = lambda_recon
        self.lr = lr
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_min = lr_min
        self.dataset = dataset
        self.total_steps = total_steps
        self.kmeans_init_after_warmup = kmeans_init_after_warmup
        self.kmeans_max_samples = kmeans_max_samples
        self.kmeans_iters = kmeans_iters
        self.kmeans_seed = kmeans_seed
        self.kmeans_max_batches = kmeans_max_batches
        self.kmeans_reset_adam = kmeans_reset_adam
        self._kmeans_init_done = False

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
            num_workers=num_workers,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.best_total_loss = float("inf")
        self._last_ckpt_path: Path | None = None

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._tee(
            f"Stage1 | Device: {self.device} | AMP: {self.use_amp} | "
            f"DataLoader workers: {self.num_workers} | Params: {n_params:,}"
        )
        if self.kmeans_init_after_warmup:
            self._tee(
                f"Stage1 | K-means codebook init: after warmup step "
                f"{self.lr_warmup_iters} (max_samples={self.kmeans_max_samples}, "
                f"iters={self.kmeans_iters})"
            )

    def _kmeans_schedule_step(self) -> int:
        """Training step index at which we run k-means (after that many optimizer steps)."""
        return self.lr_warmup_iters

    def _maybe_kmeans_init_codebooks(self, step: int) -> None:
        if not self.kmeans_init_after_warmup or self._kmeans_init_done:
            return
        if step != self._kmeans_schedule_step():
            return
        if not isinstance(self.model, VQ_VAE_2Layer):
            raise TypeError(
                "kmeans_init_after_warmup requires VQ_VAE_2Layer; got "
                f"{type(self.model).__name__}"
            )
        self._tee(
            f"Running k-means codebook init at step {step} (scanning loader) ..."
        )
        init_vqvae_codebooks_from_loader(
            self.model,
            self.loader,
            self.device,
            max_batches=self.kmeans_max_batches,
            max_samples=self.kmeans_max_samples,
            kmeans_iters=self.kmeans_iters,
            seed=self.kmeans_seed,
        )
        if self.kmeans_reset_adam:
            n_cleared = 0
            for name, p in self.model.named_parameters():
                if "_vq_coarse" in name or "_vq_fine" in name:
                    if p in self.optimizer.state:
                        self.optimizer.state[p].clear()
                        n_cleared += 1
            self._tee(f"Stage1 | Cleared Adam state for {n_cleared} VQ parameter tensors")

        self._kmeans_init_done = True
        self._tee("Stage1 | K-means codebook init done.")

    def _get_lr(self, step: int, total_steps: int) -> float:
        if step < self.lr_warmup_iters:
            return self.lr * (step + 1) / self.lr_warmup_iters
        progress = (step - self.lr_warmup_iters) / max(
            1, total_steps - self.lr_warmup_iters
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + cosine * (self.lr - self.lr_min)

    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        self._maybe_kmeans_init_codebooks(step)

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
            loss_fine, loss_coarse, recon, _, _, perp_coarse, perp_fine = out
            recon_loss = F.mse_loss(recon, x)
            total_loss = loss_fine + loss_coarse + self.lambda_recon * recon_loss

        self.scaler.scale(total_loss).backward()

        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "loss_fine": loss_fine.item(),
            "loss_coarse": loss_coarse.item(),
            "perplexity_fine": perp_fine.item(),
            "perplexity_coarse": perp_coarse.item(),
            "lr": lr,
        }

    def _log(self, avg: dict[str, float], its_sec: float) -> None:
        self._tee(
            f"[{self.global_step:>6d}] "
            f"loss={avg['total']:.4f}  recon={avg['recon']:.4f}  "
            f"loss_fine={avg['loss_fine']:.4f}  loss_coarse={avg['loss_coarse']:.4f}  "
            f"perp_fine={avg['perplexity_fine']:.2f}  perp_coarse={avg['perplexity_coarse']:.2f}  "
            f"lr={avg['lr']:.2e} ({its_sec:.1f} it/s)"
        )

    def _save_checkpoint(
        self, tag: str | None = None, avg: dict[str, float] | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_total_loss": self.best_total_loss,
            "kmeans_init_done": bool(self._kmeans_init_done),
            "target_T": int(self.dataset.target_T),
            "n_mels": int(self.dataset.data.shape[2]),
            "num_embeddings_fine": int(self.model.num_embeddings_fine),
            "num_embeddings_coarse": int(self.model.num_embeddings_coarse),
            "embedding_dim_fine": int(self.model.embedding_dim_fine),
            "embedding_dim_coarse": int(self.model.embedding_dim_coarse),
            "hidden_channels_fine": int(self.model.hidden_channels_fine),
            "hidden_channels_coarse": int(self.model.hidden_channels_coarse),
            "num_residual_layers": int(self.model.num_residual_layers),
        }
        norm_stats = getattr(self.dataset, "norm_stats", None)
        if norm_stats:
            payload["norm_stats"] = {
                mt: {"mean": mean.cpu().clone(), "std": std.cpu().clone()}
                for mt, (mean, std) in norm_stats.items()
            }
            payload["machine_types"] = list(norm_stats.keys())
        else:
            norm_mean = getattr(self.dataset, "norm_mean", None)
            norm_std = getattr(self.dataset, "norm_std", None)
            if norm_mean is not None and norm_std is not None:
                payload["norm_mean"] = norm_mean.cpu().clone()
                payload["norm_std"] = norm_std.cpu().clone()

        # 1. Delete previous latest checkpoint before saving new one
        tag = tag or f"iter_{self.global_step:06d}"
        latest_path = self.ckpt_dir / f"stage1_{self.machine_type}_{tag}.pt"
        if self._last_ckpt_path is not None and self._last_ckpt_path.exists():
            self._last_ckpt_path.unlink()
            self._tee(f"  Deleted previous checkpoint: {self._last_ckpt_path}")

        # 2. Save current model as latest
        torch.save(payload, latest_path)
        self._last_ckpt_path = latest_path
        self._tee(f"  Checkpoint saved: {latest_path}")

        # 3. Update best_model if we improved
        total_loss = avg.get("total", float("inf")) if avg else float("inf")
        if total_loss < self.best_total_loss:
            self.best_total_loss = total_loss
            best_path = self.ckpt_dir / f"stage1_{self.machine_type}_best.pt"
            torch.save(payload, best_path)
            self._tee(f"  New best model saved: {best_path} (loss={total_loss:.4f})")

    def _load_checkpoint(self, path: str) -> None:
        from ..utils.checkpoint_compat import migrate_vq_vae_state_dict

        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt["model_state_dict"]
        migrate_vq_vae_state_dict(state)
        self.model.load_state_dict(state)
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        if "best_total_loss" in ckpt:
            self.best_total_loss = ckpt["best_total_loss"]
        if self.kmeans_init_after_warmup:
            if "kmeans_init_done" in ckpt:
                self._kmeans_init_done = bool(ckpt["kmeans_init_done"])
            else:
                self._kmeans_init_done = (
                    self.global_step > self._kmeans_schedule_step()
                )
        self._tee(f"Resumed from {path} at step {self.global_step}")