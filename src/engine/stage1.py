"""
Stage 1 trainer: VQ-VAE-2 reconstruction.

Model interface: forward(x) -> (loss_fine, loss_coarse, recon, q_coarse, q_fine, perplexity_coarse, perplexity_fine)
Any encoder that returns this tuple can be used.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from ..models.vq_vae.autoencoders import VQ_VAE_2Layer

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
        lambda_recon: float = 1.0,
        lr: float = 2e-4,
        batch_size: int = 256,
        grad_clip: float | None = 1.0,
        log_every: int = 100,
        ckpt_every: int = 2000,
        ckpt_dir: str | Path = "./checkpoints",
        use_amp: bool = True,
        device: str = "cuda",
        total_steps: int = 20000,
        num_workers: int = 0,
    ) -> None:
        self.machine_type = machine_type
        self.lambda_recon = lambda_recon
        self.lr = lr
        self.dataset = dataset
        self.total_steps = total_steps

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
        ts = max(1, int(total_steps))
        m80, m90 = int(0.8 * ts), int(0.9 * ts)
        milestones = sorted({m for m in (m80, m90) if m > 0})
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1
        )
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.best_total_loss = float("inf")
        self._last_ckpt_path: Path | None = None
        self._last_finite_log: dict[str, float] | None = None
        self._nonfinite_step_warned = False

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._tee(
            f"Stage1 | Device: {self.device} | AMP: {self.use_amp} | "
            f"DataLoader workers: {self.num_workers} | Params: {n_params:,} | "
            f"LR: MultiStepLR milestones={milestones} gamma=0.1"
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model(x)
            loss_fine, loss_coarse, recon, _, _, perp_coarse, perp_fine = out

        # L2 recon in fp32 (avoids fp16 MSE blow-ups in decoder under AMP)
        with autocast(device_type=self.device.type, enabled=False):
            recon_loss = F.mse_loss(recon.float(), x.float())
            total_loss = (
                loss_fine.float()
                + loss_coarse.float()
                + self.lambda_recon * recon_loss
            )

        if torch.isfinite(total_loss).all():
            self.scaler.scale(total_loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            scale_before = float(self.scaler.get_scale()) if self.use_amp else 1.0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = float(self.scaler.get_scale()) if self.use_amp else 1.0
            if scale_after >= scale_before:
                self.lr_scheduler.step()

            lr = self._current_lr()
            metrics = {
                "total": float(total_loss.detach().item()),
                "recon": float(recon_loss.detach().item()),
                "loss_fine": float(loss_fine.detach().item()),
                "loss_coarse": float(loss_coarse.detach().item()),
                "perplexity_fine": float(perp_fine.detach().item()),
                "perplexity_coarse": float(perp_coarse.detach().item()),
                "lr": lr,
            }
            self._last_finite_log = metrics
            return metrics

        # If AMP overflowed before any scaler.step(), GradScaler may not have recorded
        # inf checks; calling update() would raise:
        #   AssertionError: No inf checks were recorded prior to update.
        # In that case we simply skip the optimizer step and keep going.
        if not self._nonfinite_step_warned:
            self._tee(
                "Warning: non-finite total loss; skipping optimizer step "
                "(decoder/recon path overflowed). If this repeats, try --no_amp or lower lr."
            )
            self._nonfinite_step_warned = True
        lr_log = self._current_lr()
        if self._last_finite_log is not None:
            fallback = dict(self._last_finite_log)
            fallback["lr"] = lr_log
            return fallback
        return {
            "total": float("nan"),
            "recon": float("nan"),
            "loss_fine": float(loss_fine.detach().item()),
            "loss_coarse": float(loss_coarse.detach().item()),
            "perplexity_fine": float(perp_fine.detach().item()),
            "perplexity_coarse": float(perp_coarse.detach().item()),
            "lr": lr_log,
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
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_total_loss": self.best_total_loss,
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
        # Normalization is disabled project-wide: checkpoints do not carry norm stats.
        payload["spectrogram_standardize"] = False
        payload["spectrogram_norm_type"] = "none"

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

        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt["model_state_dict"]
        self.model.load_state_dict(state)
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        if "best_total_loss" in ckpt:
            self.best_total_loss = ckpt["best_total_loss"]
        if "lr_scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        elif self.global_step > 0:
            self.lr_scheduler.last_epoch = int(self.global_step) - 1
            self._tee(
                "Warning: checkpoint has no lr_scheduler_state_dict; "
                "scheduler last_epoch aligned to global_step (LR may differ from original run)."
            )
        self._tee(f"Resumed from {path} at step {self.global_step}")

    def _extract_x(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    @staticmethod
    def _vq_raw_latent_sse_and_counts(
        *,
        z: torch.Tensor,
        embedding_weight: torch.Tensor,
        indices_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Compute raw latent SSE between pre-quant features and nearest embeddings.

        Matches quantizer math:
          inputs: (B, C, H, W) -> permute to (B, H, W, C) -> flatten to (N, C)
          quantized: embedding[indices] -> view back to (B, H, W, C)

        Returns:
          sse: scalar float64 tensor on device
          n: number of latent elements contributing to SSE
        """
        z32 = z.float()
        z_hw_c = z32.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = z_hw_c.shape
        n_tokens = int(B * H * W)
        emb = embedding_weight.float()
        nearest = emb[indices_flat]  # (N, C)
        nearest = nearest.view(B, H, W, C)
        diff2 = (nearest - z_hw_c).to(torch.float64).pow(2)
        sse = diff2.sum()
        n_elems = int(n_tokens * C)
        return sse, n_elems

    @staticmethod
    def _perplexity_from_counts(counts: torch.Tensor, eps: float = 1e-10) -> float:
        counts64 = counts.to(torch.float64)
        total = float(counts64.sum().item())
        if total <= 0:
            return float("nan")
        p = counts64 / total
        ent = -(p * (p + eps).log()).sum()
        return float(torch.exp(ent).item())

    def _compute_trainset_metrics_frozen(self) -> dict[str, float]:
        if not isinstance(self.model, VQ_VAE_2Layer):
            raise TypeError(
                "Stage1 final metrics (global perplexity + raw quant MSE) currently "
                f"require VQ_VAE_2Layer; got {type(self.model).__name__}"
            )

        eval_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        Kc = int(self.model.num_embeddings_coarse)
        Kf = int(self.model.num_embeddings_fine)
        counts_coarse = torch.zeros(Kc, dtype=torch.int64, device=self.device)
        counts_fine = torch.zeros(Kf, dtype=torch.int64, device=self.device)

        recon_sse = torch.zeros((), dtype=torch.float64, device=self.device)
        recon_n = 0

        commit_sum_coarse = torch.zeros((), dtype=torch.float64, device=self.device)
        commit_sum_fine = torch.zeros((), dtype=torch.float64, device=self.device)
        commit_tokens_coarse = 0
        commit_tokens_fine = 0

        raw_sse_coarse = torch.zeros((), dtype=torch.float64, device=self.device)
        raw_sse_fine = torch.zeros((), dtype=torch.float64, device=self.device)
        raw_n_coarse = 0
        raw_n_fine = 0

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                x = self._extract_x(batch).to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss_fine, loss_coarse, recon, q_coarse, q_fine, _, _ = self.model(x)

                diff = (recon.float() - x.float()).to(torch.float64)
                recon_sse += (diff * diff).sum()
                recon_n += int(x.numel())

                B = int(x.shape[0])
                Hc, Wc = int(q_coarse.shape[-2]), int(q_coarse.shape[-1])
                Hf, Wf = int(q_fine.shape[-2]), int(q_fine.shape[-1])
                n_tokens_coarse = int(B * Hc * Wc)
                n_tokens_fine = int(B * Hf * Wf)

                commit_sum_coarse += loss_coarse.to(torch.float64) * n_tokens_coarse
                commit_sum_fine += loss_fine.to(torch.float64) * n_tokens_fine
                commit_tokens_coarse += n_tokens_coarse
                commit_tokens_fine += n_tokens_fine

                _, _, _, _, z_fine, z_coarse = self.model.encode_with_prequant(x)
                idx_coarse = self.model._vq_coarse.get_indices(z_coarse)
                idx_fine = self.model._vq_fine.get_indices(z_fine)

                counts_coarse += torch.bincount(idx_coarse, minlength=Kc)
                counts_fine += torch.bincount(idx_fine, minlength=Kf)

                sse_c, n_c = self._vq_raw_latent_sse_and_counts(
                    z=z_coarse,
                    embedding_weight=self.model._vq_coarse._embedding.weight,
                    indices_flat=idx_coarse,
                )
                sse_f, n_f = self._vq_raw_latent_sse_and_counts(
                    z=z_fine,
                    embedding_weight=self.model._vq_fine._embedding.weight,
                    indices_flat=idx_fine,
                )
                raw_sse_coarse += sse_c
                raw_sse_fine += sse_f
                raw_n_coarse += n_c
                raw_n_fine += n_f

        if was_training:
            self.model.train()

        recon_mse = float((recon_sse / max(recon_n, 1)).item())
        perp_coarse = self._perplexity_from_counts(counts_coarse)
        perp_fine = self._perplexity_from_counts(counts_fine)
        commit_loss_coarse = float((commit_sum_coarse / max(commit_tokens_coarse, 1)).item())
        commit_loss_fine = float((commit_sum_fine / max(commit_tokens_fine, 1)).item())
        raw_latent_mse_coarse = float((raw_sse_coarse / max(raw_n_coarse, 1)).item())
        raw_latent_mse_fine = float((raw_sse_fine / max(raw_n_fine, 1)).item())

        return {
            "recon_mse": recon_mse,
            "perplexity_coarse": perp_coarse,
            "perplexity_fine": perp_fine,
            "commit_loss_coarse": commit_loss_coarse,
            "commit_loss_fine": commit_loss_fine,
            "raw_latent_mse_coarse": raw_latent_mse_coarse,
            "raw_latent_mse_fine": raw_latent_mse_fine,
        }

    def _on_training_end(self) -> None:
        metrics = self._compute_trainset_metrics_frozen()
        self._tee("Stage1 final train metrics (frozen model, full train set):")
        self._tee(f"  recon_mse={metrics['recon_mse']:.8f}")
        self._tee(
            f"  perplexity_fine={metrics['perplexity_fine']:.4f}  "
            f"perplexity_coarse={metrics['perplexity_coarse']:.4f}"
        )
        self._tee(
            f"  commit_loss_fine={metrics['commit_loss_fine']:.8f}  "
            f"commit_loss_coarse={metrics['commit_loss_coarse']:.8f}"
        )
        self._tee(
            f"  raw_latent_mse_fine={metrics['raw_latent_mse_fine']:.8f}  "
            f"raw_latent_mse_coarse={metrics['raw_latent_mse_coarse']:.8f}"
        )
        self._save_checkpoint(tag="final", avg=self._last_avg)