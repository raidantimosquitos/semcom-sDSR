"""
Stage 2 trainer: sDSR anomaly detection.

Model interface: forward_train(x, M_gt) -> dict with m_out, x_s, M, x
Requires pre-trained encoder (vq_vae) loaded from Stage 1.

Reconstruction and subspace restriction losses are computed on the full batch
(all samples: normal and synthetically anomalous) so that at inference the
object decoder and subspace behave correctly for both normal and anomalous inputs.
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
        m_out, x_s, M (focal target; latent-snapped), x
    Only trainable params (object_decoder, anomaly_detection) are optimized.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        machine_type: str = "unknown",
        machine_id: str | None = None,
        lambda_recon: float = 10.0,
        lambda_focal: float = 1.0,
        lambda_sub: float = 1.0,
        lr: float = 2e-4,
        lr_min: float = 1e-5,
        batch_size: int = 16,
        grad_clip: float | None = 1.0,
        log_every: int = 50,
        ckpt_every: int = 500,
        ckpt_dir: str | Path = "./checkpoints",
        use_amp: bool = True,
        device: str = "cuda",
        total_steps: int = 20000,
        val_dataset: Any | None = None,
        val_every: int = 1000,
        val_batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.lambda_recon = lambda_recon
        self.lambda_focal = lambda_focal
        self.lambda_sub = lambda_sub
        self.lr = lr
        self.lr_min = lr_min
        self.total_steps = total_steps

        # Dataset run name must match base.machine_type (single type or joined multi-type)
        if getattr(dataset, "machine_type", None) not in (None, machine_type):
            raise ValueError(f"Dataset machine_type '{getattr(dataset, 'machine_type')}' != trainer machine_type '{machine_type}'")

        ckpt_path = Path(ckpt_dir) / "stage2" / machine_type
        if machine_id is not None:
            ckpt_path = ckpt_path / machine_id
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

        # Optimizer for trainable params only (object_decoder, anomaly_detection)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        from ..models.sDSR.loss import FocalLoss
        self.focal_loss = FocalLoss(gamma=2.0)

        self.best_total_loss = float("inf")
        self._last_ckpt_path: Path | None = None

        self._val_dataset = val_dataset
        self._val_every = val_every
        self._val_batch_size = val_batch_size
        self.best_val_auc: float = 0.0
        # Previous best-val checkpoint path per metric (deleted when a new best is saved).
        self._val_best_ckpt_paths: dict[str, Path] = {}

        n_params = sum(p.numel() for p in trainable)
        self._tee(
            f"Stage2 | Device: {self.device} | AMP: {self.use_amp} | "
            f"DataLoader workers: {self.num_workers} | Trainable params: {n_params:,}"
        )

    def _get_lr(self, step: int, total_steps: int) -> float:
        progress = step / max(1, total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + cosine * (self.lr - self.lr_min)

    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        """
        One training step. Reconstruction and subspace losses are computed on the
        full batch (all samples: normal and synthetically anomalous) so that the
        object decoder and subspace restriction module learn identity-like behavior
        on normal codes and correction on anomalous codes.
        """
        x = batch["image"].to(self.device, non_blocking=True)
        M_gt = batch["anomaly_mask"].to(self.device, non_blocking=True)

        lr = self._get_lr(step, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model.forward_train(x, M_gt=M_gt)
            m_out = out["m_out"]
            x_specific = out["x_specific"]
            M = out["M"]                

            # Reconstruction: L2(x, x_specific) on full batch (normal + anomalous).
            # For normal samples: object decoder learns x_specific ≈ x. For anomalous:
            # decoder learns to reconstruct original normal x from corrected codes.
            loss_recon = F.mse_loss(out["x"], x_specific)
            loss_focal = self.focal_loss(m_out, M)

            total_loss = (
                self.lambda_recon * loss_recon
                + self.lambda_focal * loss_focal
            )

            # Subspace restriction: L2(recon_feat, q) on full batch (normal + anomalous).
            # For normal samples: subspace learns identity. For anomalous: learns to
            # map modified codes back to clean q. Requires model to return aux from
            # object decoder when subspace is enabled.
            has_subspace_aux = (
                "recon_feat_fine" in out
                and "recon_feat_coarse" in out
                and "q_fine" in out
                and "q_coarse" in out
            )
            if self.lambda_sub != 0 and has_subspace_aux:
                loss_sub_fine = F.mse_loss(
                    out["recon_feat_fine"], out["q_fine"].detach()
                )
                loss_sub_coarse = F.mse_loss(
                    out["recon_feat_coarse"], out["q_coarse"].detach()
                )
                loss_sub = loss_sub_fine + loss_sub_coarse
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
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if "best_total_loss" in ckpt:
            self.best_total_loss = ckpt["best_total_loss"]
        self._tee(f"Resumed from {path} at step {self.global_step}")

    def _on_training_end(self) -> None:
        """Skip final checkpoint; validation-selected models are preferred."""
        pass

    # ------------------------------------------------------------------
    # Validation hook: evaluate on real test set and save best models
    # ------------------------------------------------------------------

    def _post_checkpoint_hook(self) -> None:
        if self._val_dataset is None:
            return
        if self.global_step % self._val_every != 0:
            return
        self._run_validation()

    def _run_validation(self) -> None:
        from .evaluator import AnomalyEvaluator

        self.model.eval()
        val_id = self.machine_id
        evaluator = AnomalyEvaluator(
            model=self.model,
            test_dataset=self._val_dataset,
            device=str(self.device),
            batch_size=self._val_batch_size,
            train_score_stats=None,
            subset_machine_id=val_id,
        )
        results = evaluator.evaluate()

        for run_name, ids in results.items():
            tag = f" (machine_id={val_id})" if val_id else ""
            self._tee(f"  [val@{self.global_step}] {run_name}{tag}:")
            for k, v in ids.items():
                if isinstance(v, dict):
                    if val_id and k == "average":
                        continue
                    self._tee(f"    {k}: AUC={v['auc']:.4f} pAUC={v['pauc']:.4f}")

        avg = None
        for run_name, ids in results.items():
            if "average" in ids:
                avg = ids["average"]
                break

        if avg is not None:
            mean_auc = avg["auc"]
            mean_pauc = avg["pauc"]
            label = f"machine_id={val_id}" if val_id else "average"
            self._tee(
                f"  [val@{self.global_step}] val metric ({label}) AUC={mean_auc:.4f} pAUC={mean_pauc:.4f}  "
                f"(best AUC={self.best_val_auc:.4f})"
            )
            if val_id and not math.isfinite(mean_auc):
                self._tee(
                    f"  [val@{self.global_step}] warning: no test samples for machine_id={val_id!r}; "
                    "skipping val-best checkpoint update"
                )
            elif math.isfinite(mean_auc):
                self._save_val_best(mean_auc, "best_val_auc", "best_auc", "AUC")

        self.model.train()

    def _save_val_best(
        self, value: float, attr: str, suffix: str, label: str,
    ) -> None:
        if not math.isfinite(value):
            return
        prev = getattr(self, attr)
        if value <= prev:
            return
        setattr(self, attr, value)
        payload = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            f"{attr}": value,
        }
        old = self._val_best_ckpt_paths.get(suffix)
        if old is not None and old.exists():
            old.unlink()
            self._tee(f"  Deleted previous val-{label} checkpoint: {old}")
        path = (
            self.ckpt_dir
            / f"stage2_{self.machine_type}_{suffix}_iter_{self.global_step:07d}.pt"
        )
        torch.save(payload, path)
        self._val_best_ckpt_paths[suffix] = path
        self._tee(
            f"  New best val {label} model saved: {path} "
            f"({label}={value:.4f}, iter={self.global_step})"
        )
