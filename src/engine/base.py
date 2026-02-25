"""
Base trainer: shared training loop, checkpointing, and logging.

Designed for generalization: subclasses implement _step(batch, step, total_steps).
Model interfaces (for swapping encoder/DSR implementations):
  - Stage 1: forward(x) -> (loss_b, loss_t, recon, q_top, q_bot, perp_t, perp_b)
  - Stage 2: forward_train(x) -> dict with m_out, x_s, M, x
  - Evaluation: forward(x) -> M_out (B, 2, H, W) segmentation logits
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class BaseTrainer(ABC):
    """
    Abstract base trainer. Subclasses must implement _step(batch, step, total_steps).

    Checkpoint format (consistent across stages):
        {"global_step": int, "model_state_dict": ..., "optim_state_dict": ...,
         "scaler_state_dict": ..., "config": ...}
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        device: str | torch.device = "cuda",
        ckpt_dir: str | Path = "./checkpoints",
        log_every: int = 100,
        ckpt_every: int = 2000,
        batch_size: int = 64,
        grad_clip: float | None = 1.0,
        use_amp: bool = True,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.ckpt_dir / "train.log", "w", encoding="utf-8")
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        self.global_step = 0
        self._data_iter = iter(self.loader)
        self._last_avg: dict[str, float] | None = None

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "tb_logs"))
        else:
            self.writer = None

    def _tee(self, msg: str) -> None:
        """Print to terminal and append to train.log."""
        print(msg)
        self._log_file.write(msg + "\n")
        self._log_file.flush()

    def _next_batch(self) -> Any:
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.loader)
            batch = next(self._data_iter)
        return batch

    @abstractmethod
    def _step(self, batch: Any, step: int, total_steps: int) -> dict[str, float]:
        """Perform one training step. Return loss_dict for logging."""
        raise NotImplementedError

    def train(
        self,
        n_iterations: int,
        resume_from: str | None = None,
    ) -> None:
        """Run training loop for n_iterations."""
        try:
            if resume_from:
                self._load_checkpoint(resume_from)

            self.model.train()
            t0 = time.time()
            loss_accum: dict[str, float] = {}

            while self.global_step < n_iterations:
                batch = self._next_batch()
                loss_dict = self._step(batch, self.global_step, n_iterations)
                self.global_step += 1

                for k, v in loss_dict.items():
                    loss_accum[k] = loss_accum.get(k, 0.0) + v

                if self.global_step % self.log_every == 0:
                    elapsed = time.time() - t0
                    avg = {k: v / self.log_every for k, v in loss_accum.items()}
                    self._last_avg = avg
                    its_sec = self.log_every / elapsed
                    self._log(avg, its_sec)
                    if self.writer:
                        for k, v in avg.items():
                            self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    loss_accum = {}
                    t0 = time.time()

                if self.global_step % self.ckpt_every == 0:
                    self._save_checkpoint(tag=None, avg=self._last_avg)

            self._save_checkpoint(tag="final", avg=self._last_avg)
        finally:
            self._log_file.close()

    def _log(self, avg: dict[str, float], its_sec: float) -> None:
        """Override in subclass for custom logging. Default: print keys and values."""
        parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
        print(f"[{self.global_step:>6d}] {parts}  ({its_sec:.1f} it/s)")

    @abstractmethod
    def _save_checkpoint(self, tag: str | None = None, avg: dict[str, float] | None = None) -> None:
        """Save checkpoint. Subclass sets path and contents. avg contains logged metrics (e.g. total loss) for best-model tracking."""
        raise NotImplementedError

    @abstractmethod
    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint. Subclass sets how to restore state."""
        raise NotImplementedError
