"""
trainer.py — Training Engine with 3-Class Diagnostics
======================================================

Orchestrates the training loop with per-class metrics tracking.

Primary metric: mDice_soft (macro soft Dice over tumor, stroma, other).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.episodic_sampler import EpisodicSampler
from models.losses import CombinedLoss
from utils.checkpointing import save_checkpoint
from .evaluator import Evaluator


class Trainer:
    """
    Main training engine for 3-class few-shot segmentation.

    Usage:
        trainer = Trainer(model, train_sampler, val_sampler, config)
        trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_sampler: EpisodicSampler,
        val_sampler: EpisodicSampler,
        loss_fn: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: dict,
        device: str = "cuda",
        output_dir: str = "./runs",
    ):
        self.model = model
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluator
        self.evaluator = Evaluator(
            val_sampler, device=device,
            ignore_index=config.get("ignore_index", 255),
            num_classes=config.get("num_classes", 3),
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        # Training state
        self.best_metric = 0.0
        self.best_epoch = -1
        self.grad_clip = config.get("grad_clip", 5.0)

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Main training loop.

        For each epoch:
            1. Re-seed episodes
            2. Train on all episodes
            3. Evaluate with per-class metrics
            4. Save checkpoint if best mDice_soft
            5. Update learning rate schedule
        """
        print(f"\n{'='*70}")
        print(f"  Training: {num_epochs} epochs, "
              f"{len(self.train_sampler)} train / {len(self.val_sampler)} val episodes")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}, "
              f"LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"  Grad clip: {self.grad_clip}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            # ── 1. Re-seed episodes ─────────────────────────────────────
            self.train_sampler.reset_seed(epoch)

            # ── 2. Train ────────────────────────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── 3. Evaluate ─────────────────────────────────────────────
            val_metrics = self.evaluator.evaluate(
                self.model, epoch=epoch, loss_fn=self.loss_fn
            )

            # ── 4. Log ──────────────────────────────────────────────────
            elapsed = time.time() - epoch_start
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed)

            # ── 5. Checkpoint (primary metric: mDice_soft) ──────────────
            primary_metric = val_metrics.get("mDice_soft", 0.0)
            is_best = primary_metric > self.best_metric

            if is_best:
                self.best_metric = primary_metric
                self.best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, self.best_metric,
                    str(self.output_dir / "best_model.pth"),
                )
                print(f"  ★ New best: mDice_soft = {primary_metric:.4f}")

            # Also save latest
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_metrics, self.best_metric,
                str(self.output_dir / "latest.pth"),
            )

            # ── 6. LR schedule ──────────────────────────────────────────
            if self.scheduler is not None:
                self.scheduler.step()

        print(f"\n{'='*70}")
        print(f"  Training complete. Best: epoch {self.best_epoch}, "
              f"mDice_soft = {self.best_metric:.4f}")
        print(f"{'='*70}")

        self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch of episodes."""
        self.model.train()

        accum = {"loss": 0, "ce": 0, "dice": 0, "focal": 0, "grad_norm": 0}
        n = 0
        nan_count = 0

        loader = DataLoader(self.train_sampler, batch_size=None, shuffle=False, num_workers=0)
        print_every = self.config.get("print_every", 10)

        for step, episode in enumerate(loader):
            # Move to device
            sprt_imgs = episode.support_images.to(self.device)   # (K, 3, H, W)
            sprt_masks = episode.support_masks.to(self.device)   # (K, H, W)
            q_img = episode.query_image.unsqueeze(0).to(self.device)  # (1, 3, H, W)
            q_mask = episode.query_mask.unsqueeze(0).to(self.device)  # (1, H, W)

            # ── Forward ─────────────────────────────────────────────────
            output = self.model(sprt_imgs, sprt_masks, q_img)
            logits = output["logits"]  # (1, 3, H, W)

            # ── Loss ────────────────────────────────────────────────────
            loss_dict = self.loss_fn(logits, q_mask)
            loss = loss_dict["loss"]

            # ── NaN guard ───────────────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 5:
                    print(f"  [WARN] {nan_count} NaN losses in epoch {epoch}!")
                self.optimizer.zero_grad()
                continue

            # ── Backward ────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            ).item()

            if not torch.isfinite(torch.tensor(grad_norm)):
                nan_count += 1
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()

            # ── Accumulate ──────────────────────────────────────────────
            accum["loss"] += loss.item()
            accum["ce"] += loss_dict["ce"].item()
            accum["dice"] += loss_dict["dice"].item()
            accum["focal"] += loss_dict["focal"].item()
            accum["grad_norm"] += grad_norm
            n += 1

            # ── Print ───────────────────────────────────────────────────
            if (step + 1) % print_every == 0:
                avg_loss = accum["loss"] / n
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} | Step {step+1}/{len(self.train_sampler)} | "
                      f"Loss {avg_loss:.4f} | GradNorm {grad_norm:.3f} | LR {lr:.1e}")

        # Average
        avg = {k: v / max(n, 1) for k, v in accum.items()}
        if nan_count > 0:
            avg["nan_count"] = nan_count
        return avg

    def _log_epoch(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict, elapsed: float
    ):
        """Log metrics to console and TensorBoard."""

        # Console — per-class metrics
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"\n  Epoch {epoch:3d} | "
              f"train_loss {train_metrics['loss']:.4f} | "
              f"mIoU {val_metrics.get('mIoU', 0):.4f} | "
              f"mDice_soft {val_metrics.get('mDice_soft', 0):.4f} | "
              f"IoU_T {val_metrics.get('IoU_tumor', 0):.4f} "
              f"IoU_S {val_metrics.get('IoU_stroma', 0):.4f} "
              f"IoU_O {val_metrics.get('IoU_other', 0):.4f} | "
              f"pixAcc {val_metrics.get('pixel_acc', 0):.4f} | "
              f"LR {lr:.1e} | "
              f"time {elapsed:.1f}s")

        # TensorBoard
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)

        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", v, epoch)

        self.writer.add_scalar("lr", lr, epoch)
        self.writer.flush()
