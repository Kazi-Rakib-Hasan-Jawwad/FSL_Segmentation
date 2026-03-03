"""
evaluator.py — Evaluation Engine with Per-Class Metrics (3-Class)
=================================================================

Runs deterministic evaluation episodes and computes per-class metrics:
    IoU_tumor, IoU_stroma, IoU_other, mIoU, mDice_soft, etc.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data.episodic_sampler import EpisodicSampler, Episode
from utils.metrics import compute_metrics


class Evaluator:
    """
    Evaluates a FewShotSegmentor on a set of 3-class episodes.

    Usage:
        evaluator = Evaluator(val_sampler, device="cuda", num_classes=3)
        metrics = evaluator.evaluate(model, epoch=5)
    """

    def __init__(
        self,
        sampler: EpisodicSampler,
        device: str = "cuda",
        ignore_index: int = 255,
        num_classes: int = 3,
    ):
        self.sampler = sampler
        self.device = device
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        epoch: int = 0,
        loss_fn=None,
    ) -> Dict[str, float]:
        """
        Run evaluation on all episodes.

        Args:
            model: FewShotSegmentor
            epoch: Current epoch (used for deterministic seed)
            loss_fn: Optional loss function to compute val loss

        Returns:
            Dict with averaged per-class metrics:
                IoU_tumor, IoU_stroma, IoU_other, mIoU,
                mDice_soft, soft_dice_tumor, soft_dice_stroma, soft_dice_other,
                pixel_acc, val_loss
        """
        model.eval()

        # Fixed seed for reproducible validation episodes
        self.sampler.reset_seed(epoch=99999)

        accum = {}
        n_episodes = 0

        loader = DataLoader(self.sampler, batch_size=None, shuffle=False, num_workers=0)

        for episode in loader:
            support_images = episode.support_images.to(self.device)
            support_masks = episode.support_masks.to(self.device)
            query_image = episode.query_image.unsqueeze(0).to(self.device)
            query_mask = episode.query_mask.unsqueeze(0).to(self.device)

            # Forward pass
            output = model(support_images, support_masks, query_image)
            logits = output["logits"]  # (1, 3, H, W)

            # Compute per-class metrics
            episode_metrics = compute_metrics(
                logits, query_mask,
                ignore_index=self.ignore_index,
                num_classes=self.num_classes,
            )

            # Compute loss
            if loss_fn is not None:
                loss_dict = loss_fn(logits, query_mask)
                episode_metrics["val_loss"] = loss_dict["loss"].item()

            # Accumulate
            for k, v in episode_metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n_episodes += 1

        # Average
        avg_metrics = {k: v / max(n_episodes, 1) for k, v in accum.items()}
        return avg_metrics
