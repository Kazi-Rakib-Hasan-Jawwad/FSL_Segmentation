"""
losses.py — Stable Loss Functions for 3-Class Segmentation
============================================================

Provides Cross-Entropy, Dice, and Focal losses with proper numerical stability.
All losses support multi-class segmentation with ignore_index=255.

Tensor flow:
    logits: (B, C, H, W) — raw class logits from decoder (C=3)
    target: (B, H, W)    — ground truth {0, 1, 2, 255=ignore}
    ↓ Masked CE + Dice + Focal
    loss: scalar

References:
    - Focal Loss: Lin et al., ICCV 2017
    - Dice Loss: Milletari et al., 3DV 2016
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss — differentiable, handles class imbalance well.

    dice = 2 * |P ∩ G| / (|P| + |G|)
    loss = 1 - dice  (averaged over all classes, excluding ignore)
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) — raw logits
            target: (B, H, W)    — ground truth

        Returns:
            loss: scalar — mean Dice loss over all classes
        """
        B, C, H, W = logits.shape

        valid = (target != self.ignore_index)  # (B, H, W)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        target_clean = target.clone()
        target_clean[~valid] = 0
        target_onehot = F.one_hot(target_clean, C).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        # Per-class Dice
        dims = (0, 2, 3)  # Sum over batch and spatial
        intersection = (probs * target_onehot).sum(dim=dims)  # (C,)
        p_sum = probs.sum(dim=dims)
        t_sum = target_onehot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class weights.

    FL(p_t) = -α_c · (1 - p_t)^γ · log(p_t)

    When γ > 0, easy examples (high p_t) are down-weighted.
    Per-class α_c allows balancing between tumor, stroma, other.

    References:
        Lin et al., ICCV 2017
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
            class_weights: Per-class weights [w_tumor, w_stroma, w_other].
                           None = uniform weighting.
            ignore_index: Value to ignore in target.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            target: (B, H, W)

        Returns:
            loss: scalar
        """
        B, C, H, W = logits.shape
        valid = target != self.ignore_index

        logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (BHW, C)
        target_flat = target.view(-1)  # (BHW,)
        valid_flat = valid.view(-1)    # (BHW,)

        logits_v = logits_flat[valid_flat]  # (N, C)
        target_v = target_flat[valid_flat]  # (N,)

        if target_v.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(logits_v, dim=1)  # (N, C)
        log_pt = log_probs.gather(1, target_v.unsqueeze(1)).squeeze(1)  # (N,)
        pt = log_pt.exp()  # (N,)

        # Focal weighting
        loss = -((1 - pt) ** self.gamma) * log_pt  # (N,)

        # Per-class weighting
        if self.class_weights is not None:
            alpha_t = self.class_weights[target_v]  # (N,)
            loss = alpha_t * loss

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: CE + Dice + Focal.

    All loss components are properly masked for ignore_index=255.
    Supports multi-class segmentation with per-class focal weights.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(
            gamma=focal_gamma,
            class_weights=focal_class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: (B, C, H, W) — raw class logits (C=3)
            target: (B, H, W)    — ground truth

        Returns:
            Dict with "loss" (total), "ce", "dice", "focal"
        """
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        focal = self.focal(logits, target)

        total = self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal

        return {
            "loss": total,
            "ce": ce.detach(),
            "dice": dice.detach(),
            "focal": focal.detach(),
        }
