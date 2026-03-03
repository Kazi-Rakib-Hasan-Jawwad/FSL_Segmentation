"""
metrics.py — Per-Class Hard and Soft Evaluation Metrics (3-Class)
================================================================

Provides both argmax-based (hard) and probability-based (soft) metrics
for 3-class segmentation: Tumor (0), Stroma (1), Other (2).

All metrics are computed per-class and then macro-averaged.
No class is treated as "background" — all 3 are target classes.

All functions operate on tensors and return Python floats.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

CLASS_NAMES = {0: "tumor", 1: "stroma", 2: "other"}


def compute_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    num_classes: int = 3,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute per-class IoU and Dice using argmax predictions.

    Args:
        logits: (B, C, H, W) — raw logits
        target: (B, H, W)    — ground truth
        ignore_index: Value to ignore
        num_classes: Number of classes (3)

    Returns:
        Dict with mIoU, mDice, per-class IoU_tumor/stroma/other, Dice_tumor/stroma/other
    """
    pred = logits.argmax(dim=1)  # (B, H, W)
    valid = target != ignore_index
    pred_v = pred[valid]
    target_v = target[valid]

    ious = []
    dices = []
    result = {}
    for c in range(num_classes):
        p = (pred_v == c)
        t = (target_v == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        denom = p.sum().float() + t.sum().float()

        iou = (inter + eps) / (union + eps)
        dice = (2 * inter + eps) / (denom + eps)
        ious.append(iou.item())
        dices.append(dice.item())

        name = CLASS_NAMES.get(c, f"c{c}")
        result[f"IoU_{name}"] = iou.item()
        result[f"Dice_{name}"] = dice.item()

    result["mIoU"] = sum(ious) / len(ious)
    result["mDice"] = sum(dices) / len(dices)

    return result


def compute_soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> Dict[str, float]:
    """
    Macro soft Dice score over all classes using probabilities (not argmax).

    Computes per-class soft Dice and returns macro average + per-class values.

    Args:
        logits: (B, C, H, W)
        target: (B, H, W)
        num_classes: 3

    Returns:
        Dict with mDice_soft, soft_dice_tumor, soft_dice_stroma, soft_dice_other
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    valid = (target != ignore_index).float()  # (B, H, W)

    result = {}
    per_class = []

    for c in range(num_classes):
        c_prob = probs[:, c] * valid   # (B, H, W)
        c_target = (target == c).float() * valid  # (B, H, W)

        intersection = (c_prob * c_target).sum()
        dice = (2 * intersection + smooth) / (c_prob.sum() + c_target.sum() + smooth)

        name = CLASS_NAMES.get(c, f"c{c}")
        result[f"soft_dice_{name}"] = dice.item()
        per_class.append(dice.item())

    result["mDice_soft"] = sum(per_class) / len(per_class)
    return result


def compute_mean_class_prob(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Mean predicted probability for each class on its true pixels.

    For each class c: mean P(c) on pixels where target == c.

    Args:
        logits: (B, C, H, W)
        target: (B, H, W)

    Returns:
        Dict with mean_prob_tumor, mean_prob_stroma, mean_prob_other
    """
    probs = F.softmax(logits, dim=1)
    result = {}

    for c in range(num_classes):
        c_mask = (target == c) & (target != ignore_index)
        if c_mask.sum() == 0:
            result[f"mean_prob_{CLASS_NAMES.get(c, f'c{c}')}"] = 0.0
            continue
        c_prob = probs[:, c]  # (B, H, W)
        result[f"mean_prob_{CLASS_NAMES.get(c, f'c{c}')}"] = c_prob[c_mask].mean().item()

    return result


def compute_pixel_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """
    Overall pixel accuracy (fraction of correctly classified valid pixels).

    Args:
        logits: (B, C, H, W)
        target: (B, H, W)

    Returns:
        Pixel accuracy ∈ [0, 1]
    """
    pred = logits.argmax(dim=1)
    valid = target != ignore_index

    if valid.sum() == 0:
        return 0.0

    correct = (pred[valid] == target[valid]).float()
    return correct.mean().item()


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute ALL metrics (hard + soft) in a single call.

    Returns dict with:
        - Hard: mIoU, mDice, IoU_tumor, IoU_stroma, IoU_other, Dice_tumor, ...
        - Soft: mDice_soft, soft_dice_tumor, soft_dice_stroma, soft_dice_other
        - Probs: mean_prob_tumor, mean_prob_stroma, mean_prob_other
        - pixel_acc
    """
    hard = compute_iou(logits, target, ignore_index, num_classes)
    soft = compute_soft_dice(logits, target, ignore_index, num_classes)
    probs = compute_mean_class_prob(logits, target, ignore_index, num_classes)
    px_acc = compute_pixel_accuracy(logits, target, ignore_index)

    return {
        **hard,
        **soft,
        **probs,
        "pixel_acc": px_acc,
    }
