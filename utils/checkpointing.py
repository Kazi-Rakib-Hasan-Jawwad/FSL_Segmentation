"""
checkpointing.py — Checkpoint Save/Load with Metadata
=====================================================

Saves checkpoints with full training state AND metadata about the
best metric values, making it easy to resume training or select
the best model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    best_metric: float,
    save_path: str,
    extra: dict = None,
):
    """
    Save a training checkpoint.

    Args:
        model: The model (state_dict saved)
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch number
        metrics: Dict of metric values at this epoch
        best_metric: Best primary metric value so far
        save_path: Path to save the checkpoint
        extra: Any additional data to save
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "best_metric": best_metric,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cuda",
) -> Dict:
    """
    Load a checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map tensors to

    Returns:
        Dict with "epoch", "metrics", "best_metric", and any extra data.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "best_metric": checkpoint.get("best_metric", 0.0),
    }
