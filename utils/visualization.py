"""
visualization.py — 3-Class Prediction Overlays and Episode Visualization
=========================================================================

Always uses CLASS_COLORS (tumor=red, stroma=green, other=blue).
No more binary overlay mode.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Color map for TIGER classes — always used
CLASS_COLORS = {
    0: (220, 50, 50),     # Tumor — red
    1: (50, 180, 50),     # Stroma — green
    2: (50, 100, 220),    # Other — blue
    255: (128, 128, 128), # Ignore — gray
}

CLASS_NAMES = {0: "tumor", 1: "stroma", 2: "other", 255: "ignore"}


def denormalize(tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize a tensor and convert to uint8 numpy array.

    Args:
        tensor: (3, H, W) float32 normalized tensor

    Returns:
        (H, W, 3) uint8 numpy array
    """
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    colors: dict = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a colored segmentation mask on an image.

    Args:
        image: (H, W, 3) uint8
        mask:  (H, W)     integer labels
        colors: Dict mapping label → (R, G, B). Defaults to CLASS_COLORS.
        alpha: Transparency (0=no overlay, 1=opaque)

    Returns:
        (H, W, 3) uint8 — blended image
    """
    colors = colors or CLASS_COLORS
    overlay = image.copy().astype(np.float32)

    for label, color in colors.items():
        region = mask == label
        if not region.any():
            continue
        for c_idx in range(3):
            overlay[region, c_idx] = (
                alpha * color[c_idx] + (1 - alpha) * overlay[region, c_idx]
            )

    return overlay.clip(0, 255).astype(np.uint8)


def get_class_summary(mask: np.ndarray) -> str:
    """
    Get a summary string of class pixel counts in a mask.

    Args:
        mask: (H, W) integer labels

    Returns:
        e.g., "tumor:1200 stroma:3400 other:800 ignore:100"
    """
    parts = []
    for c in [0, 1, 2, 255]:
        count = (mask == c).sum()
        if count > 0:
            parts.append(f"{CLASS_NAMES.get(c, f'c{c}')}:{count}")
    return " | ".join(parts)


def visualize_episode(
    episode,
    prediction: torch.Tensor = None,
    save_path: str = None,
) -> np.ndarray:
    """
    Visualize a 3-class few-shot episode.

    Creates a grid:
        Row 1: Support images with 3-class ground truth masks
        Row 2: Query image, GT mask overlay, Prediction overlay

    Always uses CLASS_COLORS (tumor=red, stroma=green, other=blue).

    Args:
        episode: Episode dataclass
        prediction: (H, W) predicted mask tensor
        save_path: Optional path to save the visualization

    Returns:
        (H_grid, W_grid, 3) uint8 numpy array
    """
    K = episode.support_images.shape[0]
    H, W = 256, 256  # TIGER patch size

    # ── Row 1: Support images with 3-class masks ────────────────────
    support_panels = []
    for k in range(K):
        img = denormalize(episode.support_images[k])
        msk = episode.support_masks[k].cpu().numpy()
        panel = overlay_mask(img, msk, CLASS_COLORS, alpha=0.5)
        support_panels.append(panel)

    # ── Row 2: Query ────────────────────────────────────────────────
    query_img = denormalize(episode.query_image)
    query_mask = episode.query_mask.cpu().numpy()
    query_gt = overlay_mask(query_img, query_mask, CLASS_COLORS, alpha=0.5)

    if prediction is not None:
        pred_np = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
        query_pred = overlay_mask(query_img, pred_np, CLASS_COLORS, alpha=0.5)
    else:
        query_pred = query_img.copy()

    # ── Assemble grid ──────────────────────────────────────────────
    n_cols = max(K, 3)
    grid_h = 2 * H + 20  # 20px gap between rows
    grid_w = n_cols * W + (n_cols - 1) * 10  # 10px gap between columns
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # White background

    # Place support panels in row 1
    for k, panel in enumerate(support_panels):
        x_start = k * (W + 10)
        grid[0:H, x_start:x_start+W] = panel

    # Place query panels in row 2
    y_start = H + 20
    grid[y_start:y_start+H, 0:W] = query_img
    grid[y_start:y_start+H, W+10:2*W+10] = query_gt
    grid[y_start:y_start+H, 2*(W+10):3*W+20] = query_pred

    if save_path:
        Image.fromarray(grid).save(save_path)

    return grid
