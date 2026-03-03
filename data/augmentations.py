"""
augmentations.py — Histopathology-Specific Data Augmentation
============================================================

Provides albumentations transforms for training and evaluation.

Key design choices for histopathology:
    - Color jitter: H&E staining varies across labs and scanners
    - Random flip + rotate90: tissue orientation is arbitrary
    - Elastic deformation: mimics tissue deformation during slide preparation
    - NO random crop: patches are already 256×256, cropping would lose context

All transforms use albumentations which synchronizes spatial transforms
between image and mask automatically.

Tensor flow:
    Input:  image (H,W,3) uint8, mask (H,W) uint8
    Output: image (H,W,3) uint8, mask (H,W) uint8  (ToTensor done in dataset)
"""

from __future__ import annotations

import albumentations as A


def get_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline.

    Augmentations are applied with synchronized spatial transforms for
    image-mask pairs. The intensity transforms (color jitter, blur) only
    affect the image, not the mask.
    """
    return A.Compose([
        # ── Spatial transforms (applied to both image AND mask) ────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Elastic deformation: mimics tissue deformation
        # Moderate parameters to avoid destroying cell morphology
        A.ElasticTransform(
            alpha=50,                # Displacement intensity
            sigma=5,                 # Smoothness of displacement field
            p=0.3,
        ),

        # ── Intensity transforms (applied ONLY to image, not mask) ────
        # Color jitter: accounts for H&E staining variability across labs
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,               # Small hue shift — H&E has characteristic colors
            p=0.8,
        ),

        # Gaussian blur: slight defocus simulation
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2,
        ),

        # Gaussian noise: scanner noise simulation
        A.GaussNoise(
            p=0.1,
        ),
    ])


def get_val_transforms():
    """
    Validation/test transforms — NO augmentation.
    Returns None so the dataset skips the transform step entirely.
    (Albumentations v1.4.7 doesn't support empty Compose([]).)
    """
    return None
