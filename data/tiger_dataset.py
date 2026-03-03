"""
tiger_dataset.py — Base Dataset for TIGER Histopathology Patches
================================================================

Loads 256×256 PNG patches and their corresponding pixel-level masks.
The TIGER dataset provides 3 semantic classes:
    0 = tumor (invasive + in-situ)
    1 = stroma (TAS + inflamed)
    2 = other  (healthy gland + necrosis + rest)
    255 = ignore (unannotated regions)

Tensor flow:
    PNG (H,W,3) uint8 → albumentations → tensor (3,256,256) float32  [0,1] → normalized
    Mask PNG (H,W) uint8 → same spatial augments → LongTensor (256,256) ∈ {0,1,2,255}

Each sample also carries metadata: slide_id, patch_id, per-class fractions,
which enables patient-level splitting and class-balanced sampling.

References:
    - TIGER Challenge: https://tiger.grand-challenge.org/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class PatchMeta:
    """Metadata for a single patch — used by the episodic sampler."""
    patch_id: str
    slide_id: str
    class_fracs: Dict[int, float]      # {0: frac_tumor, 1: frac_stroma, 2: frac_other}
    dominant_class: int
    n_present_classes: int
    labeled_frac: float


class TigerDataset(Dataset):
    """
    Base dataset that loads TIGER patches and masks.

    Usage:
        ds = TigerDataset(data_root, split="train", fold=0, transform=...)
        img, mask, meta = ds[i]
        # img:  (3, 256, 256) float32 normalized
        # mask: (256, 256) int64 ∈ {0, 1, 2, 255}
        # meta: PatchMeta dataclass

    The dataset provides rich metadata per patch so the episodic sampler
    can build balanced, patient-disjoint episodes.
    """



    # Class names for visualization
    CLASS_NAMES = {0: "tumor", 1: "stroma", 2: "other", 255: "ignore"}
    NUM_CLASSES = 3

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fold: int = 0,
        transform=None,
        min_labeled_frac: float = 0.9,
    ):
        """
        Args:
            data_root: Path to tiger-group-256/ directory.
            split: One of "train", "val", "test".
            fold: Cross-validation fold (0–3).
            transform: Albumentations transform (applied to both image and mask).
            min_labeled_frac: Skip patches with less than this fraction annotated.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.fold = fold
        self.transform = transform

        # ── Load label map ──────────────────────────────────────────────
        label_map_path = self.data_root / "label_map.json"
        with open(label_map_path) as f:
            self.label_map = json.load(f)  # {"0": "tumor ...", "1": "stroma ...", ...}

        # ── Load split file ─────────────────────────────────────────────
        splits_dir = self.data_root / "splits_cv" / f"fold_{fold}"
        split_file = splits_dir / f"{split}_patches.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            split_patch_ids = set(line.strip() for line in f if line.strip())

        # ── Load per-patch stats and filter (fully vectorized) ───────────
        import pandas as pd
        stats_path = self.data_root / "per_patch_stats.csv"
        df = pd.read_csv(stats_path)

        # Fast filter using pandas isin + threshold
        df = df[df["patch_id"].isin(split_patch_ids)]
        df = df[df["annotated_frac"] >= min_labeled_frac]
        df = df.reset_index(drop=True)

        # Extract slide_id vectorized
        df["slide_id"] = df["patch_id"].str.split("_[", regex=False).str[0]

        # Store as arrays for fast access (no PatchMeta overhead)
        self._patch_ids = df["patch_id"].values       # numpy array of strings
        self._slide_ids = df["slide_id"].values
        self._frac_0 = df["frac_0"].values.astype(np.float32)
        self._frac_1 = df["frac_1"].values.astype(np.float32)
        self._frac_2 = df["frac_2"].values.astype(np.float32)
        self._dominant = df["dominant_class"].values.astype(np.int32)
        self._n_classes = df["n_present_classes"].values.astype(np.int32)
        self._labeled_frac = df["annotated_frac"].values.astype(np.float32)

        # Build PatchMeta objects lazily (only when accessed)
        self.patches = [None] * len(df)  # Placeholder list for compatibility
        self._n_patches = len(df)
        self.patch_id_to_idx = {pid: i for i, pid in enumerate(self._patch_ids)}

        # ── Build class → patch index mapping (vectorized) ───────────────
        self.class_to_indices: Dict[int, List[int]] = {c: [] for c in range(self.NUM_CLASSES)}
        fracs = [self._frac_0, self._frac_1, self._frac_2]
        for c in range(self.NUM_CLASSES):
            mask = fracs[c] > 0.01
            self.class_to_indices[c] = np.where(mask)[0].tolist()

        # ── Build slide → patch index mapping ────────────────────────────
        self.slide_to_indices: Dict[str, List[int]] = {}
        for idx, sid in enumerate(self._slide_ids):
            self.slide_to_indices.setdefault(sid, []).append(idx)

        print(f"[TigerDataset] {split} fold={fold}: {self._n_patches} patches, "
              f"{len(self.slide_to_indices)} slides")
        for c in range(self.NUM_CLASSES):
            print(f"  Class {c} ({self.CLASS_NAMES[c]}): {len(self.class_to_indices[c])} patches")

    def __len__(self) -> int:
        return self._n_patches

    def get_meta(self, idx: int) -> PatchMeta:
        """Lazily construct PatchMeta for the given index."""
        return PatchMeta(
            patch_id=str(self._patch_ids[idx]),
            slide_id=str(self._slide_ids[idx]),
            class_fracs={0: float(self._frac_0[idx]), 1: float(self._frac_1[idx]), 2: float(self._frac_2[idx])},
            dominant_class=int(self._dominant[idx]),
            n_present_classes=int(self._n_classes[idx]),
            labeled_frac=float(self._labeled_frac[idx]),
        )

    def __getitem__(self, idx: int):
        """
        Returns:
            image: (3, H, W) float32 tensor, normalized with ImageNet stats
            mask:  (H, W) int64 tensor with values in {0, 1, 2, 255}
            meta:  PatchMeta dataclass
        """
        patch_id = str(self._patch_ids[idx])
        slide_id = str(self._slide_ids[idx])

        # ── Load image and mask ──────────────────────────────────────────
        img_path = self.data_root / "images" / f"{patch_id}.png"
        mask_path = self.data_root / "masks" / f"{patch_id}.png"

        image = np.array(Image.open(img_path).convert("RGB"))   # (H,W,3) uint8
        mask = np.array(Image.open(mask_path))                  # (H,W) uint8

        # ── Apply augmentation ───────────────────────────────────────────
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ── Convert to tensors ───────────────────────────────────────────
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask = torch.from_numpy(mask.astype(np.int64)) if isinstance(mask, np.ndarray) else mask.long()

        meta = self.get_meta(idx)
        return image, mask, meta

    def get_slides(self) -> List[str]:
        """Return list of unique slide IDs in this split."""
        return list(self.slide_to_indices.keys())

    def get_patches_for_class(self, class_id: int) -> List[int]:
        """Return list of dataset indices containing the given class."""
        return self.class_to_indices.get(class_id, [])

    def get_patches_for_slide(self, slide_id: str) -> List[int]:
        """Return list of dataset indices from a given slide."""
        return self.slide_to_indices.get(slide_id, [])
