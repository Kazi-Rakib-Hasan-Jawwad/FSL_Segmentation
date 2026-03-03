"""
episodic_sampler.py — 3-Class Episode Generator
================================================

Constructs few-shot episodes for training and evaluation.
Episodes contain all 3 TIGER classes (tumor, stroma, other) in
their original label form — NO binary remapping.

Design (mixed-support):
    1. Per-epoch re-shuffling to prevent memorization.
    2. Patient-disjoint support/query patches.
    3. Support set: K patches chosen so all 3 classes appear across the set.
    4. Query patch: must contain at least 2 classes above a pixel threshold.

Tensor flow for one episode:
    support_images:  (K, 3, 256, 256)  float32 normalized
    support_masks:   (K, 256, 256)     int64, ORIGINAL {0=tumor, 1=stroma, 2=other, 255=ignore}
    query_image:     (1, 3, 256, 256)  float32 normalized
    query_mask:      (1, 256, 256)     int64, ORIGINAL {0=tumor, 1=stroma, 2=other, 255=ignore}

References:
    - PANet (Wang et al., 2019) — episodic training for few-shot segmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tiger_dataset import TigerDataset


@dataclass
class Episode:
    """
    A single few-shot episode with 3-class masks.

    Shapes (for K-shot, 3-way):
        support_images:  (K, 3, H, W)
        support_masks:   (K, H, W)    — original: {0=tumor, 1=stroma, 2=other, 255=ignore}
        query_image:     (3, H, W)
        query_mask:      (H, W)       — original: {0=tumor, 1=stroma, 2=other, 255=ignore}
    """
    support_images: torch.Tensor
    support_masks: torch.Tensor
    query_image: torch.Tensor
    query_mask: torch.Tensor
    support_slide_ids: List[str]    # For debugging: which slides were used
    query_slide_id: str


class EpisodicSampler(Dataset):
    """
    Generates 3-class few-shot episodes from a TigerDataset.

    Mixed-support design: K support patches are sampled to collectively
    contain all 3 classes. Query must contain at least 2 classes.

    Anti-overfitting features:
        - Different random episodes each epoch via reset_seed()
        - Patient-disjoint support/query
        - Multi-class coverage across the support set

    Usage:
        sampler = EpisodicSampler(dataset, k_shot=5, episodes=200)
        for epoch in range(100):
            sampler.reset_seed(epoch)
            for ep in DataLoader(sampler, batch_size=None):
                train(ep)
    """

    def __init__(
        self,
        base_dataset: TigerDataset,
        n_way: int = 3,
        k_shot: int = 5,
        episodes: int = 200,
        seed: int = 42,
        min_pixels_per_class: int = 100,
        patient_disjoint: bool = True,
        ignore_index: int = 255,
    ):
        self.base_ds = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.episodes = episodes
        self.base_seed = seed
        self.min_pixels_per_class = min_pixels_per_class
        self.patient_disjoint = patient_disjoint
        self.ignore_index = ignore_index
        self.num_classes = base_dataset.NUM_CLASSES  # 3

        self.rng = np.random.RandomState(seed)
        self._current_epoch = 0

        # Precompute slide → [indices] for efficient sampling
        self.slide_to_indices = {}  # {slide_id: [idx, ...]}
        for idx in range(len(base_dataset)):
            slide = str(base_dataset._slide_ids[idx])
            self.slide_to_indices.setdefault(slide, []).append(idx)

        # Precompute per-patch class pixel counts (at 256x256 = 65536 pixels)
        self._pixels_per_class = np.stack([
            base_dataset._frac_0 * 65536,
            base_dataset._frac_1 * 65536,
            base_dataset._frac_2 * 65536,
        ], axis=1).astype(np.float32)  # (N, 3)

        # Indices of patches with at least 2 classes above threshold
        self._multi_class_indices = []
        for idx in range(len(base_dataset)):
            n_classes_present = (self._pixels_per_class[idx] >= min_pixels_per_class).sum()
            if n_classes_present >= 2:
                self._multi_class_indices.append(idx)
        self._multi_class_indices = np.array(self._multi_class_indices)

        print(f"[EpisodicSampler] {len(self._multi_class_indices)} / {len(base_dataset)} "
              f"patches have ≥2 classes above {min_pixels_per_class} pixels")

    def reset_seed(self, epoch: int):
        """
        Re-seed the RNG for a new epoch → different episodes every epoch.
        This is THE key anti-overfitting mechanism.
        """
        self._current_epoch = epoch
        self.rng = np.random.RandomState((self.base_seed + epoch * 1000) % (2**32))

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, episode_idx: int) -> Episode:
        """
        Construct one 3-class episode using mixed-support design.

        Algorithm:
            1. Sample K support patches that collectively cover all 3 classes.
            2. Sample 1 query patch containing at least 2 classes.
            3. If patient_disjoint: ensure query is from a different slide than supports.
            4. Keep ORIGINAL masks {0, 1, 2, 255} — NO remapping.
        """
        ep_rng = np.random.RandomState(
            (self.base_seed + self._current_epoch * 10000 + episode_idx) % (2**32)
        )

        support_indices, query_idx, support_slides, query_slide = \
            self._sample_episode(ep_rng)

        # ── Load support ──────────────────────────────────────────────
        support_images = []
        support_masks = []
        for si in support_indices:
            img, mask, _ = self.base_ds[si]
            support_images.append(img)
            support_masks.append(mask)

        # ── Load query ────────────────────────────────────────────────
        query_img, query_mask, _ = self.base_ds[query_idx]

        return Episode(
            support_images=torch.stack(support_images),   # (K, 3, H, W)
            support_masks=torch.stack(support_masks),     # (K, H, W)
            query_image=query_img,                         # (3, H, W)
            query_mask=query_mask,                         # (H, W)
            support_slide_ids=support_slides,
            query_slide_id=query_slide,
        )

    def _sample_episode(
        self, rng: np.random.RandomState
    ) -> Tuple[List[int], int, List[str], str]:
        """
        Sample K support + 1 query ensuring:
            - Support set collectively covers at least 2 classes
            - Query has at least 2 classes
            - Patient-disjoint

        Returns:
            support_indices, query_idx, support_slides, query_slide
        """
        if self.patient_disjoint:
            return self._sample_patient_disjoint(rng)
        else:
            return self._sample_simple(rng)

    def _sample_patient_disjoint(
        self, rng: np.random.RandomState
    ) -> Tuple[List[int], int, List[str], str]:
        """Sample with patient disjointness."""
        slides = list(self.slide_to_indices.keys())
        rng.shuffle(slides)

        # Try to find a good query slide (has multi-class patches)
        query_idx = None
        query_slide = None
        support_slides_available = []

        for s in slides:
            slide_indices = self.slide_to_indices[s]
            # Check if this slide has multi-class patches
            mc_in_slide = [i for i in slide_indices
                           if i in set(self._multi_class_indices)]
            if mc_in_slide:
                query_idx = int(rng.choice(mc_in_slide))
                query_slide = s
                support_slides_available = [sl for sl in slides if sl != s]
                break

        # Fallback: just pick any patch
        if query_idx is None:
            query_slide = slides[0]
            query_idx = int(rng.choice(self.slide_to_indices[query_slide]))
            support_slides_available = slides[1:]

        # Gather support pool from other slides
        support_pool = []
        for s in support_slides_available:
            support_pool.extend(self.slide_to_indices[s])
        rng.shuffle(support_pool)

        # Greedily select K supports that maximize class coverage
        support_indices = self._greedy_class_coverage(support_pool, rng)
        support_slide_ids = [str(self.base_ds._slide_ids[i]) for i in support_indices]

        return support_indices, query_idx, support_slide_ids, query_slide

    def _sample_simple(
        self, rng: np.random.RandomState
    ) -> Tuple[List[int], int, List[str], str]:
        """Fallback: random sampling without patient disjointness."""
        all_indices = list(range(len(self.base_ds)))
        rng.shuffle(all_indices)

        # Pick multi-class query
        mc_set = set(self._multi_class_indices)
        query_idx = None
        remaining = []
        for idx in all_indices:
            if query_idx is None and idx in mc_set:
                query_idx = idx
            else:
                remaining.append(idx)
        if query_idx is None:
            query_idx = all_indices[0]
            remaining = all_indices[1:]

        support_indices = self._greedy_class_coverage(remaining, rng)
        support_slides = [str(self.base_ds._slide_ids[i]) for i in support_indices]
        query_slide = str(self.base_ds._slide_ids[query_idx])

        return support_indices, query_idx, support_slides, query_slide

    def _greedy_class_coverage(
        self, pool: List[int], rng: np.random.RandomState
    ) -> List[int]:
        """
        Greedily select K patches from pool to maximize class coverage.

        Strategy: for each slot, pick the patch that adds the most
        pixels of the LEAST-covered class so far.
        """
        selected = []
        coverage = np.zeros(self.num_classes, dtype=np.float32)  # Total pixels per class

        for _ in range(self.k_shot):
            if not pool:
                break

            best_idx = None
            best_score = -1.0

            # Sample a subset to avoid O(N*K) cost
            candidates = pool[:min(50, len(pool))]

            for idx in candidates:
                pixels = self._pixels_per_class[idx]  # (3,)
                # Score: how much does this patch help the weakest class?
                new_coverage = coverage + pixels
                min_class_coverage = new_coverage.min()
                if min_class_coverage > best_score:
                    best_score = min_class_coverage
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                coverage += self._pixels_per_class[best_idx]
                pool.remove(best_idx)

        # If we didn't get enough, fill randomly
        while len(selected) < self.k_shot and pool:
            selected.append(pool.pop(0))

        return selected
