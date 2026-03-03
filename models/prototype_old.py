"""
prototype.py — Multi-Class Prototype Extraction and Matching
=============================================================

Implements Masked Average Pooling (MAP) for extracting class-conditional
prototypes from support features. Supports all N classes (tumor, stroma, other).

Prototypes are computed per class c:
    μ_c = Σ(F_i · 1[y_i=c]) / (Σ 1[y_i=c] + ε)

With multi-scale grid pooling for richer spatial representation.

Tensor flow:
    Support features:  (K, C, h, w)  — from backbone
    Support masks:     (K, h, w)     — {0=tumor, 1=stroma, 2=other, 255=ignore}
    ↓ Masked Average Pooling per class
    Prototypes:        {0: (N0, C), 1: (N1, C), 2: (N2, C)}
    ↓ Cosine Similarity with query
    Similarity map:    (B, num_classes, h, w)

References:
    - PANet: Wang et al., ICCV 2019
    - ASGNet: Li et al., CVPR 2021
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeExtractor(nn.Module):
    """
    Extracts class-conditional prototypes from support features using MAP.

    For each class c ∈ {0, 1, 2}, computes:
        μ_c = Σ(F * M_c) / Σ(M_c)

    Optionally: multi-scale prototypes by partitioning the spatial map into
    a grid and computing prototypes per grid cell per class.

    Args:
        feature_dim: Feature channel dimension (256 for VQ-VAE).
        num_classes: Number of segmentation classes (3).
        multi_scale_grids: List of grid sizes, e.g., [1, 4].
        similarity: 'cosine' or 'learned'.
        temperature: Initial temperature τ for cosine similarity scaling.
        learnable_temp: If True, τ is a learnable parameter.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 3,
        multi_scale_grids: List[int] = None,
        similarity: str = "cosine",
        temperature: float = 0.1,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.multi_scale_grids = multi_scale_grids or [1]
        self.similarity = similarity

        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def extract_prototypes(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        ignore_index: int = 255,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract prototypes for ALL classes from support set.

        Args:
            features: (K, C, h, w)  — support features from backbone
            masks:    (K, h, w)     — multi-class masks: {0, 1, 2, 255}
            ignore_index: Value to ignore in masks.

        Returns:
            Dict mapping class_id → prototypes tensor:
                {0: (N0, C), 1: (N1, C), 2: (N2, C)}
            where N depends on multi_scale_grids.
        """
        prototypes = {}

        for class_id in range(self.num_classes):
            class_protos = []

            for grid_size in self.multi_scale_grids:
                grid_protos = self._extract_grid_prototypes(
                    features, masks, class_id, grid_size, ignore_index
                )
                if grid_protos is not None:
                    class_protos.append(grid_protos)

            if class_protos:
                # L2 normalize prototypes for stable cosine similarity
                raw = torch.cat(class_protos, dim=0)  # (N, C)
                prototypes[class_id] = F.normalize(raw, dim=1, p=2)
            else:
                # Fallback: zero prototype if no valid pixels for this class
                prototypes[class_id] = torch.zeros(1, self.feature_dim,
                                                   device=features.device)

        return prototypes

    def _extract_grid_prototypes(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        class_id: int,
        grid_size: int,
        ignore_index: int,
    ) -> Optional[torch.Tensor]:
        """
        Extract prototypes from a grid partition of the spatial map.

        For grid_size=1: standard MAP (one global prototype per class).
        For grid_size=4: partition into 4×4 grid, one prototype per cell.

        Args:
            features: (K, C, h, w)
            masks:    (K, h, w) — {0, 1, 2, 255}
            class_id: Which class to extract (0, 1, or 2)
            grid_size: Number of grid divisions along each axis
            ignore_index: 255

        Returns:
            (N_valid, C) tensor of prototypes, or None if no valid pixels
        """
        K, C, h, w = features.shape

        if grid_size == 1:
            # ── Global MAP ──────────────────────────────────────────────
            target_mask = (masks == class_id).float()  # (K, h, w)
            valid_mask = (masks != ignore_index).float()  # (K, h, w)
            select_mask = target_mask * valid_mask  # (K, h, w)

            select_mask_3d = select_mask.unsqueeze(1)  # (K, 1, h, w)
            numerator = (features * select_mask_3d).sum(dim=(0, 2, 3))  # (C,)
            denominator = select_mask.sum().clamp(min=1.0)

            proto = numerator / denominator  # (C,)
            return proto.unsqueeze(0)  # (1, C)

        else:
            # ── Grid MAP ────────────────────────────────────────────────
            cell_h = h // grid_size
            cell_w = w // grid_size
            protos = []

            for gi in range(grid_size):
                for gj in range(grid_size):
                    y1, y2 = gi * cell_h, (gi + 1) * cell_h
                    x1, x2 = gj * cell_w, (gj + 1) * cell_w

                    cell_feat = features[:, :, y1:y2, x1:x2]   # (K, C, ch, cw)
                    cell_mask = masks[:, y1:y2, x1:x2]          # (K, ch, cw)

                    target = (cell_mask == class_id).float()
                    valid = (cell_mask != ignore_index).float()
                    select = target * valid

                    n_pixels = select.sum()
                    if n_pixels < 1:
                        continue

                    select_3d = select.unsqueeze(1)
                    proto = (cell_feat * select_3d).sum(dim=(0, 2, 3)) / n_pixels
                    protos.append(proto)

            if protos:
                return torch.stack(protos, dim=0)  # (N_cells, C)
            return None

    def compute_similarity(
        self,
        query_features: torch.Tensor,
        prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute per-pixel cosine similarity between query features and prototypes.

        Args:
            query_features: (B, C, h, w) — query features from backbone
            prototypes: Dict {class_id: (N_proto, C)}

        Returns:
            similarity: (B, num_classes, h, w) — per-class similarity scores

        Each pixel gets a similarity score to the nearest prototype of each class.
        With multi-scale prototypes, we use MAX over grid prototypes:
            sim(x, class_c) = max_i cos(x, p_c_i) / τ
        """
        B, C, h, w = query_features.shape

        # Normalize query features
        q_norm = F.normalize(query_features, dim=1, p=2)  # (B, C, h, w)

        class_sims = []
        for class_id in range(self.num_classes):
            if class_id not in prototypes:
                class_sims.append(torch.zeros(B, 1, h, w, device=query_features.device))
                continue

            p = prototypes[class_id]  # (N, C)
            p_norm = F.normalize(p, dim=1, p=2)  # (N, C)

            q_flat = q_norm.view(B, C, h * w)  # (B, C, hw)
            sim = torch.einsum("nc,bcs->bns", p_norm, q_flat)  # (B, N, hw)
            sim = sim.view(B, p_norm.shape[0], h, w)

            # Max over prototypes → (B, 1, h, w)
            max_sim = sim.max(dim=1, keepdim=True).values
            class_sims.append(max_sim)

        # Stack: (B, num_classes, h, w)
        similarity = torch.cat(class_sims, dim=1) / self.temperature.clamp(min=0.01)

        return similarity

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
        query_features: torch.Tensor,
        ignore_index: int = 255,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Full prototype extraction + matching pipeline.

        Args:
            support_features: (K, C, h, w)
            support_masks:    (K, h, w) — {0, 1, 2, 255}
            query_features:   (B, C, h, w)
            ignore_index:     255

        Returns:
            similarity: (B, 3, h, w)  — per-class similarity scores
            prototypes: Dict {0: (N, C), 1: (N, C), 2: (N, C)}
        """
        prototypes = self.extract_prototypes(support_features, support_masks, ignore_index)
        similarity = self.compute_similarity(query_features, prototypes)
        return similarity, prototypes
