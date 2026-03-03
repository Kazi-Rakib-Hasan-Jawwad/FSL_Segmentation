"""
gpa.py — Guided Prototype Allocation
=====================================

Maps query features to prototype space using learned correlation.

The GPA module (from ASGNet, Li et al. CVPR 2021) takes query features and
prototypes, computes correlation, and produces prototype-allocated features
where each spatial position is a weighted combination of prototypes.

CRITICAL: Always uses SOFT assignment during training.
Lesson from Project-1: Hard assignment uses argmax which is non-differentiable
and kills gradient flow to the prototype extractor.

Tensor flow:
    query_feat:   (B, C, h, w)     — from backbone
    prototypes:   (N_proto, C)     — from prototype extractor
    ↓ Correlation: query^T · protos
    correlation:  (B, N_proto, h, w)
    ↓ Softmax (soft assignment)
    weights:      (B, N_proto, h, w)
    ↓ Weighted sum of prototypes
    allocated:    (B, C, h, w)     — prototype-allocated features
    ↓ Conv + GroupNorm
    output:       (B, C, h, w)     — refined features

References:
    - ASGNet: Li et al., "Adaptive Prototype Learning and Allocation for
      Few-Shot Segmentation", CVPR 2021
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedPrototypeAllocation(nn.Module):
    """
    GPA module: allocates prototypes to query spatial positions.

    For each query pixel, computes how strongly it correlates with each
    prototype, then produces a weighted combination.

    Design choices:
        - ALWAYS soft assignment (softmax weights) — no argmax
        - GroupNorm instead of BatchNorm (batch_size=1 in episodes)
        - Temperature-scaled correlation for sharper/softer allocation
    """

    def __init__(
        self,
        feature_dim: int = 256,
        norm_groups: int = 8,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature

        # Refinement after prototype allocation
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Initialize
        nn.init.kaiming_normal_(self.refine[0].weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        query_features: torch.Tensor,
        prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Allocate prototypes to query spatial positions.

        Args:
            query_features: (B, C, h, w) — query features from backbone
            prototypes: Dict {class_id: (N_i, C)} — prototypes per class

        Returns:
            allocated: (B, C, h, w) — prototype-allocated features

        Tensor flow:
            1. Stack all prototypes: (N_total, C)
            2. Correlation: query_feat^T · protos → (B, N_total, h, w)
            3. Softmax → (B, N_total, h, w)  [always soft, never hard]
            4. Weighted sum → (B, C, h, w)
            5. Refine with conv+norm → (B, C, h, w)
        """
        B, C, h, w = query_features.shape

        # ── 1. Stack all prototypes ─────────────────────────────────────
        all_protos = []
        for class_id in sorted(prototypes.keys()):
            all_protos.append(prototypes[class_id])  # (N_i, C)
        proto_matrix = torch.cat(all_protos, dim=0)  # (N_total, C)
        N = proto_matrix.shape[0]

        # ── 2. Compute correlation ──────────────────────────────────────
        # Normalize for cosine-like correlation
        q_norm = F.normalize(query_features, dim=1, p=2)    # (B, C, h, w)
        p_norm = F.normalize(proto_matrix, dim=1, p=2)      # (N, C)

        # Reshape query for matrix multiply
        q_flat = q_norm.view(B, C, h * w)  # (B, C, hw)

        # Correlation: p_norm @ q_flat → (B, N, hw)
        # Using einsum for clarity: batch, proto, spatial
        corr = torch.einsum("nc,bcs->bns", p_norm, q_flat)  # (B, N, hw)
        corr = corr / max(self.temperature, 0.01)

        # ── 3. Soft assignment (ALWAYS soft — lesson from Project-1) ────
        weights = F.softmax(corr, dim=1)  # (B, N, hw)

        # ── 4. Weighted sum of prototypes ───────────────────────────────
        # allocated = Σ(w_n * p_n) for each spatial position
        # proto_matrix: (N, C),  weights: (B, N, hw)
        allocated = torch.einsum("nc,bns->bcs", proto_matrix, weights)  # (B, C, hw)
        allocated = allocated.view(B, C, h, w)  # (B, C, h, w)

        # ── 5. Refine ──────────────────────────────────────────────────
        allocated = self.refine(allocated)  # (B, C, h, w)

        return allocated
