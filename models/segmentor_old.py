"""
segmentor.py — Complete Few-Shot Segmentation Pipeline (3-Class)
================================================================

Orchestrates all components into a single forward pass:
    VQ-VAE Backbone → Prototype Extraction → GPA → Decoder → Logits

Full tensor flow (3-way, 5-shot, 256×256 images):

    SUPPORT PATH:
    support_images   (5, 3, 256, 256)
    → VQ Backbone    (5, 256, 64, 64) features
    support_masks    (5, 256, 256) → downsample → (5, 64, 64)
    → Prototype Extraction → {0: (N0, 256), 1: (N1, 256), 2: (N2, 256)}

    QUERY PATH:
    query_image      (1, 3, 256, 256)
    → VQ Backbone    (1, 256, 64, 64) features
    → GPA(query_feat, prototypes) → (1, 256, 64, 64) allocated features
    → Decoder(gpa_out, skip_feat) → (1, 3, 256, 256) logits

    LOSS:
    logits (1, 3, 256, 256) vs query_mask (1, 256, 256) → CE + Dice + Focal

References:
    - PANet (Wang et al., 2019)
    - ASGNet (Li et al., 2021)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VQVAEBackbone
from .prototype import PrototypeExtractor
from .gpa import GuidedPrototypeAllocation
from .decoder import SegmentationDecoder


class FewShotSegmentor(nn.Module):
    """
    Full prototypical few-shot segmentation model (3-class).

    Given K support image-mask pairs and 1 query image, produces
    per-pixel 3-class logits for the query.

    Learnable components:
        1. Feature Projector (in backbone)
        2. GPA refinement conv
        3. Decoder + Classifier
    """

    def __init__(
        self,
        vqvae_weights: str,
        vqvae_modules_path: str,
        feature_dim: int = 256,
        num_classes: int = 3,
        projector_blocks: int = 2,
        multi_scale_grids: list = None,
        temperature: float = 0.1,
        learnable_temp: bool = True,
        aspp_rates: tuple = (6, 12, 18),
        use_attention: bool = True,
        norm_groups: int = 8,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # ── 1. VQ-VAE Backbone (frozen encoder + learnable projector) ───
        self.backbone = VQVAEBackbone(
            weights_path=vqvae_weights,
            modules_path=vqvae_modules_path,
            feature_dim=feature_dim,
            projector_blocks=projector_blocks,
            norm_groups=norm_groups,
            dropout=dropout,
            freeze=freeze_backbone,
        )

        # ── 2. Prototype Engine (3-class) ──────────────────────────────
        self.prototype = PrototypeExtractor(
            feature_dim=feature_dim,
            num_classes=num_classes,
            multi_scale_grids=multi_scale_grids or [1, 4],
            similarity="cosine",
            temperature=temperature,
            learnable_temp=learnable_temp,
        )

        # ── 3. GPA (Guided Prototype Allocation) ───────────────────────
        self.gpa = GuidedPrototypeAllocation(
            feature_dim=feature_dim,
            norm_groups=norm_groups,
            temperature=temperature,
        )

        # ── 4. Decoder ─────────────────────────────────────────────────
        self.decoder = SegmentationDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            aspp_rates=aspp_rates,
            use_attention=use_attention,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        # Print parameter counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[FewShotSegmentor] num_classes={num_classes} | "
              f"Trainable: {trainable:,} | Frozen: {frozen:,}")

    def _downsample_mask(self, mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Downsample mask using nearest-neighbor to preserve discrete labels.

        Args:
            mask: (B, H, W) or (K, H, W) — LongTensor  {0, 1, 2, 255}
            target_size: (h, w)

        Returns:
            (B, h, w) LongTensor
        """
        mask_float = mask.unsqueeze(1).float()
        mask_down = F.interpolate(mask_float, size=target_size, mode="nearest")
        return mask_down.squeeze(1).long()

    def encode_support(
        self, support_images: torch.Tensor, support_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode support images and downsample masks.

        Args:
            support_images: (K, 3, H, W)
            support_masks:  (K, H, W) — {0, 1, 2, 255}

        Returns:
            support_features: (K, C, h, w)
            support_masks_low: (K, h, w)
        """
        support_features, _, _ = self.backbone(support_images)
        h, w = support_features.shape[2:]
        support_masks_low = self._downsample_mask(support_masks, (h, w))
        return support_features, support_masks_low

    def forward(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_image: torch.Tensor,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: support → prototypes → query → logits.

        Args:
            support_images: (K, 3, H, W)
            support_masks:  (K, H, W) — {0=tumor, 1=stroma, 2=other, 255=ignore}
            query_image:    (B, 3, H, W)
            debug:          If True, return intermediate tensors

        Returns:
            Dict with:
                "logits": (B, 3, H, W) — per-pixel class logits
                "prototypes": Dict {0: (N, C), 1: (N, C), 2: (N, C)}
        """
        H, W = query_image.shape[2:]

        # ── Encode support ──────────────────────────────────────────────
        support_feat, support_masks_low = self.encode_support(support_images, support_masks)

        # ── Extract prototypes (3-class) ────────────────────────────────
        prototypes = self.prototype.extract_prototypes(
            support_feat, support_masks_low, ignore_index=255
        )
        # prototypes: {0: (N0, 256), 1: (N1, 256), 2: (N2, 256)}

        # ── Encode query ────────────────────────────────────────────────
        query_feat, query_zq, query_idx = self.backbone(query_image)

        # ── GPA: allocate prototypes to query positions ────────────────
        gpa_out = self.gpa(query_feat, prototypes)

        # ── Decode ──────────────────────────────────────────────────────
        logits = self.decoder(gpa_out, skip_features=query_feat, target_size=(H, W))
        # logits: (B, 3, 256, 256)

        result = {
            "logits": logits,
            "prototypes": prototypes,
        }

        if debug:
            similarity = self.prototype.compute_similarity(query_feat, prototypes)
            result.update({
                "support_features": support_feat.detach(),
                "query_features": query_feat.detach(),
                "gpa_output": gpa_out.detach(),
                "similarity": similarity.detach(),
            })

        return result
