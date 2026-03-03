"""segmentor.py — Complete Few-Shot Segmentation Pipeline (3-Class)
================================================================

Orchestrates all components into a single forward pass:
    VQ-VAE Backbone → Prototype Extraction → GPA → Decoder → Logits

This file is updated to optionally support:
  - Task-adaptive prototype refinement (Transformer set-to-set)
  - Context-contrastive codebook selection (using existing VQ indices)

Both upgrades require NO new data modalities.
"""

from __future__ import annotations

from typing import Dict, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VQVAEBackbone
from .prototype import PrototypeExtractor
from .gpa import GuidedPrototypeAllocation
from .decoder import SegmentationDecoder

class FewShotSegmentor(nn.Module):
    """Full prototypical few-shot segmentation model (3-class)."""

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
        # --- Task-adaptive prototype refinement ---
        use_task_adapt: bool = False,
        adapt_layers: int = 2,
        adapt_heads: int = 8,
        adapt_dropout: float = 0.0,
        adapt_gamma_init: float = 0.0,
        adapt_use_episode_token: bool = True,
        # --- Contrastive code selection (uses VQ indices) ---
        use_codebook_contrastive: bool = False,
        codebook_size: int = 4096,
        topk_codes_per_class: int = 64,
        min_target_code_count: int = 1,
        code_eps: float = 1e-6,
        code_context_mode: Literal["non_target", "all_valid"] = "non_target",
        min_selected_pixels_global: int = 16,
        min_selected_pixels_cell: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.backbone = VQVAEBackbone(
            weights_path=vqvae_weights,
            modules_path=vqvae_modules_path,
            feature_dim=feature_dim,
            projector_blocks=projector_blocks,
            norm_groups=norm_groups,
            dropout=dropout,
            freeze=freeze_backbone,
        )

        self.prototype = PrototypeExtractor(
            feature_dim=feature_dim,
            num_classes=num_classes,
            multi_scale_grids=multi_scale_grids or [1, 4],
            similarity="cosine",
            temperature=temperature,
            learnable_temp=learnable_temp,
            use_task_adapt=use_task_adapt,
            adapt_layers=adapt_layers,
            adapt_heads=adapt_heads,
            adapt_dropout=adapt_dropout,
            adapt_gamma_init=adapt_gamma_init,
            adapt_use_episode_token=adapt_use_episode_token,
            use_codebook_contrastive=use_codebook_contrastive,
            codebook_size=codebook_size,
            topk_codes_per_class=topk_codes_per_class,
            min_target_code_count=min_target_code_count,
            code_eps=code_eps,
            code_context_mode=code_context_mode,
            min_selected_pixels_global=min_selected_pixels_global,
            min_selected_pixels_cell=min_selected_pixels_cell,
        )

        self.gpa = GuidedPrototypeAllocation(
            feature_dim=feature_dim,
            norm_groups=norm_groups,
            temperature=temperature,
        )

        self.decoder = SegmentationDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            aspp_rates=aspp_rates,
            use_attention=use_attention,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[FewShotSegmentor] num_classes={num_classes} | Trainable: {trainable:,} | Frozen: {frozen:,}")

    def _downsample_mask(self, mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        mask_float = mask.unsqueeze(1).float()
        mask_down = F.interpolate(mask_float, size=target_size, mode="nearest")
        return mask_down.squeeze(1).long()

    def encode_support(
            self, support_images: torch.Tensor, support_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support_features, _, support_idx = self.backbone(support_images)
        h, w = support_features.shape[2:]
        support_masks_low = self._downsample_mask(support_masks, (h, w))
        return support_features, support_masks_low, support_idx

    def forward(
            self,
            support_images: torch.Tensor,
            support_masks: torch.Tensor,
            query_image: torch.Tensor,
            debug: bool = False,
    ):
        H, W = query_image.shape[2:]

        support_feat, support_masks_low, support_idx = self.encode_support(support_images, support_masks)

        prototypes = self.prototype.extract_prototypes(
            support_feat,
            support_masks_low,
            indices=support_idx,
            ignore_index=255,
        )

        query_feat, _, _ = self.backbone(query_image)
        gpa_out = self.gpa(query_feat, prototypes)
        logits = self.decoder(gpa_out, skip_features=query_feat, target_size=(H, W))

        result = {"logits": logits, "prototypes": prototypes}

        if debug:
            similarity = self.prototype.compute_similarity(query_feat, prototypes)
            result.update({
                "support_features": support_feat.detach(),
                "support_indices": support_idx.detach(),
                "query_features": query_feat.detach(),
                "gpa_output": gpa_out.detach(),
                "similarity": similarity.detach(),
            })
            if getattr(self.prototype, "last_selected_codes", None) is not None:
                result["selected_codes"] = self.prototype.last_selected_codes
            if getattr(self.prototype, "last_fallback_stats", None) is not None:
                result["fallback_stats"] = self.prototype.last_fallback_stats
            if getattr(self.prototype, "last_adapter_gamma", None) is not None:
                result["adapter_gamma"] = torch.tensor(self.prototype.last_adapter_gamma)

        return result