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

prototype.py — Multi-Class Prototype Extraction and Matching
=============================================================

Baseline behavior:
  - Masked Average Pooling (MAP) prototypes from support features.
  - Optional multi-scale grid pooling.

Option B upgrades (no new data required):
  1) Context-contrastive codebook selection (uses existing VQ indices + masks)
     - Filters which support pixels contribute to prototypes.
     - Robust fallback to plain MAP if filtering is too strict.

  2) Task-adaptive prototype refinement (FEAT-style set-to-set adapter)
     - Refines prototypes per-episode using a small Transformer.
     - Starts as identity via gamma_init=0.0.

Tensor flow:
  Support features:  (K, C, h, w)
  Support masks:     (K, h, w)
  Support indices:   (K, h, w)  — optional, for contrastive code selection
  → prototypes: {class_id: (N_i, C)}

  Query features:    (B, C, h, w)
  → similarity map:  (B, num_classes, h, w)  (used only for debug)


"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototype_adapters import CodebookContrastiveSelector, TaskAdaptivePrototypeAdapter


class PrototypeExtractor(nn.Module):
    """
    Extracts class-conditional prototypes from support features.

    Default: standard MAP (masked average pooling).

    Optional additions:
      - Contrastive code selection over VQ indices (filters MAP region)
      - Task-adaptive set-to-set refinement of prototype tokens (Transformer)

    Notes on stability:
      - Filtering is ALWAYS guarded by fallback to plain MAP.
      - Task-adapter starts as identity when gamma_init=0.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 3,
        multi_scale_grids: List[int] = None,
        similarity: str = "cosine",
        temperature: float = 0.1,
        learnable_temp: bool = True,
        # --- Task-adaptive prototype refinement (Transformer) ---
        use_task_adapt: bool = False,
        adapt_layers: int = 2,
        adapt_heads: int = 8,
        adapt_dropout: float = 0.0,
        adapt_gamma_init: float = 0.0,
        adapt_use_episode_token: bool = True,
        # --- Context-contrastive code selection (VQ indices) ---
        use_codebook_contrastive: bool = False,
        codebook_size: int = 4096,
        topk_codes_per_class: int = 64,
        min_target_code_count: int = 1,
        code_eps: float = 1e-6,
        code_context_mode: Literal["non_target", "all_valid"] = "non_target",
        # --- Fallback thresholds (pixels on feature grid) ---
        min_selected_pixels_global: int = 16,
        min_selected_pixels_cell: int = 4,
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

        self.min_selected_pixels_global = int(min_selected_pixels_global)
        self.min_selected_pixels_cell = int(min_selected_pixels_cell)

        self.code_selector: Optional[CodebookContrastiveSelector] = None
        if use_codebook_contrastive:
            self.code_selector = CodebookContrastiveSelector(
                codebook_size=codebook_size,
                topk=topk_codes_per_class,
                eps=code_eps,
                min_target_count=min_target_code_count,
                context_mode=code_context_mode,
            )

        self.task_adapter: Optional[TaskAdaptivePrototypeAdapter] = None
        if use_task_adapt:
            self.task_adapter = TaskAdaptivePrototypeAdapter(
                feature_dim=feature_dim,
                num_classes=num_classes,
                layers=adapt_layers,
                heads=adapt_heads,
                dropout=adapt_dropout,
                use_episode_token=adapt_use_episode_token,
                gamma_init=adapt_gamma_init,
            )

        # Debug hooks
        self.last_selected_codes: Optional[Dict[int, Optional[torch.Tensor]]] = None
        self.last_fallback_stats: Optional[Dict[str, int]] = None
        self.last_adapter_gamma: Optional[float] = None

    def extract_prototypes(
            self,
            features: torch.Tensor,  # (K,C,h,w)
            masks: torch.Tensor,  # (K,h,w)
            indices: Optional[torch.Tensor] = None,  # (K,h,w)
            ignore_index: int = 255,
    ) -> Dict[int, torch.Tensor]:
        """Extract prototypes for ALL classes from a support set."""

        stats = {
            "global_filtered": 0,
            "global_fallback": 0,
            "cell_filtered": 0,
            "cell_fallback": 0,
        }

        selected_codes: Optional[Dict[int, Optional[torch.Tensor]]] = None
        if indices is not None and self.code_selector is not None:
            selected_codes = self.code_selector(indices, masks, num_classes=self.num_classes, ignore_index=ignore_index)

        # Save for debugging (cpu-friendly)
        if selected_codes is not None:
            self.last_selected_codes = {
                k: (v.detach().cpu() if v is not None else None)
                for k, v in selected_codes.items()
            }
        else:
            self.last_selected_codes = None

        prototypes: Dict[int, torch.Tensor] = {}

        for class_id in range(self.num_classes):
            class_protos = []

            code_ok = None
            if indices is not None and selected_codes is not None and selected_codes.get(class_id, None) is not None:
                sel = selected_codes[class_id]
                lut = torch.zeros(self.code_selector.codebook_size, device=features.device, dtype=torch.bool)
                lut[sel] = True
                code_ok = lut[indices]  # (K,h,w) bool

            for grid_size in self.multi_scale_grids:
                grid_protos = self._extract_grid_prototypes(
                    features=features,
                    masks=masks,
                    class_id=class_id,
                    grid_size=grid_size,
                    ignore_index=ignore_index,
                    code_ok=code_ok,
                    stats=stats,
                )
                if grid_protos is not None:
                    class_protos.append(grid_protos)

            if class_protos:
                raw = torch.cat(class_protos, dim=0)
                prototypes[class_id] = torch.nn.functional.normalize(raw, dim=1, p=2)
            else:
                prototypes[class_id] = torch.zeros(1, self.feature_dim, device=features.device)

        # Task-adaptive refinement (episode-conditioned)
        if self.task_adapter is not None:
            prototypes = self.task_adapter(prototypes, support_features=features)
        return prototypes

    def _extract_grid_prototypes(
            self,
            features: torch.Tensor,
            masks: torch.Tensor,
            class_id: int,
            grid_size: int,
            ignore_index: int,
            code_ok: Optional[torch.Tensor],
            stats: Dict[str, int],
    ) -> Optional[torch.Tensor]:
        """Extract prototypes using global MAP (grid_size=1) or grid MAP."""

        K, C, h, w = features.shape

        if grid_size == 1:
            target_mask = masks.eq(class_id)
            valid_mask = masks.ne(ignore_index)

            base_sel = (target_mask & valid_mask).float()  # (K,h,w)
            sel = base_sel

            if code_ok is not None:
                stats["global_filtered"] += 1
                sel = sel * code_ok.float()
                if sel.sum() < self.min_selected_pixels_global:
                    stats["global_fallback"] += 1
                    sel = base_sel

            numerator = (features * sel.unsqueeze(1)).sum(dim=(0, 2, 3))
            denom = sel.sum().clamp(min=1.0)
            proto = numerator / denom
            return proto.unsqueeze(0)

        cell_h = h // grid_size
        cell_w = w // grid_size
        protos = []

        for gi in range(grid_size):
            for gj in range(grid_size):
                y1, y2 = gi * cell_h, (gi + 1) * cell_h
                x1, x2 = gj * cell_w, (gj + 1) * cell_w

                cell_feat = features[:, :, y1:y2, x1:x2]
                cell_mask = masks[:, y1:y2, x1:x2]

                target = cell_mask.eq(class_id)
                valid = cell_mask.ne(ignore_index)

                base_sel = (target & valid).float()  # (K,ch,cw)
                sel = base_sel

                if code_ok is not None:
                    stats["cell_filtered"] += 1
                    sel = sel * code_ok[:, y1:y2, x1:x2].float()
                    if sel.sum() < self.min_selected_pixels_cell:
                        stats["cell_fallback"] += 1
                        sel = base_sel

                n_pixels = sel.sum()
                if n_pixels < 1:
                    continue

                proto = (cell_feat * sel.unsqueeze(1)).sum(dim=(0, 2, 3)) / n_pixels
                protos.append(proto)

        if protos:
            return torch.stack(protos, dim=0)
        return None

    def compute_similarity(
            self,
            query_features: torch.Tensor,
            prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Cosine similarity (used for debug/inspection)."""
        B, C, h, w = query_features.shape
        q_norm = torch.nn.functional.normalize(query_features, dim=1, p=2)

        class_sims = []
        for class_id in range(self.num_classes):
            if class_id not in prototypes:
                class_sims.append(torch.zeros(B, 1, h, w, device=query_features.device))
                continue

            p = prototypes[class_id]
            p_norm = torch.nn.functional.normalize(p, dim=1, p=2)

            q_flat = q_norm.view(B, C, h * w)
            sim = torch.einsum("nc,bcs->bns", p_norm, q_flat).view(B, p_norm.shape[0], h, w)
            class_sims.append(sim.max(dim=1, keepdim=True).values)

        similarity = torch.cat(class_sims, dim=1) / self.temperature.clamp(min=0.01)
        return similarity

    def forward(
            self,
            support_features: torch.Tensor,
            support_masks: torch.Tensor,
            query_features: torch.Tensor,
            ignore_index: int = 255,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        prototypes = self.extract_prototypes(support_features, support_masks, indices=None, ignore_index=ignore_index)
        similarity = self.compute_similarity(query_features, prototypes)
        return similarity, prototypes