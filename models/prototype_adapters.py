"""prototype_adapters.py — Task-adaptive prototypes + contrastive code selection
===============================================================================

This module contains OPTIONAL add-ons for the prototypical FSL segmentation pipeline.

They do NOT require any new data/modalities.

1) CodebookContrastiveSelector
   - Uses VQ-VAE code indices + support masks to select codes enriched in class-vs-context.

2) TaskAdaptivePrototypeAdapter
   - FEAT-style set-to-set refinement (Transformer) that adapts prototype tokens per-episode.
   - Starts as identity via a learnable interpolation gate gamma (initialized to 0).
"""

from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn


class CodebookContrastiveSelector(nn.Module):
    """Context-contrastive code selection using only (indices, masks).

    For each class c, compute:
        s(k,c) = log((p(k|c)+eps) / (p(k|ctx)+eps))
    and select top-K code ids k.

    Context can be defined without extra data:
        - non_target: valid pixels where mask != c
        - all_valid : all valid pixels (includes class c)
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        topk: int = 64,
        eps: float = 1e-6,
        min_target_count: int = 1,
        context_mode: Literal["non_target", "all_valid"] = "non_target",
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.topk = int(topk)
        self.eps = float(eps)
        self.min_target_count = int(min_target_count)
        self.context_mode = context_mode

    @torch.no_grad()
    def forward(
            self,
            indices: torch.Tensor,  # (K,h,w)
            masks: torch.Tensor,  # (K,h,w)
            num_classes: int,
            ignore_index: int = 255,
    ) -> Dict[int, Optional[torch.Tensor]]:
        if indices.shape != masks.shape:
            raise ValueError(f"indices and masks must match; got {indices.shape} vs {masks.shape}")

        device = indices.device
        valid = masks.ne(ignore_index)

        flat_idx = indices.reshape(-1).long()
        flat_valid = valid.reshape(-1)

        def _hist(sel_flat_bool: torch.Tensor) -> torch.Tensor:
            sel_idx = flat_idx[sel_flat_bool]
            if sel_idx.numel() == 0:
                return torch.zeros(self.codebook_size, device=device, dtype=torch.float32)
            ones = torch.ones(sel_idx.numel(), device=device, dtype=torch.float32)
            h = torch.zeros(self.codebook_size, device=device, dtype=torch.float32)
            return h.index_add_(0, sel_idx, ones)

        all_valid_counts = None
        if self.context_mode == "all_valid":
            all_valid_counts = _hist(flat_valid)

        selected: Dict[int, Optional[torch.Tensor]] = {}
        for c in range(num_classes):
            tgt = valid & masks.eq(c)
            flat_tgt = tgt.reshape(-1)
            if int(flat_tgt.sum().item()) < 1:
                selected[c] = None
                continue

            tgt_counts = _hist(flat_tgt)

            if self.context_mode == "all_valid":
                ctx_counts = all_valid_counts
            else:
                ctx = valid & (~masks.eq(c))
                flat_ctx = ctx.reshape(-1)
                if int(flat_ctx.sum().item()) < 1:
                    selected[c] = None
                    continue
                ctx_counts = _hist(flat_ctx)

            p_t = tgt_counts / tgt_counts.sum().clamp(min=1.0)
            p_c = ctx_counts / ctx_counts.sum().clamp(min=1.0)
            score = torch.log((p_t + self.eps) / (p_c + self.eps))

            present = tgt_counts.ge(self.min_target_count)
            score = score.masked_fill(~present, float("-inf"))
            if torch.isneginf(score).all():
                selected[c] = None
                continue

            k = min(self.topk, int(present.sum().item()))
            selected[c] = torch.topk(score, k=k, largest=True).indices.long()

        return selected


class TaskAdaptivePrototypeAdapter(nn.Module):
    """FEAT-style set-to-set prototype refinement (Transformer encoder).

    Input: prototypes dict {class_id: (N_i, C)}
    Output: same dict, but each token is refined using all tokens in the episode.

    Stability: out = in + gamma*(adapt(in)-in), gamma starts at 0 -> identity.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        use_episode_token: bool = True,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.use_episode_token = bool(use_episode_token)

        self.class_emb = nn.Embedding(num_classes, feature_dim)
        nn.init.zeros_(self.class_emb.weight)  # neutral at start

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.episode_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.episode_mlp[-1].weight)
        nn.init.zeros_(self.episode_mlp[-1].bias)

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        prototypes: Dict[int, torch.Tensor],
        support_features: torch.Tensor,  # (K,C,h,w)
    ) -> Dict[int, torch.Tensor]:
        keys = sorted(prototypes.keys())
        lengths = [int(prototypes[k].shape[0]) for k in keys]

        x0 = torch.cat([prototypes[k] for k in keys], dim=0)  # (N,C)
        if x0.numel() == 0:
            return prototypes

        device = x0.device
        class_ids = torch.cat(
            [torch.full((lengths[i],), keys[i], device=device, dtype=torch.long) for i in range(len(keys))],
            dim=0,
        )

        x = x0 + self.class_emb(class_ids)  # (N,C)
        x = x.unsqueeze(0)  # (1,N,C)

        if self.use_episode_token:
            ep = support_features.mean(dim=(0, 2, 3))  # (C,)
            ep = self.episode_mlp(ep).view(1, 1, -1)   # (1,1,C)
            x_in = torch.cat([ep, x], dim=1)           # (1,1+N,C)
            x_adapt = self.encoder(x_in)[:, 1:, :]     # (1,N,C)
        else:
            x_adapt = self.encoder(x)

        x_adapt = x_adapt.squeeze(0)  # (N,C)

        g = self.gamma.clamp(0.0, 1.0)
        x_out = x0 + g * (x_adapt - x0)

        out: Dict[int, torch.Tensor] = {}
        start = 0
        for k, n in zip(keys, lengths):
            out[k] = x_out[start:start + n]
            start += n
        return out