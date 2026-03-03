"""
backbone.py — Frozen VQ-VAE Feature Backbone
=============================================

Wraps the pretrained VQ-VAE encoder as a frozen feature extractor.

The VQ-VAE (van den Oord et al., 2017) was pretrained on TIGER histopathology
patches for image reconstruction. We reuse its encoder to get rich tissue
representations without labeled data.

Architecture:
    Input:  (B, 3, 256, 256)  — RGB histopathology patch
    ↓ VQ Encoder (frozen)
    z_e:    (B, 256, 64, 64)  — continuous encoder features
    ↓ VQ Quantizer (frozen)
    z_q:    (B, 256, 64, 64)  — quantized codebook features
    idx:    (B, 64, 64)       — codebook indices ∈ {0..4095}
    ↓ Feature Projector (learnable)
    feat:   (B, 256, 64, 64)  — adapted features for segmentation

The Feature Projector is a lightweight residual MLP that adapts frozen VQ features
for the segmentation task. This was identified as critical in Project-1: frozen
features alone cannot adapt to downstream tasks.

References:
    - VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualProjector(nn.Module):
    """
    Lightweight residual block for adapting frozen VQ features.

    Design: GroupNorm → ReLU → Conv1×1 → GroupNorm → ReLU → Conv1×1 + skip

    Initialization: Kaiming (fan_out) for convolutions, zero for the second
    conv's output so the block starts as identity. This way, the projector
    initially passes through VQ features unchanged and gradually learns
    task-specific adaptations.

    Lesson from Project-1: Zero-init the SECOND conv (not both) so the residual
    branch starts at zero → output = input + 0 = identity.
    """
    def __init__(self, dim: int, hidden_dim: int = None, norm_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.block = nn.Sequential(
            nn.GroupNorm(norm_groups, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
        )

        # ── Initialize ──────────────────────────────────────────────────
        # Kaiming for first conv, zero for second → starts as identity
        nn.init.kaiming_normal_(self.block[2].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.block[6].weight)  # Second conv: zero → identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W) — adapted features
        """
        return x + self.block(x)


class FeatureProjector(nn.Module):
    """
    Stack of residual projector blocks.
    Adapts frozen VQ-VAE features for the segmentation task.
    """
    def __init__(self, dim: int = 256, num_blocks: int = 2, norm_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualProjector(dim, dim, norm_groups, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class VQVAEBackbone(nn.Module):
    """
    Frozen VQ-VAE encoder + learnable feature projector.

    Usage:
        backbone = VQVAEBackbone(weights_path, modules_path)
        feat, z_q, indices = backbone(images)
        # feat:    (B, 256, 64, 64) — projected features (learnable)
        # z_q:     (B, 256, 64, 64) — quantized features (frozen)
        # indices: (B, 64, 64)      — codebook indices
    """
    def __init__(
        self,
        weights_path: str,
        modules_path: str,
        feature_dim: int = 256,
        projector_blocks: int = 2,
        norm_groups: int = 8,
        dropout: float = 0.1,
        freeze: bool = True,
    ):
        super().__init__()

        # ── Load pretrained VQ-VAE ──────────────────────────────────────
        # Add the VQ-VAE source directory to path to import its modules
        sys.path.insert(0, str(Path(modules_path)))
        from model import VQVAEPreTraining
        from safetensors.torch import load_file as safe_load_file

        # VQ-VAE config (must match the pretrained model exactly)
        config = Namespace(**{
            "name": "VQVAE", "module": "VQVAEPreTraining", "beta": 0.01,
            "optimizer_name": "Adam", "optimizer_kwargs": {"lr": 0.0001},
            "scheduler_name": "CosineAnnealingLR", "scheduler_kwargs": {"T_max": 65536},
            "log_image_every_n_steps": 64,
            "encoder": {
                "conv1_kwargs": {"in_channels": 3, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
                "resb1_kwargs": {"in_channels": 256, "out_channels": 256, "num_groups": 32, "dropout": 0.0},
                "nlcb1_kwargs": {"channels": 256, "num_groups": 32},
                "resb2_kwargs": {"in_channels": 256, "out_channels": 256, "num_groups": 32, "dropout": 0.0},
                "norm_last_kwargs": {"num_channels": 256, "num_groups": 32},
                "actv_last_kwargs": {"inplace": True},
                "conv_last_kwargs": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
                "downsample_blocks": {"channels": [256, 256], "num_res_blocks_per_downsample": 2,
                                      "res_block_norm_num_groups": 32, "dropout": 0.0}
            },
            "quantizer": {"num_embeddings": 4096, "embedding_dim": 256},
            "decoder": {
                "conv1_kwargs": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
                "resb1_kwargs": {"in_channels": 256, "out_channels": 256, "num_groups": 32, "dropout": 0.0},
                "nlcb1_kwargs": {"channels": 256, "num_groups": 32},
                "resb2_kwargs": {"in_channels": 256, "out_channels": 256, "num_groups": 32, "dropout": 0.0},
                "norm_last_kwargs": {"num_channels": 256, "num_groups": 32},
                "actv_last_kwargs": {"inplace": True},
                "conv_last_kwargs": {"in_channels": 256, "out_channels": 3, "kernel_size": 3, "stride": 1, "padding": 1},
                "upsample_blocks": {"channels": [256, 256], "num_res_blocks_per_upsample": 2,
                                    "res_block_norm_num_groups": 32, "dropout": 0.0}
            },
            "use_ema": True, "ema_kwargs": {"gamma": 0.999, "warm_up_steps": 1024}
        })

        # Load weights
        sd = safe_load_file(weights_path, device="cpu")
        vqvae_lightning = VQVAEPreTraining(config=config)
        missing, unexpected = vqvae_lightning.load_state_dict(sd, strict=False)
        print(f"[VQVAEBackbone] Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")

        # Extract the inner VQVAE model
        self.vqvae = vqvae_lightning.model

        # ── Freeze VQ-VAE ───────────────────────────────────────────────
        if freeze:
            for p in self.vqvae.parameters():
                p.requires_grad = False
            self.vqvae.eval()

        self.freeze = freeze
        self.feature_dim = feature_dim
        self.codebook = self.vqvae.quantizer.embedding  # nn.Embedding(4096, 256)

        # ── Learnable feature projector ─────────────────────────────────
        self.projector = FeatureProjector(
            dim=feature_dim, num_blocks=projector_blocks,
            norm_groups=norm_groups, dropout=dropout,
        )

    def train(self, mode: bool = True):
        """Override: keep VQ-VAE in eval mode even when model is training."""
        super().train(mode)
        if self.freeze:
            self.vqvae.eval()
        return self

    @torch.no_grad()
    def _encode_vq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run frozen VQ-VAE encoder + quantizer.

        Args:
            x: (B, 3, 256, 256)
        Returns:
            z_e:     (B, 256, 64, 64) — continuous features before quantization
            z_q:     (B, 256, 64, 64) — quantized codebook features
            indices: (B, 64, 64)      — codebook code indices
        """
        z_e = self.vqvae.encoder(x)           # (B, 256, 64, 64) after downsampling
        z_e = self.vqvae.proj(z_e)            # (B, 256, 64, 64) projected to codebook dim

        # Quantize: find nearest codebook entry per spatial position
        # z_q, min_embeddings = self.vqvae.quantizer(z_e)
        # For indices, we compute L2 distance to codebook
        B, C, h, w = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*h*w, C)
        cb = self.codebook.weight  # (K, C)

        # Efficient L2: ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z,e>
        z2 = (z_flat * z_flat).sum(dim=1, keepdim=True)  # (N, 1)
        e2 = (cb * cb).sum(dim=1).unsqueeze(0)             # (1, K)
        dist = z2 + e2 - 2.0 * (z_flat @ cb.t())           # (N, K)
        indices = dist.argmin(dim=1).view(B, h, w)          # (B, h, w)

        # Quantized features via codebook lookup
        z_q = F.embedding(indices, cb).permute(0, 3, 1, 2).contiguous()  # (B, C, h, w)

        return z_e, z_q, indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full backbone forward pass.

        Args:
            x: (B, 3, 256, 256) — normalized RGB images

        Returns:
            feat:    (B, 256, 64, 64) — projected features (LEARNABLE, has gradients)
            z_q:     (B, 256, 64, 64) — quantized features (frozen, no gradients)
            indices: (B, 64, 64)      — codebook indices (frozen)

        Tensor flow:
            x (B,3,256,256) → VQ-Encoder → z_e (B,256,64,64) → Projector → feat (B,256,64,64)
                                          → VQ-Quantizer → z_q (B,256,64,64), idx (B,64,64)
        """
        z_e, z_q, indices = self._encode_vq(x)

        # Project continuous features (learnable path — this is where gradients flow)
        feat = self.projector(z_e)  # (B, 256, 64, 64)

        return feat, z_q, indices
