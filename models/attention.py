"""
attention.py — Spatial Attention Gates for Feature Fusion
=========================================================

Implements attention gates that learn to selectively focus on relevant
spatial regions when fusing features from different sources (e.g.,
skip connections and decoder features).

Design: Uses GroupNorm throughout (lesson from Project-1: BatchNorm fails
at batch_size=1, which is common in episodic few-shot training).

Tensor flow:
    g (gating signal):  (B, C, h, w)  — from decoder
    x (skip features):  (B, C, h, w)  — from backbone
    ↓ Additive attention
    attention_map:      (B, 1, h, w)  — sigmoid spatial weights
    ↓ Element-wise multiply
    output:             (B, C, h, w)  — attended features

References:
    - Attention U-Net: Oktay et al., "Attention U-Net: Learning Where to Look
      for the Pancreas", MIDL 2018
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Additive spatial attention gate.

    Learns to highlight informative spatial regions in skip connections
    based on the gating signal from the decoder path.

    All normalization uses GroupNorm (stable for batch_size=1).
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None, norm_groups: int = 8):
        """
        Args:
            gate_channels: Channels in gating signal (from decoder)
            skip_channels: Channels in skip connection (from encoder/backbone)
            inter_channels: Intermediate channels (default: skip_channels // 2)
            norm_groups: Number of GroupNorm groups
        """
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 1)

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(min(norm_groups, inter_channels), inter_channels),
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(min(norm_groups, inter_channels), inter_channels),
        )

        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention gate.

        Args:
            g: (B, C_gate, h, w)   — gating signal (from decoder path)
            x: (B, C_skip, h, w)   — skip features (from encoder/backbone)

        Returns:
            out: (B, C_skip, h, w) — attended skip features

        Tensor flow:
            g → W_g → (B, inter, h, w)
            x → W_x → (B, inter, h, w)
            ↓ Add + ReLU + ψ
            α → (B, 1, h, w) — attention map ∈ [0, 1]
            output = x * α
        """
        # Align spatial dimensions if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)     # (B, inter, h, w)
        x1 = self.W_x(x)     # (B, inter, h, w)
        combined = self.relu(g1 + x1)  # Additive attention
        alpha = self.psi(combined)      # (B, 1, h, w) — attention weights

        return x * alpha  # Element-wise gating
