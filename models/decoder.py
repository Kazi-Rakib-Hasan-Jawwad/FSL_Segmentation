"""
decoder.py — Progressive DeepLabV3+ Decoder
=============================================

A full progressive decoder inspired by DeepLabV3+ that upsamples through
two learned stages (64→128→256) instead of a single bilinear 4× jump.

Architecture:

    Stage 0 — ASPP + Skip fusion at 64×64:
        GPA output (B,256,64,64) → ASPP → (B,256,64,64)
        Skip features (B,256,64,64) → AttentionGate → (B,256,64,64)
        cat(aspp, attended_skip) → Conv3×3 → Conv3×3 → (B,256,64,64)

    Stage 1 — Progressive upsample to 128×128:
        (B,256,64,64) → TransposedConv(stride=2) → (B,128,128,128)
        → Conv3×3 → GroupNorm → ReLU → Conv3×3 → GroupNorm → ReLU → (B,128,128,128)

    Stage 2 — Progressive upsample to 256×256:
        (B,128,128,128) → TransposedConv(stride=2) → (B,64,256,256)
        → Conv3×3 → GroupNorm → ReLU → Conv3×3 → GroupNorm → ReLU → (B,64,256,256)

    Classifier — final 1×1 conv:
        (B,64,256,256) → Conv1×1 → (B,3,256,256)

All convolution weights are Kaiming-initialized (fan_out, ReLU).
Classifier head is small-scale initialized for balanced initial predictions.

Advantages over the old decoder:
    1. Full learned upsampling (not just bilinear interpolation)
    2. Progressive refinement at 128×128 and 256×256 resolutions
    3. Channel tapering (256→128→64) for efficiency at higher resolutions
    4. More decoder capacity → decoder learns more of the task

References:
    - DeepLab v3+: Chen et al., ECCV 2018
    - Progressive upsampling: Karnewar & Wang, TPAMI 2022 (MSG-GAN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionGate


# ─── Building Blocks ────────────────────────────────────────────────────


class ASPPConv(nn.Module):
    """Atrous convolution with GroupNorm + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int, norm_groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(norm_groups, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        out = self.pool(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Captures multi-scale context via parallel atrous convolutions
    at different dilation rates + global average pooling.

    Input:  (B, C, h, w)
    Output: (B, out_ch, h, w)
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 256, rates=(6, 12, 18), norm_groups: int = 8):
        super().__init__()
        modules = [
            # 1×1 convolution branch
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(norm_groups, out_ch),
                nn.ReLU(inplace=True),
            ),
        ]
        for rate in rates:
            modules.append(ASPPConv(in_ch, out_ch, rate, norm_groups))
        modules.append(ASPPPooling(in_ch, out_ch, norm_groups))

        self.branches = nn.ModuleList(modules)
        n_branches = len(modules)

        self.fuse = nn.Sequential(
            nn.Conv2d(n_branches * out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        cat = torch.cat(branch_outs, dim=1)
        return self.fuse(cat)


class UpsampleStage(nn.Module):
    """
    One progressive upsample stage: TransposedConv(×2) → Conv → Conv.

    Doubles spatial resolution while reducing channels:
        (B, in_ch, H, W) → (B, out_ch, 2H, 2W)

    Uses two Conv3×3 blocks after upsampling for refinement at the
    new resolution (following the DeepLabV3+ decoder philosophy).
    """

    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8, dropout: float = 0.1):
        super().__init__()

        # Learned upsampling via transposed conv
        self.upsample = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False,
        )
        self.upsample_norm = nn.GroupNorm(norm_groups, out_ch)

        # Refinement at the new resolution
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.upsample_norm(self.upsample(x)), inplace=True)
        x = self.refine(x)
        return x


# ─── Main Decoder ───────────────────────────────────────────────────────


class SegmentationDecoder(nn.Module):
    """
    Progressive DeepLabV3+ Decoder.

    Three-stage architecture with full learned upsampling:

        Stage 0 (64×64):  ASPP + attention-gated skip fusion
        Stage 1 (128×128): Learned upsample + refinement
        Stage 2 (256×256): Learned upsample + refinement
        Classifier:        Conv1×1 → 3-class logits

    All weights are Kaiming-initialized (fan_out, ReLU).
    Classifier uses small-scale init for balanced initial predictions.

    Tensor flow:
        gpa_out   (B,256,64,64) ─┐
                                  ├→ ASPP + Attn skip → fuse (B,256,64,64)
        skip_feat (B,256,64,64) ─┘
                                  → UpsampleStage (B,128,128,128)
                                  → UpsampleStage (B,64,256,256)
                                  → Conv1×1       (B,3,256,256)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 3,
        aspp_rates=(6, 12, 18),
        use_attention: bool = True,
        norm_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # ── Stage 0: ASPP + Skip fusion at 64×64 ───────────────────────
        self.aspp = ASPP(feature_dim, feature_dim, aspp_rates, norm_groups)

        if use_attention:
            self.attention_gate = AttentionGate(
                gate_channels=feature_dim,
                skip_channels=feature_dim,
                norm_groups=norm_groups,
            )

        # Fusion: cat(aspp, skip) → conv → conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, feature_dim),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: 64×64 → 128×128 (256ch → 128ch) ──────────────────
        mid_ch = feature_dim // 2   # 128
        self.up_stage1 = UpsampleStage(feature_dim, mid_ch, norm_groups, dropout)

        # ── Stage 2: 128×128 → 256×256 (128ch → 64ch) ──────────────────
        low_ch = mid_ch // 2        # 64
        self.up_stage2 = UpsampleStage(mid_ch, low_ch, norm_groups, dropout)

        # ── Classifier head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(norm_groups, low_ch), low_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(low_ch, num_classes, 1),
        )

        # ── Kaiming initialization for ALL conv weights ─────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming (fan_out) initialization for all conv layers.
        Classifier final layer gets small-scale init for balanced predictions.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Small-scale init for classifier → balanced class predictions at start
        final_conv = self.classifier[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        gpa_features: torch.Tensor,
        skip_features: torch.Tensor,
        target_size: tuple = None,
    ) -> torch.Tensor:
        """
        Decode GPA-allocated features into segmentation logits.

        Args:
            gpa_features:  (B, 256, 64, 64) — from GPA module
            skip_features: (B, 256, 64, 64) — from backbone (projected z_e)
            target_size:   (H, W) — final output resolution (default: 256×256)

        Returns:
            logits: (B, num_classes, H, W) — per-pixel class logits

        Tensor flow:
            gpa  (B,256,64,64)  → ASPP → (B,256,64,64)
            skip (B,256,64,64)  → Attn → (B,256,64,64)
            cat → fuse → (B,256,64,64)
            → Stage1 → (B,128,128,128)
            → Stage2 → (B,64,256,256)
            → Classifier → (B,3,256,256)
        """
        # ── Stage 0: ASPP + Skip fusion at 64×64 ───────────────────────
        aspp_out = self.aspp(gpa_features)

        if self.use_attention:
            attended_skip = self.attention_gate(g=aspp_out, x=skip_features)
        else:
            attended_skip = skip_features

        fused = torch.cat([aspp_out, attended_skip], dim=1)  # (B, 512, 64, 64)
        x = self.fuse_conv(fused)  # (B, 256, 64, 64)

        # ── Stage 1: 64→128 ────────────────────────────────────────────
        x = self.up_stage1(x)  # (B, 128, 128, 128)

        # ── Stage 2: 128→256 ───────────────────────────────────────────
        x = self.up_stage2(x)  # (B, 64, 256, 256)

        # ── Classifier ─────────────────────────────────────────────────
        logits = self.classifier(x)  # (B, 3, 256, 256)

        # Final resize if target_size differs from current resolution
        if target_size is not None and logits.shape[2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits
