"""Lightweight DINOv3 patch-feature decoder for Tiny-Edit-Guard.

Input:
    feature_map: (B, D, h, w), produced by backbone.to_feature_map(feats)

Output:
    mask_logits: (B, 1, H, W)

The decoder predicts one binary edited-region logit per pixel.
Sigmoid is intentionally NOT applied here because trainer.py uses
binary_cross_entropy_with_logits().
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # GroupNorm is batch-size friendly and works well for small course-project batches.
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: DINOv3 hidden dimension (= backbone.hidden_size).
            hidden_channels: decoder channel widths.
            dropout: spatial dropout before the final prediction layer.
        """
        super().__init__()

        c1, c2 = hidden_channels
        self.decoder = nn.Sequential(
            ConvBlock(in_channels, c1),
            ConvBlock(c1, c2),
            nn.Dropout2d(dropout),
            nn.Conv2d(c2, 1, kernel_size=1),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, D, h, w)
            output_size: (H, W) of the original model input

        Returns:
            mask_logits: (B, 1, H, W)
        """
        if feature_map.dim() != 4:
            raise ValueError(
                f"feature_map must have shape (B, D, h, w), got {tuple(feature_map.shape)}"
            )

        logits_lowres = self.decoder(feature_map)

        mask_logits = F.interpolate(
            logits_lowres,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return mask_logits


if __name__ == "__main__":
    # 512x512 input with patch size 16 -> 32x32 patch feature map.
    x = torch.randn(4, 384, 32, 32)
    decoder = MaskDecoder(in_channels=384)
    y = decoder(x, output_size=(512, 512))
    print("mask_logits:", tuple(y.shape))
