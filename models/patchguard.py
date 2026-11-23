"""PatchGuard model: DINOv3 backbone + Patch-Aware classifier + mask decoder.

Compatible with trainer.py.

    out = model(pixel_values, pad_mask=None)

Required model outputs:
    cls_logits:  (B,)
    mask_logits: (B, 1, H, W)

Additional analysis outputs:
    patch_logits:    (B, N)
    patch_score_map: (B, 1, h, w)
    patch_valid:     (B, N) bool  (False = padding patch)
    topk_indices:    (B, K) or None

Design notes
  - The image-level patch head always reads the LAST layer's patch tokens, i.e. the same
    features the Global baseline's CLS token comes from. Only the aggregation differs, so the
    "Patch vs Global" comparison is not confounded by extra features.
  - The mask decoder can additionally fuse several intermediate layers (backbone
    feature_layers), as in DINO-FIL; this only affects localization.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoBackbone, to_feature_map
from .mask_decoder import ConvBlock, MaskDecoder
from .patch_head import PatchHead


def patch_valid_from_pad(pad_mask: Optional[torch.Tensor], patch_size: int) -> Optional[torch.Tensor]:
    """(B,1,H,W)/(B,H,W) pixel pad mask (1 = real image) -> (B, N) bool, valid if >= half real."""
    if pad_mask is None:
        return None
    if pad_mask.dim() == 3:
        pad_mask = pad_mask.unsqueeze(1)
    cover = F.avg_pool2d(pad_mask.float(), kernel_size=patch_size, stride=patch_size)
    return (cover >= 0.5).flatten(1)


class LayerFusion(nn.Module):
    """Concatenate several (B, D, h, w) maps and project back to D channels."""

    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim * n_layers, dim, kernel_size=1, bias=False))
        self.block = ConvBlock(dim, dim)

    def forward(self, maps):
        return self.block(self.proj(torch.cat(maps, dim=1)))


class PatchGuard(nn.Module):
    def __init__(
        self,
        backbone: DinoBackbone,
        aggregation: str = "topk",
        patch_hidden_dim: int = 256,
        topk_ratio: float = 0.02,
        dropout: float = 0.1,
        decoder_hidden_channels=(128, 64),
    ):
        """
        Args:
            backbone: DinoBackbone instance (set its feature_layers to enable decoder fusion).
            aggregation: "average", "topk" or "soft_topk".
            patch_hidden_dim: MLP width used by PatchHead.
            topk_ratio: fraction of valid patches used by Top-k aggregation.
            dropout: dropout used in PatchHead / MaskDecoder.
            decoder_hidden_channels: channels of the lightweight localization decoder.
        """
        super().__init__()
        self.backbone = backbone

        self.patch_head = PatchHead(
            in_dim=backbone.hidden_size,
            aggregation=aggregation,
            hidden_dim=patch_hidden_dim,
            dropout=dropout,
            topk_ratio=topk_ratio,
        )

        n_fuse = len(backbone.feature_layers)
        self.fusion = LayerFusion(backbone.hidden_size, n_fuse) if n_fuse > 0 else None

        self.mask_decoder = MaskDecoder(
            in_channels=backbone.hidden_size,
            hidden_channels=decoder_hidden_channels,
            dropout=dropout,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            pixel_values: (B, 3, H, W), ImageNet-normalized, H/W multiples of 16.
            pad_mask: optional (B, 1, H, W) or (B, H, W), 1 = real image, 0 = padding.
        """
        H, W = pixel_values.shape[-2:]
        feats = self.backbone(pixel_values)
        patch_valid = patch_valid_from_pad(pad_mask, self.backbone.patch_size)

        # 1) Image-level Patch Average / Top-k / soft Top-k classification (last layer).
        patch_out = self.patch_head(feats["patch_tokens"], patch_valid)

        # 2) Pixel-level localization (optionally multi-layer).
        if self.fusion is not None:
            maps = [to_feature_map(feats, t) for t in feats["layer_patch_tokens"]]
            feature_map = self.fusion(maps)
        else:
            feature_map = to_feature_map(feats)
        mask_logits = self.mask_decoder(feature_map, output_size=(H, W))

        h, w = feats["grid_size"]
        patch_score_map = patch_out["patch_logits"].reshape(-1, 1, h, w)

        return {
            "cls_logits": patch_out["cls_logits"],
            "mask_logits": mask_logits,
            "patch_logits": patch_out["patch_logits"],
            "patch_score_map": patch_score_map,
            "patch_valid": patch_valid,
            "topk_indices": patch_out["topk_indices"],
        }
