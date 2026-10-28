"""PatchGuard model: DINOv3 backbone + Patch-Aware classifier + mask decoder.

Compatible with the existing Tiny-Edit-Guard trainer.py.

Required model outputs:
    cls_logits:  (B,)
    mask_logits: (B, 1, H, W)

Additional analysis outputs:
    patch_logits:    (B, N)
    patch_score_map: (B, 1, h, w)
    topk_indices:    (B, K) or None
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import DinoBackbone, to_feature_map
from .mask_decoder import MaskDecoder
from .patch_head import PatchHead


class PatchGuard(nn.Module):
    def __init__(
        self,
        backbone: DinoBackbone,
        aggregation: str = "topk",
        patch_hidden_dim: int = 256,
        topk_ratio: float = 0.10,
        dropout: float = 0.1,
        decoder_hidden_channels=(128, 64),
    ):
        """
        Args:
            backbone: existing DinoBackbone instance.
            aggregation: "average" or "topk".
            patch_hidden_dim: MLP width used by PatchHead.
            topk_ratio: fraction of patches used by Top-k aggregation.
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

        self.mask_decoder = MaskDecoder(
            in_channels=backbone.hidden_size,
            hidden_channels=decoder_hidden_channels,
            dropout=dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            pixel_values: (B, 3, H, W), ImageNet-normalized.
                          H/W must be multiples of DINOv3 patch size (16).

        Returns:
            cls_logits:      (B,)
            mask_logits:     (B, 1, H, W)
            patch_logits:    (B, N)
            patch_score_map: (B, 1, h, w)
            topk_indices:    (B, K) or None
        """
        H, W = pixel_values.shape[-2:]

        feats = self.backbone(pixel_values)

        # 1) Image-level Patch Average / Patch Top-k classification.
        patch_out = self.patch_head(feats["patch_tokens"])

        # 2) Pixel-level localization.
        feature_map = to_feature_map(feats)  # (B, D, h, w)
        mask_logits = self.mask_decoder(
            feature_map,
            output_size=(H, W),
        )

        h, w = feats["grid_size"]
        patch_score_map = patch_out["patch_logits"].reshape(
            patch_out["patch_logits"].shape[0], 1, h, w
        )

        return {
            "cls_logits": patch_out["cls_logits"],
            "mask_logits": mask_logits,
            "patch_logits": patch_out["patch_logits"],
            "patch_score_map": patch_score_map,
            "topk_indices": patch_out["topk_indices"],
        }


if __name__ == "__main__":
    # This file normally needs the gated facebook/dinov3-vits16 checkpoint.
    # Run after `huggingface-cli login`, or test patch_head.py / mask_decoder.py
    # independently before connecting the real backbone.
    print(
        "PatchGuard is ready. Instantiate it with an existing DinoBackbone, e.g.\\n"
        "  backbone = DinoBackbone(freeze=True)\\n"
        "  model = PatchGuard(backbone, aggregation='topk', topk_ratio=0.10)"
    )
