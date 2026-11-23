"""SegFormer-B0 localization baseline (binary edited-region mask).

    out = model(pixel_values)
    out["mask_logits"]  # (B, 1, H, W) raw logits at input resolution
    out["cls_logits"]   # (B,) image-level logit = mean of top-k pixel logits,
                        # so classification metrics are comparable with the other models

Uses the ImageNet-pretrained MiT-B0 encoder (`nvidia/mit-b0`); the decode head is newly
initialized with a single output channel. The loss is computed outside (trainer / losses.py),
not by the HF model, so `labels` is never passed.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .patch_head import masked_topk_mean

DEFAULT_MODEL = "nvidia/mit-b0"


class SegFormerBaseline(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        topk_ratio: float = 0.01,
        model: Optional[nn.Module] = None,
    ):
        """
        Args:
            topk_ratio: fraction of pixels averaged to get the image-level logit.
            model: an already-built SegformerForSemanticSegmentation (tests / offline use).
        """
        super().__init__()
        self.model = model if model is not None else SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        self.topk_ratio = topk_ratio

    def forward(self, pixel_values: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """pad_mask: optional (B,1,H,W)/(B,H,W), 1 = real image; padded pixels never enter top-k."""
        H, W = pixel_values.shape[-2:]
        logits = self.model(pixel_values=pixel_values).logits  # (B, 1, H/4, W/4)
        mask_logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        valid = None
        if pad_mask is not None:
            valid = (pad_mask.unsqueeze(1) if pad_mask.dim() == 3 else pad_mask).flatten(1) > 0.5
        cls_logits, _ = masked_topk_mean(mask_logits.flatten(1), valid, self.topk_ratio)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}
