"""Global baseline: image-level edited/real classification from the DINOv3 CLS token.

    out = model(pixel_values)
    out["cls_logits"]  # (B,) raw logit, sigmoid -> P(edited)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import DinoBackbone


class GlobalBaseline(nn.Module):
    def __init__(
        self,
        backbone: DinoBackbone,
        use_mean_patch: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            use_mean_patch: concat mean-pooled patch tokens to the CLS token ([CLS; mean(patches)]).
            hidden_dim: MLP hidden size; 0 means a single linear layer (linear probe).
        """
        super().__init__()
        self.backbone = backbone
        self.use_mean_patch = use_mean_patch
        in_dim = backbone.hidden_size * (2 if use_mean_patch else 1)

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 1))

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(pixel_values)
        x = feats["cls_token"]
        if self.use_mean_patch:
            x = torch.cat([x, feats["patch_tokens"].mean(dim=1)], dim=-1)
        return {"cls_logits": self.head(x).squeeze(-1)}
