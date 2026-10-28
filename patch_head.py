from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PatchHead(nn.Module):
    """SegFormer feature map -> patch scores -> image-level logit."""

    SUPPORTED_AGGREGATIONS = {"average", "topk"}

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        aggregation: str = "average",
        topk_ratio: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if aggregation not in self.SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self.SUPPORTED_AGGREGATIONS}, "
                f"got {aggregation!r}"
            )
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1].")

        self.aggregation = aggregation
        self.topk_ratio = topk_ratio

        self.patch_scorer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self, feature_map: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}"
            )

        score_map = self.patch_scorer(feature_map)   # [B,1,H,W]
        patch_scores = score_map.flatten(1)          # [B,H*W]

        topk_indices = None
        if self.aggregation == "average":
            image_logits = patch_scores.mean(dim=1)
        else:
            num_patches = patch_scores.shape[1]
            k = max(1, round(num_patches * self.topk_ratio))
            k = min(k, num_patches)

            topk_scores, topk_indices = torch.topk(
                patch_scores, k=k, dim=1, largest=True, sorted=True
            )
            image_logits = topk_scores.mean(dim=1)

        return {
            "logits": image_logits,
            "patch_scores": patch_scores,
            "score_map": score_map,
            "topk_indices": topk_indices,
        }


if __name__ == "__main__":
    x = torch.randn(4, 256, 16, 16)

    for mode in ("average", "topk"):
        head = PatchHead(aggregation=mode, topk_ratio=0.1)
        out = head(x)
        print(mode)
        print("logits:", out["logits"].shape)
        print("patch_scores:", out["patch_scores"].shape)
        print("score_map:", out["score_map"].shape)
