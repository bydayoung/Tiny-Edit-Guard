"""Patch-aware image classification head for Tiny-Edit-Guard.

Input:
    patch_tokens: (B, N, D)

Output:
    cls_logits:    (B,)    raw image-level logits
    patch_logits:  (B, N)  raw local edited-region logits
    topk_indices:  (B, K) or None

The local scorer is shared between Patch Average and Patch Top-k.
Only the aggregation rule changes.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PatchHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        aggregation: str = "topk",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        topk_ratio: float = 0.10,
    ):
        """
        Args:
            in_dim: DINOv3 patch-token dimension (= backbone.hidden_size).
            aggregation: "average" or "topk".
            hidden_dim: local patch scorer hidden dimension.
            dropout: dropout probability in the scorer.
            topk_ratio: fraction of local patch logits used by Top-k aggregation.
        """
        super().__init__()

        if aggregation not in {"average", "topk"}:
            raise ValueError(
                f"aggregation must be 'average' or 'topk', got {aggregation!r}"
            )
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        self.aggregation = aggregation
        self.topk_ratio = float(topk_ratio)

        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            patch_tokens: (B, N, D)

        Returns:
            cls_logits:   (B,)
            patch_logits: (B, N)
            topk_indices: (B, K) for Top-k, otherwise None
        """
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"patch_tokens must have shape (B, N, D), got {tuple(patch_tokens.shape)}"
            )

        patch_logits = self.scorer(patch_tokens).squeeze(-1)  # (B, N)

        if self.aggregation == "average":
            cls_logits = patch_logits.mean(dim=1)
            topk_indices = None
        else:
            n_patches = patch_logits.shape[1]
            k = max(1, int(round(n_patches * self.topk_ratio)))
            k = min(k, n_patches)

            topk = torch.topk(
                patch_logits,
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            )
            cls_logits = topk.values.mean(dim=1)
            topk_indices = topk.indices

        return {
            "cls_logits": cls_logits,
            "patch_logits": patch_logits,
            "topk_indices": topk_indices,
        }


if __name__ == "__main__":
    # DINOv3 ViT-S/16 + 512x512 example:
    # 32 x 32 = 1024 patches. hidden size is read from the backbone in real use.
    x = torch.randn(4, 1024, 384)

    for aggregation in ("average", "topk"):
        head = PatchHead(
            in_dim=384,
            aggregation=aggregation,
            topk_ratio=0.10,
        )
        out = head(x)
        print(f"[{aggregation}]")
        print(" cls_logits:  ", tuple(out["cls_logits"].shape))
        print(" patch_logits:", tuple(out["patch_logits"].shape))
        if out["topk_indices"] is not None:
            print(" topk_indices:", tuple(out["topk_indices"].shape))
