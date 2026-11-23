"""Patch-aware image classification head for Tiny-Edit-Guard.

Input:
    patch_tokens: (B, N, D)
    patch_valid:  (B, N) bool, optional — False for patches that are pure padding

Output:
    cls_logits:    (B,)    raw image-level logits
    patch_logits:  (B, N)  raw local edited-region logits
    topk_indices:  (B, K) or None

The local scorer is shared between all aggregation rules; only the pooling changes.
    average   : mean over valid patches
    topk      : mean of the top-k valid patches, k = round(n_valid * topk_ratio)
    soft_topk : softmax-weighted sum over standardised patch logits with a learnable
                sharpness kappa (ReGFLoW-style soft top-k, no fixed k to tune)

For tiny edits (1-3% of the image ≈ 10-30 of 1024 patches) topk_ratio should be about
0.01-0.02; with 0.10 the top-k mean is mostly non-edited patches.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

AGGREGATIONS = ("average", "topk", "soft_topk")


def masked_mean(values: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """values (B, N), valid (B, N) bool -> (B,)"""
    if valid is None:
        return values.mean(dim=1)
    w = valid.to(values.dtype)
    return (values * w).sum(1) / w.sum(1).clamp(min=1.0)


def masked_topk_mean(values: torch.Tensor, valid: Optional[torch.Tensor], ratio: float):
    """Mean of the top-k valid entries per row, k_i = max(1, round(n_valid_i * ratio)).

    Returns (mean (B,), indices (B, k_max)). Padded entries are never selected.
    """
    B, N = values.shape
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    n_valid = valid.sum(1).clamp(min=1)
    k = torch.round(n_valid.float() * ratio).long().clamp(min=1)
    k = torch.minimum(k, n_valid)
    k_max = int(k.max())
    masked = values.masked_fill(~valid, float("-inf"))
    top = torch.topk(masked, k=k_max, dim=1, largest=True, sorted=True)
    keep = torch.arange(k_max, device=values.device)[None, :] < k[:, None]
    vals = torch.where(keep, top.values, torch.zeros_like(top.values))
    return vals.sum(1) / k.to(values.dtype), top.indices


class PatchHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        aggregation: str = "topk",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        topk_ratio: float = 0.02,
        soft_kappa_init: float = 2.0,
    ):
        """
        Args:
            in_dim: DINOv3 patch-token dimension (= backbone.hidden_size).
            aggregation: "average", "topk" or "soft_topk".
            hidden_dim: local patch scorer hidden dimension.
            dropout: dropout probability in the scorer.
            topk_ratio: fraction of valid patches used by Top-k aggregation.
            soft_kappa_init: initial sharpness of soft_topk (learned, kept positive).
        """
        super().__init__()

        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
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
        if aggregation == "soft_topk":
            # softplus(raw) = kappa  ->  raw = log(exp(kappa) - 1)
            raw = torch.log(torch.expm1(torch.tensor(float(soft_kappa_init))))
            self.kappa_raw = nn.Parameter(raw)

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self.kappa_raw)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"patch_tokens must have shape (B, N, D), got {tuple(patch_tokens.shape)}"
            )

        patch_logits = self.scorer(patch_tokens).squeeze(-1)  # (B, N)
        topk_indices = None

        if self.aggregation == "average":
            cls_logits = masked_mean(patch_logits, patch_valid)
        elif self.aggregation == "topk":
            cls_logits, topk_indices = masked_topk_mean(patch_logits, patch_valid, self.topk_ratio)
        else:  # soft_topk
            valid = patch_valid if patch_valid is not None else torch.ones_like(patch_logits, dtype=torch.bool)
            mu = masked_mean(patch_logits, valid)[:, None]
            var = masked_mean((patch_logits - mu) ** 2, valid)[:, None]
            z = (patch_logits - mu) / (var.sqrt() + 1e-4)
            w = torch.softmax((self.kappa * z).masked_fill(~valid, float("-inf")), dim=1)
            cls_logits = (w * patch_logits.masked_fill(~valid, 0.0)).sum(1)

        return {
            "cls_logits": cls_logits,
            "patch_logits": patch_logits,
            "topk_indices": topk_indices,
        }


if __name__ == "__main__":
    # DINOv3 ViT-S/16 + 512x512 example: 32 x 32 = 1024 patches, hidden size 384.
    x = torch.randn(4, 1024, 384)
    valid = torch.ones(4, 1024, dtype=torch.bool)
    valid[0, 512:] = False  # half of image 0 is padding
    for aggregation in AGGREGATIONS:
        out = PatchHead(in_dim=384, aggregation=aggregation)(x, valid)
        print(aggregation, tuple(out["cls_logits"].shape), tuple(out["patch_logits"].shape))
