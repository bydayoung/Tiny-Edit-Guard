"""DINOv3 ViT backbone.

Output contract (shared with patch_head.py / mask_decoder.py / patchguard.py):

    feats = backbone(pixel_values)          # pixel_values: (B, 3, H, W), ImageNet-normalized
    feats["cls_token"]     # (B, D)
    feats["patch_tokens"]  # (B, N, D), N = h * w, row-major (left->right, top->bottom)
    feats["grid_size"]     # (h, w) = (H // 16, W // 16)

    to_feature_map(feats)  # (B, D, h, w), convenient for conv decoders

H and W must be multiples of the patch size (16). Register tokens are dropped.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 (LVD-1689M) checkpoints use ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        model: Optional[nn.Module] = None,
    ):
        """
        Args:
            model_name: Hugging Face checkpoint id (gated: run `huggingface-cli login` first).
            freeze: freeze all backbone weights.
            unfreeze_last_n: with freeze=True, unfreeze the last n transformer blocks + final norm.
            model: an already-built DINOv3ViTModel (for tests / offline use); skips from_pretrained.
        """
        super().__init__()
        self.model = model if model is not None else AutoModel.from_pretrained(model_name)
        cfg = self.model.config
        self.hidden_size: int = cfg.hidden_size
        self.patch_size: int = cfg.patch_size
        self.num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        self.set_trainable(freeze, unfreeze_last_n)

    def set_trainable(self, freeze: bool = True, unfreeze_last_n: int = 0) -> None:
        for p in self.model.parameters():
            p.requires_grad = not freeze
        if freeze and unfreeze_last_n > 0:
            for block in self._blocks()[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.model.norm.parameters():
                p.requires_grad = True

    def _blocks(self) -> nn.ModuleList:
        # transformers 4.x: model.layer / 5.x: model.model.layer (blocks moved into an encoder)
        if hasattr(self.model, "layer"):
            return self.model.layer
        return self.model.model.layer

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = pixel_values.shape[-2:]
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Input size {(H, W)} must be a multiple of patch size {self.patch_size}")

        hidden = self.model(pixel_values=pixel_values).last_hidden_state  # (B, 1 + R + N, D)
        start = 1 + self.num_register_tokens
        return {
            "cls_token": hidden[:, 0],
            "patch_tokens": hidden[:, start:],
            "grid_size": (H // self.patch_size, W // self.patch_size),
        }


def to_feature_map(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, N, D) patch tokens -> (B, D, h, w)."""
    h, w = feats["grid_size"]
    tokens = feats["patch_tokens"]
    B, N, D = tokens.shape
    # contiguous(): the transposed view can make CPU conv2d fail ("view size is not compatible")
    return tokens.transpose(1, 2).reshape(B, D, h, w).contiguous()
