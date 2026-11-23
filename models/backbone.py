"""DINOv3 ViT backbone.

Output contract (shared with patch_head.py / mask_decoder.py / patchguard.py):

    feats = backbone(pixel_values)          # pixel_values: (B, 3, H, W), ImageNet-normalized
    feats["cls_token"]     # (B, D)
    feats["patch_tokens"]  # (B, N, D), N = h * w, row-major (left->right, top->bottom)
    feats["grid_size"]     # (h, w) = (H // 16, W // 16)
    feats["layer_patch_tokens"]  # list of (B, N, D), only when feature_layers is set

    to_feature_map(feats)  # (B, D, h, w), convenient for conv decoders

H and W must be multiples of the patch size (16). DINOv3 uses RoPE, so any such size works
(no need to resize to a fixed resolution). Register tokens are dropped.

Fine-tuning options (all keep the pretrained weights, see train.py):
    freeze=True, unfreeze_last_n=0, lora_rank=0   -> fully frozen (old default)
    unfreeze_last_n=n                             -> last n blocks + final norm trainable
    lora_rank=r                                   -> LoRA on attention q/v projections
                                                     (DinoLizer / PatchHead style)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 (LVD-1689M) checkpoints use ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(x)); W frozen, A/B trainable, B initialised to 0."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)

    @property
    def weight(self):  # some HF code paths read .weight / .bias (dtype checks)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale


class DinoBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        model: Optional[nn.Module] = None,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_last_n: int = 0,
        lora_dropout: float = 0.0,
        feature_layers: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            model_name: Hugging Face checkpoint id (gated: run `huggingface-cli login` first).
            freeze: freeze all backbone weights.
            unfreeze_last_n: with freeze=True, unfreeze the last n transformer blocks + final norm.
            model: an already-built DINOv3ViTModel (for tests / offline use); skips from_pretrained.
            lora_rank: > 0 adds LoRA adapters to attention q/v projections (base weights stay frozen).
            lora_last_n: apply LoRA only to the last n blocks (0 = all blocks).
            feature_layers: 1-based block indices whose patch tokens are also returned as
                feats["layer_patch_tokens"] (final LayerNorm applied), e.g. (3, 6, 9, 12).
        """
        super().__init__()
        self.model = model if model is not None else AutoModel.from_pretrained(model_name)
        cfg = self.model.config
        self.hidden_size: int = cfg.hidden_size
        self.patch_size: int = cfg.patch_size
        self.num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        self.num_layers: int = len(self.blocks)
        self.feature_layers: List[int] = [self._layer_index(i) for i in (feature_layers or [])]

        self.lora_rank = lora_rank
        if lora_rank > 0:
            self._add_lora(lora_rank, lora_alpha, lora_last_n, lora_dropout)
        self.set_trainable(freeze, unfreeze_last_n)

    # ------------------------------------------------------------------ structure

    @property
    def blocks(self) -> nn.ModuleList:
        """Transformer blocks. transformers 4.5x: model.layer, 5.x: model.model.layer."""
        if hasattr(self.model, "layer"):
            return self.model.layer
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "layer"):
            return inner.layer
        raise AttributeError("Could not find DINOv3 transformer blocks (model.layer / model.model.layer)")

    @property
    def final_norm(self) -> nn.Module:
        if hasattr(self.model, "norm"):
            return self.model.norm
        return self.model.model.norm

    def _layer_index(self, i: int) -> int:
        """1-based block index (negative = from the end) -> hidden_states index."""
        n = self.num_layers
        idx = i if i > 0 else n + 1 + i
        if not 1 <= idx <= n:
            raise ValueError(f"feature layer {i} out of range for a {n}-block backbone")
        return idx

    def _add_lora(self, rank, alpha, last_n, dropout):
        blocks = list(self.blocks)
        if last_n > 0:
            blocks = blocks[-last_n:]
        for block in blocks:
            attn = block.attention
            for name in ("q_proj", "v_proj"):
                base = getattr(attn, name)
                if isinstance(base, LoRALinear):
                    continue
                setattr(attn, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))

    def set_trainable(self, freeze: bool = True, unfreeze_last_n: int = 0) -> None:
        for name, p in self.model.named_parameters():
            p.requires_grad = not freeze
        if freeze and unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.final_norm.parameters():
                p.requires_grad = True
        # LoRA adapters are always trainable; the wrapped base weights follow the rules above.
        for name, p in self.model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                p.requires_grad = True

    # ------------------------------------------------------------------ forward

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = pixel_values.shape[-2:]
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Input size {(H, W)} must be a multiple of patch size {self.patch_size}")

        want_layers = bool(self.feature_layers)
        out = self.model(pixel_values=pixel_values, output_hidden_states=want_layers)
        hidden = out.last_hidden_state  # (B, 1 + R + N, D)
        start = 1 + self.num_register_tokens
        feats = {
            "cls_token": hidden[:, 0],
            "patch_tokens": hidden[:, start:],
            "grid_size": (H // self.patch_size, W // self.patch_size),
        }
        if want_layers:
            # hidden_states[0] = embeddings, hidden_states[i] = output of block i (pre final norm)
            norm = self.final_norm
            feats["layer_patch_tokens"] = [
                norm(out.hidden_states[i])[:, start:] for i in self.feature_layers
            ]
        return feats


def to_feature_map(feats: Dict[str, torch.Tensor], tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, N, D) patch tokens -> (B, D, h, w). Uses feats["patch_tokens"] unless `tokens` is given."""
    h, w = feats["grid_size"]
    tokens = feats["patch_tokens"] if tokens is None else tokens
    B, N, D = tokens.shape
    # contiguous(): the transposed view can make CPU conv2d fail ("view size is not compatible")
    return tokens.transpose(1, 2).reshape(B, D, h, w).contiguous()
