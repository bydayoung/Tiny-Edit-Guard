from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from patch_head import PatchHead


class GlobalClassifierHead(nn.Module):
    """SegFormer 마지막 feature map의 Global Average Pooling baseline."""

    def __init__(self, in_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, 1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return self.classifier(pooled).squeeze(-1)


class TinyEditModel(nn.Module):
    """
    SegFormer-B0 기반 TinyEdit-Guard.

    classification_mode:
        - "global"
        - "patch_average"
        - "patch_topk"

    출력:
        logits, edit_prob, patch_scores, score_map,
        topk_indices, mask_logits
    """

    SUPPORTED_MODES = {"global", "patch_average", "patch_topk"}

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        classification_mode: str = "patch_topk",
        num_segmentation_labels: int = 2,
        patch_hidden_dim: int = 128,
        topk_ratio: float = 0.1,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        if classification_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"classification_mode must be one of {self.SUPPORTED_MODES}, "
                f"got {classification_mode!r}"
            )

        self.classification_mode = classification_mode

        # ADE20K의 기존 150-class head 대신
        # 0=background, 1=edited 의 2-class head를 새로 사용.
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_segmentation_labels,
            id2label={0: "background", 1: "edited"},
            label2id={"background": 0, "edited": 1},
            ignore_mismatched_sizes=True,
        )

        last_hidden_size = self.segformer.config.hidden_sizes[-1]

        self.global_head = GlobalClassifierHead(
            in_channels=last_hidden_size,
            dropout=dropout,
        )

        if classification_mode == "patch_average":
            self.patch_head = PatchHead(
                in_channels=last_hidden_size,
                hidden_dim=patch_hidden_dim,
                aggregation="average",
                topk_ratio=topk_ratio,
                dropout=dropout,
            )
        elif classification_mode == "patch_topk":
            self.patch_head = PatchHead(
                in_channels=last_hidden_size,
                hidden_dim=patch_hidden_dim,
                aggregation="topk",
                topk_ratio=topk_ratio,
                dropout=dropout,
            )
        else:
            self.patch_head = None

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for p in self.segformer.segformer.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.segformer.segformer.parameters():
            p.requires_grad = True

    @staticmethod
    def _ensure_4d_feature(feature: torch.Tensor) -> torch.Tensor:
        """
        기본 SegFormer는 [B,C,H,W] feature map을 반환한다.
        일부 버전/설정의 [B,N,C] 출력도 처리한다.
        """
        if feature.ndim == 4:
            return feature

        if feature.ndim != 3:
            raise ValueError(
                f"Expected 3D/4D feature, got {tuple(feature.shape)}"
            )

        b, n, c = feature.shape
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(
                f"Cannot reshape {n} tokens into a square feature map."
            )

        return (
            feature.transpose(1, 2)
            .reshape(b, c, side, side)
            .contiguous()
        )

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must be [B,3,H,W], got {tuple(pixel_values.shape)}"
            )

        outputs = self.segformer(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("SegFormer did not return hidden_states.")

        last_feature = self._ensure_4d_feature(
            outputs.hidden_states[-1]
        )

        patch_scores = None
        score_map = None
        topk_indices = None

        if self.classification_mode == "global":
            logits = self.global_head(last_feature)
        else:
            patch_out = self.patch_head(last_feature)
            logits = patch_out["logits"]
            patch_scores = patch_out["patch_scores"]
            score_map = patch_out["score_map"]
            topk_indices = patch_out["topk_indices"]

        # SegFormer segmentation logits는 입력보다 작을 수 있으므로
        # 입력 해상도로 upsample.
        mask_logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": logits,
            "edit_prob": torch.sigmoid(logits),
            "patch_scores": patch_scores,
            "score_map": score_map,
            "topk_indices": topk_indices,
            "mask_logits": mask_logits,
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyEditModel(
        classification_mode="patch_topk",
        topk_ratio=0.1,
        freeze_encoder=True,
    ).to(device)

    dummy = torch.randn(2, 3, 512, 512, device=device)

    with torch.no_grad():
        out = model(dummy)

    print("logits:", out["logits"].shape)
    print("edit_prob:", out["edit_prob"].shape)
    print("mask_logits:", out["mask_logits"].shape)
    if out["patch_scores"] is not None:
        print("patch_scores:", out["patch_scores"].shape)
    if out["score_map"] is not None:
        print("score_map:", out["score_map"].shape)
