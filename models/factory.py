"""Single place that turns a run config (train.py args / run_config.json) into a model.

train.py, train_hf.py, app.py and evals/predict.py all call `build_model`, so a checkpoint is
always rebuilt with exactly the architecture it was trained with (LoRA, fused layers, top-k ...).

    model = build_model(cfg)                          # downloads pretrained weights
    model = build_model(cfg, hf_model=prebuilt_hf)    # offline: weights come from the checkpoint
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from .backbone import DinoBackbone
from .global_baseline import GlobalBaseline
from .patchguard import PatchGuard
from .segformer_baseline import SegFormerBaseline

MODEL_ALIASES = {
    "global": "global_cls",
    "global_cls": "global_cls",
    "global_mean": "global_cls_mean",
    "global_cls_mean": "global_cls_mean",
    "patch_avg": "patch_avg",
    "patchguard_avg": "patch_avg",
    "patch_topk": "patch_topk",
    "patchguard_topk": "patch_topk",
    "patch_soft": "patch_soft",
    "patchguard_soft": "patch_soft",
    "segformer": "segformer",
}

AGGREGATION = {"patch_avg": "average", "patch_topk": "topk", "patch_soft": "soft_topk"}

# Defaults for keys missing from old run_config.json files (= the pre-V3 behaviour), so old
# checkpoints still load. New training defaults live in train.py.
LEGACY_DEFAULTS: Dict[str, Any] = dict(
    freeze_backbone=True,
    unfreeze_last_n=0,
    lora_rank=0,
    lora_alpha=16.0,
    lora_last_n=0,
    lora_dropout=0.0,
    feature_layers="",
    head_hidden_dim=256,
    patch_hidden_dim=256,
    dropout=0.1,
    topk_ratio=0.10,
    segformer_topk_ratio=0.01,
)


def parse_layers(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(",") if v.strip()]


def build_model(cfg: Dict[str, Any], hf_model: Optional[nn.Module] = None) -> nn.Module:
    """cfg: dict with at least "model"; missing keys fall back to LEGACY_DEFAULTS.

    hf_model: an already-built HF backbone (DINOv3ViTModel or SegformerForSemanticSegmentation),
    e.g. created from a saved config for offline inference.
    """
    c = {**LEGACY_DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    kind = MODEL_ALIASES.get(c["model"])
    if kind is None:
        raise ValueError(f"Unknown model: {c['model']}")

    if kind == "segformer":
        return SegFormerBaseline(topk_ratio=c["segformer_topk_ratio"], model=hf_model)

    is_patch = kind in AGGREGATION
    backbone = DinoBackbone(
        freeze=c["freeze_backbone"],
        unfreeze_last_n=c["unfreeze_last_n"],
        model=hf_model,
        lora_rank=c["lora_rank"],
        lora_alpha=c["lora_alpha"],
        lora_last_n=c["lora_last_n"],
        lora_dropout=c["lora_dropout"],
        # multi-layer fusion only feeds the PatchGuard mask decoder
        feature_layers=parse_layers(c["feature_layers"]) if is_patch else None,
    )

    if kind in ("global_cls", "global_cls_mean"):
        return GlobalBaseline(
            backbone=backbone,
            use_mean_patch=(kind == "global_cls_mean"),
            hidden_dim=c["head_hidden_dim"],
            dropout=c["dropout"],
        )

    return PatchGuard(
        backbone=backbone,
        aggregation=AGGREGATION[kind],
        patch_hidden_dim=c["patch_hidden_dim"],
        topk_ratio=c["topk_ratio"],
        dropout=c["dropout"],
    )
