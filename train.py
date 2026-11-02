"""Training entry point for Tiny-Edit-Guard.

Works with the current repository layout without renaming
`data/14_build_dataloader.py`.

Examples
--------
Smoke test:
    python train.py --model patch_topk --epochs 1 --max-train-batches 5 --max-val-batches 2

Patch Average:
    python train.py --model patch_avg --epochs 10 --output-dir runs/patch_avg_s0

Patch Top-k + patch supervision:
    python train.py --model patch_topk --epochs 10 --patch-weight 0.1 \
        --output-dir runs/patch_topk_patchsup_s0

Global baseline:
    python train.py --model global_cls --epochs 10 --output-dir runs/global_cls_s0

SegFormer localization baseline:
    python train.py --model segformer --epochs 10 --monitor pixel_f1 \
        --output-dir runs/segformer_s0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

import trainer as trainer_module
from losses import TinyEditLoss
from models import (
    DinoBackbone,
    GlobalBaseline,
    SegFormerBaseline,
)
from models.patchguard import PatchGuard


ROOT = Path(__file__).resolve().parent
DATA_LOADER_FILE = ROOT / "data" / "14_build_dataloader.py"


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort reproducibility.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Current data-loader compatibility
# -----------------------------------------------------------------------------

def load_current_data_module():
    """Import data/14_build_dataloader.py although the filename starts with a digit."""
    if not DATA_LOADER_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_LOADER_FILE}. "
            "Run train.py from the Tiny-Edit-Guard repository root."
        )

    spec = importlib.util.spec_from_file_location(
        "tinyedit_current_dataloader",
        DATA_LOADER_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {DATA_LOADER_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoaderAdapter:
    """Wrap the existing DataLoader.

    - Renames batch["image"] -> batch["pixel_values"] for trainer.py.
    - Optionally limits the number of batches for smoke tests.
    """

    def __init__(self, loader, max_batches: Optional[int] = None):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = getattr(loader, "dataset", None)

    def __len__(self):
        n = len(self.loader)
        if self.max_batches is None:
            return n
        return min(n, self.max_batches)

    def __iter__(self) -> Iterator[dict]:
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break

            if "pixel_values" not in batch:
                if "image" not in batch:
                    raise KeyError(
                        "DataLoader batch must contain 'image' or 'pixel_values'."
                    )
                batch = dict(batch)
                batch["pixel_values"] = batch.pop("image")

            yield batch


# -----------------------------------------------------------------------------
# Validation metric patch:
# current trainer.MetricTracker does not filter invalid localization masks.
# -----------------------------------------------------------------------------

class ValidMaskMetricTracker(trainer_module.MetricTracker):
    @torch.no_grad()
    def update(self, outputs, batch):
        # Image-level metrics: every sample.
        self.scores.append(outputs["cls_logits"].sigmoid().float().cpu())
        self.labels.append(batch["label"].float().cpu())

        if "mask_logits" not in outputs or "mask" not in batch:
            return

        valid = batch.get("valid_mask")
        if valid is None:
            valid = torch.ones(
                batch["label"].shape[0],
                dtype=torch.bool,
                device=batch["label"].device,
            )
        else:
            valid = valid.bool()

        if not valid.any():
            return

        self.has_mask = True

        pred = (
            outputs["mask_logits"][valid].sigmoid() > self.threshold
        ).float()

        gt = trainer_module._as_mask(batch["mask"][valid])

        tp = (pred * gt).flatten(1).sum(1)
        fp = (pred * (1 - gt)).flatten(1).sum(1)
        fn = ((1 - pred) * gt).flatten(1).sum(1)

        self.tp += tp.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()

        # Per-image F1 only for edited + valid-mask samples.
        edited = batch["label"][valid].bool()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)
        self.per_image_f1.extend(f1[edited].cpu().tolist())


# Trainer.validate() looks up MetricTracker from its module globals.
trainer_module.MetricTracker = ValidMaskMetricTracker
Trainer = trainer_module.Trainer


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------

def build_model(args):
    if args.model in {"global", "global_cls"}:
        backbone = DinoBackbone(
            freeze=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
        )
        return GlobalBaseline(
            backbone=backbone,
            use_mean_patch=False,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )

    if args.model == "global_cls_mean":
        backbone = DinoBackbone(
            freeze=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
        )
        return GlobalBaseline(
            backbone=backbone,
            use_mean_patch=True,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )

    if args.model in {"patch_avg", "patch_topk"}:
        backbone = DinoBackbone(
            freeze=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
        )

        aggregation = (
            "average"
            if args.model == "patch_avg"
            else "topk"
        )

        return PatchGuard(
            backbone=backbone,
            aggregation=aggregation,
            patch_hidden_dim=args.patch_hidden_dim,
            topk_ratio=args.topk_ratio,
            dropout=args.dropout,
        )

    if args.model == "segformer":
        return SegFormerBaseline(
            topk_ratio=args.segformer_topk_ratio,
        )

    raise ValueError(f"Unknown model: {args.model}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="patch_topk",
        choices=[
            "global",
            "global_cls",
            "global_cls_mean",
            "patch_avg",
            "patch_topk",
            "segformer",
        ],
    )

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)

    # Backbone / heads
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unfreeze-last-n", type=int, default=0)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--patch-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--segformer-topk-ratio", type=float, default=0.01)

    # Loss
    p.add_argument("--mask-weight", type=float, default=1.0)
    p.add_argument("--patch-weight", type=float, default=0.0)
    p.add_argument("--mask-bce-weight", type=float, default=1.0)
    p.add_argument("--mask-dice-weight", type=float, default=1.0)

    # Checkpoint selection
    p.add_argument("--monitor", type=str, default=None)
    p.add_argument(
        "--monitor-mode",
        choices=["max", "min"],
        default="max",
    )

    # Smoke-test / debugging
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()

    # The existing data code uses relative paths such as ./data/...
    os.chdir(ROOT)
    set_seed(args.seed)

    if args.patch_weight > 0.0 and args.model not in {"patch_avg", "patch_topk"}:
        raise ValueError(
            "--patch-weight > 0 is only valid for patch_avg / patch_topk."
        )

    # 1) Existing data loaders.
    data_module = load_current_data_module()

    train_loader = LoaderAdapter(
        data_module.train_loader,
        max_batches=args.max_train_batches,
    )
    val_loader = LoaderAdapter(
        data_module.val_loader,
        max_batches=args.max_val_batches,
    )

    # 2) Model.
    model = build_model(args)

    # 3) Loss.
    loss_fn = TinyEditLoss(
        mask_weight=args.mask_weight,
        patch_weight=args.patch_weight,
        mask_bce_weight=args.mask_bce_weight,
        mask_dice_weight=args.mask_dice_weight,
    )

    # GlobalBaseline has no mask output, so mask loss is simply ignored.
    # PatchGuard / SegFormer use valid_mask=True samples only.

    monitor = args.monitor
    if monitor is None:
        monitor = "pixel_f1" if args.model == "segformer" else "auc"

    output_dir = args.output_dir or f"runs/{args.model}_seed{args.seed}"
    output_path = ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the run arguments for reproducibility.
    with open(output_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Tiny-Edit-Guard training")
    print(f"model       : {args.model}")
    print(f"seed        : {args.seed}")
    print(f"train batch : {len(train_loader)}")
    print(f"val batch   : {len(val_loader)}")
    print(f"monitor     : val_{monitor}")
    print(f"output      : {output_path}")
    print("=" * 72)

    # 4) Trainer.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        monitor=monitor,
        monitor_mode=args.monitor_mode,
        output_dir=str(output_path),
        log_every=args.log_every,
    )

    if args.resume:
        trainer.load(args.resume, weights_only=False)

    # 5) Train.
    final_metrics = trainer.fit()

    print("=" * 72)
    print("Training complete")
    print(json.dumps(final_metrics, indent=2))
    print(f"Best checkpoint: {output_path / 'best.pt'}")
    print(f"Last checkpoint: {output_path / 'last.pt'}")


if __name__ == "__main__":
    main()
