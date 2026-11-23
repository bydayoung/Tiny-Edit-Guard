"""Tiny-Edit-Guard training entry point using Hugging Face Hub data directly.

This version imports:
    data/dataloader_hf.py

and therefore does not depend on:
    data/processed/
    data/final_split_metadata.csv
    data/13_preprocess_images.py
    data/14_build_dataloader.py

Example:
    python train_hf.py \
        --model patch_topk \
        --epochs 1 \
        --max-train-batches 5 \
        --max-val-batches 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import trainer as trainer_module
from data.dataloader_hf import build_dataloaders
from data.dataloader_local import build_local_dataloaders
from losses import TinyEditLoss
from models.backbone import DinoBackbone
from models.global_baseline import GlobalBaseline
from models.patchguard import PatchGuard
from models.segformer_baseline import SegFormerBaseline


ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LimitedLoader:
    """Optional batch limiter for smoke tests."""

    def __init__(self, loader, max_batches=None):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = getattr(loader, "dataset", None)

    def __len__(self):
        n = len(self.loader)
        return n if self.max_batches is None else min(n, self.max_batches)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            yield batch


class ValidMaskMetricTracker(trainer_module.MetricTracker):
    """Exclude valid_mask=False samples from localization validation metrics."""

    @torch.no_grad()
    def update(self, outputs, batch):
        self.scores.append(
            outputs["cls_logits"].sigmoid().float().cpu()
        )
        self.labels.append(
            batch["label"].float().cpu()
        )

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

        gt = trainer_module._as_mask(
            batch["mask"][valid]
        )

        tp = (pred * gt).flatten(1).sum(1)
        fp = (pred * (1 - gt)).flatten(1).sum(1)
        fn = ((1 - pred) * gt).flatten(1).sum(1)

        self.tp += tp.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()

        edited = batch["label"][valid].bool()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)
        self.per_image_f1.extend(
            f1[edited].cpu().tolist()
        )


# Patch the existing trainer's tracker without editing trainer.py.
trainer_module.MetricTracker = ValidMaskMetricTracker
Trainer = trainer_module.Trainer


def build_model(args):
    if args.model == "global_cls":
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

        return PatchGuard(
            backbone=backbone,
            aggregation=(
                "average"
                if args.model == "patch_avg"
                else "topk"
            ),
            patch_hidden_dim=args.patch_hidden_dim,
            topk_ratio=args.topk_ratio,
            dropout=args.dropout,
        )

    if args.model == "segformer":
        return SegFormerBaseline(
            topk_ratio=args.segformer_topk_ratio,
        )

    raise ValueError(args.model)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="patch_topk",
        choices=[
            "global_cls",
            "global_cls_mean",
            "patch_avg",
            "patch_topk",
            "segformer",
        ],
    )

    # Hub / data
    p.add_argument(
        "--dataset-id",
        default=os.environ.get(
            "TINYEDIT_HF_DATASET",
            "pdyoung/TinyEdit-Guard-Dataset",
        ),
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Local data folder (e.g. data/v3_processed). If set, Hub is not used.",
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="Metadata CSV for --data-root (default: the only *.csv in it, "
             "else data/final_split_metadata.csv).",
    )
    p.add_argument(
        "--input-mode",
        default=None,
        help="Keep only rows with this input_mode value (v3 metadata).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--hf-token",
        default=None,
        help="Usually unnecessary after `hf auth login`.",
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
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)

    # Backbone / heads
    p.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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

    # Checkpoint / smoke test
    p.add_argument("--monitor", default=None)
    p.add_argument(
        "--monitor-mode",
        choices=["max", "min"],
        default="max",
    )
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--skip-predict", action="store_true",
                   help="skip writing preds_val.csv / preds_test.csv after training")

    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)

    if args.patch_weight > 0 and args.model not in {
        "patch_avg", "patch_topk"
    }:
        raise ValueError(
            "--patch-weight > 0 can only be used with PatchGuard."
        )

    # 1) Load Hub dataset using normal load_dataset(), not streaming.
    #    --data-root switches to a local folder (e.g. data/v3_processed).
    if args.data_root:
        data = build_local_dataloaders(
            data_root=args.data_root,
            metadata=args.metadata,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            input_mode=args.input_mode,
        )
    else:
        data = build_dataloaders(
            dataset_id=args.dataset_id,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            token=args.hf_token,
        )

    # Pure SegFormer localization can use valid-mask-only splits.
    if args.model == "segformer":
        train_loader = data["train_seg_loader"]
        val_loader = data["val_seg_loader"]
    else:
        train_loader = data["train_loader"]
        val_loader = data["val_loader"]

    train_loader = LimitedLoader(
        train_loader,
        args.max_train_batches,
    )
    val_loader = LimitedLoader(
        val_loader,
        args.max_val_batches,
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

    monitor = args.monitor
    if monitor is None:
        monitor = "pixel_f1" if args.model == "segformer" else "auc"

    output_dir = Path(
        args.output_dir
        or f"runs/{args.model}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("Tiny-Edit-Guard / Hugging Face Hub training")
    print(f"dataset     : {args.data_root or args.dataset_id}")
    print(f"model       : {args.model}")
    print(f"train batch : {len(train_loader)}")
    print(f"val batch   : {len(val_loader)}")
    print(f"output      : {output_dir}")
    print("=" * 72)

    # 4) Train.
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
        output_dir=str(output_dir),
        log_every=args.log_every,
    )

    if args.resume:
        trainer.load(
            args.resume,
            weights_only=False,
        )

    metrics = trainer.fit()

    # 6) Per-sample predictions from best.pt -> evals/evaluate.py --trainer-csv.
    #    Full val/test loaders (real + edited), so classification metrics are complete.
    if not args.skip_predict:
        trainer.export_predictions({
            "val": LimitedLoader(data["val_loader"], args.max_val_batches),
            "test": LimitedLoader(data["test_loader"], args.max_val_batches),
        })

    print("=" * 72)
    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print("best:", output_dir / "best.pt")
    print("last:", output_dir / "last.pt")


if __name__ == "__main__":
    main()
