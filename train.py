"""Training entry point for Tiny-Edit-Guard.

Data sources
------------
    --data-source local  (default) imports a loader module (default data/14_build_dataloader.py,
                         override with --data-module path/to/v3_loader.py) and uses its
                         train_dataset / val_dataset / test_dataset.
    --data-source hf     data/dataloader_hf.py (Hugging Face Hub, --dataset-id).

Loaders are rebuilt here so every model gets the same batching:
    - PadCollate: variable-size images are padded to a multiple of 16, batch["pad_mask"] marks
      real pixels (padding is excluded from top-k pooling, losses and metrics).
    - PairBatchSampler (train): real/fake members of a group stay in the same batch.
    - PairedAugment (train): flip / JPEG / blur with identical parameters within a pair.
    - val/test default to batch size 1 so images are scored at their own resolution.

Examples
--------
Smoke test:
    python train.py --model patch_topk --epochs 1 --max-train-batches 5 --max-val-batches 2

Recommended V3 runs (same settings for every model -> fair comparison):
    python train.py --model global_cls --output-dir runs/global_cls_s0
    python train.py --model patch_avg  --output-dir runs/patch_avg_s0
    python train.py --model patch_topk --output-dir runs/patch_topk_s0
    python train.py --model patch_soft --output-dir runs/patch_soft_s0
    python train.py --model segformer  --output-dir runs/segformer_s0

Old (pre-V3) behaviour:
    python train.py --model patch_topk --topk-ratio 0.10 --patch-weight 0 --lora-rank 0 \
        --feature-layers "" --no-augment --no-pair-batches
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
from torch.utils.data import DataLoader

import trainer as trainer_module
from data.collate import GroupIdDataset, PadCollate, PairBatchSampler, PairedAugment, find_group_ids
from losses import PATCH_TARGETS, TinyEditLoss
from models.factory import AGGREGATION, MODEL_ALIASES, build_model

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_MODULE = ROOT / "data" / "14_build_dataloader.py"
Trainer = trainer_module.Trainer


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
# Data
# -----------------------------------------------------------------------------

def load_module(path: Path):
    """Import a loader module from a file path (works although the filename starts with a digit)."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run train.py from the repository root.")
    spec = importlib.util.spec_from_file_location("tinyedit_data_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_datasets(args):
    """-> (train_dataset, val_dataset, test_dataset)"""
    if args.data_source == "hf":
        from data.dataloader_hf import build_dataloaders

        d = build_dataloaders(dataset_id=args.dataset_id, batch_size=args.batch_size,
                              image_size=args.image_size, num_workers=args.num_workers,
                              token=args.hf_token)
        return d["train_dataset"], d["val_dataset"], d["test_dataset"]

    module = load_module(Path(args.data_module))
    names = ("train_dataset", "val_dataset", "test_dataset")
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        raise AttributeError(f"{args.data_module} must define {names}; missing {missing}")
    return tuple(getattr(module, n) for n in names)


def with_group_ids(dataset, name: str):
    groups = find_group_ids(dataset)
    if groups is None:
        print(f"[data] {name}: no group id column ({'/'.join(('final_group_id', 'group_id', 'pair_id'))}) "
              "-> pair batching / paired augmentation fall back to per-sample")
        return dataset, None
    return GroupIdDataset(dataset, groups), groups


def build_loaders(args):
    train_ds, val_ds, test_ds = get_datasets(args)
    train_ds, train_groups = with_group_ids(train_ds, "train")
    val_ds, _ = with_group_ids(val_ds, "val")
    test_ds, _ = with_group_ids(test_ds, "test")

    augment = PairedAugment(seed=args.seed) if args.augment else None
    train_collate = PadCollate(augment=augment)
    eval_collate = PadCollate()
    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    if args.pair_batches and train_groups is not None:
        sampler = PairBatchSampler(train_groups, args.batch_size, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=train_collate, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_collate, **common)

    def eval_loader(ds):
        return DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                          collate_fn=eval_collate, **common)

    return train_loader, eval_loader(val_ds), eval_loader(test_ds)


class LoaderAdapter:
    """Optionally limits the number of batches (smoke tests); renames image -> pixel_values."""

    def __init__(self, loader, max_batches: Optional[int] = None):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = getattr(loader, "dataset", None)

    def __len__(self):
        n = len(self.loader)
        return n if self.max_batches is None else min(n, self.max_batches)

    def __iter__(self) -> Iterator[dict]:
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            if "pixel_values" not in batch and "image" in batch:
                batch = dict(batch)
                batch["pixel_values"] = batch.pop("image")
            yield batch


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args(argv=None, data_source_default: str = "local"):
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="patch_topk", choices=sorted(MODEL_ALIASES))

    # Data
    p.add_argument("--data-source", choices=["local", "hf"], default=data_source_default)
    p.add_argument("--data-module", default=str(DEFAULT_DATA_MODULE),
                   help="[local] loader module defining train_dataset / val_dataset / test_dataset")
    p.add_argument("--dataset-id", default=os.environ.get("TINYEDIT_HF_DATASET", "pdyoung/TinyEdit-Guard-Dataset"))
    p.add_argument("--hf-token", default=None)
    p.add_argument("--image-size", type=int, default=0,
                   help="[hf] resize size in dataloader_hf.py; 0 = keep stored resolution (V3)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=1,
                   help="1 = every val/test image at its own resolution (no shared padding)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pair-batches", action=argparse.BooleanOptionalAction, default=True,
                   help="keep real/fake pairs (same group id) in the same training batch")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="paired flip / JPEG(65-100) / blur augmentation on the training set")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--lora-lr", type=float, default=2e-4)
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
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA on DINOv3 attention q/v (0 = off). Applies to every DINOv3 model "
                        "(global / patch) so they are compared under the same backbone policy")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-last-n", type=int, default=0, help="LoRA only in the last n blocks (0 = all)")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--feature-layers", type=str, default="3,6,9,12",
                   help="1-based DINOv3 blocks fused for the PatchGuard mask decoder ('' = last layer only)")
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--patch-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--topk-ratio", type=float, default=0.02,
                   help="tiny edits cover ~1-3%% of patches; 0.10 dilutes them")
    p.add_argument("--segformer-topk-ratio", type=float, default=0.01)

    # Loss
    p.add_argument("--mask-weight", type=float, default=1.0)
    p.add_argument("--patch-weight", type=float, default=0.2,
                   help="patch-level supervision for PatchGuard (ignored for global / segformer)")
    p.add_argument("--patch-target", choices=PATCH_TARGETS, default="soft")
    p.add_argument("--mask-bce-weight", type=float, default=1.0)
    p.add_argument("--mask-dice-weight", type=float, default=1.0)

    # Checkpoint selection (same metric for every model unless overridden)
    p.add_argument("--monitor", type=str, default="auroc")
    p.add_argument("--monitor-mode", choices=["max", "min"], default="max")

    # Smoke-test / debugging
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--skip-predict", action="store_true",
                   help="skip writing preds_val.csv / preds_test.csv after training")
    p.add_argument("--log-every", type=int, default=50)

    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None, data_source_default: str = "local"):
    args = parse_args(argv, data_source_default)

    # The existing data code uses relative paths such as ./data/...
    os.chdir(ROOT)
    set_seed(args.seed)

    kind = MODEL_ALIASES[args.model]
    if kind not in AGGREGATION and args.patch_weight > 0.0:
        print(f"[train] --patch-weight ignored for {kind} (PatchGuard only)")
        args.patch_weight = 0.0

    # 1) Data.
    train_loader, val_loader, test_loader = build_loaders(args)
    train_loader = LoaderAdapter(train_loader, max_batches=args.max_train_batches)
    val_loader_lim = LoaderAdapter(val_loader, max_batches=args.max_val_batches)

    # 2) Model (models/factory.py: the same function rebuilds it in app.py / evals/predict.py).
    model = build_model(vars(args))

    # 3) Loss.
    loss_fn = TinyEditLoss(
        mask_weight=args.mask_weight,
        patch_weight=args.patch_weight,
        mask_bce_weight=args.mask_bce_weight,
        mask_dice_weight=args.mask_dice_weight,
        patch_target=args.patch_target,
    )

    output_path = ROOT / (args.output_dir or f"runs/{args.model}_seed{args.seed}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the run arguments: models/factory.py rebuilds the model from this file.
    with open(output_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Tiny-Edit-Guard training")
    print(f"model       : {args.model}")
    print(f"data        : {args.data_source} ({args.data_module if args.data_source == 'local' else args.dataset_id})")
    print(f"seed        : {args.seed}")
    print(f"train batch : {len(train_loader)}  (pair batches={args.pair_batches}, augment={args.augment})")
    print(f"val batch   : {len(val_loader_lim)}")
    print(f"monitor     : val_{args.monitor}")
    print(f"output      : {output_path}")
    print("=" * 72)

    # 4) Trainer.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader_lim,
        loss_fn=loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        output_dir=str(output_path),
        log_every=args.log_every,
    )

    if args.resume:
        trainer.load(args.resume, weights_only=False)

    # 5) Train.
    final_metrics = trainer.fit()

    # 6) Per-sample predictions from best.pt -> evals/evaluate.py --trainer-csv.
    if not args.skip_predict:
        trainer.export_predictions({
            "val": LoaderAdapter(val_loader, max_batches=args.max_val_batches),
            "test": LoaderAdapter(test_loader, max_batches=args.max_val_batches),
        })

    print("=" * 72)
    print("Training complete")
    print(json.dumps(final_metrics, indent=2))
    print(f"Best checkpoint: {output_path / 'best.pt'}")
    print(f"Last checkpoint: {output_path / 'last.pt'}")


if __name__ == "__main__":
    main()
