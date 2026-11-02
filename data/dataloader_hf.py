"""Hugging Face Hub DataLoader for Tiny-Edit-Guard.

This loader uses `datasets.load_dataset(..., streaming=False)`.
You do NOT need to manually download/extract the dataset.

Important:
- Hugging Face still downloads the required dataset shards automatically
  and stores them in its cache (usually ~/.cache/huggingface).
- This is different from streaming=True: the dataset supports normal
  random access, shuffle, DataLoader workers, and repeated epochs.

Expected repository:
    pdyoung/TinyEdit-Guard-Dataset

Returned batch keys are compatible with trainer.py:
    pixel_values       (B, 3, 512, 512)
    mask               (B, 1, 512, 512)
    label              (B,)
    valid_mask         (B,)
    id                 list[str]
    dataset            list[str]
    method             list[str]
    edit_size_bin      list[str]
    actual_mask_ratio  (B,)
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_DATASET_ID = os.environ.get(
    "TINYEDIT_HF_DATASET",
    "pdyoung/TinyEdit-Guard-Dataset",
)

IMAGE_SIZE = int(os.environ.get("TINYEDIT_IMAGE_SIZE", "512"))
BATCH_SIZE = int(os.environ.get("TINYEDIT_BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("TINYEDIT_NUM_WORKERS", "0"))

# DINOv3 / ImageNet normalization.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {
        "1", "true", "t", "yes", "y"
    }


def _to_pil(value: Any, mode: str) -> Optional[Image.Image]:
    """Convert common Hugging Face Image representations to PIL."""
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return value.convert(mode)

    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert(mode)

    # Hugging Face Image can occasionally be represented as
    # {"path": ..., "bytes": ...}.
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert(mode)

        path = value.get("path")
        if path:
            return Image.open(path).convert(mode)

    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert(mode)

    raise TypeError(
        f"Unsupported image/mask value type: {type(value)!r}"
    )


def _find_column(
    columns: Sequence[str],
    candidates: Iterable[str],
    required: bool = False,
) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find any of {list(candidates)} "
            f"in dataset columns: {list(columns)}"
        )
    return None


def _column_map(dataset: Dataset) -> Dict[str, Optional[str]]:
    columns = dataset.column_names

    return {
        "image": _find_column(
            columns,
            ["image", "img", "rgb_image"],
            required=True,
        ),
        "mask": _find_column(
            columns,
            ["mask", "gt_mask", "ground_truth_mask", "segmentation_mask"],
            required=False,
        ),
        "label": _find_column(
            columns,
            ["label", "target", "is_edited", "class"],
            required=True,
        ),
        "valid_mask": _find_column(
            columns,
            ["valid_mask", "mask_valid", "is_mask_valid"],
            required=False,
        ),
        "id": _find_column(
            columns,
            ["id", "sample_id", "image_id"],
            required=False,
        ),
        "dataset": _find_column(
            columns,
            ["dataset", "source_dataset", "dataset_name"],
            required=False,
        ),
        "method": _find_column(
            columns,
            ["method", "edit_method", "generator", "model"],
            required=False,
        ),
        "edit_size_bin": _find_column(
            columns,
            ["edit_size_bin", "size_bin", "mask_ratio_bin"],
            required=False,
        ),
        "actual_mask_ratio": _find_column(
            columns,
            ["actual_mask_ratio", "mask_ratio", "edit_ratio"],
            required=False,
        ),
        "split": _find_column(
            columns,
            ["split", "data_split"],
            required=False,
        ),
    }


def _normalize_split_name(name: str) -> str:
    x = str(name).strip().lower()

    if x in {"train", "training"}:
        return "train"
    if x in {"validation", "valid", "val", "dev"}:
        return "validation"
    if x in {"test", "testing", "eval", "evaluation"}:
        return "test"

    return x


def _split_dataset(ds: DatasetDict) -> Dict[str, Dataset]:
    """Return canonical train / validation / test splits.

    Supports:
    1) Hub DatasetDict already containing train/validation/test splits.
    2) A single split containing a column named `split` with train/val/test values.
    """
    split_lookup = {
        _normalize_split_name(k): k
        for k in ds.keys()
    }

    if all(k in split_lookup for k in ("train", "validation", "test")):
        return {
            "train": ds[split_lookup["train"]],
            "validation": ds[split_lookup["validation"]],
            "test": ds[split_lookup["test"]],
        }

    # If Hub exposes only one physical split, use its "split" metadata column.
    if len(ds) == 1:
        only = next(iter(ds.values()))
        cmap = _column_map(only)
        split_col = cmap["split"]

        if split_col is None:
            raise RuntimeError(
                "The Hub dataset has only one split and no 'split' column. "
                f"Available splits={list(ds.keys())}, columns={only.column_names}"
            )

        result = {}
        for canonical in ("train", "validation", "test"):
            result[canonical] = only.filter(
                lambda row, target=canonical: (
                    _normalize_split_name(row[split_col]) == target
                )
            )
        return result

    # Some repos may use val instead of validation as physical split.
    if "train" in split_lookup and "test" in split_lookup:
        val_key = None
        for alias in ("validation", "valid", "val", "dev"):
            if alias in ds:
                val_key = alias
                break

        if val_key is not None:
            return {
                "train": ds[split_lookup["train"]],
                "validation": ds[val_key],
                "test": ds[split_lookup["test"]],
            }

    raise RuntimeError(
        "Could not resolve train/validation/test splits. "
        f"Hub splits: {list(ds.keys())}"
    )


# -----------------------------------------------------------------------------
# PyTorch Dataset wrapper
# -----------------------------------------------------------------------------

class TinyEditHFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        image_size: int = IMAGE_SIZE,
        segmentation_only: bool = False,
    ):
        self.ds = hf_dataset
        self.image_size = int(image_size)
        self.columns = _column_map(hf_dataset)

        # For the pure segmentation baseline, optionally remove samples
        # whose GT masks are invalid.
        if segmentation_only and self.columns["valid_mask"] is not None:
            valid_col = self.columns["valid_mask"]
            self.ds = self.ds.filter(
                lambda row: to_bool(row[valid_col])
            )

    def __len__(self) -> int:
        return len(self.ds)

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        x = TF.to_tensor(image)
        x = TF.normalize(
            x,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        )
        return x

    def _mask_tensor(
        self,
        mask: Optional[Image.Image],
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros(
                1,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
            )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        arr = np.asarray(mask, dtype=np.uint8)
        binary = (arr >= 128).astype(np.float32)

        return torch.from_numpy(binary).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[int(idx)]
        c = self.columns

        image = _to_pil(row[c["image"]], "RGB")
        if image is None:
            raise ValueError(f"Missing image at row {idx}")

        mask_value = (
            row[c["mask"]]
            if c["mask"] is not None
            else None
        )
        mask = _to_pil(mask_value, "L") if mask_value is not None else None

        # A sample is localization-valid only if metadata says so AND
        # an actual mask exists.
        if c["valid_mask"] is None:
            is_valid = mask is not None
        else:
            is_valid = to_bool(row[c["valid_mask"]]) and mask is not None

        label = int(row[c["label"]])

        sample_id = (
            str(row[c["id"]])
            if c["id"] is not None
            else str(idx)
        )

        source_dataset = (
            str(row[c["dataset"]])
            if c["dataset"] is not None
            else ""
        )

        method = (
            str(row[c["method"]])
            if c["method"] is not None
            else ""
        )

        size_bin = (
            str(row[c["edit_size_bin"]])
            if c["edit_size_bin"] is not None
            else ""
        )

        ratio_value = (
            row[c["actual_mask_ratio"]]
            if c["actual_mask_ratio"] is not None
            else 0.0
        )
        try:
            actual_mask_ratio = float(ratio_value)
        except (TypeError, ValueError):
            actual_mask_ratio = 0.0

        return {
            "pixel_values": self._image_tensor(image),
            "mask": self._mask_tensor(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "valid_mask": torch.tensor(is_valid, dtype=torch.bool),
            "id": sample_id,
            "dataset": source_dataset,
            "method": method,
            "edit_size_bin": size_bin,
            "actual_mask_ratio": torch.tensor(
                actual_mask_ratio,
                dtype=torch.float32,
            ),
        }


# -----------------------------------------------------------------------------
# Hub loading
# -----------------------------------------------------------------------------

def load_tinyedit_hf(
    dataset_id: str = HF_DATASET_ID,
    token: Optional[bool | str] = None,
) -> Dict[str, Dataset]:
    """Load TinyEdit-Guard from Hugging Face Hub.

    `streaming=False` is intentional. This gives a normal map-style Dataset
    suitable for shuffled mini-batch training and repeated epochs.

    If the repo is private/gated:
        hf auth login
    and call with token=True, or set HF_TOKEN.
    """
    ds = load_dataset(
        dataset_id,
        streaming=False,
        token=token,
    )

    if not isinstance(ds, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict from load_dataset, got {type(ds)!r}"
        )

    return _split_dataset(ds)


def build_dataloaders(
    dataset_id: str = HF_DATASET_ID,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
    token: Optional[bool | str] = None,
):
    splits = load_tinyedit_hf(
        dataset_id=dataset_id,
        token=token,
    )

    train_dataset = TinyEditHFDataset(
        splits["train"],
        image_size=image_size,
        segmentation_only=False,
    )
    val_dataset = TinyEditHFDataset(
        splits["validation"],
        image_size=image_size,
        segmentation_only=False,
    )
    test_dataset = TinyEditHFDataset(
        splits["test"],
        image_size=image_size,
        segmentation_only=False,
    )

    # Optional mask-valid-only datasets for pure segmentation experiments.
    train_seg_dataset = TinyEditHFDataset(
        splits["train"],
        image_size=image_size,
        segmentation_only=True,
    )
    val_seg_dataset = TinyEditHFDataset(
        splits["validation"],
        image_size=image_size,
        segmentation_only=True,
    )
    test_seg_dataset = TinyEditHFDataset(
        splits["test"],
        image_size=image_size,
        segmentation_only=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    train_seg_loader = DataLoader(
        train_seg_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_seg_loader = DataLoader(
        val_seg_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_seg_loader = DataLoader(
        test_seg_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_seg_loader": train_seg_loader,
        "val_seg_loader": val_seg_loader,
        "test_seg_loader": test_seg_loader,
    }


# Build loaders explicitly with build_dataloaders().
# This avoids loading the Hub dataset automatically just by importing this module.

if __name__ == "__main__":
    print("=" * 70)
    print("TinyEdit-Guard Hugging Face DataLoader")
    print("=" * 70)
    print("dataset:", HF_DATASET_ID)

    objects = build_dataloaders()
    train_dataset = objects["train_dataset"]
    val_dataset = objects["val_dataset"]
    test_dataset = objects["test_dataset"]
    train_loader = objects["train_loader"]

    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    print("test:", len(test_dataset))

    batch = next(iter(train_loader))

    print("\nFIRST BATCH")
    print("pixel_values:", tuple(batch["pixel_values"].shape))
    print("mask:", tuple(batch["mask"].shape))
    print("label:", tuple(batch["label"].shape))
    print("valid_mask:", tuple(batch["valid_mask"].shape))
    print("ids:", batch["id"][:3])
