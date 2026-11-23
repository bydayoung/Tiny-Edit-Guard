"""Local-folder loader for Tiny-Edit-Guard (e.g. data/v3_processed).

Produces the exact same batch dict as data/dataloader_hf.py
("pixel_values", "mask", "label", "valid_mask", "id", "dataset", "method",
"edit_size_bin", "actual_mask_ratio"), so train_hf.py / trainer.py / evals
work unchanged.

Supported layouts for --data-root:

A) `datasets` save_to_disk output (has dataset_dict.json)
       <root>/dataset_dict.json, <root>/train/, <root>/validation/, ...

B) Pre-processed image folders + metadata CSV (13_preprocess_images.py style)
       <root>/<split>/images/<safe_id>.(jpg|png|...)
       <root>/<split>/masks/<safe_id>.png          (optional per sample)
   Metadata CSV: --metadata, else the only *.csv inside <root>, else
   data/final_split_metadata.csv. Needs columns id, label, split
   (valid_mask / dataset / method / edit_size_bin / actual_mask_ratio optional).

C) Metadata CSV with explicit file paths (v3_processed style)
       <root>/metadata.csv with processed_image_path / processed_mask_path
       (or image_path / mask_path) and input_id (or id), label, split.
   Paths may be absolute or relative to <root>, its parent, or the repo root.
   If there is no valid_mask column: valid = real (all-zero GT), or edited
   with an existing mask file and actual_mask_ratio > 0.

Quick check:
    python -m data.dataloader_local data/v3_processed
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataloader_hf import (
    TinyEditHFDataset,
    _normalize_split_name,
    _split_dataset,
    to_bool,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
DEFAULT_META = Path(__file__).resolve().parent / "final_split_metadata.csv"


def make_safe_id(sample_id: Any) -> str:
    # Same rule as 13_preprocess_images.py / 14_build_dataloader.py
    return (
        str(sample_id)
        .replace(".jpg", "")
        .replace(".jpeg", "")
        .replace(".png", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _find_file(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _split_dirs(root: Path) -> Dict[str, Path]:
    out = {}
    for d in root.iterdir():
        if d.is_dir() and (d / "images").is_dir():
            out[_normalize_split_name(d.name)] = d
    return out


def _resolve_metadata(root: Path, metadata: Optional[str]) -> Path:
    if metadata:
        return Path(metadata)
    csvs = sorted(root.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) > 1:
        raise RuntimeError(
            f"Multiple CSVs in {root}: {[c.name for c in csvs]}. "
            "Pick one with --metadata."
        )
    return DEFAULT_META


def _resolve_path(value: Any, bases) -> Optional[Path]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = str(value).strip()
    if not v or v.lower() in {"nan", "none"}:
        return None
    v = v.replace("\\", "/")
    p = Path(v)
    if p.is_absolute():
        return p if p.exists() else None
    for base in bases:
        q = base / p
        if q.exists():
            return q
    # Fallback: strip leading folders one by one (e.g. "data/v3_processed/train/..." vs root)
    parts = p.parts
    for i in range(1, len(parts)):
        for base in bases:
            q = base.joinpath(*parts[i:])
            if q.exists():
                return q
    return None


class TinyEditPathDataset(TinyEditHFDataset):
    """Rows carry their own image/mask paths (resolved up-front)."""

    def __init__(self, df: pd.DataFrame, image_size: int = 512,
                 segmentation_only: bool = False):
        self.image_size = int(image_size)
        df = df.reset_index(drop=True)
        if segmentation_only:
            df = df[df["_valid"]].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        image = Image.open(row["_img"]).convert("RGB")
        mask = Image.open(row["_mask"]).convert("L") if row["_mask"] else None

        def s(col):
            v = row.get(col, "")
            return "" if pd.isna(v) else str(v)

        return {
            "pixel_values": self._image_tensor(image),
            "mask": self._mask_tensor(mask),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "valid_mask": torch.tensor(bool(row["_valid"]), dtype=torch.bool),
            "id": str(row["_id"]),
            "dataset": s("dataset"),
            "method": s("method"),
            "edit_size_bin": s("edit_size_bin"),
            "actual_mask_ratio": torch.tensor(float(row["_ratio"]), dtype=torch.float32),
        }


def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _path_datasets(root: Path, df: pd.DataFrame, meta_path: Path, image_size: int,
                   input_mode: Optional[str]):
    img_col = _first_col(df, ["processed_image_path", "image_path"])
    mask_col = _first_col(df, ["processed_mask_path", "mask_path"])
    id_col = _first_col(df, ["id", "input_id", "sample_id"])
    for col, name in ((id_col, "id/input_id"), ("label", "label"), ("split", "split")):
        if col is None or col not in df.columns:
            raise KeyError(f"{meta_path} has no '{name}' column: {list(df.columns)}")

    if "input_mode" in df.columns:
        counts = df["input_mode"].value_counts().to_dict()
        print(f"[local data] input_mode counts: {counts}")
        if input_mode:
            df = df[df["input_mode"].astype(str) == input_mode]
            print(f"[local data] using input_mode={input_mode}: {len(df)} rows")
        elif len(counts) > 1:
            print("[local data] WARNING: several input_mode values mixed; "
                  "pick one with --input-mode if they are alternative views of the same image.")

    repo = Path(__file__).resolve().parent.parent
    bases = [meta_path.parent, root, root.parent, repo, Path.cwd()]
    df = df.copy()
    df["_id"] = df[id_col].astype(str)
    df["_img"] = [_resolve_path(v, bases) for v in df[img_col]]
    df["_mask"] = ([_resolve_path(v, bases) for v in df[mask_col]]
                   if mask_col else [None] * len(df))

    n_missing = int(df["_img"].isna().sum())
    if n_missing:
        ex = df.loc[df["_img"].isna(), img_col].head(3).tolist()
        raise FileNotFoundError(
            f"{n_missing}/{len(df)} images not found (column {img_col}). "
            f"Examples: {ex}. Searched under: {[str(b) for b in bases]}"
        )

    ratio_col = _first_col(df, ["actual_mask_ratio", "mask_ratio"])
    df["_ratio"] = (pd.to_numeric(df[ratio_col], errors="coerce").fillna(0.0)
                    if ratio_col else 0.0)
    has_mask = df["_mask"].notna()
    if "valid_mask" in df.columns:
        df["_valid"] = df["valid_mask"].apply(to_bool) & has_mask
    else:
        # Real: GT is all-zero even without a file. Edited: needs a non-empty mask file.
        df["_valid"] = (df["label"].astype(int) == 0) | (has_mask & (df["_ratio"] > 0))
    df["_mask"] = df["_mask"].where(has_mask, None)

    print(f"[local data] root={root}  metadata={meta_path}  (path columns: {img_col}, {mask_col})")
    df["_split"] = df["split"].map(_normalize_split_name)
    out = {}
    for split in ("train", "validation", "test"):
        part = df[df["_split"] == split]
        if len(part) == 0:
            raise RuntimeError(f"No rows for split '{split}'. split values: {df['split'].unique().tolist()}")
        out[split] = TinyEditPathDataset(part, image_size, False)
        out[split + "_seg"] = TinyEditPathDataset(part, image_size, True)
    return out


class TinyEditFolderDataset(TinyEditHFDataset):
    """Reuses the HF dataset's resize/normalize/mask logic on local files."""

    def __init__(self, df: pd.DataFrame, split_dir: Path,
                 image_size: int = 512, segmentation_only: bool = False):
        self.image_size = int(image_size)
        self.split_dir = split_dir
        df = df.reset_index(drop=True)
        if segmentation_only and "valid_mask" in df.columns:
            df = df[df["valid_mask"].apply(to_bool)].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        stem = make_safe_id(row["id"])

        img_path = _find_file(self.split_dir / "images", stem)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {self.split_dir / 'images' / stem}.*")
        image = Image.open(img_path).convert("RGB")

        mask_path = _find_file(self.split_dir / "masks", stem)
        mask = Image.open(mask_path).convert("L") if mask_path else None

        if "valid_mask" in self.df.columns:
            is_valid = to_bool(row["valid_mask"]) and mask is not None
        else:
            is_valid = mask is not None

        try:
            ratio = float(row.get("actual_mask_ratio", 0.0))
            if np.isnan(ratio):
                ratio = 0.0
        except (TypeError, ValueError):
            ratio = 0.0

        def s(col):
            v = row.get(col, "")
            return "" if pd.isna(v) else str(v)

        return {
            "pixel_values": self._image_tensor(image),
            "mask": self._mask_tensor(mask),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "valid_mask": torch.tensor(is_valid, dtype=torch.bool),
            "id": str(row["id"]),
            "dataset": s("dataset"),
            "method": s("method"),
            "edit_size_bin": s("edit_size_bin"),
            "actual_mask_ratio": torch.tensor(ratio, dtype=torch.float32),
        }


def _folder_datasets(root: Path, metadata: Optional[str], image_size: int,
                     input_mode: Optional[str] = None):
    meta_path = _resolve_metadata(root, metadata)
    df = pd.read_csv(meta_path)
    if _first_col(df, ["processed_image_path", "image_path"]):
        return _path_datasets(root, df, meta_path, image_size, input_mode)

    dirs = _split_dirs(root)
    missing = {"train", "validation", "test"} - set(dirs)
    if missing:
        raise RuntimeError(
            f"{root}: expected <split>/images folders for train/val/test, "
            f"found {sorted(dirs)} (missing {sorted(missing)})"
        )
    for col in ("id", "label", "split"):
        if col not in df.columns:
            raise KeyError(f"{meta_path} has no '{col}' column: {list(df.columns)}")
    df["_split"] = df["split"].map(_normalize_split_name)
    print(f"[local data] root={root}  metadata={meta_path}")

    out = {}
    for split in ("train", "validation", "test"):
        part = df[df["_split"] == split]
        for seg in (False, True):
            key = split + ("_seg" if seg else "")
            out[key] = TinyEditFolderDataset(part, dirs[split], image_size, seg)
    return out


def _disk_datasets(root: Path, image_size: int):
    from datasets import load_from_disk
    splits = _split_dataset(load_from_disk(str(root)))
    print(f"[local data] load_from_disk({root})")
    out = {}
    for split in ("train", "validation", "test"):
        out[split] = TinyEditHFDataset(splits[split], image_size, False)
        out[split + "_seg"] = TinyEditHFDataset(splits[split], image_size, True)
    return out


def build_local_dataloaders(data_root: str, metadata: Optional[str] = None,
                            batch_size: int = 8, image_size: int = 512,
                            num_workers: int = 0, input_mode: Optional[str] = None):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"--data-root not found: {root.resolve()}")
    if (root / "dataset_dict.json").exists():
        ds = _disk_datasets(root, image_size)
    else:
        ds = _folder_datasets(root, metadata, image_size, input_mode)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    short = {"train": "train", "validation": "val", "test": "test"}
    out = {}
    for split, name in short.items():
        out[f"{name}_dataset"] = ds[split]
        out[f"{name}_loader"] = DataLoader(ds[split], shuffle=(split == "train"), **kw)
        out[f"{name}_seg_dataset"] = ds[split + "_seg"]
        out[f"{name}_seg_loader"] = DataLoader(ds[split + "_seg"], shuffle=(split == "train"), **kw)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/v3_processed"
    meta = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    mode = sys.argv[3] if len(sys.argv) > 3 else None
    data = build_local_dataloaders(root, meta, batch_size=4, input_mode=mode)
    for name in ("train", "val", "test"):
        d, s = data[f"{name}_dataset"], data[f"{name}_seg_dataset"]
        extra = ""
        if hasattr(d, "df"):
            vc = d.df["label"].astype(int).value_counts().to_dict()
            extra = f"  real={vc.get(0, 0)} edited={vc.get(1, 0)}"
        print(f"{name:5s}: {len(d)} samples ({len(s)} mask-valid){extra}")
    b = next(iter(data["val_loader"]))
    print("batch:", tuple(b["pixel_values"].shape), tuple(b["mask"].shape),
          b["label"].tolist(), b["valid_mask"].tolist())
    print("OK")
