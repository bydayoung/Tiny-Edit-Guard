"""Batching utilities for variable-size, paired (real/fake) data (V3).

    GroupIdDataset      wraps a dataset and adds sample["group_id"] (pair id) from its metadata
    PairBatchSampler    keeps every real/fake pair of a group in the SAME batch
    PadCollate          pads images to a common multiple-of-16 size and adds batch["pad_mask"]
                        (1 = real image pixel, 0 = padding); optional paired augmentation

Why paired batches / paired augmentation:
  B-Free (CVPR'25) and "Aligned Datasets" (ICLR'25) train on content-aligned real/fake pairs so
  the only difference left is the generation trace. Putting both members of a pair in the same
  batch, and giving them the SAME flip / JPEG quality / blur, keeps that property during training
  (different JPEG strength for real vs fake would itself become a shortcut).

Works with any dataset whose samples are dicts with "image" or "pixel_values" (3,H,W,
ImageNet-normalized) and optionally "mask" (1,H,W) / "pad_mask" (1,H,W).
"""
from __future__ import annotations

import io
import math
import random
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

GROUP_COLUMNS = ("final_group_id", "group_id", "pair_id")
IMAGE_KEYS = ("pixel_values", "image")


# ----------------------------------------------------------------------------- group ids

def find_group_ids(dataset) -> Optional[List[str]]:
    """Group ids from dataset.df (pandas, data/14_build_dataloader.py style) or dataset.ds
    (Hugging Face Dataset, data/dataloader_hf.py style). None if no group column exists."""
    df = getattr(dataset, "df", None)
    if df is not None:
        for col in GROUP_COLUMNS:
            if col in df.columns:
                return [str(v) for v in df[col].tolist()]
    ds = getattr(dataset, "ds", None)
    if ds is not None and hasattr(ds, "column_names"):
        for col in GROUP_COLUMNS:
            if col in ds.column_names:
                return [str(v) for v in ds[col]]
    return None


class GroupIdDataset(Dataset):
    """Adds sample["group_id"] so it reaches the collate fn and predictions CSV."""

    def __init__(self, dataset, group_ids: Sequence[str]):
        if len(group_ids) != len(dataset):
            raise ValueError(f"{len(group_ids)} group ids for {len(dataset)} samples")
        self.dataset = dataset
        self.group_ids = list(group_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = dict(self.dataset[idx])
        sample.setdefault("group_id", self.group_ids[idx])
        return sample


# ----------------------------------------------------------------------------- sampler

class PairBatchSampler(Sampler[List[int]]):
    """Shuffles groups (not samples) and packs whole groups into batches of ~batch_size.

    A group larger than batch_size becomes its own batch. Re-shuffles on every __iter__.
    """

    def __init__(self, group_ids: Sequence[str], batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False):
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for i, g in enumerate(group_ids):
            groups.setdefault(str(g), []).append(i)
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._len = len(self._make_batches(self.groups))

    def _make_batches(self, groups):
        batches, cur = [], []
        for g in groups:
            if cur and len(cur) + len(g) > self.batch_size:
                batches.append(cur)
                cur = []
            cur = cur + g
        if cur and not (self.drop_last and len(cur) < self.batch_size):
            batches.append(cur)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        self.epoch += 1
        yield from self._make_batches(groups)

    def __len__(self):
        return self._len


# ----------------------------------------------------------------------------- augmentation

class PairedAugment:
    """flip / JPEG / blur with ONE set of random parameters per group (pair).

    Applied to ImageNet-normalized tensors (de-normalized internally). Masks and pad masks are
    flipped together with the image. Defaults follow the CNNDetect / B-Free recipe
    (blur + JPEG), plus horizontal flip.
    """

    def __init__(self, p_flip: float = 0.5, p_jpeg: float = 0.5, jpeg_quality=(65, 100),
                 p_blur: float = 0.3, blur_sigma=(0.1, 1.5), seed: Optional[int] = None):
        self.p_flip, self.p_jpeg, self.jpeg_quality = p_flip, p_jpeg, jpeg_quality
        self.p_blur, self.blur_sigma = p_blur, blur_sigma
        self.rng = random.Random(seed)

    def draw(self) -> Dict:
        r = self.rng
        return {
            "flip": r.random() < self.p_flip,
            "jpeg": r.randint(*self.jpeg_quality) if r.random() < self.p_jpeg else None,
            "blur": r.uniform(*self.blur_sigma) if r.random() < self.p_blur else None,
        }

    @staticmethod
    def _photometric(x: torch.Tensor, jpeg: Optional[int], blur: Optional[float]) -> torch.Tensor:
        if jpeg is None and blur is None:
            return x
        rgb = (x * IMAGE_STD + IMAGE_MEAN).clamp(0, 1)
        img = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        if blur is not None:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        if jpeg is not None:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(jpeg))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (arr - IMAGE_MEAN) / IMAGE_STD

    def _reseed_per_worker(self):
        # forked DataLoader workers would otherwise share one RNG state (identical augmentations)
        info = torch.utils.data.get_worker_info()
        if info is not None and getattr(self, "_worker_id", None) != info.id:
            self._worker_id = info.id
            self.rng.seed(info.seed)

    def __call__(self, samples: List[Dict]) -> List[Dict]:
        self._reseed_per_worker()
        params: Dict[str, Dict] = {}
        out = []
        for s in samples:
            g = str(s.get("group_id", id(s)))  # no group id -> independent params per sample
            p = params.setdefault(g, self.draw())
            s = dict(s)
            key = next(k for k in IMAGE_KEYS if k in s)
            x = s[key]
            if p["flip"]:
                x = x.flip(-1)
                for k in ("mask", "pad_mask"):
                    if k in s and torch.is_tensor(s[k]):
                        s[k] = s[k].flip(-1)
            s[key] = self._photometric(x, p["jpeg"], p["blur"])
            out.append(s)
        return out


# ----------------------------------------------------------------------------- collate

class PadCollate:
    """Pads every sample to the batch's max H/W (rounded up to `multiple`) and adds pad_mask.

    Padding value 0 in normalized space (= ImageNet mean colour). Masks are padded with 0.
    If samples already carry "pad_mask" (dataset-side padding), it is padded and kept.
    The image key is renamed to "pixel_values" (what trainer.py expects).
    """

    def __init__(self, multiple: int = 16, augment: Optional[PairedAugment] = None):
        self.multiple = multiple
        self.augment = augment

    def __call__(self, samples: List[Dict]) -> Dict:
        if self.augment is not None:
            samples = self.augment(samples)
        key = next(k for k in IMAGE_KEYS if k in samples[0])
        H = max(s[key].shape[-2] for s in samples)
        W = max(s[key].shape[-1] for s in samples)
        H = int(math.ceil(H / self.multiple) * self.multiple)
        W = int(math.ceil(W / self.multiple) * self.multiple)

        padded = []
        for s in samples:
            s = dict(s)
            x = s.pop(key)
            h, w = x.shape[-2:]
            pad = (0, W - w, 0, H - h)  # right / bottom
            region = s.pop("pad_mask", None)
            if region is None:
                region = torch.ones(1, h, w)
            region = region.float().reshape(1, h, w)
            s["pixel_values"] = F.pad(x, pad, value=0.0)
            s["pad_mask"] = F.pad(region, pad, value=0.0)
            if "mask" in s and torch.is_tensor(s["mask"]):
                m = s["mask"].float()
                m = m.unsqueeze(0) if m.dim() == 2 else m
                s["mask"] = F.pad(m, pad, value=0.0)
            padded.append(s)
        return default_collate(padded)


def pad_collate(samples: List[Dict]) -> Dict:
    return PadCollate()(samples)
