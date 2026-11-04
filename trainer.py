"""Generic trainer for GlobalBaseline / SegFormerBaseline / PatchGuard.

Batch format (dict from the DataLoader):
    "pixel_values": (B, 3, H, W) float, ImageNet-normalized   ("image" is also accepted)
    "label":        (B,) int/float, 1 = edited, 0 = real
    "mask":         (B, 1, H, W) or (B, H, W) float in {0, 1}   [optional, for localization]
    "valid_mask":   (B,) bool   [optional; False = invalid/empty GT on an edited sample ->
                                 excluded from mask loss and localization metrics. Real images
                                 with empty masks are valid.]
    any other per-sample fields (e.g. "actual_mask_ratio", "method", "id") are passed through
    to `Trainer.predict()` CSV output for Tiny Edit / Unseen Method analysis.

Model output format (dict):
    "cls_logits":   (B,)            [required]
    "mask_logits":  (B, 1, H, W)    [optional; enables mask loss + localization metrics]

Loss: pass `loss_fn(outputs, batch) -> (loss, {name: float})`. losses.py can be plugged in here;
`default_loss` (BCE for classification, BCE + Dice for the mask) is used otherwise.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, set_seed

# Old metric names -> current names, so existing --monitor values keep selecting best.pt.
METRIC_ALIASES = {"auc": "auroc", "pixel_f1": "pixel_dice", "image_mean_f1": "image_mean_dice"}

LossFn = Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]


# ----------------------------------------------------------------------------- loss

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample soft Dice loss, shape (B,)."""
    prob = logits.sigmoid().flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    return 1 - (2 * inter + eps) / (prob.sum(1) + target.sum(1) + eps)


def default_loss(outputs, batch, mask_weight: float = 1.0):
    logs = {}
    loss = F.binary_cross_entropy_with_logits(outputs["cls_logits"], batch["label"].float())
    logs["loss_cls"] = loss.item()

    if "mask_logits" in outputs and "mask" in batch:
        valid = mask_valid(batch)
        if valid.any():
            mask = _as_mask(batch["mask"])[valid]
            m_logits = outputs["mask_logits"][valid]
            bce = F.binary_cross_entropy_with_logits(m_logits, mask, reduction="none").flatten(1).mean(1)
            loss_mask = (bce + dice_loss(m_logits, mask)).mean()
            logs["loss_mask"] = loss_mask.item()
            loss = loss + mask_weight * loss_mask

    logs["loss"] = loss.item()
    return loss, logs


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.unsqueeze(1) if mask.dim() == 3 else mask).float()


def pixel_values(batch) -> torch.Tensor:
    """data/dataloader_hf.py yields "pixel_values"; data/14_build_dataloader.py yields "image"."""
    return batch["pixel_values"] if "pixel_values" in batch else batch["image"]


def mask_valid(batch) -> torch.Tensor:
    if "valid_mask" in batch:
        return batch["valid_mask"].bool()
    return torch.ones_like(batch["label"], dtype=torch.bool)


# ----------------------------------------------------------------------------- metrics

def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) formula; ties get average rank."""
    labels = labels.bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    sorted_scores = scores[order]
    uniq, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    if len(uniq) < len(scores):  # average ranks over ties
        start = 0
        for c in counts.tolist():
            if c > 1:
                idx = order[start:start + c]
                ranks[idx] = ranks[idx].mean()
            start += c
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUPRC as average precision (sklearn-style step interpolation)."""
    labels = labels.bool()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    hits = labels[order].double()
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.float64)
    return float((precision * hits).sum() / n_pos)


def binned_average_precision(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
    """AP from per-bin positive/negative counts (bins in ascending score order).
    Used for pixel AUPRC, where storing every pixel score is too large."""
    tp = pos_hist.flip(0).cumsum(0)
    fp = neg_hist.flip(0).cumsum(0)
    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan")
    precision = tp / (tp + fp).clamp(min=1)
    recall_step = pos_hist.flip(0) / n_pos
    return float((precision * recall_step).sum())


class MetricTracker:
    def __init__(self, threshold: float = 0.5, pixel_bins: int = 1000):
        self.threshold = threshold
        self.scores, self.labels = [], []
        self.tp = self.fp = self.fn = 0.0
        self.per_image_dice = []
        self.per_image_f1 = self.per_image_dice  # old name; subclasses that append here still count
        self.pixel_bins = pixel_bins
        self.pos_hist = torch.zeros(pixel_bins, dtype=torch.float64)
        self.neg_hist = torch.zeros(pixel_bins, dtype=torch.float64)
        self.has_mask = False

    @torch.no_grad()
    def update(self, outputs, batch):
        self.scores.append(outputs["cls_logits"].sigmoid().float().cpu())
        self.labels.append(batch["label"].float().cpu())

        if "mask_logits" in outputs and "mask" in batch:
            valid = mask_valid(batch)
            if not valid.any():
                return
            self.has_mask = True
            prob = outputs["mask_logits"][valid].sigmoid()
            gt = _as_mask(batch["mask"])[valid]
            pred = (prob > self.threshold).float()
            tp = (pred * gt).flatten(1).sum(1)
            fp = (pred * (1 - gt)).flatten(1).sum(1)
            fn = ((1 - pred) * gt).flatten(1).sum(1)
            self.tp += tp.sum().item()
            self.fp += fp.sum().item()
            self.fn += fn.sum().item()
            # per-image Dice only on edited images (real images have empty GT masks)
            edited = batch["label"][valid].bool()
            dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)
            self.per_image_dice.extend(dice[edited].cpu().tolist())

            bins = (prob.flatten() * self.pixel_bins).long().clamp(max=self.pixel_bins - 1)
            gt_flat = gt.flatten().bool()
            self.pos_hist += torch.bincount(bins[gt_flat], minlength=self.pixel_bins).double().cpu()
            self.neg_hist += torch.bincount(bins[~gt_flat], minlength=self.pixel_bins).double().cpu()

    def compute(self) -> Dict[str, float]:
        scores, labels = torch.cat(self.scores), torch.cat(self.labels)
        pred = (scores > self.threshold).float()
        tp = (pred * labels).sum().item()
        fp = (pred * (1 - labels)).sum().item()
        fn = ((1 - pred) * labels).sum().item()
        m = {
            "auroc": binary_auc(scores, labels),
            "auprc": average_precision(scores, labels),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1e-8),
            "acc": (pred == labels).float().mean().item(),
        }
        if self.has_mask:
            if self.pos_hist.sum() > 0:  # empty when a subclass overrides update() without filling it
                m["pixel_auprc"] = binned_average_precision(self.pos_hist, self.neg_hist)
            m["pixel_dice"] = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1e-8)
            m["pixel_iou"] = self.tp / max(self.tp + self.fp + self.fn, 1e-8)
            if self.per_image_dice:
                m["image_mean_dice"] = sum(self.per_image_dice) / len(self.per_image_dice)
        return m


# ----------------------------------------------------------------------------- trainer

def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[LossFn] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        backbone_lr: float = 1e-5,
        backbone_prefixes: Iterable[str] = ("backbone.", "model.segformer."),
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        grad_clip: float = 1.0,
        amp: bool = True,
        monitor: str = "auroc",
        monitor_mode: str = "max",
        output_dir: str = "checkpoints/run",
        log_every: int = 50,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            backbone_prefixes: parameter-name prefixes trained with `backbone_lr` (pretrained
                encoders); everything else (heads/decoders) uses `lr`. Frozen params are skipped.
            monitor: validation metric used to pick `best.pt` (e.g. "auroc" for V1, "pixel_dice" for V2).
            seed: re-seeds RNGs here (DataLoader shuffle, dropout). Also call `set_seed(seed)`
                before building the model so head initialization is reproducible.
        """
        if seed is not None:
            set_seed(seed)
        self.device = device or auto_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or default_loss
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.monitor = METRIC_ALIASES.get(monitor, monitor)
        self.monitor_mode = monitor_mode
        self.log_every = log_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self._param_groups(lr, backbone_lr, tuple(backbone_prefixes), weight_decay)
        )
        total_steps = epochs * len(train_loader)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, int(warmup_ratio * total_steps), total_steps
        )

        # AMP on CUDA only (bf16 if supported, else fp16 + GradScaler).
        self.use_amp = amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.epoch = 0
        self.global_step = 0
        self.best = -math.inf if monitor_mode == "max" else math.inf

    def _param_groups(self, lr, backbone_lr, prefixes, weight_decay):
        groups = {(b, d): [] for b in (True, False) for d in (True, False)}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = name.startswith(prefixes)
            decay = p.dim() > 1  # no decay for bias / norm
            groups[(is_backbone, decay)].append(p)
        out = []
        for (is_backbone, decay), params in groups.items():
            if params:
                out.append({
                    "params": params,
                    "lr": backbone_lr if is_backbone else lr,
                    "weight_decay": weight_decay if decay else 0.0,
                })
        n = sum(p.numel() for g in out for p in g["params"])
        print(f"[trainer] trainable params: {n / 1e6:.2f}M")
        return out

    def _to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _forward(self, batch):
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(pixel_values(batch))
        # losses / metrics in fp32
        return {k: v.float() if torch.is_tensor(v) else v for k, v in outputs.items()}

    # ------------------------------------------------------------------ loops

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals, n = {}, 0
        t0 = time.time()
        for i, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            outputs = self._forward(batch)
            loss, logs = self.loss_fn(outputs, batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            for k, v in logs.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
            if self.log_every and (i + 1) % self.log_every == 0:
                avg = {k: round(v / n, 4) for k, v in totals.items()}
                print(f"[epoch {self.epoch} step {i + 1}/{len(self.train_loader)}] {avg} "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        return {f"train_{k}": v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        loader = loader or self.val_loader
        self.model.eval()
        tracker = MetricTracker()
        loss_sum, n = 0.0, 0
        for batch in loader:
            batch = self._to_device(batch)
            outputs = self._forward(batch)
            loss, _ = self.loss_fn(outputs, batch)
            loss_sum += loss.item()
            n += 1
            tracker.update(outputs, batch)
        metrics = {"val_loss": loss_sum / max(n, 1)}
        metrics.update({f"val_{k}": v for k, v in tracker.compute().items()})
        return metrics

    def fit(self) -> Dict[str, float]:
        start = self.epoch
        metrics = {}
        for epoch in range(start, self.epochs):
            self.epoch = epoch
            metrics = {"epoch": epoch, **self.train_one_epoch()}
            if self.val_loader is not None:
                metrics.update(self.validate())
            print(f"[epoch {epoch}] " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
            with open(self.output_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")

            self.epoch = epoch + 1
            self.save("last.pt")
            score = metrics.get(f"val_{self.monitor}")
            if score is not None and not math.isnan(score) and self._is_better(score):
                self.best = score
                self.save("best.pt")
                print(f"[epoch {epoch}] new best val_{self.monitor}={score:.4f}")
        return metrics

    def _is_better(self, score: float) -> bool:
        return score > self.best if self.monitor_mode == "max" else score < self.best

    # ------------------------------------------------------------------ checkpoint

    def save(self, name: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best": self.best,
            "monitor": self.monitor,
        }, self.output_dir / name)

    def load(self, path: str, weights_only: bool = False) -> None:
        """Resume training (weights_only=False) or just load model weights for evaluation."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if not weights_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.epoch = ckpt["epoch"]
            self.global_step = ckpt["global_step"]
            self.best = ckpt["best"]
        print(f"[trainer] loaded {path} (epoch {ckpt['epoch']}, best {ckpt['best']:.4f})")

    # ------------------------------------------------------------------ per-sample predictions

    @torch.no_grad()
    def predict(self, loader: DataLoader, csv_path: Optional[str] = None, threshold: float = 0.5):
        """Per-sample results for subset analysis (Tiny Edit bins, Unseen Method, robustness).

        Each row: edit_probability, label, [predicted_edit_ratio, sample_dice, sample_iou]
        plus any per-sample batch fields (actual_mask_ratio, method, id, ...).
        """
        self.model.eval()
        skip = {"pixel_values", "image", "mask"}
        rows = []
        for batch in loader:
            batch = self._to_device(batch)
            outputs = self._forward(batch)
            B = pixel_values(batch).shape[0]
            cols = {
                "edit_probability": outputs["cls_logits"].sigmoid().cpu().tolist(),
                "label": batch["label"].cpu().tolist(),
            }
            if "mask_logits" in outputs:
                pred = (outputs["mask_logits"].sigmoid() > threshold).float()
                cols["predicted_edit_ratio"] = pred.flatten(1).mean(1).cpu().tolist()
                if "mask" in batch:
                    gt = _as_mask(batch["mask"])
                    tp = (pred * gt).flatten(1).sum(1)
                    fp = (pred * (1 - gt)).flatten(1).sum(1)
                    fn = ((1 - pred) * gt).flatten(1).sum(1)
                    valid = mask_valid(batch)
                    dice = torch.where(valid, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8), torch.nan)
                    iou = torch.where(valid, tp / (tp + fp + fn).clamp(min=1e-8), torch.nan)
                    cols["sample_dice"] = dice.cpu().tolist()
                    cols["sample_iou"] = iou.cpu().tolist()
            for k, v in batch.items():
                if k in skip or k in cols:
                    continue
                if torch.is_tensor(v) and v.dim() == 1 and len(v) == B:
                    cols[k] = v.cpu().tolist()
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    cols[k] = list(v)
            rows.extend(dict(zip(cols, vals)) for vals in zip(*cols.values()))

        if csv_path and rows:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[trainer] wrote {len(rows)} predictions to {csv_path}")
        return rows

    def export_predictions(self, loaders: Dict[str, Iterable], checkpoint: str = "best.pt") -> Dict[str, Path]:
        """After fit(): reload `checkpoint` and write <output_dir>/preds_<split>.csv per loader.

        These CSVs are the input of `evals/evaluate.py --trainer-csv` (val CSV picks the threshold):
            python3 evals/evaluate.py --trainer-csv <out>/preds_val.csv
            python3 evals/evaluate.py --trainer-csv <out>/preds_test.csv --val-csv <out>/preds_val.csv --eval-split test
        """
        ckpt = self.output_dir / checkpoint
        if not ckpt.exists():  # best.pt is never written if the monitor metric was always NaN
            print(f"[trainer] {ckpt.name} not found, using last.pt")
            ckpt = self.output_dir / "last.pt"
        self.load(str(ckpt), weights_only=True)

        paths = {}
        for split, loader in loaders.items():
            paths[split] = self.output_dir / f"preds_{split}.csv"
            self.predict(loader, str(paths[split]))
        return paths
