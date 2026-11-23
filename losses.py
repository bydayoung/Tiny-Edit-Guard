"""Loss functions for Tiny-Edit-Guard.

Designed to plug directly into trainer.Trainer:

    loss_fn(outputs, batch) -> (loss, logs)

Rules:
- Classification loss uses every sample.
- Localization loss uses only samples with valid_mask=True.
    * BCE: every valid sample (real images teach "nothing is edited").
    * Dice: only edited (label=1) valid samples. On a real image the GT is empty, so the
      per-sample Dice loss is ≈1 whatever the model does (no useful gradient, inflated loss).
- Padded pixels (batch["pad_mask"] == 0) are excluded from every pixel/patch term.
- Patch supervision (optional, PatchGuard only) uses valid masks; the patch target is
  "soft" (fraction of the patch that is edited) by default. The old "max" target (positive if
  any pixel is edited) over-labels patches that a tiny edit barely touches.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

PATCH_TARGETS = ("soft", "hard", "max")


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask shape to (B, 1, H, W), float."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}"
        )
    return mask.float()


def _valid_mask(batch: Dict, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return a boolean vector of shape (B,). Missing valid_mask => all valid."""
    valid = batch.get("valid_mask")
    if valid is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)

    valid = valid.to(device=device, dtype=torch.bool).view(-1)
    if valid.numel() != batch_size:
        raise ValueError(
            f"valid_mask has {valid.numel()} entries, expected {batch_size}"
        )
    return valid


def _pad_region(batch: Dict, like: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) float, 1 = real image pixel, 0 = padding. Missing pad_mask => all ones."""
    pad = batch.get("pad_mask")
    if pad is None:
        return torch.ones_like(like[:, :1])
    pad = _as_mask(pad).to(like.device)
    if pad.shape[-2:] != like.shape[-2:]:
        pad = F.interpolate(pad, size=like.shape[-2:], mode="nearest")
    return pad


def dice_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    region: Optional[torch.Tensor] = None,
    eps: float = 1.0,
) -> torch.Tensor:
    """Binary soft-Dice loss per sample, shape (B,). `region` (B,1,H,W) zeroes out padding."""
    target = _as_mask(target)

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prob = logits.sigmoid()
    if region is not None:
        prob = prob * region
        target = target * region
    prob, target = prob.flatten(1), target.flatten(1)

    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)

    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def localization_loss(
    mask_logits: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    labels: Optional[torch.Tensor] = None,
    region: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE (valid samples) + Dice (valid edited samples), padding excluded.

    labels: (B,) image labels; when given, Dice is computed on label==1 samples only.
    region: (B, 1, H, W) 1 = real pixel, 0 = padding.
    """
    mask = _as_mask(mask)

    if mask_logits.shape[-2:] != mask.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if region is None:
        region = torch.ones_like(mask)

    if not valid_mask.any():
        # Keep the graph connected while contributing zero localization loss.
        return mask_logits.sum() * 0.0

    logits_v, mask_v, region_v = mask_logits[valid_mask], mask[valid_mask], region[valid_mask]

    bce = F.binary_cross_entropy_with_logits(logits_v, mask_v, reduction="none")
    bce_per_sample = (bce * region_v).flatten(1).sum(1) / region_v.flatten(1).sum(1).clamp(min=1.0)
    loss = bce_weight * bce_per_sample.mean()

    dice_rows = torch.ones_like(bce_per_sample, dtype=torch.bool)
    if labels is not None:
        dice_rows = labels[valid_mask].bool()
    if dice_rows.any():
        dice = dice_loss_per_sample(logits_v[dice_rows], mask_v[dice_rows], region_v[dice_rows])
        loss = loss + dice_weight * dice.mean()
    return loss


def patch_targets(mask: torch.Tensor, grid_hw, mode: str = "soft") -> Tuple[torch.Tensor, torch.Tensor]:
    """GT pixel mask (B,1,H,W) -> patch targets (B,1,h,w) and a weight map (B,1,h,w).

    soft: target = edited fraction of the patch (coverage), every patch weighted
    hard: positive if coverage >= 0.5, negative if 0, mixed boundary patches ignored
    max : positive if any pixel is edited (old behaviour)
    """
    if mode not in PATCH_TARGETS:
        raise ValueError(f"patch target mode must be one of {PATCH_TARGETS}")
    if mode == "max":
        t = F.adaptive_max_pool2d(mask, output_size=grid_hw)
        return t, torch.ones_like(t)
    cover = F.adaptive_avg_pool2d(mask, output_size=grid_hw)
    if mode == "soft":
        return cover, torch.ones_like(cover)
    target = (cover >= 0.5).float()
    weight = ((cover >= 0.5) | (cover == 0)).float()
    return target, weight


def patch_supervision_loss(
    patch_score_map: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "soft",
    region: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Patch-level BCE supervision from GT masks (padding patches excluded)."""
    mask = _as_mask(mask)

    if not valid_mask.any():
        return patch_score_map.sum() * 0.0

    score_v = patch_score_map[valid_mask]
    grid = score_v.shape[-2:]
    target, weight = patch_targets(mask[valid_mask], grid, mode)
    if region is not None:
        weight = weight * (F.adaptive_avg_pool2d(region[valid_mask], output_size=grid) >= 0.5).float()

    bce = F.binary_cross_entropy_with_logits(score_v, target, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=1.0)


class TinyEditLoss:
    """Callable loss object compatible with trainer.Trainer."""

    def __init__(
        self,
        mask_weight: float = 1.0,
        patch_weight: float = 0.0,
        mask_bce_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
        patch_target: str = "soft",
        dice_on_fake_only: bool = True,
    ):
        self.mask_weight = float(mask_weight)
        self.patch_weight = float(patch_weight)
        self.mask_bce_weight = float(mask_bce_weight)
        self.mask_dice_weight = float(mask_dice_weight)
        self.patch_target = patch_target
        self.dice_on_fake_only = dice_on_fake_only

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if "cls_logits" not in outputs:
            raise KeyError("Model output must contain 'cls_logits'.")
        if "label" not in batch:
            raise KeyError("Batch must contain 'label'.")

        cls_logits = outputs["cls_logits"]
        labels = batch["label"].float().view_as(cls_logits)

        # 1) Image-level classification: use every sample.
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, labels)

        total = loss_cls
        logs = {"loss_cls": float(loss_cls.detach().item())}

        need_mask = (self.mask_weight > 0.0 and "mask_logits" in outputs) or self.patch_weight > 0.0
        if need_mask and "mask" in batch:
            valid = _valid_mask(batch, batch_size=cls_logits.shape[0], device=cls_logits.device)
            mask = _as_mask(batch["mask"])
            region = _pad_region(batch, mask)

            # 2) Pixel-level localization: valid masks only.
            if self.mask_weight > 0.0 and "mask_logits" in outputs:
                loss_mask = localization_loss(
                    outputs["mask_logits"],
                    mask,
                    valid_mask=valid,
                    bce_weight=self.mask_bce_weight,
                    dice_weight=self.mask_dice_weight,
                    labels=labels if self.dice_on_fake_only else None,
                    region=region,
                )
                total = total + self.mask_weight * loss_mask
                logs["loss_mask"] = float(loss_mask.detach().item())
                logs["valid_mask_fraction"] = float(valid.float().mean().item())

            # 3) Optional patch supervision: PatchGuard only.
            if self.patch_weight > 0.0:
                if "patch_score_map" not in outputs:
                    raise KeyError(
                        "patch_weight > 0 but model output has no 'patch_score_map'. "
                        "Patch supervision is only for PatchGuard models."
                    )
                loss_patch = patch_supervision_loss(
                    outputs["patch_score_map"], mask, valid_mask=valid,
                    mode=self.patch_target, region=region,
                )
                total = total + self.patch_weight * loss_patch
                logs["loss_patch"] = float(loss_patch.detach().item())
        elif self.patch_weight > 0.0:
            raise KeyError("Patch supervision requires batch['mask'].")

        logs["loss"] = float(total.detach().item())
        return total, logs


def build_loss(
    mask_weight: float = 1.0,
    patch_weight: float = 0.0,
    mask_bce_weight: float = 1.0,
    mask_dice_weight: float = 1.0,
    patch_target: str = "soft",
) -> TinyEditLoss:
    return TinyEditLoss(
        mask_weight=mask_weight,
        patch_weight=patch_weight,
        mask_bce_weight=mask_bce_weight,
        mask_dice_weight=mask_dice_weight,
        patch_target=patch_target,
    )


if __name__ == "__main__":
    # Local smoke test; does not require pretrained models.
    B, H, W = 4, 64, 64

    outputs = {
        "cls_logits": torch.randn(B, requires_grad=True),
        "mask_logits": torch.randn(B, 1, H, W, requires_grad=True),
        "patch_score_map": torch.randn(B, 1, 4, 4, requires_grad=True),
    }

    batch = {
        "label": torch.tensor([0, 1, 1, 0]),
        "mask": torch.zeros(B, 1, H, W),
        "valid_mask": torch.tensor([True, True, False, True]),
        "pad_mask": torch.ones(B, 1, H, W),
    }
    batch["mask"][1, :, 10:18, 20:28] = 1.0
    batch["pad_mask"][0, :, :, 48:] = 0.0

    for mode in PATCH_TARGETS:
        loss_fn = TinyEditLoss(mask_weight=1.0, patch_weight=0.1, patch_target=mode)
        loss, logs = loss_fn(outputs, batch)
        print(mode, logs)
    loss.backward()
    print("loss backward: OK")
