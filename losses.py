"""Loss functions for Tiny-Edit-Guard.

Designed to plug directly into trainer.Trainer:

    loss_fn(outputs, batch) -> (loss, logs)

Rules:
- Classification loss uses every sample.
- Localization loss uses only samples with valid_mask=True.
- Patch-supervision loss is optional and uses only valid masks.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


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


def dice_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Binary soft-Dice loss, returned per sample as shape (B,)."""
    target = _as_mask(target)

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prob = logits.sigmoid().flatten(1)
    target = target.flatten(1)

    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)

    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def localization_loss(
    mask_logits: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """BCE + Dice computed only on valid localization samples."""
    mask = _as_mask(mask)

    if mask_logits.shape[-2:] != mask.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if not valid_mask.any():
        # Keep the graph connected while contributing zero localization loss.
        return mask_logits.sum() * 0.0

    logits_v = mask_logits[valid_mask]
    mask_v = mask[valid_mask]

    bce_per_pixel = F.binary_cross_entropy_with_logits(
        logits_v,
        mask_v,
        reduction="none",
    )
    bce_per_sample = bce_per_pixel.flatten(1).mean(dim=1)

    dice_per_sample = dice_loss_per_sample(
        logits_v,
        mask_v,
    )

    return (
        bce_weight * bce_per_sample
        + dice_weight * dice_per_sample
    ).mean()


def patch_supervision_loss(
    patch_score_map: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Patch-level BCE supervision from GT masks.

    GT mask is reduced to the patch grid with adaptive max pooling.
    A patch is positive if at least one edited pixel lies in that patch.
    """
    mask = _as_mask(mask)

    if not valid_mask.any():
        return patch_score_map.sum() * 0.0

    score_v = patch_score_map[valid_mask]
    mask_v = mask[valid_mask]

    patch_target = F.adaptive_max_pool2d(
        mask_v,
        output_size=score_v.shape[-2:],
    )

    return F.binary_cross_entropy_with_logits(
        score_v,
        patch_target,
    )


class TinyEditLoss:
    """Callable loss object compatible with trainer.Trainer."""

    def __init__(
        self,
        mask_weight: float = 1.0,
        patch_weight: float = 0.0,
        mask_bce_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
    ):
        self.mask_weight = float(mask_weight)
        self.patch_weight = float(patch_weight)
        self.mask_bce_weight = float(mask_bce_weight)
        self.mask_dice_weight = float(mask_dice_weight)

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
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_logits,
            labels,
        )

        total = loss_cls
        logs = {
            "loss_cls": float(loss_cls.detach().item()),
        }

        # 2) Pixel-level localization: valid masks only.
        if (
            self.mask_weight > 0.0
            and "mask_logits" in outputs
            and "mask" in batch
        ):
            valid = _valid_mask(
                batch,
                batch_size=cls_logits.shape[0],
                device=cls_logits.device,
            )

            loss_mask = localization_loss(
                outputs["mask_logits"],
                batch["mask"],
                valid_mask=valid,
                bce_weight=self.mask_bce_weight,
                dice_weight=self.mask_dice_weight,
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
            if "mask" not in batch:
                raise KeyError("Patch supervision requires batch['mask'].")

            valid = _valid_mask(
                batch,
                batch_size=cls_logits.shape[0],
                device=cls_logits.device,
            )

            loss_patch = patch_supervision_loss(
                outputs["patch_score_map"],
                batch["mask"],
                valid_mask=valid,
            )

            total = total + self.patch_weight * loss_patch
            logs["loss_patch"] = float(loss_patch.detach().item())

        logs["loss"] = float(total.detach().item())
        return total, logs


def build_loss(
    mask_weight: float = 1.0,
    patch_weight: float = 0.0,
    mask_bce_weight: float = 1.0,
    mask_dice_weight: float = 1.0,
) -> TinyEditLoss:
    return TinyEditLoss(
        mask_weight=mask_weight,
        patch_weight=patch_weight,
        mask_bce_weight=mask_bce_weight,
        mask_dice_weight=mask_dice_weight,
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
    }
    batch["mask"][1, :, 10:18, 20:28] = 1.0

    loss_fn = TinyEditLoss(mask_weight=1.0, patch_weight=0.1)
    loss, logs = loss_fn(outputs, batch)

    print(logs)
    loss.backward()
    print("loss backward: OK")
