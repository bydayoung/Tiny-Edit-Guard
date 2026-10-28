from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _normalize_mask_shape(masks: torch.Tensor) -> torch.Tensor:
    """
    [B,1,H,W] 또는 [B,H,W]를 [B,H,W]로 통일.
    """
    if masks.ndim == 4:
        if masks.shape[1] != 1:
            raise ValueError(
                f"4D masks must be [B,1,H,W], got {tuple(masks.shape)}"
            )
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise ValueError(
            f"masks must be [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}"
        )

    return masks


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    이미지 단위 Real/Edited 분류 loss.
    labels: 0=Real, 1=Edited
    """
    labels = labels.float().view_as(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)


def dice_loss_per_sample(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    edited class(index=1)에 대한 sample-wise soft Dice loss.
    """
    masks = _normalize_mask_shape(masks)

    if mask_logits.ndim != 4 or mask_logits.shape[1] != 2:
        raise ValueError(
            f"mask_logits must be [B,2,H,W], got {tuple(mask_logits.shape)}"
        )

    edited_prob = torch.softmax(mask_logits, dim=1)[:, 1]
    target = masks.float()

    intersection = (edited_prob * target).sum(dim=(1, 2))
    denominator = (
        edited_prob.sum(dim=(1, 2))
        + target.sum(dim=(1, 2))
    )

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def segmentation_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Localization loss = Cross Entropy + Dice.

    mask_valid=False인 sample은 localization loss에서 제외.
    """
    masks = _normalize_mask_shape(masks).long()

    if mask_logits.shape[-2:] != masks.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    ce_pixel = F.cross_entropy(
        mask_logits,
        masks,
        reduction="none",
    )
    ce_sample = ce_pixel.mean(dim=(1, 2))

    dice_sample = dice_loss_per_sample(
        mask_logits,
        masks,
    )

    per_sample = (
        ce_weight * ce_sample
        + dice_weight * dice_sample
    )

    if mask_valid is None:
        return per_sample.mean()

    mask_valid = mask_valid.to(
        device=per_sample.device,
        dtype=torch.bool,
    ).view(-1)

    if mask_valid.shape[0] != per_sample.shape[0]:
        raise ValueError("mask_valid batch size mismatch.")

    if not mask_valid.any():
        return per_sample.sum() * 0.0

    return per_sample[mask_valid].mean()


def patch_supervision_loss(
    score_map: torch.Tensor,
    masks: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    선택적 patch supervision.

    GT mask를 score_map 크기로 adaptive max pooling.
    한 patch 안에 edited pixel이 하나라도 있으면 positive patch.
    """
    if score_map is None:
        raise ValueError(
            "score_map is None. Patch supervision requires "
            "patch_average or patch_topk mode."
        )

    masks = _normalize_mask_shape(masks).float()

    patch_targets = F.adaptive_max_pool2d(
        masks.unsqueeze(1),
        output_size=score_map.shape[-2:],
    )

    loss_map = F.binary_cross_entropy_with_logits(
        score_map,
        patch_targets,
        reduction="none",
    )

    per_sample = loss_map.mean(dim=(1, 2, 3))

    if mask_valid is None:
        return per_sample.mean()

    mask_valid = mask_valid.to(
        device=per_sample.device,
        dtype=torch.bool,
    ).view(-1)

    if not mask_valid.any():
        return per_sample.sum() * 0.0

    return per_sample[mask_valid].mean()


def compute_loss(
    outputs: Dict[str, Optional[torch.Tensor]],
    labels: torch.Tensor,
    masks: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    lambda_mask: float = 1.0,
    lambda_patch: float = 0.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    전체 loss.

    V1 권장:
        lambda_mask=1.0
        lambda_patch=0.0

    Patch supervision 실험:
        lambda_patch > 0
    """
    cls = classification_loss(
        outputs["logits"],
        labels,
    )

    mask = segmentation_loss(
        outputs["mask_logits"],
        masks,
        mask_valid=mask_valid,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )

    total = cls + lambda_mask * mask

    if lambda_patch > 0.0:
        patch = patch_supervision_loss(
            outputs["score_map"],
            masks,
            mask_valid=mask_valid,
        )
        total = total + lambda_patch * patch
    else:
        patch = total.new_zeros(())

    return {
        "loss": total,
        "cls_loss": cls.detach(),
        "mask_loss": mask.detach(),
        "patch_loss": patch.detach(),
    }


if __name__ == "__main__":
    batch_size = 4
    h = w = 512

    outputs = {
        "logits": torch.randn(batch_size, requires_grad=True),
        "mask_logits": torch.randn(
            batch_size, 2, h, w, requires_grad=True
        ),
        "score_map": torch.randn(
            batch_size, 1, 16, 16, requires_grad=True
        ),
    }

    labels = torch.tensor([0, 1, 1, 0])

    masks = torch.zeros(
        batch_size, h, w, dtype=torch.long
    )
    masks[1, 100:120, 200:220] = 1
    masks[2, 300:310, 300:315] = 1

    mask_valid = torch.tensor(
        [True, True, True, False]
    )

    losses = compute_loss(
        outputs,
        labels,
        masks,
        mask_valid=mask_valid,
        lambda_mask=1.0,
        lambda_patch=0.1,
    )

    print({
        k: float(v)
        for k, v in losses.items()
    })

    losses["loss"].backward()
    print("Backward test: OK")
