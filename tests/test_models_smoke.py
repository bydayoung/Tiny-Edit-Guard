"""Offline smoke tests for models / losses / batching / trainer (no pretrained download).

Tiny randomly initialised DINOv3 and SegFormer configs stand in for the real backbones.

    python -m pytest tests/test_models_smoke.py -q      (or: python tests/test_models_smoke.py)
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from transformers import DINOv3ViTConfig, DINOv3ViTModel, SegformerConfig, SegformerForSemanticSegmentation  # noqa: E402

from data.collate import GroupIdDataset, PadCollate, PairBatchSampler, PairedAugment  # noqa: E402
from losses import TinyEditLoss  # noqa: E402
from models.factory import build_model  # noqa: E402
from models.patch_head import PatchHead, masked_topk_mean  # noqa: E402
import trainer as trainer_module  # noqa: E402


def tiny_dino():
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                          intermediate_size=64, num_register_tokens=4, patch_size=16)
    return DINOv3ViTModel(cfg)


def tiny_segformer():
    cfg = SegformerConfig(num_labels=1, hidden_sizes=[8, 16, 24, 32], decoder_hidden_size=16,
                          depths=[1, 1, 1, 1], num_attention_heads=[1, 1, 1, 1])
    return SegformerForSemanticSegmentation(cfg)


V3_CFG = dict(lora_rank=4, feature_layers="2,4", topk_ratio=0.05, unfreeze_last_n=0)


# ----------------------------------------------------------------------------- models

def test_backbone_lora_and_layers():
    m = build_model({"model": "patch_topk", **V3_CFG}, hf_model=tiny_dino())
    bb = m.backbone
    trainable = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable), trainable
    feats = bb(torch.randn(2, 3, 48, 80))
    assert feats["patch_tokens"].shape == (2, 15, 32)
    assert len(feats["layer_patch_tokens"]) == 2


def test_unfreeze_last_n_works_on_this_transformers_version():
    m = build_model({"model": "global_cls", "unfreeze_last_n": 1}, hf_model=tiny_dino())
    names = [n for n, p in m.backbone.named_parameters() if p.requires_grad]
    assert any("layer.3." in n for n in names) and not any("layer.0." in n for n in names)


def test_masked_topk_never_picks_padding():
    v = torch.tensor([[5.0, 1.0, 9.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    valid = torch.tensor([[True, True, False, True], [True, True, True, True]])
    mean, idx = masked_topk_mean(v, valid, ratio=0.5)
    assert torch.allclose(mean, torch.tensor([(5 + 2) / 2, (4 + 3) / 2]))
    assert 2 not in idx[0, :2].tolist()


def test_all_models_forward_with_padding():
    x = torch.randn(2, 3, 64, 96)
    pad = torch.ones(2, 1, 64, 96)
    pad[1, :, :, 48:] = 0
    for kind in ("global_cls", "global_cls_mean", "patch_avg", "patch_topk", "patch_soft"):
        out = build_model({"model": kind, **V3_CFG}, hf_model=tiny_dino())(x, pad_mask=pad)
        assert out["cls_logits"].shape == (2,), kind
        if "mask_logits" in out:
            assert out["mask_logits"].shape == (2, 1, 64, 96)
            assert out["patch_valid"][1].sum() == 12  # 4x3 real patches of 4x6
    seg = build_model({"model": "segformer"}, hf_model=tiny_segformer())(x, pad_mask=pad)
    assert seg["cls_logits"].shape == (2,)


def test_legacy_config_rebuilds_old_architecture():
    m = build_model({"model": "patch_topk"}, hf_model=tiny_dino())
    assert m.patch_head.topk_ratio == 0.10 and m.fusion is None and m.backbone.lora_rank == 0


def test_soft_topk_kappa_learnable():
    head = PatchHead(in_dim=8, aggregation="soft_topk")
    head(torch.randn(2, 10, 8))["cls_logits"].sum().backward()
    assert head.kappa_raw.grad is not None


# ----------------------------------------------------------------------------- loss

def test_loss_dice_only_on_fakes_and_padding_ignored():
    B, H, W = 2, 32, 32
    out = {"cls_logits": torch.zeros(B, requires_grad=True),
           "mask_logits": torch.zeros(B, 1, H, W, requires_grad=True),
           "patch_score_map": torch.zeros(B, 1, 2, 2, requires_grad=True)}
    batch = {"label": torch.tensor([0, 0]), "mask": torch.zeros(B, 1, H, W),
             "valid_mask": torch.tensor([True, True])}
    loss_fn = TinyEditLoss(patch_weight=0.2)
    _, logs_real = loss_fn(out, batch)
    # all-real batch: only BCE (=log 2 at logit 0), no Dice term of ≈1
    assert abs(logs_real["loss_mask"] - 0.6931) < 1e-3, logs_real
    batch["pad_mask"] = torch.zeros(B, 1, H, W)
    batch["pad_mask"][..., :16, :] = 1
    loss, _ = loss_fn(out, batch)
    loss.backward()


# ----------------------------------------------------------------------------- batching

class PairDS(Dataset):
    """2 groups x (real, fake), different image sizes per group."""

    def __init__(self):
        self.items = []
        for g, (h, w) in enumerate([(32, 48), (48, 32), (40, 40)]):
            for label in (0, 1):
                mask = torch.zeros(1, h, w)
                if label:
                    mask[:, 4:12, 4:12] = 1
                self.items.append({"image": torch.randn(3, h, w), "mask": mask,
                                   "label": torch.tensor(label), "valid_mask": torch.tensor(True),
                                   "id": f"g{g}_{label}", "dataset": "toy",
                                   "edit_size_bin": "tiny" if label else "real",
                                   "actual_mask_ratio": float(mask.mean())})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def toy_groups():
    return [f"g{i // 2}" for i in range(6)]


def test_pair_sampler_keeps_pairs_together():
    sampler = PairBatchSampler(toy_groups(), batch_size=4, shuffle=True, seed=1)
    for batch in sampler:
        groups = [toy_groups()[i] for i in batch]
        for g in set(groups):
            assert groups.count(g) == 2


def test_pad_collate_and_paired_flip():
    ds = GroupIdDataset(PairDS(), toy_groups())
    aug = PairedAugment(p_flip=1.0, p_jpeg=1.0, p_blur=0.0, seed=0)
    batch = PadCollate(augment=aug)([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape == (3, 3, 48, 48)
    assert batch["pad_mask"][0, 0, :32, :48].all() and batch["pad_mask"][0, 0, 32:].sum() == 0
    # fake of group 0 was flipped: its 8x8 edit moved to the right edge (w=48 -> cols 36..43)
    assert batch["mask"][1, 0, 4:12, 36:44].all()


# ----------------------------------------------------------------------------- trainer end-to-end

def test_trainer_fit_predict_and_pair_accuracy():
    import metrics as M

    ds = GroupIdDataset(PairDS(), toy_groups())
    train = DataLoader(ds, batch_sampler=PairBatchSampler(toy_groups(), 4),
                       collate_fn=PadCollate(augment=PairedAugment(seed=0)))
    val = DataLoader(ds, batch_size=1, collate_fn=PadCollate())
    model = build_model({"model": "patch_soft", **V3_CFG}, hf_model=tiny_dino())
    with tempfile.TemporaryDirectory() as tmp:
        tr = trainer_module.Trainer(model, train, val, loss_fn=TinyEditLoss(patch_weight=0.2),
                                    epochs=1, lora_lr=1e-3, amp=False, output_dir=tmp,
                                    log_every=0, device=torch.device("cpu"))
        metrics = tr.fit()
        assert "val_auroc" in metrics and "val_pixel_dice" in metrics
        paths = tr.export_predictions({"val": val})
        rows = list(csv.DictReader(open(paths["val"])))
        assert len(rows) == 6 and "group_id" in rows[0]
        acc, n = M.pair_accuracy([r["group_id"] for r in rows], [int(float(r["label"])) for r in rows],
                                 [float(r["edit_probability"]) for r in rows])
        assert n == 3 and 0.0 <= acc <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
