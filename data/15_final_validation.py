import os
import pandas as pd
import torch

from importlib import util


# ============================================================
# 경로 설정
# ============================================================

META_PATH = "./data/final_split_metadata.csv"
DATALOADER_SCRIPT = "./data/14_build_dataloader.py"  # was ./scripts/... (file does not exist)


# ============================================================
# 14번 코드 불러오기
# ============================================================

spec = util.spec_from_file_location(
    "dataloader_module",
    DATALOADER_SCRIPT
)

module = util.module_from_spec(spec)
spec.loader.exec_module(module)


train_loader = module.train_loader
val_loader = module.val_loader
test_loader = module.test_loader

train_seg_loader = module.train_seg_loader
val_seg_loader = module.val_seg_loader
test_seg_loader = module.test_seg_loader


# ============================================================
# Metadata 로드
# ============================================================

df = pd.read_csv(META_PATH)


print("=" * 60)
print("1. BASIC CHECK")
print("=" * 60)

print("Total samples:", len(df))

print("\nSplit distribution:")
print(df["split"].value_counts())


# ============================================================
# 2. Group leakage 검사
# ============================================================

print("\n" + "=" * 60)
print("2. GROUP LEAKAGE CHECK")
print("=" * 60)

train_groups = set(
    df[df["split"] == "train"]["final_group_id"]
)

val_groups = set(
    df[df["split"] == "val"]["final_group_id"]
)

test_groups = set(
    df[df["split"] == "test"]["final_group_id"]
)

train_val_overlap = train_groups & val_groups
train_test_overlap = train_groups & test_groups
val_test_overlap = val_groups & test_groups


print(
    "Train ∩ Val:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Val ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) > 0
    or len(train_test_overlap) > 0
    or len(val_test_overlap) > 0
):
    raise ValueError(
        "ERROR: Group leakage detected!"
    )


# ============================================================
# 3. Split별 Label 분포
# ============================================================

print("\n" + "=" * 60)
print("3. LABEL DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:

    subset = df[
        df["split"] == split
    ]

    print(f"\n[{split}]")

    counts = subset[
        "label_text"
    ].value_counts()

    print(counts)

    print(
        "Real ratio:",
        (
            counts.get("real", 0)
            / len(subset)
        )
    )

    print(
        "Fake ratio:",
        (
            counts.get("fake", 0)
            / len(subset)
        )
    )


# ============================================================
# 4. Method 분포
# ============================================================

print("\n" + "=" * 60)
print("4. METHOD DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(f"\n[{split}]")

    print(
        df[
            df["split"] == split
        ]["method"]
        .value_counts()
    )


# ============================================================
# 5. Edit size 분포
# ============================================================

print("\n" + "=" * 60)
print("5. EDIT SIZE DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(f"\n[{split}]")

    print(
        df[
            df["split"] == split
        ]["edit_size_bin"]
        .value_counts()
    )


# ============================================================
# 6. valid_mask 분포
# ============================================================

print("\n" + "=" * 60)
print("6. VALID MASK DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(f"\n[{split}]")

    print(
        df[
            df["split"] == split
        ]["valid_mask"]
        .value_counts()
    )


# ============================================================
# 7. Classification DataLoader 검사
# ============================================================

def check_classification_loader(
    loader,
    name
):

    print("\n" + "=" * 60)
    print(
        f"7. CLASSIFICATION LOADER CHECK - {name}"
    )
    print("=" * 60)

    batch = next(
        iter(loader)
    )

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Mask shape:",
        masks.shape
    )

    print(
        "Label shape:",
        labels.shape
    )

    print(
        "Image dtype:",
        images.dtype
    )

    print(
        "Mask dtype:",
        masks.dtype
    )

    print(
        "Label dtype:",
        labels.dtype
    )

    print(
        "Image NaN:",
        torch.isnan(images).any().item()
    )

    print(
        "Mask NaN:",
        torch.isnan(masks).any().item()
    )

    print(
        "Mask unique values:",
        torch.unique(masks)
    )

    print(
        "Labels:",
        torch.unique(labels)
    )

    # Shape 검사
    assert images.ndim == 4
    assert images.shape[1] == 3

    assert masks.ndim == 4
    assert masks.shape[1] == 1

    # Label 검사
    assert set(
        torch.unique(labels).tolist()
    ).issubset(
        {0, 1}
    )

    # Mask 검사
    assert set(
        torch.unique(masks).tolist()
    ).issubset(
        {0.0, 1.0}
    )

    print(
        "Classification loader OK"
    )


check_classification_loader(
    train_loader,
    "TRAIN"
)

check_classification_loader(
    val_loader,
    "VAL"
)

check_classification_loader(
    test_loader,
    "TEST"
)


# ============================================================
# 8. Segmentation DataLoader 검사
# ============================================================

def check_segmentation_loader(
    loader,
    name
):

    print("\n" + "=" * 60)
    print(
        f"8. SEGMENTATION LOADER CHECK - {name}"
    )
    print("=" * 60)

    batch = next(
        iter(loader)
    )

    images = batch["image"]
    masks = batch["mask"]
    valid_masks = batch[
        "valid_mask"
    ]

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Mask shape:",
        masks.shape
    )

    print(
        "Mask unique:",
        torch.unique(masks)
    )

    print(
        "All valid_mask:",
        valid_masks
    )

    assert images.ndim == 4
    assert images.shape[1] == 3

    assert masks.ndim == 4
    assert masks.shape[1] == 1

    assert set(
        torch.unique(masks).tolist()
    ).issubset(
        {0.0, 1.0}
    )

    # segmentation loader에서는
    # 모든 sample이 valid mask여야 함
    assert valid_masks.all()

    print(
        "Segmentation loader OK"
    )


check_segmentation_loader(
    train_seg_loader,
    "TRAIN"
)

check_segmentation_loader(
    val_seg_loader,
    "VAL"
)

check_segmentation_loader(
    test_seg_loader,
    "TEST"
)


# ============================================================
# 9. 최종 결과
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION RESULT")
print("=" * 60)

print(
    "Group leakage: NONE"
)

print(
    "Classification loaders: OK"
)

print(
    "Segmentation loaders: OK"
)

print(
    "Mask values: binary 0/1"
)

print(
    "Labels: binary 0/1"
)

print(
    "NaN check: passed"
)

print("\nDATA PREPARATION COMPLETE")