import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 설정
# ============================================================

META_PATH = "./data/final_split_metadata.csv"
PROCESSED_ROOT = "./data/processed"

BATCH_SIZE = 8

# ImageNet pretrained backbone 기준
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============================================================
# CSV의 bool 값을 안전하게 변환
# ============================================================

def to_bool(value):

    if isinstance(value, bool):
        return value

    return str(value).lower() == "true"


# ============================================================
# 파일명 변환
#
# 13번 preprocessing 코드에서 사용했던 방식과 동일
# ============================================================

def make_safe_id(sample_id):

    sample_id = str(sample_id)

    return (
        sample_id
        .replace(".jpg", "")
        .replace(".jpeg", "")
        .replace(".png", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ============================================================
# Image Transform
# ============================================================

image_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])


# ============================================================
# Dataset
# ============================================================

class TinyEditDataset(Dataset):

    def __init__(
        self,
        metadata,
        processed_root,
        segmentation_only=False
    ):

        self.df = metadata.copy()
        self.processed_root = processed_root

        # ----------------------------------------------------
        # Segmentation 학습에서는
        # 정상적인 mask가 있는 데이터만 사용
        # ----------------------------------------------------

        if segmentation_only:

            self.df["valid_mask"] = (
                self.df["valid_mask"]
                .apply(to_bool)
            )

            self.df = self.df[
                self.df["valid_mask"] == True
            ].reset_index(drop=True)

        else:

            self.df = self.df.reset_index(
                drop=True
            )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        split = row["split"]

        sample_id = make_safe_id(
            row["id"]
        )

        image_path = os.path.join(
            self.processed_root,
            split,
            "images",
            f"{sample_id}.jpg"
        )

        mask_path = os.path.join(
            self.processed_root,
            split,
            "masks",
            f"{sample_id}.png"
        )

        # ----------------------------------------------------
        # 파일 존재 확인
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not os.path.exists(mask_path):

            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image_transform(
            image
        )

        # 결과:
        # [3, 512, 512]


        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.array(
            mask,
            dtype=np.uint8
        )

        mask = (
            mask >= 128
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # 결과:
        # [1, 512, 512]


        # ----------------------------------------------------
        # Label
        #
        # 0 = Real
        # 1 = Fake
        # ----------------------------------------------------

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Mask 사용 가능 여부
        # ----------------------------------------------------

        valid_mask = torch.tensor(
            to_bool(row["valid_mask"]),
            dtype=torch.bool
        )


        # ----------------------------------------------------
        # 결과 반환
        # ----------------------------------------------------

        return {

            "image": image,

            "mask": mask,

            "label": label,

            "valid_mask": valid_mask,

            "id": str(row["id"]),

            "dataset": str(row["dataset"]),

            "method": str(row["method"]),

            "edit_size_bin":
                str(row["edit_size_bin"]),

            "actual_mask_ratio":
                float(row["actual_mask_ratio"])
        }


# ============================================================
# Metadata 불러오기
# ============================================================

metadata = pd.read_csv(
    META_PATH
)

train_meta = metadata[
    metadata["split"] == "train"
].copy()

val_meta = metadata[
    metadata["split"] == "val"
].copy()

test_meta = metadata[
    metadata["split"] == "test"
].copy()


# ============================================================
# Classification Dataset
#
# valid_mask=False도 사용
# ============================================================

train_dataset = TinyEditDataset(
    train_meta,
    PROCESSED_ROOT,
    segmentation_only=False
)

val_dataset = TinyEditDataset(
    val_meta,
    PROCESSED_ROOT,
    segmentation_only=False
)

test_dataset = TinyEditDataset(
    test_meta,
    PROCESSED_ROOT,
    segmentation_only=False
)


# ============================================================
# Segmentation Dataset
#
# valid_mask=True만 사용
# ============================================================

train_seg_dataset = TinyEditDataset(
    train_meta,
    PROCESSED_ROOT,
    segmentation_only=True
)

val_seg_dataset = TinyEditDataset(
    val_meta,
    PROCESSED_ROOT,
    segmentation_only=True
)

test_seg_dataset = TinyEditDataset(
    test_meta,
    PROCESSED_ROOT,
    segmentation_only=True
)


# ============================================================
# DataLoader
#
# Windows 환경이므로 num_workers=0부터 시작
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# Segmentation용

train_seg_loader = DataLoader(
    train_seg_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_seg_loader = DataLoader(
    val_seg_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_seg_loader = DataLoader(
    test_seg_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 간단한 확인
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATASET SIZE")
    print("=" * 60)

    print("\nClassification")

    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Val:",
        len(val_dataset)
    )

    print(
        "Test:",
        len(test_dataset)
    )


    print("\nSegmentation")

    print(
        "Train:",
        len(train_seg_dataset)
    )

    print(
        "Val:",
        len(val_seg_dataset)
    )

    print(
        "Test:",
        len(test_seg_dataset)
    )


    # --------------------------------------------------------
    # 실제 batch 하나 확인
    # --------------------------------------------------------

    batch = next(
        iter(train_loader)
    )

    print("\n" + "=" * 60)
    print("FIRST BATCH")
    print("=" * 60)

    print(
        "Image shape:",
        batch["image"].shape
    )

    print(
        "Mask shape:",
        batch["mask"].shape
    )

    print(
        "Label shape:",
        batch["label"].shape
    )

    print(
        "Labels:",
        batch["label"]
    )

    print(
        "Valid mask:",
        batch["valid_mask"]
    )