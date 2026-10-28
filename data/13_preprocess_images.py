from datasets import load_dataset
from PIL import Image
import numpy as np
import pandas as pd
import os

CACHE_DIR = "./data/hf_cache"
META_PATH = "./data/final_split_metadata.csv"

OUTPUT_ROOT = "./data/processed"

TARGET_SIZE = (512, 512)

dataset = load_dataset(
    "KhangTruong/Tampered-Images-Combined-Dataset",
    cache_dir=CACHE_DIR,
    verification_mode="no_checks"
)

ds = dataset["cross_test"]

meta = pd.read_csv(META_PATH)


# ============================================================
# 폴더 생성
# ============================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        f"{OUTPUT_ROOT}/{split}/images",
        exist_ok=True
    )

    os.makedirs(
        f"{OUTPUT_ROOT}/{split}/masks",
        exist_ok=True
    )


# ============================================================
# 전처리
# ============================================================

for i, row in meta.iterrows():

    split = row["split"]
    row_index = int(row["row_index"])

    sample = ds[row_index]

    image = sample["image"].convert("RGB")
    mask = sample["mask"].convert("L")

    # --------------------------------------------------------
    # Image resize
    # --------------------------------------------------------

    image = image.resize(
        TARGET_SIZE,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Mask resize
    #
    # 반드시 NEAREST 사용
    # --------------------------------------------------------

    mask = mask.resize(
        TARGET_SIZE,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(mask)

    # --------------------------------------------------------
    # Binary mask
    #
    # 0   = 정상 영역
    # 255 = 편집 영역
    # --------------------------------------------------------

    binary_mask = (
        mask_array >= 128
    ).astype(np.uint8) * 255

    binary_mask = Image.fromarray(
        binary_mask
    )

    # --------------------------------------------------------
    # 파일명
    # --------------------------------------------------------

    # id 안에 .jpg 등이 있을 수 있으므로 제거
    sample_id = str(row["id"])

    safe_id = (
        sample_id
        .replace(".jpg", "")
        .replace(".jpeg", "")
        .replace(".png", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    image_path = (
        f"{OUTPUT_ROOT}/{split}/images/"
        f"{safe_id}.jpg"
    )

    mask_path = (
        f"{OUTPUT_ROOT}/{split}/masks/"
        f"{safe_id}.png"
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    image.save(
        image_path,
        quality=95
    )

    binary_mask.save(
        mask_path
    )

    if (i + 1) % 500 == 0:

        print(
            f"{i + 1} / {len(meta)} processed"
        )


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Total:", len(meta))
print("Target size:", TARGET_SIZE)

print("\nSaved to:")
print(OUTPUT_ROOT)