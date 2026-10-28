# TinyEdit-Guard Data

TinyEdit-Guard 프로젝트에서 사용하는 데이터 전처리 및 분할 정보를 정리한 폴더입니다.

## 1. Dataset
- Source: `KhangTruong/Tampered-Images-Combined-Dataset`
- Raw samples: **9,164**
- Real: **4,595**
- Edited: **4,569**
- Tasks:
  - Image-level Real / Edited classification
  - Pixel-level edited-region localization

원본 데이터는 GitHub에 포함하지 않으며, 전처리된 데이터셋은 Hugging Face에서 관리합니다.

- Hugging Face: `pdyoung/TinyEdit-Guard-Dataset`

## 2. Data Cleaning
원본 데이터의 mask와 metadata를 검증한 결과:
- Image / Mask size mismatch: 없음
- Real image with non-zero mask: 없음
- `TGIF2_flux1dev`의 Edited sample **1,010개는 GT mask가 비어 있음**
- 원본 `mask_ratio`와 실제 GT mask 기반 비율이 일치하지 않는 문제가 확인됨

따라서 localization에 사용할 수 없는 mask는 `valid_mask=False`로 표시하고,
편집 영역 비율은 GT mask에서 직접 다시 계산한 `actual_mask_ratio`를 사용합니다.

## 3. actual_mask_ratio
`actual_mask_ratio`는 GT mask에서 편집 영역으로 판단된 픽셀의 비율입니다.

```text
actual_mask_ratio = edited pixels / total pixels
```

Binary mask 생성 기준:

```python
binary_mask = mask_array >= 128
actual_mask_ratio = binary_mask.mean()
```

즉, mask 값이 128 이상인 픽셀을 edited pixel로 간주합니다.

## 4. Data Split
Train / Validation / Test 분할 시 동일한 source에서 생성된 이미지가 서로 다른 split에 들어가지 않도록
**source-level group split**을 적용했습니다.

이를 통해 동일하거나 매우 유사한 원본 이미지가 Train과 Test에 동시에 포함되는 data leakage를 방지합니다.

최종 검증 결과:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

## 5. Image Preprocessing
```text
RGB conversion
→ Resize to 512 × 512
→ ToTensor
→ ImageNet Normalization
```

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Final image tensor shape:

```text
[3, 512, 512]
```

## 6. Mask Preprocessing
```text
Grayscale conversion
→ Resize to 512 × 512 using Nearest Neighbor
→ Binary threshold (>= 128)
→ Tensor
```

Final mask tensor shape:

```text
[1, 512, 512]
```

Mask values:

```text
0 = normal region
1 = edited region
```

## 7. Metadata
`final_split_metadata.csv` contains the final metadata used for training and evaluation.

| Field | Description |
|---|---|
| `id` | Sample identifier |
| `label` | `0 = Real`, `1 = Edited` |
| `dataset` | Original dataset source |
| `method` | Editing / generation method |
| `actual_mask_ratio` | GT mask 기반 실제 edited pixel 비율 |
| `edit_size_bin` | 편집 영역 크기 구간 |
| `valid_mask` | Localization에 사용할 수 있는 GT mask 여부 |
| `final_group_id` | Source-level grouping identifier |
| `split` | `train`, `val`, or `test` |

## 8. Classification vs. Localization
Classification에서는 Real / Edited sample을 사용합니다.

Localization에서는 GT mask가 필요한 관계로:

```text
valid_mask == True
```

인 sample만 사용합니다.

특히 GT mask가 비어 있는 `TGIF2_flux1dev` Edited sample 1,010개는
classification에는 사용할 수 있지만 localization 학습 및 평가에서는 제외합니다.

## 9. Validation
최종 데이터 검증 결과:

```text
Group leakage: NONE
Classification loaders: OK
Segmentation loaders: OK
Mask values: binary 0/1
Labels: binary 0/1
NaN check: passed
```

## 10. Repository Policy
GitHub에는 전처리 코드와 metadata만 저장합니다.

대용량 image / mask 파일은 GitHub에 포함하지 않으며,
Hugging Face Dataset을 통해 공유합니다.
