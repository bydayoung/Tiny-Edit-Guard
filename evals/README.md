# evals — 평가 파이프라인

> **원칙: 팀 코드는 수정하지 않는다.** 이 폴더는 팀 모델/데이터 코드를 import만 한다.
> 구조: **추론 1회(predict) → 채점 N회(evaluate) → 시드 집계(aggregate)** + MLflow 실험 추적

## 전체 흐름

```
trainer.py (학습)             evals/ (평가)
best.pt ──────► predict.py ──► preds/<run>/ ──► evaluate.py ──► results/*.csv + MLflow
                 (GPU 1회)      probs.csv         (CPU, 몇 초)        │
                                masks/*.png                           ▼
                                run_meta.json                  aggregate_seeds.py
                                (latency/메모리)                (3시드 mean±std)
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `predict.py` | best.pt 로드 → test/val셋 추론 → 예측 저장 + **latency/peak GPU memory 측정(NFR-3)**. 5개 모델 변형 지원: `global`, `global_mean`, `segformer`, `patchguard_avg`, `patchguard_topk` |
| `evaluate.py` | 전체/edit_size_bin별/method별 Detection(AUROC·AUPRC·F1 + bootstrap CI) + Localization(Dice·IoU·Pixel AUPRC). `--mlflow`로 실험 추적 |
| `metrics.py` | 지표 단일 모듈 — trainer validation과 공유 권장 |
| `aggregate_seeds.py` | 3시드 mean±std 집계 (NFR-4) |

## 예측 파일 계약 (모델 ↔ 평가 인터페이스)

```
preds/<run_name>/
├── probs.csv            # id, edit_probability (0~1)
├── masks/<safe_id>.png  # 0~255 확률맵 (cls만 출력하는 global 계열은 생성 안 함)
└── run_meta.json        # latency, peak memory, device 등 (predict.py가 자동 기록)
```

파일명은 `make_safe_id()` 규칙(= `data/14_build_dataloader.py`와 동일) 적용.

## 사용법 (레포 루트에서, GPU 환경)

```bash
# 1) 추론 1회 — best.pt에서 예측 내보내기
python3 evals/predict.py --model patchguard_topk \
    --checkpoint runs/pg_topk/best.pt \
    --metadata data/final_split_metadata.csv \
    --processed-root data/processed \
    --out preds/pg_topk_s0

# 2) 채점 (CPU, 몇 초) + MLflow 기록 — 기본은 val 채점(모니터링 모드)
python3 evals/evaluate.py --pred-dir preds/pg_topk_s0 --mlflow

# 최종(모델 동결 후 1회)에만 test 개봉
python3 evals/evaluate.py --pred-dir preds/pg_topk_s0 --eval-split test --mlflow

# robustness 조건은 tag로 구분
python3 evals/evaluate.py --pred-dir preds/pg_topk_s0_jpeg75 --tag jpeg_q75

# 3) 3시드 집계
python3 evals/aggregate_seeds.py results/pg_topk_s{0,1,2}__original__test__detection.csv
```

MLflow UI: `pip install mlflow && mlflow ui` → http://localhost:5000 (experiment: `tinyedit-guard-eval`)

## 평가 프로토콜 (변경 시 팀 합의 필요)

1. threshold는 **validation에서 F1 최대화로 선정** → test에 적용만
2. 구간별 AUROC = **해당 구간 fake vs 전체 real**
3. Localization은 **label==1 & valid_mask==True** 만, **per-image 계산 후 평균**
4. 구간별 결과에 **부트스트랩 신뢰구간** 동반 (샘플 적은 구간 과신 방지)
5. **Test 개봉 정책: 모델 동결 후 1회** — evaluate 기본값이 val 채점(모니터링 모드)이라 실수로 test를 볼 수 없음. 최종에만 `--eval-split test` 명시. 중간발표는 validation 기준 숫자로 보고 (슬라이드에 명시)
6. 보고서·발표의 최종 숫자는 이 파이프라인 출력만 사용 (trainer 내부 지표는 학습 모니터링용)

## trainer.py와의 관계

- trainer의 MetricTracker(acc/F1@0.5/AUROC/pixel f1·iou): **epoch 모니터링용**
- 본 파이프라인(+AUPRC·Pixel AUPRC·Dice, val 기준 threshold, per-image 평균, 구간·method 분해, CI): **최종 보고용**
- 연결: 학습 종료 후 `predict.py` 1회 실행(또는 trainer 말미에 호출 추가)이면 끝

## TODO

- [ ] JPEG/Resize 변형 test 이미지 생성 스크립트
- [ ] 돈 그래프(구간별 AUROC 곡선 + CI 음영) 플롯 스크립트
- [ ] smoke 테스트셋 50장 고정 + CI 연동
