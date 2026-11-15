"""평가 러너: 저장된 예측을 채점한다 (GPU 불필요, 몇 초).

입력 두 가지 모드
  A) --trainer-csv : trainer.predict()가 만든 per-sample CSV를 직접 읽는다 (권장)
       한 파일에 edit_probability/label/method/edit_size_bin/sample_dice/sample_iou가 다 들어 있음.
       threshold 선정을 위해 val CSV는 --val-csv로 함께 준다.
       Pixel AUPRC는 불가(trainer가 mask를 저장 안 함) — Dice/IoU는 trainer가 계산한 값 사용.
  B) --pred-dir    : evals/predict.py가 만든 probs.csv + masks/ 를 읽는다.
       mask PNG가 있어 Pixel AUPRC까지 계산, run_meta.json의 latency/메모리도 함께 보고.

공통 규칙
  - threshold는 validation에서 선정해 채점 대상에 적용 (test에서 선정 금지)
  - 구간별/method별 AUROC는 "해당 부분집합 fake vs 전체 real"
  - Localization은 label==1 & valid_mask==True 에서만, per-image 평균
  - 모든 detection 결과에 부트스트랩 신뢰구간 동반
  - Test 개봉은 --eval-split test 를 명시할 때만 (동결 후 1회 정책)

사용 예 (레포 루트에서)
  # A) trainer CSV 모드 — 개발 중 val 채점
  python3 evals/evaluate.py --trainer-csv preds/pg_topk_val.csv            # MLflow 자동 기록(기본)
  # A) 최종 test 채점 (val CSV로 threshold 선정)
  python3 evals/evaluate.py --trainer-csv preds/pg_topk_test.csv --val-csv preds/pg_topk_val.csv --eval-split test
  python3 evals/evaluate.py --trainer-csv preds/pg_topk_val.csv --no-mlflow   # 기록 끄기
  # B) predict.py 산출물 모드 (Pixel AUPRC + latency 포함)
  python3 evals/evaluate.py --pred-dir preds/pg_topk_s0
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import metrics as M

BASE = Path(__file__).parent          # evals/
REPO = BASE.parent                    # 레포 루트


def make_safe_id(sample_id):
    """파일명 규칙 — data/14_build_dataloader.py의 make_safe_id와 동일해야 한다."""
    return (
        str(sample_id)
        .replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        .replace("/", "_").replace("\\", "_")
    )


def to_bool(series):
    """valid_mask가 bool/문자열/0·1 어떤 형태로 와도 일관되게 bool로."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_mask(path):
    return np.asarray(Image.open(path), dtype=np.float64) / 255.0


def add_detection_row(results, scope, labels, probs, threshold, note=""):
    lo, hi = M.bootstrap_auroc_ci(labels, probs)
    results.append(dict(
        scope=scope,
        n_fake=int(np.sum(np.asarray(labels) == 1)),
        n_real=int(np.sum(np.asarray(labels) == 0)),
        auroc=round(M.image_auroc(labels, probs), 4),
        auroc_ci_low=round(lo, 4), auroc_ci_high=round(hi, 4),
        auprc=round(M.image_auprc(labels, probs), 4),
        f1=round(M.image_f1(labels, probs, threshold), 4),
        note=note,
    ))


def score_detection(test, threshold, split_name):
    """Detection: 전체 + edit_size_bin별 + method별 (부분집합 fake vs 전체 real)."""
    results = []
    add_detection_row(results, f"{split_name}_overall", test.label, test.edit_probability, threshold)
    reals = test[test.label == 0]
    fakes = test[test.label == 1]
    if "edit_size_bin" in test.columns:
        for b in sorted(fakes.edit_size_bin.dropna().unique()):
            sub = pd.concat([fakes[fakes.edit_size_bin == b], reals])
            add_detection_row(results, f"bin:{b}", sub.label, sub.edit_probability, threshold,
                              note="bin fake vs all real")
    if "method" in test.columns:
        for m in sorted(fakes.method.dropna().unique()):
            sub = pd.concat([fakes[fakes.method == m], reals])
            add_detection_row(results, f"method:{m}", sub.label, sub.edit_probability, threshold,
                              note="method fake vs all real")
    return results


def loc_row(scope, vals):
    arr = np.array(vals, dtype=float)
    pap = arr[:, 2]
    has_pap = arr.shape[1] > 2 and not np.all(np.isnan(pap))
    return dict(scope=scope, n=len(vals),
                dice=round(float(np.nanmean(arr[:, 0])), 4),
                iou=round(float(np.nanmean(arr[:, 1])), 4),
                pixel_auprc=round(float(np.nanmean(pap)), 4) if has_pap else float("nan"))


def loc_from_csv(fakes):
    """trainer CSV 모드: sample_dice/sample_iou(이미 계산됨)를 bin별 평균. Pixel AUPRC는 없음."""
    if "sample_dice" not in fakes.columns:
        return []  # 분류 전용 모델(Global 등) — localization 출력 없음
    valid = fakes[to_bool(fakes.valid_mask)] if "valid_mask" in fakes.columns else fakes
    valid = valid.dropna(subset=["sample_dice"])
    if not len(valid):
        return []
    rows = [loc_row("loc_overall",
                    list(zip(valid.sample_dice, valid.sample_iou, [np.nan] * len(valid))))]
    if "edit_size_bin" in valid.columns:
        for b, g in valid.groupby("edit_size_bin"):
            rows.append(loc_row(f"loc_bin:{b}",
                                list(zip(g.sample_dice, g.sample_iou, [np.nan] * len(g)))))
    return rows


def loc_from_masks(fakes, gt_mask_dir, mask_dir):
    """predict.py 모드: mask PNG에서 Dice/IoU/Pixel AUPRC 직접 계산."""
    valid = fakes[to_bool(fakes.valid_mask)] if "valid_mask" in fakes.columns else fakes
    per_bin, overall, missing = {}, [], 0
    for _, r in valid.iterrows():
        sid = make_safe_id(r["id"])
        gt_p, pr_p = Path(gt_mask_dir) / f"{sid}.png", Path(mask_dir) / f"{sid}.png"
        if not (gt_p.exists() and pr_p.exists()):
            missing += 1
            continue
        gt = load_mask(gt_p) >= 0.5
        pm = load_mask(pr_p)
        d, i = M.dice_iou(pm >= 0.5, gt)
        p = M.pixel_auprc(pm, gt)
        overall.append((d, i, p))
        per_bin.setdefault(r.get("edit_size_bin", "all"), []).append((d, i, p))
    if missing:
        print(f"[warn] mask 파일을 못 찾은 localization 대상 {missing}개 제외")
    rows = [loc_row("loc_overall", overall)] if overall else []
    for b, vals in sorted(per_bin.items()):
        rows.append(loc_row(f"loc_bin:{b}", vals))
    return rows


def sanitize(name):
    """MLflow metric 이름 규칙에 맞게 특수문자 치환."""
    return re.sub(r"[^0-9A-Za-z_\-\./ ]", "_", name)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trainer-csv", help="trainer.predict()가 만든 채점 대상 CSV")
    src.add_argument("--pred-dir", help="predict.py 산출물 폴더 (probs.csv + masks/)")
    ap.add_argument("--val-csv", help="[trainer-csv 모드] threshold 선정용 val CSV")
    ap.add_argument("--metadata", default=str(REPO / "data/final_split_metadata.csv"),
                    help="[pred-dir 모드] id로 라벨/구간 정보를 조인할 메타데이터")
    ap.add_argument("--eval-split", choices=["val", "test"], default="val",
                    help="채점 대상 라벨. 개발 중 val(기본), test는 모델 동결 후 최종 1회만")
    ap.add_argument("--gt-mask-dir", default=None,
                    help="[pred-dir 모드] 미지정 시 data/processed/<eval-split>/masks")
    ap.add_argument("--run-name", default=None, help="결과 파일/기록에 쓸 이름 (미지정 시 입력에서 추론)")
    ap.add_argument("--tag", default="original",
                    help="평가 조건 (original / jpeg_q75 / resize_0.5 / unseen_glide 등)")
    ap.add_argument("--out-dir", default=str(BASE / "results"))
    ap.add_argument("--no-mlflow", dest="mlflow", action="store_false",
                    help="MLflow 기록 끄기 (기본은 항상 기록)")
    ap.set_defaults(mlflow=True)
    args = ap.parse_args()

    run_meta = {}

    # ---------------- 입력 모드 분기 ----------------
    if args.trainer_csv:
        test = pd.read_csv(args.trainer_csv)
        run_name = args.run_name or Path(args.trainer_csv).stem
        if args.val_csv:
            val = pd.read_csv(args.val_csv)
            threshold = M.select_threshold(val.label, val.edit_probability)
        else:
            threshold = 0.5
            print("[warn] --val-csv 미지정 → threshold=0.5. 최종 보고에는 val CSV로 threshold 선정 권장")
        loc_rows = loc_from_csv(test[test.label == 1])
        loc_note = "Dice/IoU는 trainer가 threshold 0.5에서 계산한 값 (Pixel AUPRC 없음)"
    else:  # pred-dir 모드
        pred_dir = Path(args.pred_dir)
        run_name = args.run_name or pred_dir.name
        gt_dir = args.gt_mask_dir or str(REPO / f"data/processed/{args.eval_split}/masks")
        meta = pd.read_csv(args.metadata)
        probs = pd.read_csv(pred_dir / "probs.csv")
        df = meta.merge(probs, on="id", how="inner")
        if len(df) < len(probs):
            print(f"[warn] metadata에 없는 예측 {len(probs) - len(df)}개 제외")
        val = df[df.split == "val"]
        test = df[df.split == args.eval_split]
        threshold = M.select_threshold(val.label, val.edit_probability) if len(val) else 0.5
        if not len(val):
            print("[warn] val 예측이 없어 threshold=0.5 사용")
        loc_rows = loc_from_masks(test[test.label == 1], gt_dir, pred_dir / "masks")
        loc_note = "Pixel 확률맵에서 직접 계산 (Pixel AUPRC 포함)"
        meta_path = pred_dir / "run_meta.json"
        if meta_path.exists():
            run_meta = json.loads(meta_path.read_text())

    # ---------------- 공통: 경고/안내 ----------------
    if args.eval_split == "test":
        print("⚠️  TEST 개봉 모드 — 정책: 모델 동결 후 1회만. 이 결과를 보고 모델을 수정하지 않는다.")
    else:
        print("[info] 모니터링 모드(val 채점) — threshold가 같은 val에서 선정되어 수치가 약간 낙관적일 수 있음")
    print(f"[info] run={run_name} tag={args.tag} | threshold={threshold:.3f} | n={len(test)}")

    results = score_detection(test, threshold, args.eval_split)

    # ---------------- 저장 ----------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = pd.DataFrame(results)
    det.insert(0, "tag", args.tag); det.insert(0, "run", run_name)
    det_path = out_dir / f"{run_name}__{args.tag}__{args.eval_split}__detection.csv"
    det.to_csv(det_path, index=False)
    loc = pd.DataFrame(loc_rows)
    loc_path = None
    if len(loc):
        loc.insert(0, "tag", args.tag); loc.insert(0, "run", run_name)
        loc_path = out_dir / f"{run_name}__{args.tag}__{args.eval_split}__localization.csv"
        loc.to_csv(loc_path, index=False)

    # ---------------- 출력 ----------------
    print("\n=== Detection (image-level) ===")
    print(det.drop(columns=["run", "tag"]).to_string(index=False))
    if len(loc):
        print(f"\n=== Localization (fake & valid_mask only) — {loc_note} ===")
        print(loc.drop(columns=["run", "tag"]).to_string(index=False))
    if run_meta:
        print("\n=== 운영 지표 (run_meta.json) ===")
        print(f"latency {run_meta.get('latency_ms_per_image')} ms/img | "
              f"throughput {run_meta.get('throughput_img_per_s')} img/s | "
              f"peak GPU {run_meta.get('peak_gpu_mem_gb')} GB | device {run_meta.get('device')}")
    print(f"\n[info] saved to {out_dir}/")

    # ---------------- MLflow ----------------
    if args.mlflow:
        try:
            import mlflow
        except ImportError:
            print("[info] mlflow 미설치 — 기록만 건너뜀 (결과 CSV는 정상 저장됨). "
                  "기록하려면 `pip install mlflow`")
            return
        mlflow.set_experiment("tinyedit-guard-eval")
        with mlflow.start_run(run_name=f"{run_name}__{args.tag}__{args.eval_split}"):
            mlflow.log_params({
                "run": run_name, "tag": args.tag, "eval_split": args.eval_split,
                "threshold": threshold, "n_eval": len(test),
                "input_mode": "trainer_csv" if args.trainer_csv else "pred_dir",
                **{f"meta_{k}": v for k, v in run_meta.items() if v is not None},
            })
            for r in results:
                s = sanitize(r["scope"])
                for k in ("auroc", "auprc", "f1"):
                    if not np.isnan(r[k]):
                        mlflow.log_metric(f"{k}__{s}", r[k])
            for r in loc_rows:
                s = sanitize(r["scope"])
                for k in ("dice", "iou", "pixel_auprc"):
                    if not np.isnan(r[k]):
                        mlflow.log_metric(f"{k}__{s}", r[k])
            mlflow.log_artifact(str(det_path))
            if loc_path:
                mlflow.log_artifact(str(loc_path))
        print("[info] MLflow 기록 완료 — `mlflow ui`로 확인")


if __name__ == "__main__":
    main()
