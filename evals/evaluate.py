"""평가 러너: 저장된 예측(probs.csv + masks/)을 metadata와 대조해 채점한다.

설계 원칙
  - 추론(비쌈)과 채점(쌈)의 분리: GPU 불필요, predict.py가 저장한 예측만 읽는다
  - threshold는 validation에서 선정해 test에 적용 (test에서 선정 금지)
  - 구간별 AUROC는 "해당 구간 fake vs 전체 real" 규칙
  - Localization 지표는 label==1 & valid_mask==True 에서만, per-image 계산 후 평균
  - 모든 구간 결과에 부트스트랩 신뢰구간 동반

사용 예 (레포 루트에서)
  python3 evals/evaluate.py --pred-dir preds/pg_topk_s0 --mlflow              # 개발 중: val 채점
  python3 evals/evaluate.py --pred-dir preds/pg_topk_s0 --eval-split test --mlflow   # 최종: 동결 후 1회
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


def sanitize(name):
    """MLflow metric 이름 규칙에 맞게 특수문자 치환."""
    return re.sub(r"[^0-9A-Za-z_\-\./ ]", "_", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="probs.csv와 masks/가 있는 폴더")
    ap.add_argument("--metadata", default=str(REPO / "data/final_split_metadata.csv"))
    ap.add_argument("--eval-split", choices=["val", "test"], default="val",
                    help="채점 대상. 개발 중에는 val(기본). test는 모델 동결 후 최종 1회만")
    ap.add_argument("--gt-mask-dir", default=None,
                    help="미지정 시 data/processed/<eval-split>/masks 자동 사용")
    ap.add_argument("--tag", default="original",
                    help="평가 조건 라벨 (original / jpeg_q75 / resize_0.5 / unseen_glide 등)")
    ap.add_argument("--out-dir", default=str(BASE / "results"))
    ap.add_argument("--mlflow", action="store_true", help="MLflow에 기록 (pip install mlflow)")
    args = ap.parse_args()

    if args.gt_mask_dir is None:
        args.gt_mask_dir = str(REPO / f"data/processed/{args.eval_split}/masks")
    pred_dir = Path(args.pred_dir)
    run_name = pred_dir.name
    meta = pd.read_csv(args.metadata)
    probs = pd.read_csv(pred_dir / "probs.csv")
    df = meta.merge(probs, on="id", how="inner")
    if len(df) < len(probs):
        print(f"[warn] metadata에 없는 예측 {len(probs) - len(df)}개 제외")

    # ---- threshold: validation에서만 선정 ----
    val = df[df.split == "val"]
    test = df[df.split == args.eval_split]   # 채점 대상 (기본 val = 모니터링 모드)
    threshold = M.select_threshold(val.label, val.edit_probability) if len(val) else 0.5
    if not len(val):
        print("[warn] val 예측이 없어 threshold=0.5 사용 — 최종 보고에는 val 포함 예측 필요")
    if args.eval_split == "test":
        print("⚠️  TEST 개봉 모드 — 정책: 모델 동결 후 1회만. 이 결과를 보고 모델을 수정하지 않는다.")
    else:
        print("[info] 모니터링 모드(val 채점) — threshold가 같은 val에서 선정되므로 수치가 약간 낙관적일 수 있음")
    print(f"[info] run={run_name} tag={args.tag} | threshold(val 선정)={threshold:.3f} "
          f"| {args.eval_split} n={len(test)}")

    results = []
    add_detection_row(results, f"{args.eval_split}_overall", test.label, test.edit_probability, threshold)

    # ---- 구간별 / method별: "fake 부분집합 vs 전체 real" ----
    reals = test[test.label == 0]
    fakes = test[test.label == 1]
    for b in sorted(fakes.edit_size_bin.dropna().unique()):
        sub_f = fakes[fakes.edit_size_bin == b]
        sub = pd.concat([sub_f, reals])
        add_detection_row(results, f"bin:{b}", sub.label, sub.edit_probability, threshold,
                          note="bin fake vs all real")
    for m in sorted(fakes.method.dropna().unique()):
        sub_f = fakes[fakes.method == m]
        sub = pd.concat([sub_f, reals])
        add_detection_row(results, f"method:{m}", sub.label, sub.edit_probability, threshold,
                          note="method fake vs all real")

    # ---- Localization: fake & valid_mask, per-image 평균 ----
    mask_dir = pred_dir / "masks"
    loc_targets = fakes[fakes.valid_mask == True]  # noqa: E712
    dices, ious, paps, per_bin = [], [], [], {}
    missing = 0
    for _, r in loc_targets.iterrows():
        sid = make_safe_id(r["id"])
        gt_p = Path(args.gt_mask_dir) / f"{sid}.png"
        pr_p = mask_dir / f"{sid}.png"
        if not (gt_p.exists() and pr_p.exists()):
            missing += 1
            continue
        gt = load_mask(gt_p) >= 0.5
        pm = load_mask(pr_p)
        d, i = M.dice_iou(pm >= 0.5, gt)
        p = M.pixel_auprc(pm, gt)
        dices.append(d); ious.append(i); paps.append(p)
        per_bin.setdefault(r["edit_size_bin"], []).append((d, i, p))
    if missing:
        print(f"[warn] mask 파일을 못 찾은 localization 대상 {missing}개 제외")

    def loc_row(scope, vals):
        arr = np.array(vals, dtype=float)
        return dict(scope=scope, n=len(vals),
                    dice=round(float(np.nanmean(arr[:, 0])), 4),
                    iou=round(float(np.nanmean(arr[:, 1])), 4),
                    pixel_auprc=round(float(np.nanmean(arr[:, 2])), 4))

    all_vals = list(zip(dices, ious, paps))
    loc_rows = [loc_row("loc_overall", all_vals)] if all_vals else []
    for b, vals in sorted(per_bin.items()):
        loc_rows.append(loc_row(f"loc_bin:{b}", vals))

    # ---- 운영 지표 (predict.py의 run_meta.json) ----
    run_meta = {}
    meta_path = pred_dir / "run_meta.json"
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text())

    # ---- 저장 ----
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

    # ---- 출력 ----
    print("\n=== Detection (image-level) ===")
    print(det.drop(columns=["run", "tag"]).to_string(index=False))
    if len(loc):
        print("\n=== Localization (pixel-level, fake & valid_mask only) ===")
        print(loc.drop(columns=["run", "tag"]).to_string(index=False))
    if run_meta:
        print("\n=== 운영 지표 (run_meta.json) ===")
        print(f"latency {run_meta.get('latency_ms_per_image')} ms/img | "
              f"throughput {run_meta.get('throughput_img_per_s')} img/s | "
              f"peak GPU {run_meta.get('peak_gpu_mem_gb')} GB | device {run_meta.get('device')}")
    print(f"\n[info] saved to {out_dir}/")

    # ---- MLflow (선택) ----
    if args.mlflow:
        try:
            import mlflow
        except ImportError:
            print("[warn] mlflow 미설치 — `pip install mlflow` 후 다시 실행")
            return
        mlflow.set_experiment("tinyedit-guard-eval")
        with mlflow.start_run(run_name=f"{run_name}__{args.tag}__{args.eval_split}"):
            mlflow.log_params({
                "run": run_name, "tag": args.tag, "eval_split": args.eval_split, "threshold": threshold,
                "n_eval": len(test), "metadata": Path(args.metadata).name,
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
