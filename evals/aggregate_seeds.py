"""여러 시드 run의 결과를 mean ± std로 집계 (NFR-4: 재현성 보고).

사용 예
  python3 aggregate_seeds.py results/pg_topk_s0__original__detection.csv \
                             results/pg_topk_s1__original__detection.csv \
                             results/pg_topk_s2__original__detection.csv \
                             --out results/pg_topk__3seeds.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="같은 모델·다른 시드의 결과 CSV들")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    num_cols = [c for c in ("auroc", "auprc", "f1", "dice", "iou", "pixel_auprc")
                if c in df.columns]
    agg = df.groupby("scope")[num_cols].agg(["mean", "std"]).round(4)

    print(f"{len(args.files)}개 run 집계 (mean ± std) — NFR-4: 시드 간 std 0.03 이하 목표")
    print(agg.to_string())
    if args.out:
        agg.to_csv(args.out)
        print(f"[info] saved to {args.out}")


if __name__ == "__main__":
    main()
