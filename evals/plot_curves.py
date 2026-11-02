"""돈 그래프: edit_size_bin별 AUROC 곡선 + 신뢰구간 음영 (모델 비교).

evaluate.py가 만든 detection CSV 여러 개를 받아, 모델별로 한 곡선씩 겹쳐 그린다.
"작아질수록 Global과 Patch-Aware의 격차가 벌어지는가"가 한 장에 보이게 하는 게 목적.

사용 예 (레포 루트에서)
  python3 evals/plot_curves.py \
      results/global__original__test__detection.csv \
      results/pg_avg__original__test__detection.csv \
      results/pg_topk__original__test__detection.csv \
      --out evals/results/curve_test.png

범례 이름은 CSV의 run 컬럼을 쓰며, --labels로 바꿀 수 있다.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# 편집 크기 순서 (작은 → 큰). 팀 binning에 맞춰 조정.
BIN_ORDER = ["tiny", "small", "medium", "large"]


def bin_label(scope):
    return scope.split("bin:", 1)[1] if scope.startswith("bin:") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="evaluate.py가 만든 detection CSV들")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc", "f1"])
    ap.add_argument("--labels", nargs="*", default=None, help="범례 이름 (CSV 순서대로)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="evals/results/curve.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = {b: i for i, b in enumerate(BIN_ORDER)}

    for idx, csv in enumerate(args.csvs):
        df = pd.read_csv(csv)
        df["bin"] = df["scope"].map(bin_label)
        df = df[df["bin"].isin(BIN_ORDER)].copy()
        df["x"] = df["bin"].map(xpos)
        df = df.sort_values("x")
        if df.empty:
            print(f"[warn] {csv}: bin 행이 없어 건너뜀")
            continue

        name = (args.labels[idx] if args.labels and idx < len(args.labels)
                else str(df["run"].iloc[0]) if "run" in df.columns else Path(csv).stem)
        line, = ax.plot(df["x"], df[args.metric], marker="o", label=name)
        # 신뢰구간 음영 (auroc일 때만 CI 컬럼 존재)
        if args.metric == "auroc" and {"auroc_ci_low", "auroc_ci_high"} <= set(df.columns):
            ax.fill_between(df["x"], df["auroc_ci_low"], df["auroc_ci_high"],
                            color=line.get_color(), alpha=0.15)
        # 각 점에 n 표기 (샘플 수 — 구간별 신뢰도 투명하게)
        if "n_fake" in df.columns:
            for _, r in df.iterrows():
                ax.annotate(f"n={int(r['n_fake'])}", (r["x"], r[args.metric]),
                            textcoords="offset points", xytext=(0, -12),
                            fontsize=7, ha="center", color="gray")

    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels([f"{b}\n(작음→큼)" if b == "tiny" else b for b in BIN_ORDER])
    ax.set_xlabel("편집 영역 크기 (edit_size_bin)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(args.title or f"편집 크기별 {args.metric.upper()} — 모델 비교 (음영=95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[info] saved → {out}")


if __name__ == "__main__":
    main()
