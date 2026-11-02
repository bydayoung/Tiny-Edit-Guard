"""학습된 체크포인트 → 샘플별 예측 내보내기 (평가 계약 포맷).

팀 코드는 한 줄도 수정하지 않고 import만 한다. 레포 루트 기준 실행 예:

  python3 tinyedit_eval/predict.py \
      --model patchguard_topk \
      --checkpoint runs/pg_topk/best.pt \
      --metadata data/final_split_metadata.csv \
      --processed-root data/processed \
      --out preds/pg_topk_s0

출력 (계약 포맷 — evaluate.py의 입력):
  preds/<name>/probs.csv        컬럼: id, edit_probability (0~1)
  preds/<name>/masks/<id>.png   0~255 확률맵 (mask_logits 없는 모델은 생성 안 함)
  preds/<name>/run_meta.json    latency / peak GPU memory / 환경 기록 (NFR-3 근거)

GPU 환경(Colab 등)에서 실행. 로컬에는 torch가 없어도 evaluate.py는 독립 동작한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# data/14_build_dataloader.py 와 동일한 상수/규칙 (전처리 일치 필수)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def make_safe_id(sample_id):
    """파일명 규칙 — data/14_build_dataloader.py의 make_safe_id와 동일해야 한다."""
    return (
        str(sample_id)
        .replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        .replace("/", "_").replace("\\", "_")
    )


class TestImages(Dataset):
    """전처리 완료된 512×512 이미지 폴더를 읽는 추론 전용 경량 로더."""

    def __init__(self, meta: pd.DataFrame, processed_root):
        self.meta = meta.reset_index(drop=True)
        self.root = Path(processed_root)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.root / row["split"] / "images" / f"{make_safe_id(row['id'])}.jpg"
        img = Image.open(path).convert("RGB")
        return self.tf(img), str(row["id"])


def build_model(kind: str, repo_root: Path):
    """팀 모델을 그대로 구성한다 (클래스 시그니처: 2026-10 doyeon/main 기준)."""
    sys.path.insert(0, str(repo_root))
    from models.backbone import DinoBackbone

    if kind == "global":
        from models.global_baseline import GlobalBaseline
        return GlobalBaseline(DinoBackbone())
    if kind == "global_mean":
        from models.global_baseline import GlobalBaseline
        return GlobalBaseline(DinoBackbone(), use_mean_patch=True)
    if kind == "segformer":
        from models.segformer_baseline import SegFormerBaseline
        return SegFormerBaseline()
    if kind in ("patchguard_avg", "patchguard_topk"):
        from patchguard import PatchGuard
        agg = "average" if kind.endswith("avg") else "topk"
        return PatchGuard(DinoBackbone(), aggregation=agg)
    raise ValueError(f"unknown model kind: {kind}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["global", "global_mean", "segformer",
                             "patchguard_avg", "patchguard_topk"])
    ap.add_argument("--checkpoint", required=True, help="trainer가 저장한 best.pt")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--split", default="val,test",
                    help="쉼표로 복수 지정 가능. threshold 선정에 val 예측이 필요하므로 기본은 val+test")
    ap.add_argument("--out", required=True, help="preds/<run_name>")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--limit", type=int, default=0, help="smoke 테스트용 샘플 제한 (0=전체)")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = build_model(args.model, Path(args.repo_root))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    meta = pd.read_csv(args.metadata)
    splits = [s.strip() for s in args.split.split(",")]
    meta = meta[meta.split.isin(splits)]
    if args.limit:
        meta = meta.head(args.limit)
    dl = DataLoader(TestImages(meta, args.processed_root),
                    batch_size=args.batch_size, num_workers=args.num_workers)

    out = Path(args.out)
    mask_dir = out / "masks"
    out.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, times, n_timed = [], [], 0
    wrote_mask = False
    for bi, (images, ids) in enumerate(dl):
        images = images.to(device)
        t0 = time.perf_counter()
        outputs = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if bi > 0:  # 첫 배치는 워밍업으로 간주해 latency에서 제외
            times.append(dt)
            n_timed += len(ids)

        probs = torch.sigmoid(outputs["cls_logits"]).float().cpu().numpy()
        masks = outputs.get("mask_logits")
        if masks is not None:
            masks = torch.sigmoid(masks).float().cpu().numpy()
        for j, sid in enumerate(ids):
            rows.append({"id": sid, "edit_probability": float(probs[j])})
            if masks is not None:
                arr = (np.clip(masks[j, 0], 0, 1) * 255).astype(np.uint8)
                Image.fromarray(arr).save(mask_dir / f"{make_safe_id(sid)}.png")
                wrote_mask = True

    pd.DataFrame(rows).to_csv(out / "probs.csv", index=False)

    run_meta = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(rows),
        "device": device,
        "batch_size": args.batch_size,
        "latency_ms_per_image": round(1000 * sum(times) / n_timed, 2) if n_timed else None,
        "throughput_img_per_s": round(n_timed / sum(times), 2) if times and sum(times) > 0 else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "torch_version": torch.__version__,
        "exports_mask": wrote_mask,
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(json.dumps(run_meta, indent=2))
    print(f"[info] saved → {out}/probs.csv" + (" + masks/" if wrote_mask else " (mask 미출력 모델)"))


if __name__ == "__main__":
    main()
