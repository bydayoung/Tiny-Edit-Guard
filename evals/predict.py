"""학습된 체크포인트 → 샘플별 예측 내보내기 (평가 계약 포맷).

모델은 models/factory.py (train.py 와 같은 함수)로 run_config.json 대로 구성한다. 실행 예:

  python3 evals/predict.py \
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
    """전처리 완료된 이미지 폴더를 읽는 추론 전용 경량 로더.

    V3 데이터는 PNG·원본 해상도(크기 제각각)이므로 .png → .jpg 순으로 찾고,
    배치는 data/collate.PadCollate 로 16 배수 패딩 + pad_mask 를 만든다.
    """

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
        base = self.root / row["split"] / "images" / make_safe_id(row["id"])
        path = next((base.with_suffix(ext) for ext in (".png", ".jpg") if base.with_suffix(ext).exists()),
                    base.with_suffix(".png"))
        img = Image.open(path).convert("RGB")
        return {"image": self.tf(img), "id": str(row["id"])}


def load_run_config(checkpoint: Path) -> dict:
    cfg_path = Path(checkpoint).parent / "run_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}


def build_model(kind: str, repo_root: Path, cfg: dict = None):
    """train.py 와 같은 함수(models/factory.build_model)로, 체크포인트의 run_config 대로 구성한다."""
    sys.path.insert(0, str(repo_root))
    from models.factory import build_model as factory_build

    return factory_build({**(cfg or {}), "model": kind})


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    choices=["global", "global_mean", "segformer",
                             "patchguard_avg", "patchguard_topk", "patchguard_soft"],
                    help="미지정 시 체크포인트 옆 run_config.json 의 model 사용")
    ap.add_argument("--checkpoint", required=True, help="trainer가 저장한 best.pt")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--split", default="val,test",
                    help="쉼표로 복수 지정 가능. threshold 선정에 val 예측이 필요하므로 기본은 val+test")
    ap.add_argument("--out", required=True, help="preds/<run_name>")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 = 이미지마다 원본 해상도 (크기가 다르면 큰 배치는 패딩이 늘어남)")
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

    cfg = load_run_config(Path(args.checkpoint))
    kind = args.model or cfg.get("model")
    if not kind:
        raise SystemExit("--model 을 지정하거나 체크포인트 옆에 run_config.json 을 두세요")
    model = build_model(kind, Path(args.repo_root), cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    meta = pd.read_csv(args.metadata)
    splits = [s.strip() for s in args.split.split(",")]
    meta = meta[meta.split.isin(splits)]
    if args.limit:
        meta = meta.head(args.limit)
    sys.path.insert(0, str(Path(args.repo_root)))
    from data.collate import PadCollate

    dl = DataLoader(TestImages(meta, args.processed_root), batch_size=args.batch_size,
                    num_workers=args.num_workers, collate_fn=PadCollate())

    out = Path(args.out)
    mask_dir = out / "masks"
    out.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, times, n_timed = [], [], 0
    wrote_mask = False
    for bi, batch in enumerate(dl):
        images, ids = batch["pixel_values"].to(device), batch["id"]
        pad_mask = batch["pad_mask"].to(device)
        t0 = time.perf_counter()
        outputs = model(images, pad_mask=pad_mask)
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
                region = batch["pad_mask"][j, 0].numpy() > 0.5  # 패딩 제거 → 원본 크기로 저장
                h, w = int(region.any(1).sum()), int(region.any(0).sum())
                arr = (np.clip(masks[j, 0, :h, :w], 0, 1) * 255).astype(np.uint8)
                Image.fromarray(arr).save(mask_dir / f"{make_safe_id(sid)}.png")
                wrote_mask = True

    pd.DataFrame(rows).to_csv(out / "probs.csv", index=False)

    run_meta = {
        "model": kind,
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
