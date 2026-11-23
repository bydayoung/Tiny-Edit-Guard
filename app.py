"""Tiny-Edit-Guard 웹 데모 서버 (Flask).

학습된 체크포인트(best.pt)를 불러와 업로드된 이미지에 대해
  - AI 편집 여부 / 편집 확률 (cls_logits)
  - 편집 영역 마스크, 오버레이, 확대 크롭 (mask_logits)
을 계산해 프론트(templates/index.html)에 JSON으로 돌려준다.

실행 예 (레포 루트에서):

    # 학습 결과 폴더의 best.pt + run_config.json 을 자동으로 읽음
    python app.py --checkpoint runs/patch_topk_s0/best.pt

    # run_config.json 이 없으면 모델 종류를 직접 지정
    python app.py --checkpoint path/to/best.pt --model patch_topk

    # 체크포인트 없이 화면만 확인 (가짜 결과, 화면에 '데모 모드' 표시)
    python app.py --demo

브라우저: http://127.0.0.1:7860

Vercel 등 서버리스 배포 (python app.py 를 실행하지 않고 모듈을 import 하는 경우):
  모듈 맨 아래의 전역 `app` 을 쓰며, 설정은 환경변수로 받는다.
    TEG_CHECKPOINT       로컬 체크포인트 경로 (예: weights/deploy.pt)
    TEG_CHECKPOINT_URL   또는 다운로드할 체크포인트 URL
    TEG_HF_REPO          또는 Hugging Face Hub 레포 id (TEG_HF_FILE 기본 deploy.pt, HF_TOKEN 사용)
    TEG_MODEL            run_config 가 없을 때 모델 종류
    TEG_CLS_THRESHOLD / TEG_MASK_THRESHOLD   임계값 (기본 0.5)
    TEG_DEMO=1           체크포인트 없이 데모 모드
  체크포인트는 export_for_deploy.py 로 만든 deploy.pt 를 권장한다
  (백본 config 가 들어 있어 실행 시 gated DINOv3 가중치를 내려받지 않는다).

전처리는 체크포인트가 학습된 방식에 맞춘다 (run_config.json 으로 판별):
  - V3 이후 train.py 로 학습한 체크포인트: 원본 해상도 유지 → 16 배수로 패딩(pad_mask 전달).
    긴 변이 MAX_INFER_SIDE 보다 크면 속도를 위해 그 크기로만 축소한다.
  - 이전 체크포인트: RGB → 512×512 BILINEAR resize (예전 학습 규칙)
  공통: ToTensor → ImageNet mean/std 정규화
모델 구성은 models/factory.py 의 build_model 로 train.py 와 동일하게 만든다.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent

INPUT_SIZE = 512                 # 이전(512 resize) 체크포인트용 입력 크기
MAX_INFER_SIDE = 1024            # 원본 해상도 추론 시 긴 변 상한 (CPU 속도)
PATCH_MULTIPLE = 16
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_DISPLAY_SIDE = 1024          # 프론트로 보내는 이미지의 최대 변 길이
MIN_REGION_RATIO = 0.0005        # 이보다 작은 연결 영역(전체 대비)은 박스에서 제외
MAX_REGIONS = 5                  # 박스로 표시할 최대 영역 수
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# train.py 의 --model 이름과 evals/predict.py 의 이름을 모두 받아준다 (models/factory.py 와 공유).
from models.factory import MODEL_ALIASES  # noqa: E402


def is_native_resolution(cfg: Dict) -> bool:
    """V3 이후 train.py 는 run_config 에 eval_batch_size 를 남긴다 = 원본 해상도로 학습/평가."""
    return "eval_batch_size" in cfg


# =============================================================================
# 모델 로딩
# =============================================================================

def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_run_config(checkpoint: Path) -> Dict:
    """best.pt 와 같은 폴더의 run_config.json (train.py 가 저장) 을 읽는다."""
    cfg_path = checkpoint.parent / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def hf_model_from_config(hf_config: Dict, segformer: bool = False):
    """저장해 둔 HF config(dict)로 가중치 다운로드 없이 빈 모델을 만든다."""
    from transformers import AutoConfig, AutoModel

    d = dict(hf_config)
    config = AutoConfig.for_model(d.pop("model_type"), **d)
    if segformer:
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation(config)
    return AutoModel.from_config(config)


def build_model(kind: str, cfg: Dict, hf_config: Optional[Dict] = None):
    """train.py 와 같은 함수(models/factory.build_model)로 모델을 만든다.

    hf_config 가 있으면 (export_for_deploy.py 로 만든 체크포인트) 백본을 config 로만 만들고,
    가중치는 이후 load_state_dict 로 채운다. 없으면 기존처럼 from_pretrained 를 쓴다.
    LoRA / 레이어 융합 / top-k 비율 등은 run_config 값 그대로 재현된다.
    """
    from models.factory import build_model as factory_build

    hf_model = None
    if hf_config is not None:
        hf_model = hf_model_from_config(hf_config, segformer=(kind == "segformer"))
    return factory_build({**cfg, "model": kind}, hf_model=hf_model)


class Detector:
    """체크포인트를 들고 있다가 PIL 이미지를 받아 확률/마스크를 돌려준다."""

    def __init__(self, checkpoint: str, model_kind: Optional[str], device: str = "auto"):
        import torch

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # export_for_deploy.py 로 만든 체크포인트는 run_config / hf_config 를 안에 들고 있다.
        embedded = isinstance(ckpt, dict) and "hf_config" in ckpt
        cfg = (ckpt.get("run_config") or {}) if embedded else load_run_config(ckpt_path)
        hf_config = ckpt["hf_config"] if embedded else None
        kind = model_kind or cfg.get("model")
        if not kind:
            raise ValueError(
                "모델 종류를 알 수 없습니다. run_config.json 이 체크포인트 옆에 없다면 "
                "--model 로 지정하세요 (patch_topk / patch_avg / global_cls / global_cls_mean / segformer)."
            )
        if kind not in MODEL_ALIASES:
            raise ValueError(f"지원하지 않는 모델 종류: {kind}")
        self.kind = MODEL_ALIASES[kind]

        self.device = pick_device(device)
        self.native = is_native_resolution(cfg)
        self.model = build_model(self.kind, cfg, hf_config)

        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.checkpoint = str(ckpt_path)
        self.epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
        print(f"[app] loaded {ckpt_path} | model={self.kind} | device={self.device}")

    def preprocess(self, img: Image.Image):
        """-> (x (1,3,H,W), pad_mask (1,1,H,W) or None, (h, w) of the real image inside x)."""
        import torch

        x = img.convert("RGB")
        if not self.native:  # 이전 체크포인트: 학습 때처럼 512×512 로 늘림
            x = x.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            arr = (np.asarray(x, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
            return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0), None, (INPUT_SIZE, INPUT_SIZE)

        w, h = x.size
        s = min(1.0, MAX_INFER_SIDE / max(w, h))
        if s < 1.0:
            x = x.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
            w, h = x.size
        arr = (np.asarray(x, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        H = -(-h // PATCH_MULTIPLE) * PATCH_MULTIPLE
        W = -(-w // PATCH_MULTIPLE) * PATCH_MULTIPLE
        canvas = np.zeros((H, W, 3), dtype=np.float32)  # 0 = ImageNet 평균색 (학습 패딩과 동일)
        canvas[:h, :w] = arr
        pad = np.zeros((1, 1, H, W), dtype=np.float32)
        pad[..., :h, :w] = 1.0
        return (torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0),
                torch.from_numpy(pad), (h, w))

    def predict(self, img: Image.Image) -> Tuple[float, Optional[np.ndarray]]:
        """Returns (edit_probability, prob_map[h, w] or None)."""
        import torch

        x, pad_mask, (h, w) = self.preprocess(img)
        kwargs = {"pad_mask": pad_mask.to(self.device)} if pad_mask is not None else {}
        with torch.no_grad():
            out = self.model(x.to(self.device), **kwargs)
        prob = float(torch.sigmoid(out["cls_logits"]).float().cpu()[0])
        mask = out.get("mask_logits")
        if mask is None:
            return prob, None
        prob_map = torch.sigmoid(mask).float().cpu().numpy()[0, 0, :h, :w]  # 패딩 제거
        return prob, prob_map

    def info(self) -> Dict:
        return {
            "mode": "model",
            "model": self.kind,
            "device": str(self.device),
            "checkpoint": Path(self.checkpoint).name,
            "epoch": self.epoch,
            "supports_mask": self.kind != "global_cls" and self.kind != "global_cls_mean",
        }


class DemoDetector:
    """체크포인트 없이 프론트 확인용. 결과는 가짜이며 화면에 데모 모드로 표시된다."""

    kind = "demo"

    def predict(self, img: Image.Image) -> Tuple[float, Optional[np.ndarray]]:
        rng = np.random.default_rng(abs(hash(img.tobytes()[:4096])) % (2**32))
        yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
        cy, cx = rng.uniform(0.2, 0.8, size=2) * INPUT_SIZE
        r = rng.uniform(15, 45)
        prob_map = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))).astype(np.float32)
        return float(rng.uniform(0.6, 0.99)), prob_map

    def info(self) -> Dict:
        return {"mode": "demo", "model": "demo", "device": "-", "checkpoint": None,
                "epoch": None, "supports_mask": True}


# =============================================================================
# 후처리 (마스크 → 박스 / 오버레이 / 크롭)
# =============================================================================

def fit_display(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(1.0, MAX_DISPLAY_SIDE / max(w, h))
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    return img


def to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def find_regions(binary: np.ndarray) -> List[List[float]]:
    """이진 마스크의 연결 영역별 박스 [x0, y0, x1, y1] (0~1 정규화), 큰 순서."""
    H, W = binary.shape
    if not binary.any():
        return []
    try:
        from scipy import ndimage

        labels, n = ndimage.label(binary)
        slices = ndimage.find_objects(labels)
        areas = ndimage.sum(binary, labels, index=range(1, n + 1))
        order = np.argsort(areas)[::-1]
        boxes = []
        for i in order[:MAX_REGIONS]:
            if areas[i] < MIN_REGION_RATIO * H * W and boxes:
                break
            sy, sx = slices[i]
            boxes.append([sx.start / W, sy.start / H, sx.stop / W, sy.stop / H])
        return boxes
    except ImportError:  # scipy 가 없으면 전체를 감싸는 박스 하나
        ys, xs = np.where(binary)
        return [[xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]]


def pad_box(box: List[float], aspect: float, pad: float = 0.35, min_side: float = 0.08) -> List[float]:
    """박스를 여유 있게 넓힌다. aspect = W/H (정규화 좌표를 정사각형 감각으로 맞추기 위함)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw = max(x1 - x0, min_side / max(aspect, 1e-6) ** 0.5)
    bh = max(y1 - y0, min_side * max(aspect, 1e-6) ** 0.5)
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    x0, x1 = max(0.0, cx - bw / 2), min(1.0, cx + bw / 2)
    y0, y1 = max(0.0, cy - bh / 2), min(1.0, cy + bh / 2)
    return [x0, y0, x1, y1]


def union_box(boxes: List[List[float]]) -> List[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def build_result(img: Image.Image, prob: float, prob_map: Optional[np.ndarray],
                 cls_threshold: float, mask_threshold: float) -> Dict:
    disp = fit_display(img)
    W, H = disp.size
    is_edited = prob >= cls_threshold

    result = {
        "is_edited": bool(is_edited),
        "label": "AI 편집 이미지" if is_edited else "실제 이미지",
        "probability": round(prob * 100, 1),
        "cls_threshold": cls_threshold,
        "image": to_data_url(disp),
        "image_size": [W, H],
        "has_mask": prob_map is not None,
    }
    if prob_map is None:
        return result

    # 확률맵 → 표시 크기로 맞춤 (512 stretch 체크포인트는 역변환, 원본 해상도 체크포인트는 단순 축척)
    pm = Image.fromarray((np.clip(prob_map, 0, 1) * 255).astype(np.uint8))
    pm = pm.resize((W, H), Image.Resampling.BILINEAR)
    pm_arr = np.asarray(pm, dtype=np.float32) / 255.0
    binary = pm_arr >= mask_threshold

    ratio = float(binary.mean())
    boxes = find_regions(binary)
    padded = [pad_box(b, W / H, pad=0.25, min_side=0.05) for b in boxes]

    # 오버레이: 편집 영역을 빨간색으로 칠한다
    alpha = (binary.astype(np.float32) * 0.6)[..., None]
    red = np.array([230, 40, 40], dtype=np.float32)
    base = np.asarray(disp, dtype=np.float32)
    overlay = Image.fromarray((base * (1 - alpha) + red * alpha).astype(np.uint8))
    # 경계선은 살짝 진하게
    edge = Image.fromarray(binary.astype(np.uint8) * 255).filter(ImageFilter.FIND_EDGES)
    overlay.paste((200, 20, 20), mask=edge)

    # 크롭: 모든 편집 영역을 감싸는 박스를 넉넉히 잘라 확대
    crop_url, crop_box = None, None
    if boxes:
        crop_box = pad_box(union_box(boxes), W / H, pad=0.6, min_side=0.15)
        x0, y0, x1, y1 = crop_box
        crop = disp.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
        cw, ch = crop.size
        scale = 512 / max(cw, ch)
        if scale > 1:
            crop = crop.resize((round(cw * scale), round(ch * scale)), Image.Resampling.LANCZOS)
        crop_url = to_data_url(crop)

    result.update({
        "edit_ratio": round(ratio * 100, 2),
        "mask_threshold": mask_threshold,
        "regions": padded,
        "mask": to_data_url(pm, fmt="PNG"),
        "overlay": to_data_url(overlay),
        "crop": crop_url,
        "crop_box": crop_box,
    })
    return result


# =============================================================================
# Flask 앱
# =============================================================================

def create_app(detector, cls_threshold: float, mask_threshold: float,
               examples_dir: Path) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.json.ensure_ascii = False

    def run(img: Image.Image, filename: str):
        t0 = time.perf_counter()
        prob, prob_map = detector.predict(ImageOps.exif_transpose(img).convert("RGB"))
        res = build_result(img, prob, prob_map, cls_threshold, mask_threshold)
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        res["filename"] = filename
        res["demo"] = detector.info()["mode"] == "demo"
        return jsonify(res)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        try:
            d = detector.info()
        except Exception as e:  # noqa: BLE001  (서버리스에서 모델 로드 실패를 화면에 보여주기 위함)
            app.logger.exception("model load failed")
            return jsonify({"mode": "error", "error": f"모델을 불러오지 못했습니다: {e}"}), 500
        return jsonify({**d, "cls_threshold": cls_threshold, "mask_threshold": mask_threshold})

    @app.get("/api/examples")
    def examples():
        files = []
        if examples_dir.is_dir():
            files = sorted(p.name for p in examples_dir.iterdir()
                           if p.suffix.lower() in ALLOWED_EXT)
        return jsonify([{"name": f, "url": f"/examples/{f}"} for f in files])

    @app.get("/examples/<path:name>")
    def example_file(name):
        return send_from_directory(examples_dir, name)

    @app.post("/api/predict")
    def predict():
        try:
            if "image" in request.files:
                f = request.files["image"]
                img = Image.open(f.stream)
                img.load()
                return run(img, f.filename or "upload")
            body = request.get_json(silent=True) or {}
            name = body.get("example")
            if name:
                path = (examples_dir / name).resolve()
                if examples_dir.resolve() not in path.parents or not path.is_file():
                    return jsonify({"error": "예시 이미지를 찾을 수 없습니다."}), 404
                return run(Image.open(path), name)
            return jsonify({"error": "이미지가 전송되지 않았습니다."}), 400
        except RequestEntityTooLarge:
            raise  # 아래 413 핸들러로 보낸다
        except Image.UnidentifiedImageError:
            return jsonify({"error": "이미지 파일을 읽을 수 없습니다. JPG/PNG 파일을 올려주세요."}), 400
        except Exception as e:  # noqa: BLE001
            app.logger.exception("prediction failed")
            return jsonify({"error": f"추론 중 오류가 발생했습니다: {e}"}), 500

    @app.errorhandler(413)
    def too_large(_):
        mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return jsonify({"error": f"파일이 너무 큽니다 (최대 {mb:g}MB)."}), 413

    return app


# =============================================================================
# 서버리스(Vercel) 진입점: 환경변수로 설정, 모델은 첫 요청 때 로드
# =============================================================================

class LazyDetector:
    """import 시점엔 아무것도 하지 않고, 첫 요청에서 Detector 를 만든다.

    서버리스에서는 모듈 import 가 실패하면 함수 전체가 죽으므로, 로드 오류를
    요청 단위의 JSON 에러로 돌려주기 위해 지연 로딩한다.
    """

    def __init__(self, factory):
        self._factory = factory
        self._detector = None
        self._lock = threading.Lock()

    def get(self):
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    self._detector = self._factory()
        return self._detector

    def predict(self, img):
        return self.get().predict(img)

    def info(self):
        return self.get().info()


def resolve_checkpoint() -> str:
    """환경변수에서 체크포인트 위치를 찾아 로컬 경로로 돌려준다 (필요하면 /tmp 로 다운로드)."""
    local = os.environ.get("TEG_CHECKPOINT")
    if local:
        p = Path(local)
        if not p.is_absolute():
            p = ROOT / p
        return str(p)

    cache_dir = Path(os.environ.get("TEG_CACHE_DIR", "/tmp/teg"))  # 서버리스에서 쓸 수 있는 곳은 /tmp 뿐
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo = os.environ.get("TEG_HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo,
            filename=os.environ.get("TEG_HF_FILE", "deploy.pt"),
            repo_type=os.environ.get("TEG_HF_REPO_TYPE", "model"),
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        )

    url = os.environ.get("TEG_CHECKPOINT_URL")
    if url:
        import urllib.request

        dst = cache_dir / (Path(url.split("?")[0]).name or "checkpoint.pt")
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
        return str(dst)

    raise RuntimeError("TEG_CHECKPOINT / TEG_HF_REPO / TEG_CHECKPOINT_URL 중 하나를 설정하세요 "
                       "(화면만 보려면 TEG_DEMO=1).")


def app_from_env() -> Flask:
    if os.environ.get("TEG_DEMO", "").lower() in ("1", "true", "yes"):
        detector = DemoDetector()
    else:
        detector = LazyDetector(lambda: Detector(
            resolve_checkpoint(), os.environ.get("TEG_MODEL"), os.environ.get("TEG_DEVICE", "cpu")))
    flask_app = create_app(
        detector,
        float(os.environ.get("TEG_CLS_THRESHOLD", 0.5)),
        float(os.environ.get("TEG_MASK_THRESHOLD", 0.5)),
        Path(os.environ.get("TEG_EXAMPLES_DIR", ROOT / "static" / "examples")),
    )
    if os.environ.get("VERCEL"):
        # Vercel Functions 는 요청 본문이 4.5MB 로 제한된다 (프론트에서 미리 줄여서 보냄).
        flask_app.config["MAX_CONTENT_LENGTH"] = int(4.4 * 1024 * 1024)
    return flask_app


def parse_args():
    p = argparse.ArgumentParser(description="Tiny-Edit-Guard web demo")
    p.add_argument("--checkpoint", default=os.environ.get("TEG_CHECKPOINT"),
                   help="trainer 가 저장한 best.pt 경로 (옆에 run_config.json 이 있으면 자동 사용)")
    p.add_argument("--model", default=os.environ.get("TEG_MODEL"),
                   choices=sorted(MODEL_ALIASES),
                   help="run_config.json 이 없을 때 모델 종류 지정")
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--cls-threshold", type=float, default=0.5,
                   help="AI 편집 판정 임계값 (validation F1 최대 threshold 권장)")
    p.add_argument("--mask-threshold", type=float, default=0.5,
                   help="편집 영역 픽셀 판정 임계값")
    p.add_argument("--examples-dir", default=str(ROOT / "static" / "examples"))
    p.add_argument("--demo", action="store_true", help="체크포인트 없이 가짜 결과로 화면만 확인")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.demo:
        detector = DemoDetector()
        print("[app] 데모 모드: 실제 모델이 아닌 가짜 결과를 보여줍니다.")
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint 경로를 지정하세요. (화면만 보려면 --demo)")
        detector = Detector(args.checkpoint, args.model, args.device)

    app = create_app(detector, args.cls_threshold, args.mask_threshold, Path(args.examples_dir))
    print(f"[app] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


# Vercel 은 이 파일을 import 해서 전역 변수 `app` (WSGI 앱) 을 찾는다.
# (만들기만 하고 모델은 첫 요청 때 로드하므로, python app.py 로 실행할 때는 비용이 거의 없다.)
app = app_from_env()


if __name__ == "__main__":
    main()
