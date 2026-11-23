"""지표 계산 단일 모듈.

train.py의 validation 점검과 evaluate.py의 test 채점이 반드시 이 모듈의
같은 함수를 쓴다 (학습-평가 지표 불일치 방지).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ---------- Image-level (Detection) ----------

def image_auroc(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def image_auprc(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, probs))


def select_threshold(labels, probs):
    """F1을 최대화하는 threshold 선정.

    반드시 validation set으로만 호출할 것 (test에서 선정하면 반칙).
    """
    labels, probs = np.asarray(labels), np.asarray(probs)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(probs, 3)):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def image_f1(labels, probs, threshold):
    labels, probs = np.asarray(labels), np.asarray(probs)
    return float(f1_score(labels, (probs >= threshold).astype(int), zero_division=0))


def bootstrap_auroc_ci(labels, probs, n_boot=1000, seed=0, alpha=0.05):
    """AUROC의 부트스트랩 신뢰구간. 구간(bin)별 샘플이 적을 때 반드시 함께 보고."""
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    n, stats = len(labels), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        l = labels[idx]
        if len(np.unique(l)) < 2:
            continue
        stats.append(roc_auc_score(l, probs[idx]))
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def pair_accuracy(group_ids, labels, probs):
    """쌍(pair) 정확도: 같은 group 안에서 fake 점수 > real 점수인 (fake, real) 조합의 비율.

    V3 처럼 real/fake 가 같은 원본에서 나온 exact pair 일 때, 내용·출처 차이 없이
    '편집 흔적'만으로 구분하는지 보는 지표. 동점은 0.5. 쌍이 없으면 (nan, 0).
    반환: (accuracy, 비교한 조합 수)
    """
    g = np.asarray(group_ids).astype(str)
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    wins, n = 0.0, 0
    for gid in np.unique(g):
        idx = g == gid
        f, r = probs[idx & (labels == 1)], probs[idx & (labels == 0)]
        if len(f) and len(r):
            diff = f[:, None] - r[None, :]
            wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            n += diff.size
    return (wins / n if n else float("nan")), n


# ---------- Pixel-level (Localization) ----------
# 규칙: 편집(Fake) 이미지 + valid_mask=True 에서만 계산. per-image 계산 후 평균.

def dice_iou(pred_bin, gt_bin):
    pred, gt = np.asarray(pred_bin).astype(bool), np.asarray(gt_bin).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = 2.0 * inter / denom if denom > 0 else float("nan")
    iou = inter / union if union > 0 else float("nan")
    return float(dice), float(iou)


def pixel_auprc(pred_prob, gt_bin):
    """예측 확률맵(0~1) vs GT 이진 mask. GT가 비어 있으면 정의 불가(nan)."""
    gt = np.asarray(gt_bin).astype(int).ravel()
    if gt.max() == 0:
        return float("nan")
    return float(average_precision_score(gt, np.asarray(pred_prob).ravel()))
