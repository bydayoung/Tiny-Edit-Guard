"""Smoke(회귀) 테스트: 평가 파이프라인이 조용히 깨지는 걸 막는다.

고정 fixture(tests/fixture_{val,test}.csv)로 evaluate.py를 돌려
핵심 성질이 유지되는지 검사한다. torch/GPU 불필요 — numpy/pandas/sklearn만.

실행
  python3 evals/tests/test_smoke.py        # 레포 루트에서
  (GitHub Actions가 push/PR마다 자동 실행)

fixture는 'fake일수록 확률이 높게' 구성돼 있어, 파이프라인이 정상이면
전체 AUROC가 높게 나와야 한다. 지표 계산이 깨지면 이 값이 무너지거나 에러가 난다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS))

import metrics as M          # noqa: E402
from evaluate import score_detection, loc_from_csv, to_bool  # noqa: E402

FIX = Path(__file__).parent


def fail(msg):
    print(f"❌ SMOKE FAIL: {msg}")
    sys.exit(1)


def main():
    val = pd.read_csv(FIX / "fixture_val.csv")
    test = pd.read_csv(FIX / "fixture_test.csv")

    # 1) threshold 선정이 동작하고 범위 안
    thr = M.select_threshold(val.label, val.edit_probability)
    if not (0.0 <= thr <= 1.0):
        fail(f"threshold 범위 벗어남: {thr}")

    # 2) detection이 에러 없이 돌고, 기대한 scope들이 나온다
    rows = score_detection(test, thr, "test")
    scopes = {r["scope"] for r in rows}
    if "test_overall" not in scopes:
        fail("test_overall 행 없음")
    for b in ("bin:tiny", "bin:large"):
        if b not in scopes:
            fail(f"{b} 행 없음 — 구간별 분해가 깨짐")
    if not any(s.startswith("method:") for s in scopes):
        fail("method별 행 없음")

    # 3) 모든 AUROC/AUPRC/F1이 [0,1]
    for r in rows:
        for k in ("auroc", "auprc", "f1"):
            v = r[k]
            if not (np.isnan(v) or 0.0 <= v <= 1.0):
                fail(f"{r['scope']} {k}={v} 범위 벗어남")

    # 4) 분별력: fixture는 fake 확률이 높게 설계됨 → 전체 AUROC가 충분히 높아야
    overall = next(r for r in rows if r["scope"] == "test_overall")
    if overall["auroc"] < 0.85:
        fail(f"전체 AUROC가 비정상적으로 낮음({overall['auroc']}) — 지표 계산 회귀 의심")

    # 5) 신뢰구간이 점추정을 감싼다 (low <= auroc <= high)
    if not (overall["auroc_ci_low"] <= overall["auroc"] <= overall["auroc_ci_high"]):
        fail("신뢰구간이 점추정을 감싸지 않음")

    # 6) localization: valid_mask 샘플만, Dice가 [0,1]
    loc = loc_from_csv(test[test.label == 1])
    if not loc:
        fail("localization 행 없음")
    for r in loc:
        if not (np.isnan(r["dice"]) or 0.0 <= r["dice"] <= 1.0):
            fail(f"{r['scope']} dice={r['dice']} 범위 벗어남")

    # 7) valid_mask 변환이 real(빈 mask)을 거른다
    n_valid = int(to_bool(test.valid_mask).sum())
    n_fake = int((test.label == 1).sum())
    if n_valid != n_fake:  # fixture는 fake=valid_mask 설계
        fail(f"valid_mask 수({n_valid}) != fake 수({n_fake})")

    print(f"✅ SMOKE PASS — overall AUROC={overall['auroc']}, "
          f"threshold={thr:.3f}, loc rows={len(loc)}")


if __name__ == "__main__":
    main()
