"""급속 shadow warn 을 ETA별 임계값으로 나눌 가치가 있는지 검토.

현재는 공통 thr 1개(config.RAPID_SHADOW_WARN_THRESHOLD_COMMON=0.58, 정책 R>=0.70).
공통값은 h15 에서는 과하게 엄격하고 h30 이상에서는 목표 미달이다.
같은 홀드아웃(analyze_rapid_thresholds 와 동일한 8:2 날짜 분할 + 동일 HGB)에서
ETA별 곡선을 전부 뽑아 세 가지 정책을 비교한다.

  A 공통 0.58 (현행)
  B ETA별 R>=0.70 달성 최소 thr
  C ETA별 warn 발화율 <= 12% 안에서 R 최대

정본 아티팩트는 건드리지 않는다(분석 전용). 자체적으로 홀드아웃 HGB 를 다시 적합하므로
joblib 를 읽지 않는다 — 즉 "현재 DB 데이터 기준" 곡선이 나온다.

출처: claude/data-accumulation-results 워크트리에만 있던 파일을 이 브랜치로 들여왔다.
config.py 가 경로를 참조하는데 main 에 없어서 재학습 때마다 찾을 수 없었다.
들여오며 고친 것: (1) config 상수 개명 반영 RAPID_SHADOW_WARN_THRESHOLD → _COMMON,
(2) 저장소 루트 하드코딩 제거(파일 위치 기준으로 해석).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from recommend_api.config import HORIZONS, RAPID_SHADOW_WARN_THRESHOLD_COMMON
from recommend_api.eval_metrics import split_by_date
from recommend_api.holdout_eval import _fit_predict
from recommend_api.model_store import (
    build_horizon_dataset,
    load_joined,
    load_label_series,
    make_xy,
)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eta_thresholds.json")
GRID = np.round(np.arange(0.05, 0.996, 0.01), 2)
TARGET_R = 0.70
WARN_CAP = 0.12


def curve(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """thr 별 사용불가 recall/precision/warn 발화율. 사용불가 예측 = p < thr."""
    unavail = y == 0
    n_unavail = int(unavail.sum())
    rows = []
    for thr in GRID:
        pred = p < thr
        tp = int((pred & unavail).sum())
        n_pred = int(pred.sum())
        rows.append(
            {
                "threshold": float(thr),
                "unavailable_recall": tp / n_unavail if n_unavail else None,
                "unavailable_precision": tp / n_pred if n_pred else None,
                "warning_rate": n_pred / len(y),
            }
        )
    return rows


def pick_target(rows: list[dict], target: float) -> dict | None:
    return next((r for r in rows if (r["unavailable_recall"] or 0) >= target), None)


def pick_capped(rows: list[dict], cap: float) -> dict:
    ok = [r for r in rows if r["warning_rate"] <= cap]
    return max(ok, key=lambda r: r["unavailable_recall"] or 0)


def at(rows: list[dict], thr: float) -> dict:
    return min(rows, key=lambda r: abs(r["threshold"] - thr))


def apply_policy(y, p, eta, thr_by_eta: dict[int, float]) -> dict:
    pred = np.zeros(len(y), dtype=bool)
    for h, thr in thr_by_eta.items():
        m = eta == h
        pred[m] = p[m] < thr
    unavail = y == 0
    tp = int((pred & unavail).sum())
    return {
        "unavailable_recall": tp / int(unavail.sum()),
        "unavailable_precision": tp / int(pred.sum()) if pred.sum() else None,
        "warning_rate": float(pred.mean()),
        "n_warn": int(pred.sum()),
    }


def main() -> None:
    print("JOIN 로드 중...", flush=True)
    df = load_joined()
    print(f"JOIN: {len(df):,}행", flush=True)
    label_df = load_label_series()
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon: {len(hz):,}", flush=True)
    del df, label_df

    X, y, _med, meta = make_xy(hz, fill_median=False)
    del hz
    is_train, train_dates, test_dates = split_by_date(meta, ratio=0.8)
    print(f"train {len(train_dates)}일 / test {len(test_dates)}일 → {test_dates}", flush=True)

    print("홀드아웃 HGB fit...", flush=True)
    proba, y_te, te_index = _fit_predict(X, y, is_train)
    del X, y

    m_te = meta.loc[te_index]
    fast = pd.to_numeric(m_te["is_fast"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    y_arr = np.asarray(y_te)[fast]
    p_arr = np.asarray(proba, dtype=float)[fast]
    eta_arr = np.asarray(m_te["eta_minutes"])[fast].astype(int)
    print(f"급속 test n={len(y_arr):,} 사용불가율={1 - y_arr.mean():.4f}", flush=True)

    per_eta = {}
    thr_B: dict[int, float] = {}
    thr_C: dict[int, float] = {}
    print("\n ETA |      n | 불가율 | 현행0.58 R/P/warn | R>=0.70 thr(P,warn) | warn<=12% R(thr,P)")
    for h in sorted(set(eta_arr.tolist())):
        m = eta_arr == h
        rows = curve(y_arr[m], p_arr[m])
        cur = at(rows, RAPID_SHADOW_WARN_THRESHOLD_COMMON)
        tgt = pick_target(rows, TARGET_R)
        cap = pick_capped(rows, WARN_CAP)
        thr_B[int(h)] = float(tgt["threshold"]) if tgt else float(GRID[-1])
        thr_C[int(h)] = float(cap["threshold"])
        per_eta[int(h)] = {
            "n": int(m.sum()),
            "unavailable_rate": float(1 - y_arr[m].mean()),
            "at_current": cur,
            "target_r70": tgt,
            "warn_capped": cap,
        }
        tgt_s = (
            f"{tgt['threshold']:.2f} (P{tgt['unavailable_precision']:.3f}, {tgt['warning_rate']*100:4.1f}%)"
            if tgt
            else "달성불가"
        )
        print(
            f" h{h:<3}|{int(m.sum()):>8,}| {1-y_arr[m].mean():.3f} | "
            f"{cur['unavailable_recall']:.3f}/{cur['unavailable_precision']:.3f}/{cur['warning_rate']*100:4.1f}% | "
            f"{tgt_s} | {cap['unavailable_recall']:.3f} (thr{cap['threshold']:.2f}, P{cap['unavailable_precision']:.3f})",
            flush=True,
        )

    policies = {
        "A_common_current": apply_policy(
            y_arr, p_arr, eta_arr, {h: RAPID_SHADOW_WARN_THRESHOLD_COMMON for h in per_eta}
        ),
        "B_per_eta_r70": apply_policy(y_arr, p_arr, eta_arr, thr_B),
        "C_per_eta_warn_cap12": apply_policy(y_arr, p_arr, eta_arr, thr_C),
    }
    print("\n== 급속 전체 정책 비교 ==")
    for k, v in policies.items():
        print(
            f" {k:<22} R={v['unavailable_recall']:.4f} P={v['unavailable_precision']:.4f} "
            f"warn={v['warning_rate']*100:.2f}% (n_warn={v['n_warn']:,})"
        )

    OUT.write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "test_dates": [str(d) for d in test_dates],
                "n_rapid_test": int(len(y_arr)),
                "rapid_unavailable_rate": float(1 - y_arr.mean()),
                "current_common_threshold": RAPID_SHADOW_WARN_THRESHOLD_COMMON,
                "policy_target_recall": TARGET_R,
                "warn_cap": WARN_CAP,
                "per_eta": per_eta,
                "thresholds_B_r70": thr_B,
                "thresholds_C_warn_cap": thr_C,
                "policy_comparison": policies,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
