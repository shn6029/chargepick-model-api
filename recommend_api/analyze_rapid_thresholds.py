"""급속 holdout에서 사용불가 Recall 목표별 임계값 분석 (정본 joblib 미갱신).

사용불가 예측 = available_prob < thr

사용:
  py -m recommend_api.analyze_rapid_thresholds
  py -m recommend_api.analyze_rapid_thresholds --max-rows 50000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import ARTIFACTS_DIR, HORIZONS, MATCH_TOLERANCE_MIN
from recommend_api.eval_metrics import evaluate_classifier, split_by_date
from recommend_api.holdout_eval import _fit_predict, save_metrics
from recommend_api.model_store import (
    build_horizon_dataset,
    load_joined,
    load_label_series,
    make_xy,
)

OUT_PATH = ARTIFACTS_DIR / "rapid_threshold_analysis.json"
RECALL_TARGETS = [0.60, 0.70, 0.80]
# fine grid for finding first thr that hits target recall
THR_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
ETA_FOCUS = [15, 20, 30]


def _metrics_at_thr(y: np.ndarray, proba: np.ndarray, thr: float) -> dict[str, Any]:
    m = evaluate_classifier(y, proba, threshold=thr)
    warning_rate = float((proba < thr).mean())
    return {
        "threshold": float(thr),
        "unavailable_recall": m["unavailable_recall"],
        "unavailable_precision": m["unavailable_precision"],
        "unavailable_f1": m["unavailable_f1"],
        "warning_rate": warning_rate,
        "n": m["n"],
    }


def _find_targets(
    y: np.ndarray, proba: np.ndarray, targets: list[float]
) -> list[dict[str, Any]]:
    """각 목표 Recall을 최초로 달성하는 가장 낮은 thr (thr↑ → 가용 판정 엄격 → 불가 예측↑)."""
    rows = []
    for thr in THR_GRID:
        rows.append(_metrics_at_thr(y, proba, float(thr)))

    out = []
    for target in targets:
        hit = next((r for r in rows if r["unavailable_recall"] >= target), None)
        if hit is None:
            # best achievable
            best = max(rows, key=lambda r: r["unavailable_recall"])
            out.append(
                {
                    "target_unavailable_recall": target,
                    "achieved": False,
                    "best_available": best,
                    "note": "목표 Recall 미달 — grid 내 최대 Recall 보고",
                }
            )
        else:
            out.append(
                {
                    "target_unavailable_recall": target,
                    "achieved": True,
                    **hit,
                    "interpretation": (
                        "warning_rate~0.10 적용 여지 / ~0.30 오경고 과다 가능"
                    ),
                }
            )
    return out


def _slice_report(y: pd.Series, proba: np.ndarray, eta: pd.Series | None) -> dict[str, Any]:
    y_arr = np.asarray(y)
    p_arr = np.asarray(proba, dtype=float)
    base = {
        "n": int(len(y_arr)),
        "unavailable_rate": float(1.0 - y_arr.mean()) if len(y_arr) else None,
        "at_0.5": _metrics_at_thr(y_arr, p_arr, 0.5),
        "targets": _find_targets(y_arr, p_arr, RECALL_TARGETS),
        "sweep_sample": [
            _metrics_at_thr(y_arr, p_arr, t)
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        ],
    }
    if eta is not None:
        by_eta = []
        for h in ETA_FOCUS:
            mask = np.asarray(eta) == h
            if not mask.any():
                continue
            by_eta.append(
                {
                    "eta_minutes": int(h),
                    "n": int(mask.sum()),
                    "unavailable_rate": float(1.0 - y_arr[mask].mean()),
                    "at_0.5": _metrics_at_thr(y_arr[mask], p_arr[mask], 0.5),
                    "targets": _find_targets(y_arr[mask], p_arr[mask], RECALL_TARGETS),
                }
            )
        base["by_eta"] = by_eta
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rapid-only threshold analysis for unavailable recall targets"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    print("JOIN 로드 중...")
    df = load_joined(limit=args.max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise SystemExit("학습 데이터가 없습니다.")

    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). 여기서 나온 임계값이
    # config.RAPID_SHADOW_WARN_THRESHOLD 의 근거이므로 라벨 경로가 달라지면 안 된다.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon: {len(hz):,}")
    if hz.empty:
        raise SystemExit("horizon 샘플이 없습니다.")

    X, y, _med, meta = make_xy(hz, fill_median=False)
    is_train, train_dates, test_dates = split_by_date(meta, ratio=0.8)
    if len(test_dates) == 0:
        raise SystemExit("holdout test 날짜가 없습니다.")

    print("통합 HGB holdout fit...")
    proba, y_te, te_index = _fit_predict(X, y, is_train)
    is_fast_te = (
        pd.to_numeric(meta.loc[te_index, "is_fast"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    rapid_mask = is_fast_te == 1
    if not rapid_mask.any():
        raise SystemExit("급속 test 표본이 없습니다.")

    y_r = y_te.iloc[np.flatnonzero(rapid_mask)]
    p_r = proba[rapid_mask]
    eta_r = meta.loc[te_index, "eta_minutes"].iloc[np.flatnonzero(rapid_mask)]

    print(f"급속 test n={len(y_r):,} unavail={1 - y_r.mean():.1%}")
    report = _slice_report(y_r, p_r, eta_r)
    for t in report["targets"]:
        if t.get("achieved"):
            print(
                f"  target R>={t['target_unavailable_recall']}: "
                f"thr={t['threshold']}  P={t['unavailable_precision']:.3f}  "
                f"warn={t['warning_rate']:.1%}  R={t['unavailable_recall']:.3f}"
            )
        else:
            best = t.get("best_available") or {}
            print(
                f"  target R>={t['target_unavailable_recall']}: NOT achieved  "
                f"best_R={best.get('unavailable_recall')}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_tolerance_min": MATCH_TOLERANCE_MIN,
        "speed_definition": "is_fast = (output_kw >= 50)",
        "canonical_joblib_unchanged": True,
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
        "definition": "unavailable_pred = (available_prob < threshold)",
        "policy_note": (
            "rank_threshold는 0.5 유지 권장. "
            "급속 warn_threshold만 별도 검토. artifact thresholds 자동 교체 없음."
        ),
        "rapid_holdout": report,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(payload, OUT_PATH)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
