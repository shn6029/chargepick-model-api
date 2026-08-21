"""오프라인 급속 충전소 단위 top1 실패율 (정본 joblib 미갱신).

동일 created_at × eta_minutes 시각을 하나의 '요청 시각'으로 보고,
급속(비-stale) 충전기 max(p)로 충전소를 순위한 뒤 1위 충전소에
도착 시 가용(y=1) 충전기가 없으면 failure.

사용:
  py -m recommend_api.eval_station_top1
  py -m recommend_api.eval_station_top1 --min-output-kw 50
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

from recommend_api.config import (
    ARTIFACTS_DIR,
    HORIZONS,
    MATCH_TOLERANCE_MIN,
    VERY_STALE_EXCLUDE_MIN,
)
from recommend_api.eval_metrics import split_by_date
from recommend_api.holdout_eval import _fit_predict, save_metrics
from recommend_api.model_store import (
    build_horizon_dataset,
    load_joined,
    load_label_series,
    make_xy,
)

OUT_PATH = ARTIFACTS_DIR / "station_top1_metrics.json"


def _station_top1_failure(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """frame columns: created_at, eta_minutes, stat_id, y, proba"""
    if frame.empty:
        return {"n_snapshots": 0, "n_with_stations": 0, "failure_rate": None}

    # snapshot = 동일 시각 × ETA (요청 시뮬)
    frame = frame.copy()
    frame["snapshot_id"] = (
        pd.to_datetime(frame["created_at"]).astype("int64").astype(str)
        + "|"
        + frame["eta_minutes"].astype(int).astype(str)
    )

    # station score = max proba; station success = any(y==1)
    station = (
        frame.groupby(["snapshot_id", "stat_id"], as_index=False)
        .agg(score=("proba", "max"), success=("y", "max"), n_chargers=("y", "size"))
    )
    # top1 per snapshot
    station = station.sort_values(
        ["snapshot_id", "score", "stat_id"], ascending=[True, False, True]
    )
    top1 = station.groupby("snapshot_id", as_index=False).head(1)
    n = int(len(top1))
    n_fail = int((top1["success"] < 1).sum()) if n else 0
    return {
        "n_snapshots": int(frame["snapshot_id"].nunique()),
        "n_with_stations": n,
        "n_failed": n_fail,
        "failure_rate": float(n_fail / n) if n else None,
        "success_definition": (
            "snapshot(created_at×eta)별 max(proba) 1위 충전소에서 "
            "급속 충전기 중 y==1(도착 가용)이 1대 이상이면 성공"
        ),
        "mean_top1_score": float(top1["score"].mean()) if n else None,
        "mean_top1_chargers": float(top1["n_chargers"].mean()) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rapid station top1 failure")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--min-output-kw",
        type=float,
        default=50.0,
        help="호환 최소 kW (기본 50=급속)",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="stale 충전기 포함 (기본은 fresh만)",
    )
    args = parser.parse_args()

    print("JOIN 로드 중...")
    df = load_joined(limit=args.max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise SystemExit("학습 데이터가 없습니다.")

    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). model_store.load_label_series 참고.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon: {len(hz):,}")
    if hz.empty:
        raise SystemExit("horizon 샘플이 없습니다.")

    X, y, _med, meta = make_xy(hz, fill_median=False)
    is_train, train_dates, test_dates = split_by_date(meta, ratio=0.8)
    print("통합 HGB holdout fit...")
    proba, y_te, te_index = _fit_predict(X, y, is_train)

    te = meta.loc[te_index].copy()
    te["y"] = y_te.to_numpy()
    te["proba"] = proba

    # rapid + output filter
    te["is_fast"] = pd.to_numeric(te["is_fast"], errors="coerce").fillna(0).astype(int)
    mask = te["is_fast"] == 1
    if "output_kw" in te.columns and args.min_output_kw is not None:
        out_kw = pd.to_numeric(te["output_kw"], errors="coerce").fillna(0)
        mask = mask & (out_kw >= float(args.min_output_kw))

    if not args.allow_stale:
        if "is_stale_status" in te.columns:
            mask = mask & (te["is_stale_status"].fillna(0).astype(int) == 0)
        if "status_update_age_min" in te.columns:
            age = pd.to_numeric(te["status_update_age_min"], errors="coerce")
            mask = mask & (age.isna() | (age < VERY_STALE_EXCLUDE_MIN))

    rapid = te.loc[mask]
    if rapid.empty or "stat_id" not in rapid.columns:
        raise SystemExit("급속(fresh) test 표본 또는 stat_id가 없습니다.")

    print(
        f"급속 fresh test rows={len(rapid):,}  "
        f"chargers={rapid.groupby(['stat_id','chger_id']).ngroups:,}  "
        f"stations={rapid['stat_id'].nunique():,}"
    )

    overall = _station_top1_failure(rapid)
    print(
        f"station top1 failure: rate={overall.get('failure_rate')}  "
        f"n={overall.get('n_with_stations')}  failed={overall.get('n_failed')}"
    )

    by_eta = []
    for h in (15, 20, 30, 45, 60):
        sub = rapid[rapid["eta_minutes"] == h]
        if sub.empty:
            continue
        m = _station_top1_failure(sub)
        m["eta_minutes"] = int(h)
        by_eta.append(m)
        print(f"  ETA {h}: fail={m.get('failure_rate')} n={m.get('n_with_stations')}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_tolerance_min": MATCH_TOLERANCE_MIN,
        "canonical_joblib_unchanged": True,
        "filters": {
            "is_fast": True,
            "min_output_kw": args.min_output_kw,
            "exclude_stale": not args.allow_stale,
            "very_stale_exclude_min": VERY_STALE_EXCLUDE_MIN,
        },
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
        "overall": overall,
        "by_eta": by_eta,
        "note": (
            "오프라인 holdout 시뮬. 운영 로그 matched 축적 후 "
            "ops_metrics.top1_station_failure_rapid와 병행."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(payload, OUT_PATH)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
