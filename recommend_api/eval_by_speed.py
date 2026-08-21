"""급속/완속 holdout 분리 평가 (정본 joblib 미갱신).

사용:
  cd f:\\dev\\scheduler
  py -m recommend_api.eval_by_speed
  py -m recommend_api.eval_by_speed --coverage-only
  py -m recommend_api.eval_by_speed --max-rows 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import ARTIFACTS_DIR, HORIZONS, MATCH_TOLERANCE_MIN
from recommend_api.coverage_stats import compute_speed_coverage
from recommend_api.eval_metrics import plot_calibration_curves, split_by_date
from recommend_api.holdout_eval import (
    _print_speed_segment_summary,
    _public_segment,
    evaluate_date_holdout,
    save_metrics,
)
from recommend_api.model_store import (
    build_horizon_dataset,
    load_joined,
    load_label_series,
    make_xy,
)

OUT_PATH = ARTIFACTS_DIR / "speed_segment_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holdout overall / rapid_only / slow_only (does not overwrite joblib)"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="holdout 생략, 커버리지만 계산해 JSON에 병합/저장",
    )
    args = parser.parse_args()

    print("JOIN 로드 중...")
    df = load_joined(limit=args.max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise SystemExit("학습 데이터가 없습니다.")

    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). 생략하면 snapshot 앵커가 빠져
    # LOCF 사슬이 끊기는데, delta 보고가 드문 완속에서 더 심하게 망가진다.
    # 속도군 비교가 목적인 스크립트에서 이건 치명적이다.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon 샘플: {len(hz):,} | 사용가능 비율 {hz['y'].mean():.1%}")
    if hz.empty:
        raise SystemExit("horizon 샘플이 없습니다.")

    X, y, _med, meta = make_xy(hz, fill_median=False)
    if "is_fast" not in meta.columns:
        raise SystemExit("meta에 is_fast가 없습니다.")

    # coverage용: horizon에 is_fast / ids 정렬
    hz_cov = hz.copy()
    hz_cov["is_fast"] = meta["is_fast"].values
    if "stat_id" not in hz_cov.columns and "stat_id" in meta.columns:
        hz_cov["stat_id"] = meta["stat_id"].values
        hz_cov["chger_id"] = meta["chger_id"].values

    n_rapid = int((meta["is_fast"] == 1).sum())
    n_slow = int((meta["is_fast"] == 0).sum())
    print(f"급속(is_fast=1): {n_rapid:,}  완속: {n_slow:,}")

    is_train, _train_dates, _test_dates = split_by_date(meta, ratio=0.8)
    test_mask = ~is_train
    coverage = compute_speed_coverage(hz_cov, test_mask=test_mask)
    for key in ("rapid", "slow"):
        c = coverage[key]
        print(
            f"coverage {key}: chargers={c['n_unique_chargers']:,}  "
            f"stations={c['n_unique_stations']:,}  "
            f"rows={c['n_horizon_rows']:,}  "
            f"med_samples/charger={c['samples_per_charger_median']}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.coverage_only:
        payload: dict = {}
        if OUT_PATH.exists():
            try:
                payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "match_tolerance_min": MATCH_TOLERANCE_MIN,
                "horizons": HORIZONS,
                "speed_definition": "is_fast = (output_kw >= 50)",
                "n_horizon_samples": int(len(hz)),
                "n_rapid_samples": n_rapid,
                "n_slow_samples": n_slow,
                "canonical_joblib_unchanged": True,
                "coverage": coverage,
            }
        )
        save_metrics(payload, OUT_PATH)
        print(f"저장(coverage-only): {OUT_PATH}")
        return

    print("날짜 홀드아웃(+급속/완속) 평가 중...")
    date_holdout = evaluate_date_holdout(X, y, meta)
    if date_holdout is None:
        raise SystemExit("날짜 holdout을 만들 수 없습니다.")

    _print_speed_segment_summary(date_holdout)

    rapid = date_holdout.get("rapid_only")
    if rapid and rapid.get("_y") is not None and len(rapid["_y"]) > 0:
        plot_calibration_curves(
            {"rapid_only": (np.asarray(rapid["_y"]), np.asarray(rapid["_proba"]))},
            ARTIFACTS_DIR / "calibration_rapid_only.png",
        )
        print(f"급속 calibration: {ARTIFACTS_DIR / 'calibration_rapid_only.png'}")

    date_public = {
        k: v
        for k, v in date_holdout.items()
        if not str(k).startswith("_") and k not in ("rapid_only", "slow_only")
    }
    date_public["rapid_only"] = _public_segment(date_holdout.get("rapid_only"))
    date_public["slow_only"] = _public_segment(date_holdout.get("slow_only"))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_tolerance_min": MATCH_TOLERANCE_MIN,
        "horizons": HORIZONS,
        "speed_definition": "is_fast = (output_kw >= 50)",
        "n_horizon_samples": int(len(hz)),
        "n_rapid_samples": n_rapid,
        "n_slow_samples": n_slow,
        "canonical_joblib_unchanged": True,
        "coverage": coverage,
        "date_holdout": date_public,
    }
    save_metrics(payload, OUT_PATH)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
