"""수집 간격·stat_upd_dt 품질 리포트.

사용:
  py -m recommend_api.data_quality
  py -m recommend_api.data_quality --sample-chargers 2000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymysql

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import (
    ARTIFACTS_DIR,
    OUTCOME_MATCH_TOLERANCE_MIN,
    STALE_UPDATE_AGE_MIN,
)
from recommend_api.model_store import get_connection

OUT_PATH = ARTIFACTS_DIR / "data_quality.json"


def _percentiles(arr: np.ndarray) -> dict[str, float | None]:
    if arr.size == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_intervals(sample_chargers: int = 2000) -> dict[str, Any]:
    """충전기 샘플의 연속 created_at 간격(초)."""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT stat_id, chger_id
                FROM (
                    SELECT DISTINCT stat_id, chger_id
                    FROM ev_charger_status
                ) t
                ORDER BY RAND()
                LIMIT %s
                """,
                (int(sample_chargers),),
            )
            keys = cur.fetchall()
            intervals: list[float] = []
            for row in keys:
                cur.execute(
                    """
                    SELECT created_at
                    FROM ev_charger_status
                    WHERE stat_id = %s AND chger_id = %s
                    ORDER BY created_at
                    """,
                    (row["stat_id"], row["chger_id"]),
                )
                times = [r["created_at"] for r in cur.fetchall()]
                for i in range(1, len(times)):
                    if times[i] is None or times[i - 1] is None:
                        continue
                    sec = (times[i] - times[i - 1]).total_seconds()
                    if 0 < sec < 86400 * 2:  # 비정상 장간격 제외(2일+)
                        intervals.append(float(sec))
    finally:
        conn.close()

    arr = np.asarray(intervals, dtype=float)
    stats = _percentiles(arr)
    # 5분 수집 가정: 1~20분 구간만으로 tolerance 제안 (장기 공백 제외)
    near = arr[(arr >= 60) & (arr <= 1200)] if arr.size else arr
    near_stats = _percentiles(near)
    suggested = None
    if near_stats.get("p95") is not None:
        suggested = int(math.ceil(near_stats["p95"] / 60.0) + 1)
    elif stats.get("p95") is not None:
        suggested = int(math.ceil(min(stats["p95"], 1200) / 60.0) + 1)
    return {
        "sample_chargers": sample_chargers,
        "interval_seconds": stats,
        "interval_seconds_near_nominal_60_1200": near_stats,
        "current_outcome_tolerance_min": OUTCOME_MATCH_TOLERANCE_MIN,
        "suggested_outcome_tolerance_min": suggested,
        "note": (
            "설정 변경은 수동. suggested는 60~1200초 간격 p95 기반 "
            "(장기 미수집 공백 제외)."
        ),
    }


def collect_stat_upd_dt() -> dict[str, Any]:
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(stat_upd_dt IS NULL) AS null_count,
                    SUM(stat_upd_dt > created_at) AS future_count,
                    SUM(
                        stat_upd_dt IS NOT NULL
                        AND created_at >= stat_upd_dt
                        AND TIMESTAMPDIFF(MINUTE, stat_upd_dt, created_at) >= {int(STALE_UPDATE_AGE_MIN)}
                    ) AS stale_count,
                    SUM(
                        stat_upd_dt IS NULL OR stat_upd_dt > created_at
                    ) AS invalid_status_update_time_count
                FROM ev_charger_status
                """
            )
            row = cur.fetchone() or {}
    finally:
        conn.close()
    total = int(row.get("total_count") or 0)
    return {
        "total_count": total,
        "null_count": int(row.get("null_count") or 0),
        "future_count": int(row.get("future_count") or 0),
        "stale_count": int(row.get("stale_count") or 0),
        "invalid_status_update_time_count": int(
            row.get("invalid_status_update_time_count") or 0
        ),
        "stale_threshold_min": STALE_UPDATE_AGE_MIN,
        "null_rate": (int(row.get("null_count") or 0) / total) if total else None,
        "future_rate": (int(row.get("future_count") or 0) / total) if total else None,
    }


def run_report(*, sample_chargers: int = 2000, skip_intervals: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stat_upd_dt": collect_stat_upd_dt(),
    }
    if skip_intervals:
        report["collection_intervals"] = {"skipped": True}
    else:
        print("[data_quality] 수집 간격 샘플링 중...")
        report["collection_intervals"] = collect_intervals(sample_chargers)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Status data quality report")
    parser.add_argument("--sample-chargers", type=int, default=500)
    parser.add_argument("--skip-intervals", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    report = run_report(
        sample_chargers=args.sample_chargers,
        skip_intervals=args.skip_intervals,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"저장: {args.out}")
    print("stat_upd_dt:", report["stat_upd_dt"])
    if not args.skip_intervals:
        print("intervals:", report["collection_intervals"].get("interval_seconds"))
        print(
            "suggested_tolerance_min:",
            report["collection_intervals"].get("suggested_outcome_tolerance_min"),
        )


if __name__ == "__main__":
    main()
