"""급속/완속 horizon 샘플 커버리지(고유 충전기·충전소·밀도)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _segment_coverage(df: pd.DataFrame, label: str) -> dict[str, Any]:
    if df.empty or "stat_id" not in df.columns or "chger_id" not in df.columns:
        return {
            "label": label,
            "n_horizon_rows": int(len(df)),
            "n_unique_chargers": 0,
            "n_unique_stations": 0,
            "samples_per_charger_p25": None,
            "samples_per_charger_median": None,
            "samples_per_charger_p75": None,
        }
    keys = df.groupby(["stat_id", "chger_id"], sort=False).size()
    stations = df["stat_id"].nunique()
    counts = keys.to_numpy(dtype=float)
    return {
        "label": label,
        "n_horizon_rows": int(len(df)),
        "n_unique_chargers": int(len(keys)),
        "n_unique_stations": int(stations),
        "samples_per_charger_p25": float(np.percentile(counts, 25)),
        "samples_per_charger_median": float(np.median(counts)),
        "samples_per_charger_p75": float(np.percentile(counts, 75)),
    }


def compute_speed_coverage(
    horizon_df: pd.DataFrame,
    *,
    is_fast: pd.Series | None = None,
    test_mask: pd.Series | None = None,
) -> dict[str, Any]:
    """전체 / 급속 / 완속 커버리지. test_mask가 있으면 holdout test도 집계."""
    hz = horizon_df.copy()
    if is_fast is not None:
        hz["is_fast"] = pd.to_numeric(is_fast, errors="coerce").fillna(0).astype(int)
    elif "is_fast" not in hz.columns:
        raise ValueError("is_fast 컬럼이 필요합니다.")

    rapid = hz[hz["is_fast"] == 1]
    slow = hz[hz["is_fast"] == 0]
    out: dict[str, Any] = {
        "all": _segment_coverage(hz, "all"),
        "rapid": _segment_coverage(rapid, "rapid"),
        "slow": _segment_coverage(slow, "slow"),
        "note": (
            "n_horizon_rows = 충전기×수집시점×ETA 샘플 수. "
            "고유 충전기/충전소와 충전기당 샘플 중앙값으로 대표성을 확인."
        ),
    }
    if test_mask is not None:
        tm = test_mask.reindex(hz.index).fillna(False).astype(bool)
        hz_te = hz.loc[tm]
        out["holdout_test"] = {
            "all": _segment_coverage(hz_te, "holdout_test_all"),
            "rapid": _segment_coverage(hz_te[hz_te["is_fast"] == 1], "holdout_test_rapid"),
            "slow": _segment_coverage(hz_te[hz_te["is_fast"] == 0], "holdout_test_slow"),
        }
    return out
