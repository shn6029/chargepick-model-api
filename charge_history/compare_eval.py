# -*- coding: utf-8 -*-
"""Expanding vs Rolling(3·6개월) × naive_sw / naive_blend / ridge_base / ridge_seasonal.

월 단위 평가: 테스트 월 실측은 lag/rolling에 사용하지 않음 (month-frozen).
naive_blend fallback: blend → station×weekday → district×speed×weekday → global weekday.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from . import ARTIFACTS_DIR, MONTHS, scope_label, scope_prefix
from .features import load_or_build_features

WindowMode = Literal["expanding", "rolling_3", "rolling_6"]


def _geo_col(df: pd.DataFrame) -> str:
    """전국은 시군구명 충돌 방지를 위해 district_key 우선."""
    if "district_key" in df.columns:
        return "district_key"
    return "district"

WINDOW_MODES: list[WindowMode] = ["expanding", "rolling_3", "rolling_6"]
MODEL_NAMES = ("naive_sw", "naive_blend", "ridge_base", "ridge_seasonal")
BLEND_MIN_WEEKS = 2  # 최근 4주 동요일 중 non-null 최소 개수

SEASONAL_NUM_COLS = [
    "weekday",
    "power_kw",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "is_long_weekend",
    "sessions_lag_7d",
    "sessions_lag_14d",
    "sessions_lag_28d",
    "kwh_lag_7d",
    "kwh_lag_14d",
    "kwh_lag_28d",
    "sessions_rolling_4w_same_weekday",
    "sessions_recent_trend",
]

WEATHER_NUM_COLS = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "precipitation",
    "weather_missing",
]

LAG_FEATURE_COLS = [
    "sessions_lag_7d",
    "sessions_lag_14d",
    "sessions_lag_28d",
    "kwh_lag_7d",
    "kwh_lag_14d",
    "kwh_lag_28d",
    "sessions_rolling_4w_same_weekday",
    "kwh_rolling_4w_same_weekday",
    "sessions_recent_trend",
    "recent_weeks_used",
]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = y_true > 1e-6
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _tier_mae(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_vol: pd.Series,
) -> dict[str, float | None]:
    if train_vol.empty:
        return {"top20_mae": None, "bottom20_mae": None}
    q80 = train_vol.quantile(0.8)
    q20 = train_vol.quantile(0.2)
    stations = df["station_name"].to_numpy()
    top = np.isin(stations, train_vol[train_vol >= q80].index)
    bot = np.isin(stations, train_vol[train_vol <= q20].index)
    return {
        "top20_mae": float(np.mean(np.abs(y_true[top] - y_pred[top]))) if top.any() else None,
        "bottom20_mae": float(np.mean(np.abs(y_true[bot] - y_pred[bot]))) if bot.any() else None,
    }


def train_months_for(test_month: str, mode: WindowMode) -> list[str]:
    if test_month not in MONTHS:
        raise ValueError(f"unknown test_month: {test_month}")
    ti = MONTHS.index(test_month)
    prior = MONTHS[:ti]
    if not prior:
        return []
    if mode == "expanding":
        return list(prior)
    n = 3 if mode == "rolling_3" else 6
    return list(prior[-n:])


def month_start_ts(test_month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{test_month}-01")


def _one_hot(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_tr = enc.fit_transform(train[cols].astype(str))
    X_te = enc.transform(test[cols].astype(str))
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Month-frozen lag / rolling (no test-month leakage)
# ---------------------------------------------------------------------------


def freeze_test_lag_features(
    hist: pd.DataFrame,
    test: pd.DataFrame,
    month_start: pd.Timestamp,
) -> pd.DataFrame:
    """테스트 행 lag/rolling을 month_start 이전 이력만으로 재계산."""
    month_start = pd.Timestamp(month_start).normalize()
    h = hist[pd.to_datetime(hist["day"]).dt.normalize() < month_start].copy()
    h["day"] = pd.to_datetime(h["day"]).dt.normalize()
    out = test.copy()
    out["day"] = pd.to_datetime(out["day"]).dt.normalize()

    for c in LAG_FEATURE_COLS:
        out[c] = np.nan

    if h.empty:
        out["recent_weeks_used"] = 0
        return out

    # station → day-indexed sessions/kwh
    sess_by_st: dict[str, pd.Series] = {}
    kwh_by_st: dict[str, pd.Series] = {}
    for st, g in h.groupby("station_name", sort=False):
        g = g.sort_values("day")
        sess_by_st[st] = g.set_index("day")["n_sessions"]
        kwh_by_st[st] = g.set_index("day")["kwh"]

    # (station, weekday) → last up to 4 *observed* same-weekday values before month_start
    recent_sess: dict[tuple[str, int], list[float]] = defaultdict(list)
    recent_kwh: dict[tuple[str, int], list[float]] = defaultdict(list)
    h_obs = h[h["is_observed"]] if "is_observed" in h.columns else h
    h_sorted = h_obs.sort_values("day")
    for row in h_sorted.itertuples():
        key = (row.station_name, int(row.weekday))
        recent_sess[key].append(float(row.n_sessions))
        recent_kwh[key].append(float(row.kwh))
    for d in (recent_sess, recent_kwh):
        for k, vals in list(d.items()):
            d[k] = vals[-4:]

    lag_days = {
        "sessions_lag_7d": (7, "n_sessions"),
        "sessions_lag_14d": (14, "n_sessions"),
        "sessions_lag_28d": (28, "n_sessions"),
        "kwh_lag_7d": (7, "kwh"),
        "kwh_lag_14d": (14, "kwh"),
        "kwh_lag_28d": (28, "kwh"),
    }
    lag_vals: dict[str, dict[str, float]] = {c: {} for c in lag_days}
    for st in sess_by_st:
        for col, (offset, kind) in lag_days.items():
            series = sess_by_st[st] if kind == "n_sessions" else kwh_by_st[st]
            day = month_start - pd.Timedelta(days=offset)
            if day in series.index:
                lag_vals[col][st] = float(series.loc[day])
            else:
                # nearest prior day within densify? densify should have exact day
                lag_vals[col][st] = float("nan")

    s_lag7 = []
    s_lag14 = []
    s_lag28 = []
    k_lag7 = []
    k_lag14 = []
    k_lag28 = []
    s_roll = []
    k_roll = []
    trend = []
    weeks_used = []

    for row in out.itertuples():
        st = row.station_name
        wd = int(row.weekday)
        key = (st, wd)
        rs = recent_sess.get(key, [])
        rk = recent_kwh.get(key, [])
        n_w = len(rs)
        weeks_used.append(n_w)
        s_roll.append(float(np.mean(rs)) if n_w else float("nan"))
        k_roll.append(float(np.mean(rk)) if len(rk) else float("nan"))

        v7 = lag_vals["sessions_lag_7d"].get(st, float("nan"))
        v14 = lag_vals["sessions_lag_14d"].get(st, float("nan"))
        v28 = lag_vals["sessions_lag_28d"].get(st, float("nan"))
        s_lag7.append(v7)
        s_lag14.append(v14)
        s_lag28.append(v28)
        k_lag7.append(lag_vals["kwh_lag_7d"].get(st, float("nan")))
        k_lag14.append(lag_vals["kwh_lag_14d"].get(st, float("nan")))
        k_lag28.append(lag_vals["kwh_lag_28d"].get(st, float("nan")))
        if np.isfinite(v7) and np.isfinite(v28):
            trend.append(float(v7 - v28))
        else:
            trend.append(float("nan"))

    out["sessions_lag_7d"] = s_lag7
    out["sessions_lag_14d"] = s_lag14
    out["sessions_lag_28d"] = s_lag28
    out["kwh_lag_7d"] = k_lag7
    out["kwh_lag_14d"] = k_lag14
    out["kwh_lag_28d"] = k_lag28
    out["sessions_rolling_4w_same_weekday"] = s_roll
    out["kwh_rolling_4w_same_weekday"] = k_roll
    out["sessions_recent_trend"] = trend
    out["recent_weeks_used"] = weeks_used

    # Leakage assert: history used must be < month_start (by construction)
    assert (h["day"] < month_start).all()
    return out


def assert_no_test_month_in_frozen(
    hist: pd.DataFrame, month_start: pd.Timestamp, test_month: str
) -> None:
    """히스토리에 테스트 월이 섞이지 않았는지 확인."""
    h = hist[pd.to_datetime(hist["day"]).dt.normalize() < pd.Timestamp(month_start)]
    if "year_month" in h.columns and len(h):
        leaked = h[h["year_month"] == test_month]
        if len(leaked):
            raise AssertionError(
                f"leakage: {len(leaked)} hist rows in test_month={test_month}"
            )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def naive_sw_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """station×weekday → station → global (legacy helper)."""
    sw = train.groupby(["station_name", "weekday"])[target].mean()
    s_mean = train.groupby("station_name")[target].mean()
    global_mean = float(train[target].mean()) if len(train) else 0.0
    preds = []
    for row in test.itertuples():
        key = (row.station_name, int(row.weekday))
        if key in sw.index:
            preds.append(float(sw.loc[key]))
        elif row.station_name in s_mean.index:
            preds.append(float(s_mean.loc[row.station_name]))
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=float)


# tier codes for naive_blend (higher = coarser fallback)
TIER_BLEND = 0
TIER_STATION_WD = 1
TIER_DISTRICT_SPEED_WD = 2
TIER_SPEED_WD = 3
TIER_GLOBAL_WD = 4
TIER_GLOBAL = 5

TIER_NAMES = {
    TIER_BLEND: "blend",
    TIER_STATION_WD: "station_weekday",
    TIER_DISTRICT_SPEED_WD: "district_speed_weekday",
    TIER_SPEED_WD: "speed_weekday",
    TIER_GLOBAL_WD: "global_weekday",
    TIER_GLOBAL: "global",
}


def naive_blend_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    recent_weight: float = 0.7,
    min_weeks: int = BLEND_MIN_WEEKS,
    fallback_train: pd.DataFrame | None = None,
    use_speed_weekday: bool = False,
    return_tiers: bool = False,
) -> tuple[np.ndarray, dict[str, int]] | tuple[np.ndarray, dict[str, int], np.ndarray]:
    """blend → station×wd → geo×speed×wd → [speed×wd] → global wd.

    station_train=train 에서 blend/station 조회.
    fallback_train(기본=train) 에서 geo/speed/global 조회 (하이브리드 B용).
    """
    station_train = train
    fb = fallback_train if fallback_train is not None else train
    if "district_key" in test.columns or "district_key" in fb.columns:
        geo = "district_key"
    else:
        geo = "district"

    sw = (
        station_train.groupby(["station_name", "weekday"], sort=False)[target]
        .mean()
        .rename("sw_val")
        .reset_index()
    )
    dsw = (
        fb.groupby([geo, "speed_class", "weekday"], sort=False)[target]
        .mean()
        .rename("dsw_val")
        .reset_index()
    )
    spd = (
        fb.groupby(["speed_class", "weekday"], sort=False)[target]
        .mean()
        .rename("spd_val")
        .reset_index()
        if use_speed_weekday
        else None
    )
    gwd = fb.groupby("weekday", sort=False)[target].mean()
    global_mean = float(fb[target].mean()) if len(fb) else 0.0

    roll_col = (
        "sessions_rolling_4w_same_weekday"
        if target == "n_sessions"
        else "kwh_rolling_4w_same_weekday"
    )
    te = test.reset_index(drop=True)
    if geo not in te.columns and geo == "district_key" and "district" in te.columns:
        region = te["region"] if "region" in te.columns else "unknown"
        te = te.copy()
        te["district_key"] = (
            region.astype(str) + " " + te["district"].astype(str)
        ).str.strip()

    recent = (
        te[roll_col].to_numpy(dtype=float)
        if roll_col in te.columns
        else np.full(len(te), np.nan)
    )
    weeks = (
        te["recent_weeks_used"].to_numpy(dtype=float)
        if "recent_weeks_used" in te.columns
        else np.full(len(te), np.nan)
    )

    merge_cols = ["station_name", "weekday", geo, "speed_class"]
    for c in merge_cols:
        if c not in te.columns:
            te[c] = "unknown" if c != "weekday" else 0
    m = te[merge_cols].merge(sw, on=["station_name", "weekday"], how="left")
    m = m.merge(dsw, on=[geo, "speed_class", "weekday"], how="left")
    if spd is not None:
        m = m.merge(spd, on=["speed_class", "weekday"], how="left")
        spd_vals = m["spd_val"].to_numpy(dtype=float)
    else:
        spd_vals = np.full(len(te), np.nan)

    sw_vals = m["sw_val"].to_numpy(dtype=float)
    dsw_vals = m["dsw_val"].to_numpy(dtype=float)
    wd = te["weekday"].to_numpy(dtype=int)
    gwd_vals = pd.Series(wd).map(gwd).to_numpy(dtype=float)

    out = np.full(len(te), global_mean, dtype=float)
    tier = np.full(len(te), TIER_GLOBAL, dtype=np.int8)

    has_gwd = np.isfinite(gwd_vals)
    out[has_gwd] = gwd_vals[has_gwd]
    tier[has_gwd] = TIER_GLOBAL_WD

    if use_speed_weekday:
        has_spd = np.isfinite(spd_vals)
        out[has_spd] = spd_vals[has_spd]
        tier[has_spd] = TIER_SPEED_WD

    has_dsw = np.isfinite(dsw_vals)
    out[has_dsw] = dsw_vals[has_dsw]
    tier[has_dsw] = TIER_DISTRICT_SPEED_WD

    has_sw = np.isfinite(sw_vals)
    out[has_sw] = sw_vals[has_sw]
    tier[has_sw] = TIER_STATION_WD

    n_w = np.where(np.isfinite(weeks), weeks, 0)
    use_blend = has_sw & np.isfinite(recent) & (n_w >= min_weeks)
    out[use_blend] = (
        recent_weight * recent[use_blend]
        + (1.0 - recent_weight) * sw_vals[use_blend]
    )
    tier[use_blend] = TIER_BLEND

    counts = {
        name: int((tier == code).sum())
        for code, name in TIER_NAMES.items()
        if int((tier == code).sum()) > 0
    }
    preds = np.clip(out, 0, None)
    if return_tiers:
        return preds, counts, tier
    return preds, counts


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


def _encode_ridge_base(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    geo = _geo_col(train)
    X_cat_tr, X_cat_te = _one_hot(train, test, [geo, "speed_class"])
    st_sess = train.groupby("station_name")["n_sessions"].mean()
    st_kwh = train.groupby("station_name")["kwh"].mean()
    global_s = float(train["n_sessions"].mean())
    global_k = float(train["kwh"].mean())

    def num_block(df: pd.DataFrame) -> np.ndarray:
        wd = df["weekday"].to_numpy().astype(float)
        ang = 2 * np.pi * wd / 7.0
        pk = df["power_kw"].fillna(0).to_numpy().astype(float)
        ss = df["station_name"].map(st_sess).fillna(global_s).to_numpy()
        sk = df["station_name"].map(st_kwh).fillna(global_k).to_numpy()
        return np.column_stack([wd, np.sin(ang), np.cos(ang), pk, ss, sk])

    return np.hstack([X_cat_tr, num_block(train)]), np.hstack(
        [X_cat_te, num_block(test)]
    )


def _fill_train_medians(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    te = test.copy()
    for c in cols:
        if c not in tr.columns:
            tr[c] = np.nan
            te[c] = np.nan
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        if not np.isfinite(med):
            med = 0.0
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)
    return tr, te


def _encode_ridge_seasonal(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    geo = _geo_col(train)
    X_cat_tr, X_cat_te = _one_hot(
        train, test, [geo, "speed_class", "season"]
    )
    st_sess = train.groupby("station_name")["n_sessions"].mean()
    st_kwh = train.groupby("station_name")["kwh"].mean()
    global_s = float(train["n_sessions"].mean())
    global_k = float(train["kwh"].mean())

    num_cols = list(SEASONAL_NUM_COLS)
    if "weather_missing" in train.columns:
        miss_rate = float(train["weather_missing"].mean())
        if miss_rate < 0.95:
            num_cols = num_cols + WEATHER_NUM_COLS

    tr, te = _fill_train_medians(train, test, num_cols)

    def num_block(df: pd.DataFrame) -> np.ndarray:
        wd = df["weekday"].to_numpy().astype(float)
        ang_wd = 2 * np.pi * wd / 7.0
        pk = df["power_kw"].fillna(0).to_numpy().astype(float)
        ss = df["station_name"].map(st_sess).fillna(global_s).to_numpy()
        sk = df["station_name"].map(st_kwh).fillna(global_k).to_numpy()
        extras = []
        for c in num_cols:
            if c in ("weekday", "power_kw"):
                continue
            extras.append(df[c].to_numpy(dtype=float))
        return np.column_stack(
            [wd, np.sin(ang_wd), np.cos(ang_wd), pk, ss, sk, *extras]
        )

    X_tr = np.hstack([X_cat_tr, num_block(tr)])
    X_te = np.hstack([X_cat_te, num_block(te)])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te


def ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    variant: Literal["base", "seasonal"],
) -> np.ndarray:
    if len(train) < 50:
        return naive_sw_predict(train, test, target)
    if variant == "base":
        X_tr, X_te = _encode_ridge_base(train, test)
    else:
        X_tr, X_te = _encode_ridge_seasonal(train, test)
    y = train[target].to_numpy(dtype=float)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y)
    return np.clip(model.predict(X_te), 0, None)


def _metrics_block(
    test: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    train_vol: pd.Series,
) -> dict[str, Any]:
    return {
        "mae": _mae(y, pred),
        "mape": _mape(y, pred),
        **_tier_mae(test, y, pred, train_vol),
    }


def eval_target_mixed(
    train_obs: pd.DataFrame,
    train_dense: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    train_vol: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    """Returns (metrics_dict, naive_blend_predictions)."""
    y = test[target].to_numpy(dtype=float)
    blend_pred, blend_counts = naive_blend_predict(train_obs, test, target)
    preds = {
        "naive_sw": naive_sw_predict(train_obs, test, target),
        "naive_blend": blend_pred,
        "ridge_base": ridge_predict(train_dense, test, target, variant="base"),
        "ridge_seasonal": ridge_predict(
            train_dense, test, target, variant="seasonal"
        ),
    }
    out: dict[str, Any] = {
        "n": int(len(test)),
        "blend_fallback_counts": blend_counts,
    }
    for name, pred in preds.items():
        out[name] = _metrics_block(test, y, pred, train_vol)

    sw_mae = out["naive_sw"]["mae"]
    blend_mae = out["naive_blend"]["mae"]
    for name in ("naive_blend", "ridge_base", "ridge_seasonal"):
        m = out[name]["mae"]
        out[f"{name}_vs_naive_sw"] = (
            float((sw_mae - m) / sw_mae) if sw_mae and sw_mae > 0 else None
        )
        out[f"{name}_vs_naive_blend"] = (
            float((blend_mae - m) / blend_mae) if blend_mae and blend_mae > 0 else None
        )
    return out, blend_pred


# ---------------------------------------------------------------------------
# Segment report (expanding · naive_blend)
# ---------------------------------------------------------------------------


def _build_segment_report(residual_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not residual_rows:
        return {}
    df = pd.DataFrame(residual_rows)
    geo = "district_key" if "district_key" in df.columns else "district"
    if geo not in df.columns:
        df[geo] = "unknown"
    st = (
        df.groupby("station_name", as_index=False)
        .agg(
            mae=("abs_err", "mean"),
            n=("abs_err", "count"),
            district=(geo, "first"),
            speed_class=("speed_class", "first"),
            train_volume=("train_volume", "first"),
        )
    )
    station_maes = st["mae"].to_numpy()
    p90 = float(np.quantile(station_maes, 0.9))
    high = st[st["mae"] >= p90]

    # volume tiers from train_volume
    vol = st["train_volume"].replace(0, np.nan)
    q80 = vol.quantile(0.8)
    q20 = vol.quantile(0.2)
    top_vol = st[st["train_volume"] >= q80]["mae"]
    bot_vol = st[st["train_volume"] <= q20]["mae"]

    by_speed = (
        st.groupby("speed_class")["mae"].agg(["mean", "median", "count"]).reset_index()
    )
    by_district = (
        st.groupby("district")["mae"].agg(["mean", "median", "count"]).reset_index()
    )
    by_district = by_district.sort_values("mean", ascending=False)
    worst5 = by_district.head(5)
    best5 = (
        by_district.sort_values("mean", ascending=True)
        .loc[lambda d: ~d["district"].isin(worst5["district"])]
        .head(5)
    )

    return {
        "n_stations": int(len(st)),
        "station_mae_median": float(np.median(station_maes)),
        "station_mae_mean": float(np.mean(station_maes)),
        "station_mae_p90": p90,
        "high_error_top10pct_mae_mean": float(high["mae"].mean()) if len(high) else None,
        "high_error_top10pct_n": int(len(high)),
        "high_volume_mae_mean": float(top_vol.mean()) if len(top_vol) else None,
        "low_volume_mae_mean": float(bot_vol.mean()) if len(bot_vol) else None,
        "by_speed_class": [
            {
                "speed_class": str(rec["speed_class"]),
                "mae_mean": float(rec["mean"]),
                "mae_median": float(rec["median"]),
                "n_stations": int(rec["count"]),
            }
            for rec in by_speed.to_dict("records")
        ],
        "by_district_worst5": [
            {
                "district": str(rec["district"]),
                "mae_mean": float(rec["mean"]),
                "mae_median": float(rec["median"]),
                "n_stations": int(rec["count"]),
            }
            for rec in worst5.to_dict("records")
        ],
        "by_district_best5": [
            {
                "district": str(rec["district"]),
                "mae_mean": float(rec["mean"]),
                "mae_median": float(rec["median"]),
                "n_stations": int(rec["count"]),
            }
            for rec in best5.to_dict("records")
        ],
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_compare_eval(
    panel: pd.DataFrame | None = None,
    *,
    force_features: bool = False,
    out_json: Path | None = None,
    out_md: Path | None = None,
    eval_observed_only: bool = True,
    scope: str = "daegu",
) -> dict[str, Any]:
    feat = load_or_build_features(
        force=force_features, panel=panel, scope=scope
    )
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()
    if "district_key" not in feat.columns and "district" in feat.columns:
        dist = feat["district"].astype(str)
        if "region" in feat.columns:
            feat["district_key"] = (
                feat["region"].astype(str) + " " + dist
            ).str.strip()
        else:
            feat["district_key"] = dist
    geo = _geo_col(feat)

    by_window: dict[str, list[dict[str, Any]]] = {m: [] for m in WINDOW_MODES}
    # expanding · n_sessions · naive_blend residuals for segment report
    residual_rows: list[dict[str, Any]] = []
    blend_counts_total: Counter[str] = Counter()

    for mode in WINDOW_MODES:
        for i in range(len(MONTHS) - 1):
            test_month = MONTHS[i + 1]
            train_months = train_months_for(test_month, mode)
            if not train_months:
                by_window[mode].append(
                    {
                        "train_months": train_months,
                        "test_month": test_month,
                        "window": mode,
                        "skipped": True,
                        "reason": "empty train months",
                    }
                )
                continue

            m_start = month_start_ts(test_month)
            # History for frozen lags: all days before test month (full panel)
            hist = feat[feat["day"] < m_start]
            assert_no_test_month_in_frozen(hist, m_start, test_month)

            train_dense = feat[feat["year_month"].isin(train_months)].copy()
            test_raw = feat[feat["year_month"] == test_month].copy()
            if eval_observed_only:
                test_raw = test_raw[test_raw["is_observed"]]
            train_obs = train_dense[train_dense["is_observed"]]

            if train_dense.empty or test_raw.empty or train_obs.empty:
                by_window[mode].append(
                    {
                        "train_months": train_months,
                        "train_through": train_months[-1],
                        "test_month": test_month,
                        "window": mode,
                        "skipped": True,
                        "reason": "empty train or test",
                    }
                )
                continue

            # Month-frozen lag features on test (no test-month leakage)
            test = freeze_test_lag_features(hist, test_raw, m_start)
            # Train lag features: already from panel shift; OK within train months.
            # For ridge seasonal consistency near month edge, freeze is only needed on test.

            train_vol = train_obs.groupby("station_name")["n_sessions"].sum()
            ns_metrics, blend_pred = eval_target_mixed(
                train_obs, train_dense, test, "n_sessions", train_vol
            )
            kwh_metrics, _ = eval_target_mixed(
                train_obs, train_dense, test, "kwh", train_vol
            )

            if mode == "expanding":
                for k, v in (ns_metrics.get("blend_fallback_counts") or {}).items():
                    blend_counts_total[k] += int(v)
                y = test["n_sessions"].to_numpy(dtype=float)
                abs_err = np.abs(y - blend_pred)
                for j, row in enumerate(test.itertuples()):
                    residual_rows.append(
                        {
                            "station_name": row.station_name,
                            "district": getattr(row, "district", "unknown"),
                            "district_key": getattr(row, geo, getattr(row, "district", "unknown")),
                            "speed_class": row.speed_class,
                            "abs_err": float(abs_err[j]),
                            "train_volume": float(
                                train_vol.get(row.station_name, 0.0)
                            ),
                            "test_month": test_month,
                        }
                    )

            fold: dict[str, Any] = {
                "train_months": train_months,
                "train_through": train_months[-1],
                "test_month": test_month,
                "window": mode,
                "month_frozen_lags": True,
                "n_train_rows": int(len(train_dense)),
                "n_train_observed": int(len(train_obs)),
                "n_test_rows": int(len(test)),
                "n_train_stations": int(train_dense["station_name"].nunique()),
                "n_test_stations": int(test["station_name"].nunique()),
                "n_sessions": ns_metrics,
                "kwh": kwh_metrics,
                "skipped": False,
            }
            by_window[mode].append(fold)
            ns = fold["n_sessions"]
            print(
                f"[{mode}] train≤{train_months[-1]} ({len(train_months)}m) "
                f"→ {test_month} | "
                f"sw={ns['naive_sw']['mae']:.3f} "
                f"blend={ns['naive_blend']['mae']:.3f} "
                f"base={ns['ridge_base']['mae']:.3f} "
                f"seas={ns['ridge_seasonal']['mae']:.3f} "
                f"fb={ns.get('blend_fallback_counts')}"
            )

    segment = _build_segment_report(residual_rows)
    summary = _summarize_all(by_window)
    summary["blend_fallback_counts_expanding"] = dict(blend_counts_total)
    summary["segment_report_expanding_naive_blend"] = segment

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "세션 수요 비교 검증: expanding vs rolling_3/6, "
            "naive_sw / naive_blend(fallback) / ridge_base / ridge_seasonal. "
            "테스트 월 lag는 month-frozen. ETA 가용확률 모델과 별개."
        ),
        "scope": scope_label(scope),
        "scope_prefix": scope_prefix(scope),
        "months": MONTHS,
        "eval_observed_only": eval_observed_only,
        "month_frozen_lags": True,
        "blend_min_weeks": BLEND_MIN_WEEKS,
        "windows": WINDOW_MODES,
        "models": list(MODEL_NAMES),
        "folds_by_window": by_window,
        "summary": summary,
    }

    prefix = scope_prefix(scope)
    suffix = "" if prefix == "daegu" else f"_{prefix}"
    date_tag = datetime.now().strftime("%m%d")
    out_json = out_json or (
        ARTIFACTS_DIR / f"charge_history_compare_eval{suffix}.json"
    )
    out_md = out_md or (
        ARTIFACTS_DIR / f"charge_history_compare_eval{suffix}_{date_tag}.md"
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[compare_eval] wrote {out_json}")
    print(f"[compare_eval] wrote {out_md}")

    # Refresh human summary (대구 결과와 분리)
    write_weather_eval_summary(result)
    return result


def _summarize_all(
    by_window: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode, folds in by_window.items():
        active = [f for f in folds if not f.get("skipped")]
        block: dict[str, Any] = {"n_active_folds": len(active)}
        if not active:
            summary[mode] = block
            continue

        for target in ("n_sessions", "kwh"):
            for model in MODEL_NAMES:
                maes = [
                    float(f[target][model]["mae"])
                    for f in active
                    if f[target][model].get("mae") is not None
                ]
                block[f"{target}_{model}_mae_mean"] = (
                    float(np.mean(maes)) if maes else None
                )

            blend_beats_sw = sum(
                1
                for f in active
                if (f[target].get("naive_blend_vs_naive_sw") or 0) > 0
            )
            seas_beats_blend = sum(
                1
                for f in active
                if (f[target].get("ridge_seasonal_vs_naive_blend") or 0) > 0
            )
            seas_beats_sw = sum(
                1
                for f in active
                if (f[target].get("ridge_seasonal_vs_naive_sw") or 0) > 0
            )
            if target == "n_sessions":
                block["naive_blend_beat_sw_folds"] = blend_beats_sw
                block["ridge_seasonal_beat_blend_folds"] = seas_beats_blend
                block["ridge_seasonal_beat_sw_folds"] = seas_beats_sw

        summary[mode] = block
    return summary


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 충전이력 Expanding vs Rolling 비교 검증",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        f"범위: **{result['scope']}** · observed-only: `{result['eval_observed_only']}` · "
        f"month-frozen lags: `{result.get('month_frozen_lags')}`",
        "",
        "## 요약 (n_sessions MAE 평균)",
        "",
        "| window | naive_sw | naive_blend | ridge_base | ridge_seasonal | blend>sw | seas>blend |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in result["windows"]:
        s = (result.get("summary") or {}).get(mode) or {}
        lines.append(
            "| {w} | {sw} | {bl} | {rb} | {rs} | {b}/{n} | {sb}/{n} |".format(
                w=mode,
                sw=_fmt(s.get("n_sessions_naive_sw_mae_mean")),
                bl=_fmt(s.get("n_sessions_naive_blend_mae_mean")),
                rb=_fmt(s.get("n_sessions_ridge_base_mae_mean")),
                rs=_fmt(s.get("n_sessions_ridge_seasonal_mae_mean")),
                b=s.get("naive_blend_beat_sw_folds", "—"),
                sb=s.get("ridge_seasonal_beat_blend_folds", "—"),
                n=s.get("n_active_folds", 0),
            )
        )

    fb = (result.get("summary") or {}).get("blend_fallback_counts_expanding") or {}
    if fb:
        lines.extend(
            [
                "",
                "## naive_blend fallback 사용 (expanding, n_sessions)",
                "",
                "| tier | n |",
                "|---|---:|",
            ]
        )
        for k in (
            "blend",
            "station_weekday",
            "district_speed_weekday",
            "global_weekday",
            "global",
        ):
            if k in fb:
                lines.append(f"| {k} | {fb[k]} |")

    seg = (result.get("summary") or {}).get(
        "segment_report_expanding_naive_blend"
    ) or {}
    if seg:
        lines.extend(
            [
                "",
                "## 충전소별 성능 분포 (expanding · naive_blend · sessions)",
                "",
                f"- station MAE 중앙값: **{_fmt(seg.get('station_mae_median'))}**",
                f"- station MAE 평균: {_fmt(seg.get('station_mae_mean'))}",
                f"- 상위 10% 고오차 (P90={_fmt(seg.get('station_mae_p90'))}): "
                f"평균 MAE **{_fmt(seg.get('high_error_top10pct_mae_mean'))}** "
                f"(n={seg.get('high_error_top10pct_n')})",
                f"- 이용량 상위 20% 충전소 MAE: {_fmt(seg.get('high_volume_mae_mean'))}",
                f"- 이용량 하위 20% 충전소 MAE: {_fmt(seg.get('low_volume_mae_mean'))}",
                "",
                "### 급속 vs 완속 (speed_class)",
                "",
                "| speed_class | MAE mean | MAE median | n_stations |",
                "|---|---:|---:|---:|",
            ]
        )
        for r in seg.get("by_speed_class") or []:
            lines.append(
                f"| {r['speed_class']} | {r['mae_mean']:.3f} | "
                f"{r['mae_median']:.3f} | {r['n_stations']} |"
            )
        lines.extend(
            [
                "",
                "### 구·군 (MAE 높은 5 / 낮은 5)",
                "",
                "| district | MAE mean | n_stations |",
                "|---|---:|---:|",
            ]
        )
        for r in (seg.get("by_district_worst5") or []) + (
            seg.get("by_district_best5") or []
        ):
            lines.append(
                f"| {r['district']} | {r['mae_mean']:.3f} | {r['n_stations']} |"
            )

    for mode in result["windows"]:
        lines.extend(
            [
                "",
                f"## 폴드별 · {mode} (n_sessions MAE)",
                "",
                "| train≤ | #mo | test | sw | blend | base | seas |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for f in result["folds_by_window"][mode]:
            if f.get("skipped"):
                lines.append(
                    f"| {f.get('train_through', '—')} | — | {f['test_month']} | — | — | — | skipped |"
                )
                continue
            ns = f["n_sessions"]
            lines.append(
                f"| {f['train_through']} | {len(f['train_months'])} | {f['test_month']} | "
                f"{ns['naive_sw']['mae']:.3f} | {ns['naive_blend']['mae']:.3f} | "
                f"{ns['ridge_base']['mae']:.3f} | {ns['ridge_seasonal']['mae']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## 모델",
            "",
            "- **naive_sw:** station×weekday 평균",
            "- **naive_blend:** 0.7×최근4주 동요일(≥2주) + 0.3×sw "
            "→ district×speed×weekday → 전체 weekday",
            "- **ridge_base:** district·speed_class + weekday 주기 + station 평균 + power_kw",
            "- **ridge_seasonal:** base + month_sin/cos·season·holiday·month-frozen lag·날씨",
            "",
            "## 참고",
            "",
            "- 테스트 MAE는 `is_observed` 행만",
            "- lag/rolling은 **테스트 월 1일 이전** 이력만 사용 (month-frozen)",
            (
                "- 기상: 대구 ASOS 143 일별 대표값 (충전소별 미세 기상 미반영)"
                if result.get("scope_prefix") == "daegu"
                else "- 기상: 시도 대표 ASOS (region→stn) 일별 기온·강수"
            ),
            "",
            "산출 JSON: `charge_history_compare_eval.json`",
            "",
        ]
    )
    return "\n".join(lines)


def write_weather_eval_summary(result: dict[str, Any]) -> Path:
    """발표용 요약 MD 갱신."""
    s = result.get("summary") or {}
    prefix = result.get("scope_prefix") or "daegu"
    suffix = "" if prefix == "daegu" else f"_{prefix}"
    date_tag = datetime.now().strftime("%m%d")
    path = ARTIFACTS_DIR / f"charge_history_weather_eval_summary{suffix}_{date_tag}.md"
    weather_note = (
        "기상청 ASOS Open API · 대구(143) · 2025-01 ~ 2026-05"
        if prefix == "daegu"
        else "기상청 ASOS Open API · 시도 대표 지점(region→stn) · weather_missing=0 목표"
    )
    lines = [
        "# 충전이력 수요 예측 — 날씨 포함 재평가 요약"
        if prefix == "daegu"
        else "# 충전이력 수요 예측 — 전국 비교 검증 요약",
        "",
        f"**생성 기준:** `charge_history_compare_eval{suffix}` ({result.get('generated_at')})",
        f"**날씨:** {weather_note}",
        f"**범위:** {result.get('scope')} · observed-only · month-frozen lags=`{result.get('month_frozen_lags')}`",
        "",
        "> ETA 가용확률(status) 모델과 별개. 세션 이력 기반 일별 `n_sessions` / `kWh` 예측.",
        "",
        "---",
        "",
        "## 세션 MAE 평균 (낮을수록 좋음)",
        "",
        "| window | naive_sw | naive_blend | ridge_base | ridge_seasonal |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in WINDOW_MODES:
        b = s.get(mode) or {}
        lines.append(
            "| {w} | {sw} | {bl} | {rb} | {rs} |".format(
                w=mode,
                sw=_fmt(b.get("n_sessions_naive_sw_mae_mean")),
                bl=_fmt(b.get("n_sessions_naive_blend_mae_mean")),
                rb=_fmt(b.get("n_sessions_ridge_base_mae_mean")),
                rs=_fmt(b.get("n_sessions_ridge_seasonal_mae_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## kWh MAE 평균",
            "",
            "| window | naive_sw | naive_blend | ridge_base | ridge_seasonal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in WINDOW_MODES:
        b = s.get(mode) or {}
        lines.append(
            "| {w} | {sw} | {bl} | {rb} | {rs} |".format(
                w=mode,
                sw=_fmt(b.get("kwh_naive_sw_mae_mean")),
                bl=_fmt(b.get("kwh_naive_blend_mae_mean")),
                rb=_fmt(b.get("kwh_ridge_base_mae_mean")),
                rs=_fmt(b.get("kwh_ridge_seasonal_mae_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## 폴드 승패 (세션, 16 folds)",
            "",
            "| window | blend이 sw 이김 | seasonal이 blend 이김 | seasonal이 sw 이김 |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in WINDOW_MODES:
        b = s.get(mode) or {}
        n = b.get("n_active_folds", 0)
        lines.append(
            f"| {mode} | {b.get('naive_blend_beat_sw_folds')}/{n} | "
            f"{b.get('ridge_seasonal_beat_blend_folds')}/{n} | "
            f"{b.get('ridge_seasonal_beat_sw_folds')}/{n} |"
        )

    fb = s.get("blend_fallback_counts_expanding") or {}
    if fb:
        lines.extend(
            [
                "",
                "## naive_blend fallback (expanding)",
                "",
                "| tier | n |",
                "|---|---:|",
            ]
        )
        for k in (
            "blend",
            "station_weekday",
            "district_speed_weekday",
            "global_weekday",
            "global",
        ):
            if k in fb:
                lines.append(f"| {k} | {fb[k]} |")

    seg = s.get("segment_report_expanding_naive_blend") or {}
    if seg:
        lines.extend(
            [
                "",
                "## 충전소별 성능 분포 (expanding · naive_blend)",
                "",
                "| 지표 | 값 |",
                "|---|---|",
                f"| station MAE 중앙값 | {_fmt(seg.get('station_mae_median'))} |",
                f"| station MAE 평균 | {_fmt(seg.get('station_mae_mean'))} |",
                f"| 상위 10% 고오차 MAE (P90 이상) | {_fmt(seg.get('high_error_top10pct_mae_mean'))} "
                f"(n={seg.get('high_error_top10pct_n')}) |",
                f"| 이용량 상위 20% MAE | {_fmt(seg.get('high_volume_mae_mean'))} |",
                f"| 이용량 하위 20% MAE | {_fmt(seg.get('low_volume_mae_mean'))} |",
                "",
                "### 급속 vs 완속",
                "",
                "| speed_class | MAE mean | MAE median | n |",
                "|---|---:|---:|---:|",
            ]
        )
        for r in seg.get("by_speed_class") or []:
            lines.append(
                f"| {r['speed_class']} | {r['mae_mean']:.3f} | "
                f"{r['mae_median']:.3f} | {r['n_stations']} |"
            )
        lines.extend(
            [
                "",
                "### 구·군 (오차 높은 5곳 / 낮은 5곳)",
                "",
                "| district | MAE mean | n |",
                "|---|---:|---:|",
            ]
        )
        for r in (seg.get("by_district_worst5") or []) + (
            seg.get("by_district_best5") or []
        ):
            lines.append(
                f"| {r['district']} | {r['mae_mean']:.3f} | {r['n_stations']} |"
            )

    lines.extend(
        [
            "",
            "## 모델 정의",
            "",
            "| 모델 | 설명 |",
            "|---|---|",
            "| **naive_sw** | station × weekday 평균 |",
            "| **naive_blend** | 0.7×최근4주 동요일(≥2) + 0.3×sw → district×speed×wd → 전체 wd |",
            "| **ridge_base** | district · speed_class + weekday 주기 + station 평균 + power_kw |",
            "| **ridge_seasonal** | base + month_sin/cos · season · holiday · month-frozen lag · 날씨 |",
            "",
            "### 데이터 누수 방지",
            "",
            "- 월별 평가에서 lag·rolling·최근 4주 평균은 **테스트 월 1일 이전** 이력만 사용한다.",
            "- 예: 2026-05 예측 시 2026-05 실측 세션은 피처에 포함하지 않는다.",
            "",
            "### 날씨 데이터의 한계",
            "",
            (
                "기상 데이터는 **대구 ASOS 대표 관측소(143)의 일별 값**을 사용했으며, "
                "충전소별 미세 기상 차이는 반영하지 않았다."
                if prefix == "daegu"
                else "전국 스코프에서는 **시도(region) → 대표 ASOS 지점** 매핑으로 "
                "일별 기온·강수를 붙였다. 시군구·충전소 단위 미세 기상은 반영하지 않는다."
            ),
            "",
            "---",
            "",
            "## 최종 평가",
            "",
        ]
    )

    exp = s.get("expanding") or {}
    blend_m = exp.get("n_sessions_naive_blend_mae_mean")
    seas_m = exp.get("n_sessions_ridge_seasonal_mae_mean")
    sw_m = exp.get("n_sessions_naive_sw_mae_mean")
    base_m = exp.get("n_sessions_ridge_base_mae_mean")
    if blend_m is not None:
        lines.append(
            "월별 확장 검증(테스트 월 실측 미사용, month-frozen) 결과 기준 expanding 세션 MAE는 "
            f"naive_blend {_fmt(blend_m)} · ridge_seasonal {_fmt(seas_m)} · "
            f"naive_sw {_fmt(sw_m)} · ridge_base {_fmt(base_m)} 이다. "
            "상세 폴드·세그먼트는 compare_eval 산출물을 본다."
        )
    else:
        lines.append("요약 수치가 비어 있다. compare_eval JSON을 확인한다.")

    art_suffix = "" if prefix == "daegu" else f"_{prefix}"
    date_tag = datetime.now().strftime("%m%d")
    lines.extend(
        [
            "",
            "---",
            "",
            "## 관련 산출물",
            "",
            f"- 상세 폴드표: `recommend_api/artifacts/charge_history_compare_eval{art_suffix}_{date_tag}.md`",
            f"- JSON: `recommend_api/artifacts/charge_history_compare_eval{art_suffix}.json`",
            (
                "- 날씨 캐시: `charge_history/cache/daegu_asos_daily.parquet`"
                if prefix == "daegu"
                else "- 날씨 캐시: `charge_history/cache/nationwide_asos_daily.parquet`"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare_eval] wrote {path}")
    return path


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"
