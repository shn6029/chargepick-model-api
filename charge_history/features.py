# -*- coding: utf-8 -*-
"""station×day densify + calendar / holiday / lag / weather 피처."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CACHE_DIR, SCHEDULER_ROOT, scope_prefix
from .panel import load_or_build_panel

try:
    import holidays
except ImportError:  # pragma: no cover
    holidays = None

WEATHER_STN_ID = "143"  # 대구


def features_parquet_path(
    cache_dir: Path = CACHE_DIR, *, scope: str = "daegu"
) -> Path:
    return cache_dir / f"{scope_prefix(scope)}_station_day_features.parquet"


def get_kr_holidays(years: set[int]) -> set:
    if holidays is None or not years:
        return set()
    kr = holidays.country_holidays("KR", years=sorted(years))
    return set(kr.keys())


def _db_config() -> dict:
    """DB 접속 설정 — 전부 환경변수. 폴백 접속정보를 두지 않는다.

    구조가 `try/except` 였는데, except 쪽에 실제 접속정보가 박혀 있어서
    dotenv 임포트가 실패하면 **운영 DB로 조용히 붙었다.** 지금은 두 갈래가
    같은 곳(환경변수)을 보고, 비어 있으면 연결에서 실패한다.
    """
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv(SCHEDULER_ROOT / ".env")
    except ImportError:
        pass  # dotenv 가 없어도 환경변수가 이미 있으면 동작한다

    return {
        "host": os.getenv("DB_HOST", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", ""),
        "charset": "utf8mb4",
    }


def load_daily_weather(
    day_min: pd.Timestamp | None = None,
    day_max: pd.Timestamp | None = None,
    *,
    scope: str = "daegu",
) -> pd.DataFrame:
    """ASOS 일별 기온/강수.

    daegu: 지점 143 (API 캐시 → MySQL weather)
    nationwide: 시도 대표 지점들 (stn_id 포함)
    """
    from . import scope_prefix
    from .weather_asos import load_or_fetch_daily

    pref = scope_prefix(scope)
    try:
        daily = load_or_fetch_daily(
            day_min=day_min,
            day_max=day_max,
            fetch_if_missing=True,
            scope=pref,
        )
        if not daily.empty:
            ok = int(daily["temperature_mean"].notna().sum())
            n_stn = (
                daily["stn_id"].nunique() if "stn_id" in daily.columns else 1
            )
            print(
                f"[features] weather from ASOS API cache: "
                f"{ok} day-rows stn={n_stn} scope={pref}"
            )
            return daily
    except Exception as exc:
        print(f"[features] ASOS weather failed: {exc}")

    if pref == "nationwide":
        return pd.DataFrame()

    # daegu fallback: DB
    try:
        import pymysql
    except ImportError:
        print("[features] pymysql 없음 → weather skip")
        return pd.DataFrame()

    try:
        conn = pymysql.connect(**_db_config(), connect_timeout=40)
        try:
            q = """
                SELECT observed_at, ta, rn
                FROM weather
                WHERE stn_id = %s
            """
            weather = pd.read_sql(q, conn, params=(WEATHER_STN_ID,))
        finally:
            conn.close()
    except Exception as exc:
        print(f"[features] weather DB load failed: {exc}")
        return pd.DataFrame()

    if weather.empty:
        return pd.DataFrame()

    weather["observed_at"] = pd.to_datetime(weather["observed_at"], errors="coerce")
    weather = weather.dropna(subset=["observed_at"])
    weather["day"] = weather["observed_at"].dt.normalize()
    weather["ta"] = pd.to_numeric(weather["ta"], errors="coerce")
    weather["rn"] = pd.to_numeric(weather["rn"], errors="coerce")

    if day_min is not None:
        weather = weather[weather["day"] >= pd.Timestamp(day_min).normalize()]
    if day_max is not None:
        weather = weather[weather["day"] <= pd.Timestamp(day_max).normalize()]

    daily = (
        weather.groupby("day", as_index=False)
        .agg(
            temperature_mean=("ta", "mean"),
            temperature_min=("ta", "min"),
            temperature_max=("ta", "max"),
            precipitation=("rn", "sum"),
        )
    )
    print(f"[features] weather from DB: {len(daily)} days")
    return daily


def densify_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """충전소별 관측 구간 calendar densify (이용 없음 = 0)."""
    if panel.empty:
        return panel.copy()

    p = panel.copy()
    p["day"] = pd.to_datetime(p["day"]).dt.normalize()
    p["is_observed"] = True

    if "region" not in p.columns:
        p["region"] = "unknown"
    if "district_key" not in p.columns:
        p["district_key"] = (
            p["region"].astype(str) + " " + p["district"].astype(str)
        ).str.strip()

    meta = (
        p.groupby("station_name", as_index=False)
        .agg(
            region=("region", "first"),
            district=("district", "first"),
            district_key=("district_key", "first"),
            speed_class=("speed_class", "first"),
            power_kw=("power_kw", "first"),
            day_min=("day", "min"),
            day_max=("day", "max"),
        )
    )

    frames: list[pd.DataFrame] = []
    for row in meta.itertuples(index=False):
        days = pd.date_range(row.day_min, row.day_max, freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "station_name": row.station_name,
                    "day": days,
                    "region": row.region,
                    "district": row.district,
                    "district_key": row.district_key,
                    "speed_class": row.speed_class,
                    "power_kw": row.power_kw,
                }
            )
        )
    grid = pd.concat(frames, ignore_index=True)

    keep = [
        "station_name",
        "day",
        "n_sessions",
        "kwh",
        "duration_min_mean",
        "is_observed",
    ]
    sparse = p[[c for c in keep if c in p.columns]]
    dense = grid.merge(sparse, on=["station_name", "day"], how="left")
    dense["n_sessions"] = dense["n_sessions"].fillna(0.0)
    dense["kwh"] = dense["kwh"].fillna(0.0)
    dense["is_observed"] = dense["is_observed"].fillna(False).astype(bool)
    dense["year_month"] = dense["day"].dt.strftime("%Y-%m")
    dense["weekday"] = dense["day"].dt.weekday.astype(int)
    return dense.sort_values(["station_name", "day"]).reset_index(drop=True)


def add_calendar_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day = pd.to_datetime(out["day"])
    month = day.dt.month.astype(int)
    out["month"] = month
    ang = 2 * np.pi * month / 12.0
    out["month_sin"] = np.sin(ang)
    out["month_cos"] = np.cos(ang)
    # 1=봄(3-5), 2=여름(6-8), 3=가을(9-11), 4=겨울(12,1,2)
    out["season"] = np.where(
        month.isin([3, 4, 5]),
        1,
        np.where(
            month.isin([6, 7, 8]),
            2,
            np.where(month.isin([9, 10, 11]), 3, 4),
        ),
    ).astype(int)
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    years = set(day.dt.year.dropna().astype(int).unique())
    holiday_set = get_kr_holidays(years)
    dates = day.dt.date
    out["is_holiday"] = dates.map(lambda d: 1 if d in holiday_set else 0).astype(int)

    # eve: 다음 날이 공휴일
    next_day = (day + pd.Timedelta(days=1)).dt.date
    out["is_holiday_eve"] = next_day.map(lambda d: 1 if d in holiday_set else 0).astype(
        int
    )

    # long weekend: 금·월이 공휴일이거나, 토·일 인접 공휴일
    wd = out["weekday"].to_numpy()
    hol = out["is_holiday"].to_numpy()
    is_long = (
        ((wd == 4) & (hol == 1))  # Friday holiday
        | ((wd == 0) & (hol == 1))  # Monday holiday
        | ((wd == 5) & (hol == 1))
        | ((wd == 6) & (hol == 1))
    )
    eve = out["is_holiday_eve"].to_numpy()
    is_long = is_long | ((wd == 4) & (eve == 1))
    out["is_long_weekend"] = is_long.astype(int)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """station별 시간순 shift — leakage 없음."""
    out = df.sort_values(["station_name", "day"]).copy()
    g = out.groupby("station_name", sort=False)

    for lag in (7, 14, 28):
        out[f"sessions_lag_{lag}d"] = g["n_sessions"].shift(lag)
        out[f"kwh_lag_{lag}d"] = g["kwh"].shift(lag)

    # 최근 4주 같은 요일 평균: lag 7,14,21,28
    same_wd_s = pd.concat(
        [g["n_sessions"].shift(7 * k) for k in (1, 2, 3, 4)],
        axis=1,
    )
    same_wd_k = pd.concat(
        [g["kwh"].shift(7 * k) for k in (1, 2, 3, 4)],
        axis=1,
    )
    out["sessions_rolling_4w_same_weekday"] = same_wd_s.mean(axis=1)
    out["kwh_rolling_4w_same_weekday"] = same_wd_k.mean(axis=1)

    out["sessions_recent_trend"] = out["sessions_lag_7d"] - out["sessions_lag_28d"]
    return out


def attach_weather(
    df: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    *,
    scope: str = "daegu",
) -> pd.DataFrame:
    """daegu: day 기준 merge / nationwide: region→stn_id 후 (stn_id, day) merge."""
    from . import scope_prefix
    from .weather_asos import region_to_stn_id

    out = df.copy()
    out["day"] = pd.to_datetime(out["day"]).dt.normalize()
    pref = scope_prefix(scope)

    if weather is None:
        weather = load_daily_weather(
            out["day"].min(), out["day"].max(), scope=pref
        )
    if weather is None or weather.empty:
        out["temperature_mean"] = np.nan
        out["temperature_min"] = np.nan
        out["temperature_max"] = np.nan
        out["precipitation"] = np.nan
        out["weather_missing"] = 1
        out["asos_stn_id"] = None
        return out

    w = weather.copy()
    w["day"] = pd.to_datetime(w["day"]).dt.normalize()

    if pref == "nationwide":
        if "region" not in out.columns:
            out["region"] = "unknown"
        out["asos_stn_id"] = out["region"].map(region_to_stn_id)
        if "stn_id" not in w.columns:
            print("[features] nationwide weather missing stn_id → weather skip")
            out["temperature_mean"] = np.nan
            out["temperature_min"] = np.nan
            out["temperature_max"] = np.nan
            out["precipitation"] = np.nan
            out["weather_missing"] = 1
            return out
        w = w.rename(columns={"stn_id": "asos_stn_id"})
        w["asos_stn_id"] = w["asos_stn_id"].astype(str)
        out["asos_stn_id"] = out["asos_stn_id"].astype("string")
        out = out.merge(w, on=["asos_stn_id", "day"], how="left")
    else:
        # 대구: day만 조인 (기존과 동일). stn_id 컬럼이 있으면 143만 사용
        if "stn_id" in w.columns:
            w = w[w["stn_id"].astype(str) == WEATHER_STN_ID].drop(
                columns=["stn_id"], errors="ignore"
            )
        out["asos_stn_id"] = WEATHER_STN_ID
        out = out.merge(w, on="day", how="left")

    out["weather_missing"] = out["temperature_mean"].isna().astype(int)
    # 추가 날씨 컬럼: 없으면 NaN 유지
    for col in ("humidity_mean", "wind_speed_mean", "snowfall_sum"):
        if col not in out.columns:
            out[col] = float("nan")
    return out


def build_feature_panel(
    panel: pd.DataFrame | None = None,
    *,
    scope: str = "daegu",
) -> pd.DataFrame:
    if panel is None:
        panel = load_or_build_panel(scope=scope)
    dense = densify_panel(panel)
    dense = add_calendar_holiday_features(dense)
    dense = add_lag_features(dense)
    dense = attach_weather(dense, scope=scope)
    if holidays is None:
        print("[features] 경고: holidays 미설치 → is_holiday=0 고정. pip install holidays")
    print(
        f"[features] rows={len(dense):,} stations={dense['station_name'].nunique():,} "
        f"observed={int(dense['is_observed'].sum()):,} "
        f"weather_ok={int((dense['weather_missing'] == 0).sum()):,} "
        f"scope={scope_prefix(scope)}"
    )
    return dense


def load_or_build_features(
    *,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    panel: pd.DataFrame | None = None,
    scope: str = "daegu",
) -> pd.DataFrame:
    path = features_parquet_path(cache_dir, scope=scope)
    if path.exists() and not force and panel is None:
        return pd.read_parquet(path)
    if panel is None:
        panel = load_or_build_panel(cache_dir=cache_dir, force=False, scope=scope)
    feat = build_feature_panel(panel, scope=scope)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(path, index=False)
    print(f"[features] → {path}")
    return feat
