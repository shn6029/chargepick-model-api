# -*- coding: utf-8 -*-
"""세션 → station×day 패널 (n_sessions, kwh)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CACHE_DIR, scope_prefix
from .ingest import load_sessions


def build_station_day_panel(sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    if sessions is None:
        sessions = load_sessions()
    if sessions.empty:
        return pd.DataFrame()

    s = sessions.copy()
    s["start_at"] = pd.to_datetime(s["start_at"], errors="coerce")
    s = s.dropna(subset=["start_at", "station_name"])
    s["day"] = s["start_at"].dt.normalize()
    s["year_month"] = s["start_at"].dt.strftime("%Y-%m")
    s["weekday"] = s["start_at"].dt.weekday

    meta = (
        s.groupby("station_name", as_index=False)
        .agg(
            region=("region", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
            district=("district", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
            speed_class=(
                "speed_class",
                lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown",
            ),
            power_kw=("power_kw", "median"),
        )
    )

    daily = (
        s.groupby(["station_name", "day", "year_month", "weekday"], as_index=False)
        .agg(
            n_sessions=("start_at", "count"),
            kwh=("kwh", "sum"),
            duration_min_mean=("duration_min", "mean"),
        )
    )
    daily["kwh"] = daily["kwh"].fillna(0.0)
    daily = daily.merge(meta, on="station_name", how="left")
    daily["region"] = daily["region"].fillna("unknown")
    daily["district"] = daily["district"].fillna("unknown")
    daily["speed_class"] = daily["speed_class"].fillna("unknown")
    daily["power_kw"] = daily["power_kw"].fillna(0.0)
    # 전국에서 시군구명 충돌 방지 (예: 중구)
    daily["district_key"] = (
        daily["region"].astype(str) + " " + daily["district"].astype(str)
    ).str.strip()
    return daily.sort_values(["day", "station_name"]).reset_index(drop=True)


def build_hour_distribution(sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """월×시간대 세션 비중 (리포트용)."""
    if sessions is None:
        sessions = load_sessions()
    s = sessions.dropna(subset=["start_at"]).copy()
    s["start_at"] = pd.to_datetime(s["start_at"], errors="coerce")
    s["year_month"] = s["start_at"].dt.strftime("%Y-%m")
    s["hour"] = s["start_at"].dt.hour
    counts = s.groupby(["year_month", "hour"]).size().rename("n").reset_index()
    totals = counts.groupby("year_month")["n"].transform("sum")
    counts["share"] = counts["n"] / totals.replace(0, np.nan)
    return counts


def panel_parquet_path(
    cache_dir: Path = CACHE_DIR, *, scope: str = "daegu"
) -> Path:
    return cache_dir / f"{scope_prefix(scope)}_station_day.parquet"


def load_or_build_panel(
    *,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    scope: str = "daegu",
    downloads=None,
) -> pd.DataFrame:
    path = panel_parquet_path(cache_dir, scope=scope)
    if path.exists() and not force:
        return pd.read_parquet(path)
    kw = {"cache_dir": cache_dir, "scope": scope}
    if downloads is not None:
        kw["downloads"] = downloads
    panel = build_station_day_panel(load_sessions(**kw))
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    print(
        f"[panel] rows={len(panel):,} stations={panel['station_name'].nunique():,} "
        f"scope={scope_prefix(scope)} → {path.name}"
    )
    return panel
