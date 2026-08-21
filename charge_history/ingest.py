# -*- coding: utf-8 -*-
"""월별 xlsx → 통일 스키마 → (선택) 지역 필터 → parquet 캐시."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import (
    CACHE_DIR,
    DEFAULT_DOWNLOADS,
    MONTH_FILES,
    MONTHS,
    SESSION_COLS,
    scope_prefix,
)

# 활용현황(2025.01–09): pandas가 중복 컬럼을 시설구분(대).1 로 붙임
RENAME_UTIL = {
    "충전소명": "station_name",
    "충전기ID": "charger_unit_id",
    "시설구분(대)": "facility_type_major",
    "시설구분(대).1": "facility_type_minor",
    "시설구분(소)": "facility_type_minor",
    "지역": "region",
    "시군구": "district",
    "주소": "address",
    "충전기타입": "connector_type",
    "충전용량": "power_label",
    "충전시작일시": "start_raw",
    "충전종료일시": "end_raw",
    "충전시간": "duration_text",
    "충전량": "kwh",
}

RENAME_HISTORY = {
    "충전소명": "station_name",
    "충전기ID": "charger_unit_id",
    "충전소 유형(대분류)": "facility_type_major",
    "충전소 유형(소분류)": "facility_type_minor",
    "지역": "region",
    "시군구": "district",
    "주소": "address",
    "충전기타입": "connector_type",
    "충전기용량(KW)": "power_label",
    "충전시작일시": "start_raw",
    "충전종료일시": "end_raw",
    "충전시간": "duration_text",
    "충전량": "kwh",
}


def _is_util_month(year_month: str) -> bool:
    y, m = year_month.split("-")
    return y == "2025" and int(m) <= 9


def _speed_class(power_kw: pd.Series) -> pd.Series:
    return pd.cut(
        power_kw,
        bins=[-0.1, 20, 100, 200, 1000],
        labels=["완속급", "급속", "고출력급속", "초급속"],
    ).astype(str)


def normalize_frame(
    df: pd.DataFrame,
    year_month: str,
    *,
    region_contains: str | None = "대구",
) -> pd.DataFrame:
    rename = RENAME_UTIL if _is_util_month(year_month) else RENAME_HISTORY
    cols = {k: v for k, v in rename.items() if k in df.columns}
    out = df.rename(columns=cols).copy()

    for c in (
        "station_name",
        "charger_unit_id",
        "facility_type_major",
        "facility_type_minor",
        "region",
        "district",
        "address",
        "connector_type",
        "power_label",
        "start_raw",
        "end_raw",
        "duration_text",
        "kwh",
    ):
        if c not in out.columns:
            out[c] = pd.NA

    if region_contains:
        out = out[
            out["region"].astype(str).str.contains(region_contains, na=False)
        ].copy()
    if out.empty:
        out = out.reindex(columns=SESSION_COLS)
        return out

    out["start_at"] = pd.to_datetime(
        out["start_raw"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    out["end_at"] = pd.to_datetime(
        out["end_raw"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce")
    out["duration_min"] = (out["end_at"] - out["start_at"]).dt.total_seconds() / 60.0
    out["start_hour"] = out["start_at"].dt.hour
    out["start_weekday"] = out["start_at"].dt.weekday
    out["start_date"] = out["start_at"].dt.strftime("%Y-%m-%d")
    out["year_month"] = year_month
    out["power_kw"] = (
        out["power_label"].astype(str).str.extract(r"(\d+)\s*kW", expand=False).astype(float)
    )
    out["speed_class"] = _speed_class(out["power_kw"])
    out["charger_unit_id"] = out["charger_unit_id"].astype(str)

    return out[SESSION_COLS].sort_values("start_at").reset_index(drop=True)


def month_parquet_path(
    year_month: str,
    cache_dir: Path = CACHE_DIR,
    *,
    scope: str = "daegu",
) -> Path:
    prefix = scope_prefix(scope)
    return cache_dir / f"{prefix}_sessions_{year_month}.parquet"


def sessions_parquet_path(
    cache_dir: Path = CACHE_DIR, *, scope: str = "daegu"
) -> Path:
    prefix = scope_prefix(scope)
    return cache_dir / f"{prefix}_sessions_all.parquet"


def ingest_month(
    year_month: str,
    *,
    downloads: Path = DEFAULT_DOWNLOADS,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    scope: str = "daegu",
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = month_parquet_path(year_month, cache_dir, scope=scope)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    fname = MONTH_FILES[year_month]
    src = downloads / fname
    if not src.exists():
        raise FileNotFoundError(f"월 {year_month} 파일 없음: {src}")

    region_contains = None if scope_prefix(scope) == "nationwide" else "대구"
    print(f"[ingest] {year_month} ← {src.name} scope={scope_prefix(scope)}")
    raw = pd.read_excel(src)
    out = normalize_frame(raw, year_month, region_contains=region_contains)
    out.to_parquet(out_path, index=False)
    print(f"[ingest] {year_month} rows={len(out):,} → {out_path.name}")
    return out


def ingest_all(
    *,
    downloads: Path = DEFAULT_DOWNLOADS,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    months: list[str] | None = None,
    scope: str = "daegu",
) -> pd.DataFrame:
    months = months or MONTHS
    parts: list[pd.DataFrame] = []
    for ym in months:
        parts.append(
            ingest_month(
                ym,
                downloads=downloads,
                cache_dir=cache_dir,
                force=force,
                scope=scope,
            )
        )
    all_df = pd.concat(parts, ignore_index=True)
    all_path = sessions_parquet_path(cache_dir, scope=scope)
    all_df.to_parquet(all_path, index=False)
    print(
        f"[ingest] ALL rows={len(all_df):,} months={len(months)} "
        f"scope={scope_prefix(scope)} → {all_path.name}"
    )
    return all_df


def load_sessions(
    cache_dir: Path = CACHE_DIR,
    force_ingest: bool = False,
    *,
    scope: str = "daegu",
    downloads: Path = DEFAULT_DOWNLOADS,
) -> pd.DataFrame:
    path = sessions_parquet_path(cache_dir, scope=scope)
    if force_ingest or not path.exists():
        return ingest_all(
            downloads=downloads, cache_dir=cache_dir, force=force_ingest, scope=scope
        )
    return pd.read_parquet(path)
