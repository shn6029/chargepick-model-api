# -*- coding: utf-8 -*-
"""기상청 ASOS 시간자료 → 일별 기온/강수 캐시.

API: AsosHourlyInfoService/getWthrDataList
- daegu: 지점 143
- nationwide: 시도 대표 지점 (region → stn_id)
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from . import CACHE_DIR, MONTHS, SCHEDULER_ROOT

load_dotenv(SCHEDULER_ROOT / ".env")

API_BASE = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
STN_ID = "143"  # 대구
DEFAULT_SERVICE_KEY = os.getenv("KMA_SERVICE_KEY", "")

# 충전소 region(시도) → 대표 ASOS 지점
# 참고: 시도 대표 관측소(서울108·수원119·춘천101·청주131·홍성177·전주146·목포165·안동136·창원155·제주184 등)
REGION_ASOS_STATIONS: dict[str, str] = {
    "서울특별시": "108",
    "부산광역시": "159",
    "대구광역시": "143",
    "인천광역시": "112",
    "광주광역시": "156",
    "대전광역시": "133",
    "울산광역시": "152",
    "세종특별자치시": "133",  # 인근 대전
    "경기도": "119",
    "강원특별자치도": "101",
    "강원도": "101",
    "충청북도": "131",
    "충청남도": "177",
    "전북특별자치도": "146",
    "전라북도": "146",
    "전라남도": "165",
    "경상북도": "136",
    "경상남도": "155",
    "제주특별자치도": "184",
    "제주도": "184",
}

STN_LABELS: dict[str, str] = {
    "108": "서울",
    "159": "부산",
    "143": "대구",
    "112": "인천",
    "156": "광주",
    "133": "대전",
    "152": "울산",
    "119": "수원",
    "101": "춘천",
    "131": "청주",
    "177": "홍성",
    "146": "전주",
    "165": "목포",
    "136": "안동",
    "155": "창원",
    "184": "제주",
}


def service_key() -> str:
    return os.getenv("KMA_SERVICE_KEY") or os.getenv("SERVICE_KEY") or DEFAULT_SERVICE_KEY


def region_to_stn_id(region: str | None) -> str | None:
    if region is None:
        return None
    key = str(region).strip()
    return REGION_ASOS_STATIONS.get(key)


def unique_nationwide_stn_ids() -> list[str]:
    return sorted(set(REGION_ASOS_STATIONS.values()))


def hourly_parquet_path(
    cache_dir: Path = CACHE_DIR, *, scope: str = "daegu"
) -> Path:
    prefix = "nationwide" if scope == "nationwide" else "daegu"
    return cache_dir / f"{prefix}_asos_hourly.parquet"


def daily_parquet_path(
    cache_dir: Path = CACHE_DIR, *, scope: str = "daegu"
) -> Path:
    prefix = "nationwide" if scope == "nationwide" else "daegu"
    return cache_dir / f"{prefix}_asos_daily.parquet"


def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    if child is None or child.text is None:
        return None
    v = child.text.strip()
    return v or None


def _parse_tm(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def fetch_hourly_range(
    start_dt: datetime,
    end_dt: datetime,
    *,
    stn_id: str = STN_ID,
    key: str | None = None,
) -> pd.DataFrame:
    """start~end (시 단위) ASOS 조회. end는 전일(D-1)까지."""
    params = {
        "serviceKey": key or service_key(),
        "pageNo": "1",
        "numOfRows": "900",
        "dataType": "XML",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_dt.strftime("%Y%m%d"),
        "startHh": start_dt.strftime("%H"),
        "endDt": end_dt.strftime("%Y%m%d"),
        "endHh": end_dt.strftime("%H"),
        "stnIds": stn_id,
    }

    last_error: Exception | None = None
    response = None
    for attempt in range(1, 4):
        try:
            response = requests.get(API_BASE, params=params, timeout=90)
            if response.status_code in (502, 503, 504):
                raise requests.HTTPError(
                    f"{response.status_code} gateway", response=response
                )
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
            response = None
    if response is None:
        raise RuntimeError(f"ASOS API 요청 실패: {last_error}")

    root = ET.fromstring(response.content)
    code = root.findtext(".//header/resultCode") or root.findtext(".//resultCode")
    if code and code not in ("00", "0"):
        msg = root.findtext(".//header/resultMsg") or root.findtext(".//resultMsg")
        raise RuntimeError(f"ASOS API error {code}: {msg}")

    rows = []
    for item in root.findall(".//item") or root.findall(".//items/item"):
        observed_at = _parse_tm(_text(item, "tm"))
        if not observed_at:
            continue
        rows.append(
            {
                "observed_at": observed_at,
                "stn_id": str(_text(item, "stnId") or stn_id),
                "stn_nm": _text(item, "stnNm"),
                "ta": pd.to_numeric(_text(item, "ta"), errors="coerce"),
                "rn": pd.to_numeric(_text(item, "rn"), errors="coerce"),
                "hm": pd.to_numeric(_text(item, "hm"), errors="coerce"),
                "ws": pd.to_numeric(_text(item, "ws"), errors="coerce"),
                "dsnw": pd.to_numeric(_text(item, "dsnw"), errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _month_chunks(day_min: datetime, day_max: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cur = day_min.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= day_max:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        start = max(cur, day_min.replace(hour=0, minute=0, second=0, microsecond=0))
        end = min(
            nxt - timedelta(hours=1),
            day_max.replace(hour=23, minute=0, second=0, microsecond=0),
        )
        if start <= end:
            chunks.append((start, end))
        cur = nxt
    return chunks


def _resolve_day_bounds(
    day_min: str | datetime | None,
    day_max: str | datetime | None,
) -> tuple[datetime, datetime]:
    if day_min is None:
        day_min = f"{MONTHS[0]}-01"
    if day_max is None:
        y, m = map(int, MONTHS[-1].split("-"))
        if m == 12:
            day_max = datetime(y + 1, 1, 1) - timedelta(days=1)
        else:
            day_max = datetime(y, m + 1, 1) - timedelta(days=1)

    d0 = pd.Timestamp(day_min).to_pydatetime().replace(hour=0)
    d1 = pd.Timestamp(day_max).to_pydatetime().replace(hour=0)
    api_max = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=1
    )
    if d1 > api_max:
        d1 = api_max
    return d0, d1


def backfill_asos(
    *,
    day_min: str | datetime | None = None,
    day_max: str | datetime | None = None,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    sleep_s: float = 0.35,
    scope: str = "daegu",
    stn_ids: list[str] | None = None,
) -> pd.DataFrame:
    """MONTHS 구간 ASOS 시간자료 수집 → hourly/daily parquet.

    scope=daegu → 143만
    scope=nationwide → 시도 대표 지점들
    """
    d0, d1 = _resolve_day_bounds(day_min, day_max)
    if scope == "nationwide":
        stn_ids = stn_ids or unique_nationwide_stn_ids()
    else:
        stn_ids = stn_ids or [STN_ID]
    stn_ids = [str(s) for s in stn_ids]

    hourly_path = hourly_parquet_path(cache_dir, scope=scope)
    existing = pd.DataFrame()
    if hourly_path.exists() and not force:
        existing = pd.read_parquet(hourly_path)
        existing["observed_at"] = pd.to_datetime(existing["observed_at"])
        existing["stn_id"] = existing["stn_id"].astype(str)

    # 대구 캐시가 있으면 전국 백필 시 143 시드로 재사용
    if scope == "nationwide" and (existing.empty or force):
        daegu_path = hourly_parquet_path(cache_dir, scope="daegu")
        if daegu_path.exists():
            seed = pd.read_parquet(daegu_path)
            seed["observed_at"] = pd.to_datetime(seed["observed_at"])
            seed["stn_id"] = seed["stn_id"].astype(str)
            existing = seed if existing.empty else pd.concat([existing, seed], ignore_index=True)
            print(f"[asos] seeded from daegu cache: {len(seed):,} rows")

    chunks = _month_chunks(d0, d1)
    frames: list[pd.DataFrame] = []
    if not existing.empty:
        frames.append(existing)

    for stn_id in stn_ids:
        label_stn = STN_LABELS.get(stn_id, stn_id)
        for start, end in chunks:
            label = start.strftime("%Y-%m")
            if not force and not existing.empty:
                mask = (
                    (existing["stn_id"] == stn_id)
                    & (existing["observed_at"] >= start)
                    & (existing["observed_at"] <= end)
                )
                if int(mask.sum()) >= 24 * 20:
                    print(
                        f"[asos] skip cached {label_stn}({stn_id}) {label} "
                        f"({int(mask.sum())} rows)"
                    )
                    continue
            try:
                df = fetch_hourly_range(start, end, stn_id=stn_id)
                print(
                    f"[asos] {label_stn}({stn_id}) {label}: {len(df)} rows "
                    f"({start} ~ {end})"
                )
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"[asos] {label_stn}({stn_id}) {label} failed: {exc}")
            time.sleep(sleep_s)

    if not frames:
        return pd.DataFrame()

    hourly = pd.concat(frames, ignore_index=True)
    hourly["observed_at"] = pd.to_datetime(hourly["observed_at"])
    hourly["stn_id"] = hourly["stn_id"].astype(str)
    hourly = hourly.drop_duplicates(["observed_at", "stn_id"]).sort_values(
        ["stn_id", "observed_at"]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(hourly_path, index=False)

    daily = hourly_to_daily(hourly, by_station=True)
    daily_path = daily_parquet_path(cache_dir, scope=scope)
    daily.to_parquet(daily_path, index=False)
    print(
        f"[asos] scope={scope} stn={len(stn_ids)} "
        f"hourly={len(hourly):,} daily={len(daily):,} "
        f"→ {hourly_path.name} / {daily_path.name}"
    )
    return daily


def hourly_to_daily(
    hourly: pd.DataFrame, *, by_station: bool = False
) -> pd.DataFrame:
    base_cols = [
        "day",
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "precipitation",
        "humidity_mean",
        "wind_speed_mean",
        "snowfall_sum",
    ]
    if by_station:
        base_cols = ["stn_id"] + base_cols
    if hourly.empty:
        return pd.DataFrame(columns=base_cols)

    h = hourly.copy()
    h["day"] = pd.to_datetime(h["observed_at"]).dt.normalize()
    h["ta"] = pd.to_numeric(h["ta"], errors="coerce")
    h["rn"] = pd.to_numeric(h["rn"], errors="coerce")
    h["hm"] = pd.to_numeric(h.get("hm"), errors="coerce")
    h["ws"] = pd.to_numeric(h.get("ws"), errors="coerce")
    h["dsnw"] = pd.to_numeric(h.get("dsnw"), errors="coerce")
    h["stn_id"] = h["stn_id"].astype(str) if "stn_id" in h.columns else STN_ID

    group_keys = ["stn_id", "day"] if by_station else ["day"]

    agg_dict: dict = {
        "temperature_mean": ("ta", "mean"),
        "temperature_min": ("ta", "min"),
        "temperature_max": ("ta", "max"),
        "precipitation": ("rn", "sum"),
    }
    # 습도·풍속·적설은 컬럼이 있을 때만 집계
    if "hm" in h.columns:
        agg_dict["humidity_mean"] = ("hm", "mean")
    if "ws" in h.columns:
        agg_dict["wind_speed_mean"] = ("ws", "mean")
    if "dsnw" in h.columns:
        agg_dict["snowfall_sum"] = ("dsnw", "sum")

    out = (
        h.groupby(group_keys, as_index=False)
        .agg(**agg_dict)
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    # 없는 컬럼은 NaN으로 채움
    for col in ("humidity_mean", "wind_speed_mean", "snowfall_sum"):
        if col not in out.columns:
            out[col] = float("nan")

    if not by_station and "stn_id" in out.columns:
        out = out.drop(columns=["stn_id"])
    return out


def load_or_fetch_daily(
    *,
    day_min: pd.Timestamp | None = None,
    day_max: pd.Timestamp | None = None,
    cache_dir: Path = CACHE_DIR,
    fetch_if_missing: bool = True,
    scope: str = "daegu",
) -> pd.DataFrame:
    """일별 날씨. 캐시 부족 시 ASOS API 백필."""
    daily_path = daily_parquet_path(cache_dir, scope=scope)
    daily = pd.DataFrame()
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
        if "stn_id" in daily.columns:
            daily["stn_id"] = daily["stn_id"].astype(str)

    need_fetch = fetch_if_missing
    if not daily.empty and day_min is not None and day_max is not None:
        d0 = pd.Timestamp(day_min).normalize()
        d1 = pd.Timestamp(day_max).normalize()
        cover = daily[(daily["day"] >= d0) & (daily["day"] <= d1)]
        expected_days = (d1 - d0).days + 1
        if scope == "nationwide":
            n_stn = max(len(unique_nationwide_stn_ids()), 1)
            # 지점×일 커버리지
            if len(cover) >= expected_days * n_stn * 0.80:
                need_fetch = False
        else:
            if len(cover) >= expected_days * 0.85:
                need_fetch = False

    if need_fetch and fetch_if_missing:
        daily = backfill_asos(
            day_min=day_min or (MONTHS[0] + "-01"),
            day_max=day_max,
            cache_dir=cache_dir,
            force=False,
            scope=scope,
        )

    if daily.empty:
        return daily

    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    if "stn_id" in daily.columns:
        daily["stn_id"] = daily["stn_id"].astype(str)
    if day_min is not None:
        daily = daily[daily["day"] >= pd.Timestamp(day_min).normalize()]
    if day_max is not None:
        daily = daily[daily["day"] <= pd.Timestamp(day_max).normalize()]
    return daily.reset_index(drop=True)
