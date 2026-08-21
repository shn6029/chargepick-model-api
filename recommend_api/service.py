from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLS,
    HGB_BLEND_WEIGHT,
    RADIUS_EXPAND_ABS_CAP_KM,
    RADIUS_EXPAND_MAX_EXTRA_KM,
    RADIUS_EXPAND_STEP_KM,
    REMAINING_BLEND_WEIGHT,
    SHADOW_WARN_ENABLED,
    USABLE_STATS,
    VERY_STALE_EXCLUDE_MIN,
    rapid_shadow_warn_threshold,
)
from .model_store import (
    align_features,
    enrich_frame,
    get_connection,
    load_artifact,
    _holiday_feature_select,
    _context_select_and_join,
)
from .prediction_log import log_recommendations
from .remaining_time import (
    free_by_eta_score,
    model_available as remaining_model_available,
    predict_remaining_minutes,
)
from .scoring import (
    score_station,
    should_expand_radius,
    station_access_payload_from_group,
)
from .soc import resolve_min_output_kw
from .parking import load_latest_occupancy_for_stations
from .use_time import operating_at

ChargeMode = Literal["external", "home"]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lng2 - lng1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * r * asin(sqrt(a))


def load_latest_nearby(dest_lat: float, dest_lng: float, radius_km: float) -> pd.DataFrame:
    """도착지 근처 충전기의 최신 status+features+info 스냅샷."""
    delta = radius_km / 111.0
    lng_delta = radius_km / max(111.0 * abs(cos(radians(dest_lat))), 1e-6)

    conn = get_connection()
    try:
        holiday_sel = _holiday_feature_select(conn)
        context_sel, context_join = _context_select_and_join(conn)
        query = f"""
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                s.stat_upd_dt,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                {holiday_sel},
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.kind_detail, i.parking_free,
                i.limit_yn, i.limit_detail, i.traffic_yn, i.use_time,
                i.stat_nm, i.addr, i.lat, i.lng,
                {context_sel}
            FROM ev_charger_status s
            INNER JOIN (
                -- '가장 최신 행'이 아니라 '피처가 붙은 가장 최신 행'을 고른다.
                -- 전량 스냅샷(10분 주기)이 모든 충전기의 최신 행을 동시에 갈아치우므로,
                -- 최신 행만 고집하면 피처 배치가 조금만 밀려도 후보가 한꺼번에 0이 된다.
                -- 신선도는 아래 WHERE 의 created_at 기준으로 따로 거른다.
                -- log_id(auto_increment)로 잡아 동일 created_at 중복도 피한다.
                SELECT s2.stat_id, s2.chger_id, MAX(s2.log_id) AS latest_log_id
                FROM ev_charger_status s2
                INNER JOIN ev_charger_features f2 ON s2.log_id = f2.log_id
                INNER JOIN ev_charger_info i2
                    ON s2.stat_id = i2.stat_id AND s2.chger_id = i2.chger_id
                WHERE i2.lat BETWEEN %s AND %s
                  AND i2.lng BETWEEN %s AND %s
                  AND (i2.del_yn IS NULL OR i2.del_yn <> 'Y')
                GROUP BY s2.stat_id, s2.chger_id
            ) latest
                ON s.log_id = latest.latest_log_id
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            {context_join}
            WHERE i.lat BETWEEN %s AND %s
              AND i.lng BETWEEN %s AND %s
              AND (i.del_yn IS NULL OR i.del_yn <> 'Y')
        """
        params = (
            dest_lat - delta,
            dest_lat + delta,
            dest_lng - lng_delta,
            dest_lng + lng_delta,
            dest_lat - delta,
            dest_lat + delta,
            dest_lng - lng_delta,
            dest_lng + lng_delta,
        )
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()

    df = enrich_frame(df)
    if df.empty:
        return df

    df["distance_m"] = df.apply(
        lambda r: haversine_m(dest_lat, dest_lng, float(r["lat"]), float(r["lng"]))
        if pd.notna(r["lat"]) and pd.notna(r["lng"])
        else np.nan,
        axis=1,
    )
    df = df[df["distance_m"].notna() & (df["distance_m"] <= radius_km * 1000)]
    return df.reset_index(drop=True)


def filter_usable_chargers(df: pd.DataFrame) -> pd.DataFrame:
    """운영중지·점검·통신이상 등 하드 사용불가 제외. 대기(2)·충전중(3)만 남김."""
    if df.empty:
        return df
    mask = df["stat"].astype(int).isin(USABLE_STATS)
    return df.loc[mask].reset_index(drop=True)


def filter_compatible_output(
    df: pd.DataFrame, min_output_kw: float | None
) -> pd.DataFrame:
    if df.empty or min_output_kw is None:
        return df
    out = pd.to_numeric(df.get("output_kw"), errors="coerce")
    return df.loc[out.fillna(0) >= float(min_output_kw)].reset_index(drop=True)


def filter_rapid_only(df: pd.DataFrame) -> pd.DataFrame:
    """급속만 (output_kw >= 50 / is_fast==1)."""
    if df.empty:
        return df
    if "is_fast" in df.columns:
        return df.loc[df["is_fast"].fillna(0).astype(int) == 1].reset_index(drop=True)
    out = pd.to_numeric(df.get("output_kw"), errors="coerce").fillna(0)
    return df.loc[out >= 50].reset_index(drop=True)


def filter_very_stale(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status_update_age_min" not in df.columns:
        return df
    age = pd.to_numeric(df["status_update_age_min"], errors="coerce")
    keep = age.isna() | (age < VERY_STALE_EXCLUDE_MIN)
    return df.loc[keep].reset_index(drop=True)


def filter_by_mode(
    df: pd.DataFrame,
    mode: ChargeMode,
    registered_stat_ids: list[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    registered = {str(x) for x in (registered_stat_ids or []) if x}
    if mode == "home":
        return df.loc[df["stat_id"].astype(str).isin(registered)].reset_index(drop=True)
    return df


def filter_closed_at_arrival(df: pd.DataFrame) -> pd.DataFrame:
    """도착 시 운영 종료가 명확한 후보만 제외한다.

    UNKNOWN(NaN)은 운영시간 원천이 불완전한 경우이므로 보수적으로 유지한다.
    이 값은 예측 피처가 아니라 이미 아는 이용 가능 조건이라 서빙 필터에 둔다.
    """
    if df.empty or "is_operating_at_arrival" not in df.columns:
        return df
    operating = pd.to_numeric(df["is_operating_at_arrival"], errors="coerce")
    keep = operating.isna() | operating.ne(0)
    return df.loc[keep].reset_index(drop=True)


def _compute_arrival_holiday_flags(arrival_ts: pd.Timestamp) -> dict[str, int]:
    """도착 시각 기준 공휴일 파생변수 계산."""
    try:
        import holidays as _holidays
        _yrs = {arrival_ts.year - 1, arrival_ts.year, arrival_ts.year + 1}
        _kr = set(_holidays.country_holidays("KR", years=sorted(_yrs)).keys())
    except Exception:
        _kr = set()

    from datetime import date as _date, timedelta as _td

    d: _date = arrival_ts.date()

    def _is_off(x: _date) -> bool:
        return x in _kr or x.weekday() >= 5

    is_holiday = 1 if _is_off(d) else 0
    is_holiday_eve = 1 if _is_off(d + _td(days=1)) else 0
    is_after_holiday = 1 if _is_off(d - _td(days=1)) else 0

    # consecutive_holiday_days: 현재 날짜 포함 연속 휴일 길이
    if _is_off(d):
        start = d
        while _is_off(start - _td(days=1)):
            start -= _td(days=1)
        end = d
        while _is_off(end + _td(days=1)):
            end += _td(days=1)
        consec = (end - start).days + 1
    else:
        consec = 0

    return {
        "is_holiday": is_holiday,
        "is_holiday_eve": is_holiday_eve,
        "is_after_holiday": is_after_holiday,
        "consecutive_holiday_days": consec,
        "is_long_weekend": 1 if consec >= 3 else 0,
    }


def prepare_inference_frame(
    latest: pd.DataFrame,
    eta_minutes: float,
    arrival_at: str | None,
) -> pd.DataFrame:
    frame = latest.copy()
    if arrival_at:
        try:
            arrival_ts = pd.to_datetime(arrival_at)
        except Exception:
            arrival_ts = pd.Timestamp.now() + pd.Timedelta(minutes=eta_minutes)
    else:
        arrival_ts = pd.Timestamp.now() + pd.Timedelta(minutes=eta_minutes)

    if arrival_ts.tzinfo is not None:
        arrival_ts = arrival_ts.tz_localize(None)

    frame["eta_minutes"] = float(eta_minutes)
    frame["arrival_hour"] = int(arrival_ts.hour)
    frame["arrival_weekday"] = int(arrival_ts.weekday())

    # 도착 시점 기준 공휴일 파생변수 (현재 시각 기준 DB 값 대신 도착 시각으로 덮어쓰기)
    holiday_flags = _compute_arrival_holiday_flags(arrival_ts)
    for col, val in holiday_flags.items():
        frame[col] = val

    # 도착 시점 운영 여부 (Y=1 / N=0 / UNKNOWN=NaN). 학습과 동일 로직.
    if "use_time" in frame.columns:
        frame["is_operating_at_arrival"] = operating_at(
            frame["use_time"], arrival_ts.to_pydatetime()
        )
    else:
        frame["is_operating_at_arrival"] = float("nan")

    return frame


def build_feature_matrix(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    type_dummies = pd.get_dummies(frame["chger_type"].fillna("UNK"), prefix="ctype")
    kind_dummies = pd.get_dummies(frame["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(frame["busi_id"].fillna("UNK"), prefix="busi")
    X = pd.concat([frame[FEATURE_COLS], type_dummies, kind_dummies, busi_dummies], axis=1)
    return align_features(X, artifact["feature_columns"], artifact["medians"])


def blend_with_remaining_time(frame: pd.DataFrame, eta_minutes: float) -> pd.DataFrame:
    out = frame.copy()
    out["pred_remaining_min"] = np.nan
    out["free_by_eta_score"] = np.nan
    out["hgb_available_prob"] = out["available_prob"]
    out["remaining_source"] = "none"

    charging = out["stat"].astype(int) == 3
    if not charging.any():
        return out

    rem = predict_remaining_minutes(out.loc[charging])
    out.loc[charging, "pred_remaining_min"] = rem
    scores = rem.map(lambda r: free_by_eta_score(float(r), eta_minutes))
    out.loc[charging, "free_by_eta_score"] = scores
    out.loc[charging, "remaining_source"] = (
        "ml" if remaining_model_available() else "power_lookup"
    )

    # REMAINING_BLEND_WEIGHT 는 2026-08-12 부터 0 이다(config.py 근거표 참고).
    # 성분은 표시용으로 계속 계산해 응답에 싣되 점수에는 섞지 않는다.
    if REMAINING_BLEND_WEIGHT == 0.0:
        return out

    # 성분이 비유한(NaN/inf)이면 그 행은 HGB 값을 그대로 둔다.
    # w=0 일 때 0.0 * NaN = NaN 이라, 곱셈만 믿으면 멀쩡한 확률이 조용히 사라진다.
    hgb = out.loc[charging, "hgb_available_prob"]
    usable = np.isfinite(scores.to_numpy(dtype=float))
    blended = hgb.where(
        ~usable,
        HGB_BLEND_WEIGHT * hgb + REMAINING_BLEND_WEIGHT * scores,
    )
    out.loc[charging, "available_prob"] = blended
    return out


def _confidence_level(eta_minutes: float) -> str:
    """ETA 구간별 표현 신뢰도 (모델 재학습 없음).

    ≤15 high · 16–20 medium · ≥21 low
    """
    eta = float(eta_minutes)
    if eta <= 15:
        return "high"
    if eta <= 20:
        return "medium"
    return "low"


def _horizon_note(
    eta_minutes: float,
    artifact: dict[str, Any],
    rem_on: bool,
    empty: bool = False,
    radius_note: str | None = None,
) -> str:
    base = f"약 {int(round(eta_minutes))}분 뒤 도착 기준 예상"
    if empty:
        note = f"{base} (근처 사용가능 충전기 없음)"
    else:
        rem_note = "잔여시간ML" if rem_on else "잔여시간lookup(kW)"
        note = (
            f"{base} (학습 {artifact.get('n_samples', '?')}샘플, {rem_note}, "
            f"100점 랭킹)"
        )
    eta = float(eta_minutes)
    if 16 <= eta <= 20:
        note += " — 도착 시각이 멀어 변동 가능성이 있습니다"
    elif 21 <= eta <= 30:
        note += " — 중거리: 예측 점수 반영 비중을 낮춰 해석하세요"
    elif eta > 30:
        note += " — 장거리: 현재 혼잡도 참고 수준으로 보세요"
    if radius_note:
        note += f" — {radius_note}"
    return note


def _build_charger_items(
    g: pd.DataFrame, shadow_warn_thr: float | None = None
) -> list[dict[str, Any]]:
    """shadow_warn_thr 은 요청 ETA 로 결정된 값이다(config.rapid_shadow_warn_threshold).

    None 이면 warn 을 끈다. 임계값이 ETA마다 다르므로 여기서 다시 계산하지 않고
    호출부에서 받아 쓴다 — 응답 meta·예측로그와 같은 값이어야 하기 때문이다.
    """
    chargers: list[dict[str, Any]] = []
    for _, row in g.iterrows():
        stale = bool(int(row.get("is_stale_status", 0)))
        long_state = bool(int(row.get("is_long_state_duration", 0)))
        is_fast = bool(int(row.get("is_fast", 0)))
        avail_prob = float(row["available_prob"])
        shadow_warn = (
            SHADOW_WARN_ENABLED
            and shadow_warn_thr is not None
            and is_fast
            and avail_prob < shadow_warn_thr
        )
        item: dict[str, Any] = {
            "chger_id": str(row["chger_id"]),
            "chger_type": None if pd.isna(row.get("chger_type")) else str(row["chger_type"]),
            "output_kw": None if pd.isna(row.get("output_kw")) else float(row["output_kw"]),
            "current_stat": int(row["stat"]),
            "available_prob": avail_prob,
            "hgb_available_prob": float(row["hgb_available_prob"]),
            "current_state_duration": (
                None
                if pd.isna(row.get("current_state_duration"))
                else int(row["current_state_duration"])
            ),
            "status_update_age_min": (
                None
                if pd.isna(row.get("status_update_age_min"))
                else round(float(row["status_update_age_min"]), 1)
            ),
            "is_long_state_duration": long_state,
            "is_stale_status": stale,
            "is_invalid_status_update_time": bool(
                int(row.get("is_invalid_status_update_time", 0))
            ),
            "shadow_unavailable_warn": shadow_warn,
        }
        if long_state:
            item["long_state_note"] = "동일한 상태가 장시간 유지되고 있습니다"
        if stale:
            item["stale_note"] = "상태 갱신이 오래된 충전기입니다"
        if pd.notna(row.get("pred_remaining_min")):
            item["pred_remaining_min"] = round(float(row["pred_remaining_min"]), 1)
            item["free_by_eta_score"] = round(float(row["free_by_eta_score"]), 4)
            item["remaining_source"] = str(row.get("remaining_source", "none"))
        chargers.append(item)
    return chargers


def _rank_stations_from_frame(
    frame: pd.DataFrame,
    *,
    mode: ChargeMode,
    registered_set: set[str],
    dest_lat: float,
    dest_lng: float,
    origin_lat: float | None,
    origin_lng: float | None,
    current_soc: float | None,
    vehicle_model_id: str | None,
    rank_thr: float,
    shadow_warn_thr: float | None = None,
    parking_by_stat: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    stations: list[dict[str, Any]] = []
    parking_by_stat = parking_by_stat or {}
    for (stat_id, stat_nm), g in frame.groupby(["stat_id", "stat_nm"], dropna=False):
        g = g.sort_values("chger_id")
        total = len(g)
        if total == 0:
            continue

        avg_prob = float(g["available_prob"].mean())
        pred_available = int((g["available_prob"] >= rank_thr).sum())
        availability_rate = pred_available / total if total else 0.0
        distance_m = float(g["distance_m"].min())
        has_stale = bool((g["is_stale_status"] == 1).any())
        has_long = bool((g["is_long_state_duration"] == 1).any())

        ages = pd.to_numeric(g.get("status_update_age_min"), errors="coerce")
        worst_age = float(ages.max()) if ages.notna().any() else None
        max_kw = None
        kw_series = pd.to_numeric(g.get("output_kw"), errors="coerce")
        if kw_series.notna().any():
            max_kw = float(kw_series.max())

        row0 = g.iloc[0]
        access = station_access_payload_from_group(g)
        sid = str(stat_id)
        registered = sid in registered_set

        park = parking_by_stat.get(sid) or {}
        occ_rate = park.get("occupancy_rate")
        try:
            occ_rate_f = float(occ_rate) if occ_rate is not None else None
        except (TypeError, ValueError):
            occ_rate_f = None
        if occ_rate_f is None or occ_rate_f != occ_rate_f:
            # 폴백: 잔여/총면수로 직접 계산.
            #
            # parking_realtime_status.occupancy_rate 는 대체로 채워져 있지만
            # (최근 1일 99.1%) 빠지는 경우가 있고, 하필 **만차인 주차장에서
            # 비는 경향**이 있다. 그대로 두면 "잔여 0면" 을 보여주면서 혼잡
            # 계수는 1.0(여유)이 되어 정확히 거꾸로 된 안내가 나간다.
            # total 은 map 테이블(parking_lot_info 유래, 항상 채워짐)을 쓴다 —
            # 실시간 쪽 total_spaces 는 rate 가 빌 때 같이 비기 때문이다.
            # 변수명에 park_ 접두사 필수 — 이 스코프의 `total` 은 호환 충전기 수라
            # 그대로 쓰면 덮어써서 score_station(n_compatible=None) 으로 터진다.
            park_rem = park.get("remaining_spaces")
            park_total = park.get("parking_total_spaces")
            try:
                rem_f, total_f = float(park_rem), float(park_total)
                if total_f > 0 and rem_f >= 0:
                    occ_rate_f = min(100.0, max(0.0, (1 - rem_f / total_f) * 100.0))
            except (TypeError, ValueError):
                pass

        scored = score_station(
            mode=mode,
            registered=registered,
            avg_available_prob=avg_prob,
            n_compatible=total,
            max_output_kw=max_kw,
            worst_status_age_min=worst_age,
            distance_m=distance_m,
            dest_lat=dest_lat,
            dest_lng=dest_lng,
            station_lat=float(row0["lat"]),
            station_lng=float(row0["lng"]),
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            current_soc=current_soc,
            vehicle_model_id=vehicle_model_id,
            access_type=str(access["access_type"]),
            is_housing=bool(access["is_housing"]),
            haversine_m_fn=haversine_m,
            parking_occupancy_rate=occ_rate_f,
        )
        if scored["excluded"]:
            continue

        rec_score = float(scored["recommendation_score"])
        station: dict[str, Any] = {
            "stat_id": sid,
            "stat_nm": str(stat_nm),
            "addr": None if pd.isna(row0.get("addr")) else str(row0["addr"]),
            "lat": float(row0["lat"]),
            "lng": float(row0["lng"]),
            "distance_m": round(distance_m, 1),
            "score": round(rec_score / 100.0, 4),
            "recommendation_score": rec_score,
            "recommendation_label": scored["recommendation_label"],
            "access_coefficient": scored["access_coefficient"],
            "score_breakdown": scored["score_breakdown"],
            "avg_available_prob": round(avg_prob, 4),
            "pred_available": pred_available,
            "total_chargers": total,
            "availability_rate": round(availability_rate, 4),
            "detour_minutes": scored.get("detour_minutes"),
            "extra_distance_km": scored.get("extra_distance_km"),
            "access_type": access["access_type"],
            "is_housing": access["is_housing"],
            "access_warning": access["access_warning"],
            "badges": {
                "has_fast": bool((g["is_fast"] == 1).any()),
                "parking_free": bool((g["parking_free_yn"] == 1).any()),
                "has_stale_charger": has_stale or bool(scored.get("stale_info")),
                "has_long_state_charger": has_long,
            },
            "chargers": _build_charger_items(g, shadow_warn_thr),
        }
        if scored.get("arrival_soc_pct") is not None:
            station["arrival_soc_pct"] = scored["arrival_soc_pct"]
        if park:
            rem = park.get("remaining_spaces")
            station["parking"] = {
                "pklt_id": park.get("pklt_id"),
                "parking_nm": park.get("parking_nm"),
                "total_spaces": park.get("parking_total_spaces"),
                "remaining_spaces": None if rem is None else int(rem),
                "occupancy_rate": occ_rate_f,
                "congestion_status": park.get("congestion_status"),
                "fee_type": park.get("parking_fee_type"),
                "is_24h": bool(park.get("parking_24h")),
            }
        stations.append(station)

    stations.sort(key=lambda s: (-s["recommendation_score"], s["distance_m"]))
    return stations


def recommend(
    dest_lat: float,
    dest_lng: float,
    eta_minutes: float,
    radius_km: float = 2.0,
    top_k: int = 10,
    arrival_at: str | None = None,
    log_predictions: bool = True,
    mode: ChargeMode = "external",
    registered_stat_ids: list[str] | None = None,
    origin_lat: float | None = None,
    origin_lng: float | None = None,
    current_soc: float | None = None,
    vehicle_model_id: str | None = None,
    min_output_kw: float | None = None,
    include_slow: bool = False,
) -> dict[str, Any]:
    artifact = load_artifact()
    model_version = artifact.get("model_version")
    confidence = _confidence_level(eta_minutes)
    rem_on = remaining_model_available()
    registered_set = {str(x) for x in (registered_stat_ids or []) if x}
    resolved_min_kw = resolve_min_output_kw(min_output_kw, vehicle_model_id)
    # 기본: 급속만. include_slow면 차량/요청 min_output만 적용
    if not include_slow:
        floor = 50.0
        if resolved_min_kw is None:
            resolved_min_kw = floor
        else:
            resolved_min_kw = max(float(resolved_min_kw), floor)

    rank_thr = 0.5
    thr_info = artifact.get("thresholds") or {}
    if isinstance(thr_info, dict) and "rank_threshold" in thr_info:
        rank_thr = float(thr_info["rank_threshold"])

    # shadow warn 임계값은 ETA마다 다르다(config 곡선). 응답 meta·예측로그·판정이
    # 같은 값을 봐야 하므로 여기서 한 번만 계산해서 내려보낸다.
    shadow_warn_thr = (
        rapid_shadow_warn_threshold(eta_minutes) if SHADOW_WARN_ENABLED else None
    )

    requested_radius = float(radius_km)
    max_radius = min(
        requested_radius + RADIUS_EXPAND_MAX_EXTRA_KM,
        RADIUS_EXPAND_ABS_CAP_KM,
    )
    used_radius = requested_radius
    radius_expanded = False
    radius_note: str | None = None
    stations: list[dict[str, Any]] = []
    frame_for_log: pd.DataFrame | None = None

    while True:
        latest = load_latest_nearby(dest_lat, dest_lng, used_radius)
        latest = filter_usable_chargers(latest)
        latest = filter_compatible_output(latest, resolved_min_kw)
        if not include_slow:
            latest = filter_rapid_only(latest)
        latest = filter_very_stale(latest)
        latest = filter_by_mode(latest, mode, registered_stat_ids)

        if latest.empty:
            stations = []
            frame_for_log = latest
        else:
            frame = prepare_inference_frame(latest, eta_minutes, arrival_at)
            frame = filter_closed_at_arrival(frame)
            if frame.empty:
                stations = []
                frame_for_log = frame
            else:
                X = build_feature_matrix(frame, artifact)
                proba = artifact["model"].predict_proba(X)[:, 1]
                frame = frame.copy()
                frame["available_prob"] = proba
                frame = blend_with_remaining_time(frame, eta_minutes)
                frame_for_log = frame
                parking_by_stat = load_latest_occupancy_for_stations(
                    frame["stat_id"].astype(str).unique().tolist()
                )
                stations = _rank_stations_from_frame(
                    frame,
                    mode=mode,
                    registered_set=registered_set,
                    dest_lat=dest_lat,
                    dest_lng=dest_lng,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    current_soc=current_soc,
                    vehicle_model_id=vehicle_model_id,
                    rank_thr=rank_thr,
                    shadow_warn_thr=shadow_warn_thr,
                    parking_by_stat=parking_by_stat,
                )

        if not should_expand_radius(stations):
            break
        next_radius = round(used_radius + RADIUS_EXPAND_STEP_KM, 1)
        if next_radius > max_radius + 1e-9:
            break
        used_radius = next_radius
        radius_expanded = True
        radius_note = (
            "가까운 곳에 적합한 충전소가 없어 범위를 넓혔습니다."
        )

    ranked = [{"rank": i, **station} for i, station in enumerate(stations[:top_k], start=1)]
    empty = len(ranked) == 0

    payload = {
        "meta": {
            "dest_lat": dest_lat,
            "dest_lng": dest_lng,
            "eta_minutes": eta_minutes,
            "arrival_at": arrival_at,
            "model": artifact.get("model_name", "HistGradientBoosting"),
            "model_version": model_version,
            "confidence_level": confidence,
            "include_slow": include_slow,
            "min_output_kw": resolved_min_kw,
            "remaining_model": rem_on,
            "rank_threshold": rank_thr,
            "shadow_warn_threshold": shadow_warn_thr,
            "horizon_note": _horizon_note(
                eta_minutes, artifact, rem_on, empty=empty, radius_note=radius_note
            ),
            "mode": mode,
            "radius_km": used_radius,
            "radius_expanded": radius_expanded,
            "radius_note": radius_note,
        },
        "recommendations": ranked,
    }

    if log_predictions and frame_for_log is not None and not frame_for_log.empty:
        frame_by_key = {
            (str(r.stat_id), str(r.chger_id)): {
                "current_state_duration": r.current_state_duration,
                "is_stale_status": int(r.is_stale_status),
                "is_long_state_duration": int(r.is_long_state_duration),
                "status_update_age_min": float(r.status_update_age_min)
                if pd.notna(r.status_update_age_min)
                else None,
            }
            for r in frame_for_log.itertuples()
        }
        req_id = log_recommendations(
            payload,
            dest_lat=dest_lat,
            dest_lng=dest_lng,
            eta_minutes=eta_minutes,
            radius_km=used_radius,
            top_k=top_k,
            frame_by_key=frame_by_key,
            vehicle_model_id=vehicle_model_id,
            min_output_kw=resolved_min_kw,
            n_candidate_stations=len(stations),
        )
        if req_id:
            payload["meta"]["request_id"] = req_id
    elif empty:
        payload["meta"]["request_id"] = str(uuid4())

    return payload
