"""후보 제외 → 100점 기본점수 → 접근성 계수."""

from __future__ import annotations

from typing import Any, Literal

from .access import (
    access_coefficient,
    classify_access,
)
from .config import (
    PROXY_SPEED_M_PER_MIN,
    SCORE_AVAIL_MAX,
    SCORE_BATTERY_MAX,
    SCORE_CHARGER_COUNT_MAX,
    SCORE_EXPAND_THRESHOLD,
    SCORE_FRESHNESS_MAX,
    SCORE_ROUTE_DIST_MAX,
    SCORE_ROUTE_TIME_MAX,
    SCORE_SPEED_MAX,
    SOC_HARD_EXCLUDE_PCT,
    VERY_STALE_EXCLUDE_MIN,
)
from .soc import compute_station_soc

ChargeMode = Literal["external", "home"]
SpeedClass = Literal["ultra", "fast", "mid", "slow"]


def speed_class_from_kw(power_kw: float | None) -> SpeedClass:
    if power_kw is None or not (power_kw == power_kw):
        return "slow"
    kw = float(power_kw)
    if kw >= 300:
        return "ultra"
    if kw >= 100:
        return "fast"
    if kw >= 50:
        return "mid"
    return "slow"


def avail_prob_points(avg_prob: float) -> float:
    return float(max(0.0, min(1.0, avg_prob))) * SCORE_AVAIL_MAX


def route_time_points(detour_minutes: float) -> float:
    m = max(0.0, float(detour_minutes))
    if m <= 2:
        return 15.0
    if m <= 5:
        return 12.0
    if m <= 10:
        return 8.0
    if m <= 15:
        return 4.0
    return 0.0


def route_distance_points(extra_km: float) -> float:
    d = max(0.0, float(extra_km))
    if d <= 0.5:
        return 5.0
    if d <= 1.0:
        return 4.0
    if d <= 2.0:
        return 2.0
    return 0.0


def battery_points(arrival_soc_pct: float | None) -> tuple[float, bool, bool]:
    """(점수, 주의표시, 하드제외). SOC 미제공 시 만점(중립)."""
    if arrival_soc_pct is None:
        return SCORE_BATTERY_MAX, False, False
    soc = float(arrival_soc_pct)
    if soc < SOC_HARD_EXCLUDE_PCT:
        return 0.0, True, True
    if soc >= 20:
        return 10.0, False, False
    if soc >= 15:
        return 8.0, False, False
    if soc >= 10:
        return 5.0, False, False
    if soc >= 5:
        return 2.0, True, False
    return 0.0, True, True


def charger_count_points(n_compatible: int) -> tuple[float, bool]:
    """(점수, 하드제외)."""
    n = int(n_compatible)

    if n <= 0:
        return 0.0, True
    if n >= 4:
        return 15.0, False
    if n == 3:
        return 12.0, False
    if n == 2:
        return 9.0, False
    return 5.0, False


def speed_fit_points(speed: SpeedClass, mode: ChargeMode) -> float:
    if mode == "home":
        # 완속 우선
        table = {"slow": 10.0, "mid": 6.0, "fast": 3.0, "ultra": 2.0}
    else:
        table = {"ultra": 10.0, "fast": 9.0, "mid": 6.0, "slow": 3.0}
    return table.get(speed, 3.0)


def freshness_points(age_min: float | None) -> tuple[float, bool, bool]:
    """(점수, 정보지연표시, 하드제외). age 미상 → 중간값 2점."""
    if age_min is None or not (age_min == age_min):
        return 2.0, False, False
    age = float(age_min)
    if age >= VERY_STALE_EXCLUDE_MIN:
        return 0.0, True, True
    if age <= 5:
        return 5.0, False, False
    if age <= 10:
        return 4.0, False, False
    if age <= 20:
        return 2.0, False, False
    return 0.0, True, False


def recommendation_label(score: float) -> str:
    s = float(score)
    if s >= 80:
        return "매우 추천"
    if s >= 65:
        return "추천"
    if s >= 50:
        return "조건부 추천"
    if s >= 35:
        return "주의·대안 부족"
    return "추천 어려움"


def estimate_route_metrics(
    *,
    distance_m: float,
    origin_lat: float | None,
    origin_lng: float | None,
    dest_lat: float,
    dest_lng: float,
    station_lat: float,
    station_lng: float,
    haversine_m_fn,
) -> tuple[float, float]:
    """(detour_minutes, extra_km). TMAP 없이 직선 프록시.

    origin 있음: (o→s + s→d) - (o→d)
    origin 없음: 목적지↔충전소 거리/ETA를 우회·추가거리로 사용
    """
    speed = max(float(PROXY_SPEED_M_PER_MIN), 1.0)
    if origin_lat is not None and origin_lng is not None:
        o_s = haversine_m_fn(origin_lat, origin_lng, station_lat, station_lng)
        s_d = haversine_m_fn(station_lat, station_lng, dest_lat, dest_lng)
        o_d = haversine_m_fn(origin_lat, origin_lng, dest_lat, dest_lng)
        extra_m = max(0.0, o_s + s_d - o_d)
        detour_min = extra_m / speed
        return float(detour_min), float(extra_m / 1000.0)

    # 주변 검색: 목적지–충전소 직선거리를 경로 비용으로 사용
    d_m = max(0.0, float(distance_m))
    return float(d_m / speed), float(d_m / 1000.0)


def score_station(
    *,
    mode: ChargeMode,
    registered: bool,
    avg_available_prob: float,
    n_compatible: int,
    max_output_kw: float | None,
    worst_status_age_min: float | None,
    distance_m: float,
    dest_lat: float,
    dest_lng: float,
    station_lat: float,
    station_lng: float,
    origin_lat: float | None,
    origin_lng: float | None,
    current_soc: float | None,
    vehicle_model_id: str | None,
    access_type: str,
    is_housing: bool,
    haversine_m_fn,
    parking_occupancy_rate: float | None = None,
) -> dict[str, Any]:
    """한 충전소 점수. excluded=True면 recommendation_score 없음."""
    exclude_reasons: list[str] = []

    count_pts, count_exclude = charger_count_points(n_compatible)
    if count_exclude:
        exclude_reasons.append("compatible_chargers_zero")

    detour_min, extra_km = estimate_route_metrics(
        distance_m=distance_m,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        dest_lat=dest_lat,
        dest_lng=dest_lng,
        station_lat=station_lat,
        station_lng=station_lng,
        haversine_m_fn=haversine_m_fn,
    )

    travel_m = distance_m
    if origin_lat is not None and origin_lng is not None:
        travel_m = haversine_m_fn(origin_lat, origin_lng, station_lat, station_lng)

    # 도착 예상 배터리는 추천점수와 후보 제외에 사용하지 않는다.
    soc_info = None
    arrival_soc = None
    bat_warn = False

    fresh_pts, fresh_warn, fresh_exclude = freshness_points(worst_status_age_min)
    if fresh_exclude:
        exclude_reasons.append("status_very_stale")

    coef = access_coefficient(access_type, is_housing, registered=registered)
    # 주차 점유율과 충전기 가용률의 시간대 제거 후 잔차 상관은 -0.073
    # (R² 0.54%)로 예측 신호가 없었다. 응답 호환을 위해 계수 필드는 남기되
    # 추천 순위에는 반영하지 않고, 만차 여부는 UI 안내용 원자료로만 전달한다.
    park_coef = 1.0
    if coef is None:
        exclude_reasons.append("access_restricted")

    if exclude_reasons:
        return {
            "excluded": True,
            "exclude_reasons": exclude_reasons,
            "recommendation_score": None,
            "score_breakdown": None,
            "access_coefficient": coef,
            "parking_occupancy_coefficient": park_coef,
            "recommendation_label": None,
            "arrival_soc_pct": arrival_soc,
            "detour_minutes": round(detour_min, 1),
            "extra_distance_km": round(extra_km, 3),
            "battery_caution": bat_warn,
            "stale_info": fresh_warn,
            "soc_info": soc_info,
        }

    speed = speed_class_from_kw(max_output_kw)
    avail_pts = avail_prob_points(avg_available_prob)
    time_pts = route_time_points(detour_min)
    dist_pts = route_distance_points(extra_km)
    speed_pts = speed_fit_points(speed, mode)

    base = avail_pts + time_pts + dist_pts + count_pts + speed_pts + fresh_pts
    combined_coef = float(coef)
    final = round(float(base) * combined_coef, 1)
    breakdown = {
        "availability": round(avail_pts, 2),
        "route_time": round(time_pts, 2),
        "route_distance": round(dist_pts, 2),
        # 기존 API 계약 호환용이며 실제 점수에는 반영하지 않는다.
        "battery": 0.0,
        "charger_count": round(count_pts, 2),
        "speed_fit": round(speed_pts, 2),
        "freshness": round(fresh_pts, 2),
        "base_total": round(base, 2),
        "access_coefficient": float(coef),
        "parking_occupancy_coefficient": float(park_coef),
        "parking_occupancy_rate": (
            None if parking_occupancy_rate is None else float(parking_occupancy_rate)
        ),
    }
    return {
        "excluded": False,
        "exclude_reasons": [],
        "recommendation_score": final,
        "score_breakdown": breakdown,
        "access_coefficient": combined_coef,
        "parking_occupancy_coefficient": float(park_coef),
        "recommendation_label": recommendation_label(final),
        "arrival_soc_pct": arrival_soc,
        "detour_minutes": round(detour_min, 1),
        "extra_distance_km": round(extra_km, 3),
        "battery_caution": bat_warn,
        "stale_info": fresh_warn,
        "soc_info": soc_info,
        "speed_class": speed,
    }


def should_expand_radius(stations: list[dict[str, Any]]) -> bool:
    if not stations:
        return True
    scores = [
        float(s["recommendation_score"])
        for s in stations
        if s.get("recommendation_score") is not None
    ]
    if not scores:
        return True
    return max(scores) < SCORE_EXPAND_THRESHOLD


def station_access_payload(row: Any) -> dict[str, Any]:
    return classify_access(
        limit_yn=row.get("limit_yn") if hasattr(row, "get") else getattr(row, "limit_yn", None),
        limit_detail=(
            row.get("limit_detail") if hasattr(row, "get") else getattr(row, "limit_detail", None)
        ),
        kind=row.get("kind") if hasattr(row, "get") else getattr(row, "kind", None),
        kind_detail=(
            row.get("kind_detail") if hasattr(row, "get") else getattr(row, "kind_detail", None)
        ),
        stat_nm=row.get("stat_nm") if hasattr(row, "get") else getattr(row, "stat_nm", None),
        addr=row.get("addr") if hasattr(row, "get") else getattr(row, "addr", None),
    )


def station_access_payload_from_group(rows: Any) -> dict[str, Any]:
    """충전소의 모든 충전기 정보를 합쳐 가장 보수적인 접근 유형을 반환한다.

    같은 stat_id 안에서도 limit_yn/limit_detail 이 섞인 사례가 있으므로 첫 번째
    충전기만 보면 제한 충전소가 PUBLIC으로 빠질 수 있다. 우선순위는
    RESIDENT > RESTRICTED > UNKNOWN > PUBLIC이며, 주거시설 여부는 any로 묶는다.
    """
    if hasattr(rows, "iterrows"):
        items = (row for _, row in rows.iterrows())
    else:
        items = iter(rows)

    payloads = [station_access_payload(row) for row in items]
    if not payloads:
        return classify_access()

    priority = {"PUBLIC": 0, "UNKNOWN": 1, "RESTRICTED": 2, "RESIDENT": 3}
    selected = max(payloads, key=lambda p: priority.get(str(p["access_type"]), 1))
    access_type = str(selected["access_type"])
    warning = next(
        (
            p.get("access_warning")
            for p in payloads
            if str(p["access_type"]) == access_type and p.get("access_warning")
        ),
        selected.get("access_warning"),
    )
    return {
        "access_type": access_type,
        "is_housing": any(bool(p.get("is_housing")) for p in payloads),
        "access_warning": warning,
    }
