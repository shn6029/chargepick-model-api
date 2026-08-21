"""scoring 티어·계수 단위 테스트 (DB/모델 불필요)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.access import access_coefficient, classify_access
from recommend_api.config import (
    RAPID_SHADOW_WARN_THRESHOLD_BY_ETA,
    rapid_shadow_warn_threshold,
)
from recommend_api.scoring import (
    avail_prob_points,
    battery_points,
    charger_count_points,
    freshness_points,
    recommendation_label,
    route_distance_points,
    route_time_points,
    score_station,
    speed_fit_points,
    speed_class_from_kw,
)


def test_avail_and_route_tiers():
    assert avail_prob_points(0.78) == 39.0  # SCORE_AVAIL_MAX 45->50 (2026-08-13 배점 개편)
    assert route_time_points(1) == 15
    assert route_time_points(4) == 12
    assert route_time_points(8) == 8
    assert route_time_points(12) == 4
    assert route_time_points(20) == 0
    assert route_distance_points(0.4) == 5
    assert route_distance_points(0.8) == 4
    assert route_distance_points(1.5) == 2
    assert route_distance_points(3) == 0


def test_battery_charger_freshness():
    assert battery_points(55)[0] == 10
    assert battery_points(17)[0] == 8
    assert battery_points(12)[0] == 5
    pts, warn, excl = battery_points(7)
    assert pts == 2 and warn and not excl
    pts, warn, excl = battery_points(3)
    assert excl
    assert charger_count_points(0)[1] is True
    assert charger_count_points(4)[0] == 15
    assert charger_count_points(1)[0] == 5
    assert freshness_points(3)[0] == 5
    assert freshness_points(8)[0] == 4
    assert freshness_points(15)[0] == 2
    assert freshness_points(25)[0] == 0
    assert freshness_points(60)[2] is True


def test_speed_and_labels():
    assert speed_class_from_kw(350) == "ultra"
    assert speed_class_from_kw(100) == "fast"
    assert speed_class_from_kw(50) == "mid"
    assert speed_class_from_kw(7) == "slow"
    assert speed_fit_points("ultra", "external") == 10
    assert speed_fit_points("slow", "external") == 3
    assert speed_fit_points("slow", "home") == 10
    assert recommendation_label(83) == "매우 추천"
    assert recommendation_label(70) == "추천"
    assert recommendation_label(55) == "조건부 추천"
    assert recommendation_label(40) == "주의·대안 부족"
    assert recommendation_label(20) == "추천 어려움"


def test_access_coefficient():
    assert access_coefficient("PUBLIC", False) == 1.0
    assert access_coefficient("PUBLIC", True) == 0.9
    assert access_coefficient("UNKNOWN", True) == 0.6
    assert access_coefficient("RESIDENT", True) is None
    assert access_coefficient("RESIDENT", True, registered=True) == 1.0
    acc = classify_access(limit_yn="N", kind="A", addr="대구 수성구")
    assert acc["access_type"] == "PUBLIC"


def _hav(a, b, c, d):
    return 500.0  # 고정 거리 프록시


def test_score_station_public_example():
    out = score_station(
        mode="external",
        registered=False,
        avg_available_prob=0.78,
        n_compatible=2,
        max_output_kw=100,
        worst_status_age_min=3,
        distance_m=300,
        dest_lat=35.84,
        dest_lng=128.68,
        station_lat=35.841,
        station_lng=128.681,
        origin_lat=None,
        origin_lng=None,
        current_soc=60,
        vehicle_model_id="generic",
        access_type="PUBLIC",
        is_housing=False,
        haversine_m_fn=_hav,
    )
    assert out["excluded"] is False
    assert out["recommendation_score"] is not None
    bd = out["score_breakdown"]
    assert abs(bd["availability"] - 39.0) < 1e-6  # SCORE_AVAIL_MAX 45->50
    assert abs(bd["base_total"] * bd["access_coefficient"] - out["recommendation_score"]) < 0.15


def test_score_station_ignores_low_soc():
    """도착 SOC 는 2026-08-13 배점 개편에서 점수·후보제외 양쪽에서 빠졌다.

    구 동작은 arrival_soc < 5 이면 `arrival_soc_below_5` 로 하드 제외였다.
    지금은 제외하지도, 점수에 넣지도 않는다(`score_breakdown.battery` 는
    API 계약 호환용으로 0.0 만 남는다). 되살리려면 여기부터 깨진다.
    """
    out = score_station(
        mode="external",
        registered=False,
        avg_available_prob=0.9,
        n_compatible=3,
        max_output_kw=100,
        worst_status_age_min=3,
        distance_m=50_000,
        dest_lat=35.84,
        dest_lng=128.68,
        station_lat=36.0,
        station_lng=129.0,
        origin_lat=None,
        origin_lng=None,
        current_soc=8,
        vehicle_model_id="generic",
        access_type="PUBLIC",
        is_housing=False,
        haversine_m_fn=lambda a, b, c, d: 80_000.0,
    )
    assert out["excluded"] is False
    assert "arrival_soc_below_5" not in out["exclude_reasons"]
    assert out["score_breakdown"]["battery"] == 0.0


def test_rapid_shadow_warn_threshold_curve():
    # 격자점은 그대로
    for eta, thr in RAPID_SHADOW_WARN_THRESHOLD_BY_ETA.items():
        assert rapid_shadow_warn_threshold(eta) == thr

    # 양끝 clamp — 로그상 1·2·4분 요청이 실제로 들어온다
    assert rapid_shadow_warn_threshold(1) == RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[5]
    assert rapid_shadow_warn_threshold(0) == RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[5]
    assert rapid_shadow_warn_threshold(90) == RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[60]

    # 격자 사이는 선형 보간 (h15 0.59 ~ h20 0.63)
    assert rapid_shadow_warn_threshold(17.5) == 0.61
    assert abs(rapid_shadow_warn_threshold(16) - 0.598) < 1e-9
    # h20 0.63 ~ h30 0.71 의 중간
    assert rapid_shadow_warn_threshold(25) == 0.67

    # 단조 증가 — 곡선이 뒤집히면 장거리에서 경고가 줄어드는 옛 문제로 돌아간다
    prev = 0.0
    for eta in range(0, 91):
        cur = rapid_shadow_warn_threshold(eta)
        assert cur >= prev
        prev = cur


if __name__ == "__main__":
    test_rapid_shadow_warn_threshold_curve()
    test_avail_and_route_tiers()
    test_battery_charger_freshness()
    test_speed_and_labels()
    test_access_coefficient()
    test_score_station_public_example()
    test_score_station_ignores_low_soc()
    print("scoring tests OK")
