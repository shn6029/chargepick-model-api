"""시간대·접근제한·주차 중립화 서빙 정책 테스트 (DB/모델 불필요)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommend_api.scoring import score_station, station_access_payload_from_group
from recommend_api.service import filter_closed_at_arrival


def test_filter_closed_at_arrival_keeps_open_and_unknown():
    frame = pd.DataFrame(
        {
            "chger_id": ["01", "02", "03"],
            "is_operating_at_arrival": [1.0, 0.0, np.nan],
        }
    )

    filtered = filter_closed_at_arrival(frame)

    assert filtered["chger_id"].tolist() == ["01", "03"]


def test_station_access_uses_most_restrictive_charger():
    station = pd.DataFrame(
        [
            {
                "limit_yn": "N",
                "limit_detail": "",
                "kind": "A0",
                "stat_nm": "공용 충전소",
                "addr": "대구광역시",
            },
            {
                "limit_yn": "Y",
                "limit_detail": "입주민 전용",
                "kind": "H0",
                "stat_nm": "공동주택 충전소",
                "addr": "대구광역시",
            },
        ]
    )

    access = station_access_payload_from_group(station)

    assert access["access_type"] == "RESIDENT"
    assert access["is_housing"] is True


def _score(parking_occupancy_rate: float):
    return score_station(
        mode="external",
        registered=False,
        avg_available_prob=0.8,
        n_compatible=3,
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
        haversine_m_fn=lambda *_: 500.0,
        parking_occupancy_rate=parking_occupancy_rate,
    )


def test_parking_occupancy_does_not_change_recommendation_score():
    empty = _score(0.0)
    full = _score(99.0)

    assert empty["recommendation_score"] == full["recommendation_score"]
    assert empty["parking_occupancy_coefficient"] == 1.0
    assert full["parking_occupancy_coefficient"] == 1.0
    assert full["score_breakdown"]["parking_occupancy_rate"] == 99.0

