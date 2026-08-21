"""
대구광역시 교통소통정보·돌발정보 수집 → ev_charger_traffic_snapshot 테이블
충전소 주변 도로 혼잡도 및 사고/공사 정보를 이력으로 저장.

수집 주기: 5~10분 (run.py와 동일 패턴)
모델 투입 조건: 최소 4주 이력 누적 후 feature importance 검증 필요

API:
  - 대구광역시 교통소통정보: data.go.kr B552061/getTrafficSpeedInfoGu
  - 대구광역시 돌발정보:     data.go.kr B552061/getAccidentInfoGu
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import pymysql
import requests
import schedule
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

TRAFFIC_SERVICE_KEY = os.getenv(
    "TRAFFIC_SERVICE_KEY",
    os.getenv("SERVICE_KEY", ""),
)

TRAFFIC_API_BASE = (
    "https://apis.data.go.kr/B552061/getTrafficSpeedInfoGu"
)
INCIDENT_API_BASE = (
    "https://apis.data.go.kr/B552061/getAccidentInfoGu"
)

# 충전소 주변 집계 반경
NEARBY_RADIUS_M = 1000
COLLECT_INTERVAL_SEC = 300  # 5분

CREATE_SNAPSHOT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_traffic_snapshot (
    id                      BIGINT       NOT NULL AUTO_INCREMENT,
    stat_id                 VARCHAR(20)  NOT NULL COMMENT '충전소 ID',
    observed_at             DATETIME     NOT NULL COMMENT '관측 시각(5분 단위)',
    nearby_avg_speed        DECIMAL(6,2) NULL     COMMENT '주변 1km 도로 평균속도(km/h)',
    nearby_min_speed        DECIMAL(6,2) NULL     COMMENT '주변 1km 도로 최저속도(km/h)',
    congestion_level        TINYINT      NULL     COMMENT '혼잡 수준(0=원활,1=서행,2=정체)',
    incident_count_1km      SMALLINT     NOT NULL DEFAULT 0 COMMENT '1km 내 돌발 건수',
    has_accident_1km        TINYINT      NOT NULL DEFAULT 0 COMMENT '1km 내 사고 여부',
    has_roadwork_1km        TINYINT      NOT NULL DEFAULT 0 COMMENT '1km 내 공사 여부',
    segment_count           SMALLINT     NOT NULL DEFAULT 0 COMMENT '집계 구간 수',
    collected_at            TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_stat_obs (stat_id, observed_at),
    KEY idx_observed (observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전소 주변 교통 스냅샷 이력 (모델 투입 전 검증용)'
"""

INSERT_SNAPSHOT_SQL = """
INSERT INTO ev_charger_traffic_snapshot (
    stat_id, observed_at,
    nearby_avg_speed, nearby_min_speed, congestion_level,
    incident_count_1km, has_accident_1km, has_roadwork_1km, segment_count
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    a = (
        sin((p2 - p1) / 2) ** 2
        + cos(p1) * cos(p2) * sin(radians(lng2 - lng1) / 2) ** 2
    )
    return 2 * r * asin(sqrt(a))


def _speed_to_congestion(speed_kmh: float) -> int:
    """속도(km/h) → 혼잡 수준 (0=원활 ≥40, 1=서행 20~39, 2=정체 <20)."""
    if speed_kmh >= 40:
        return 0
    if speed_kmh >= 20:
        return 1
    return 2


def _text(el: ET.Element, tag: str) -> Optional[str]:
    child = el.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip() or None


def _flt(el: ET.Element, tag: str) -> Optional[float]:
    v = _text(el, tag)
    try:
        return float(v) if v else None
    except (ValueError, TypeError):
        return None


def fetch_traffic_segments() -> list[dict]:
    """대구 교통소통 구간 목록 조회."""
    if not TRAFFIC_SERVICE_KEY:
        return []
    params = {
        "serviceKey": TRAFFIC_SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": 1000,
        "type": "xml",
    }
    try:
        resp = requests.get(TRAFFIC_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        segments = []
        for item in root.findall(".//item"):
            lat = _flt(item, "strtLat") or _flt(item, "lat")
            lng = _flt(item, "strtLot") or _flt(item, "lot")
            speed = _flt(item, "spd") or _flt(item, "speed")
            if lat is None or lng is None or speed is None:
                continue
            segments.append({"lat": lat, "lng": lng, "speed": speed})
        return segments
    except Exception as exc:
        print(f"[교통] 소통정보 API 오류: {exc}")
        return []


def fetch_incidents() -> list[dict]:
    """대구 돌발정보 (사고·공사·행사 등) 조회."""
    if not TRAFFIC_SERVICE_KEY:
        return []
    params = {
        "serviceKey": TRAFFIC_SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": 500,
        "type": "xml",
    }
    try:
        resp = requests.get(INCIDENT_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        incidents = []
        for item in root.findall(".//item"):
            lat = _flt(item, "lat") or _flt(item, "startY")
            lng = _flt(item, "lot") or _flt(item, "startX")
            kind = _text(item, "kind") or _text(item, "accdtdvcd") or ""
            if lat is None or lng is None:
                continue
            incidents.append({"lat": lat, "lng": lng, "kind": kind})
        return incidents
    except Exception as exc:
        print(f"[교통] 돌발정보 API 오류: {exc}")
        return []


def load_stations() -> list[tuple[str, float, float]]:
    """ev_charger_info에서 (stat_id, lat, lng) 목록 로드."""
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stat_id, AVG(lat) AS lat, AVG(lng) AS lng
                FROM ev_charger_info
                WHERE (del_yn IS NULL OR del_yn <> 'Y')
                  AND lat IS NOT NULL AND lng IS NOT NULL
                GROUP BY stat_id
                """
            )
            return [(str(r[0]), float(r[1]), float(r[2])) for r in cur.fetchall()]
    finally:
        conn.close()


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_SNAPSHOT_TABLE_SQL)
    conn.commit()


def aggregate_for_stations(
    stations: list[tuple[str, float, float]],
    segments: list[dict],
    incidents: list[dict],
    observed_at: datetime,
) -> list[tuple]:
    rows: list[tuple] = []
    for stat_id, s_lat, s_lng in stations:
        nearby_segs = [
            seg for seg in segments
            if haversine_m(s_lat, s_lng, seg["lat"], seg["lng"]) <= NEARBY_RADIUS_M
        ]
        nearby_incs = [
            inc for inc in incidents
            if haversine_m(s_lat, s_lng, inc["lat"], inc["lng"]) <= NEARBY_RADIUS_M
        ]

        speeds = [seg["speed"] for seg in nearby_segs]
        avg_speed: Optional[float] = None
        min_speed: Optional[float] = None
        congestion: Optional[int] = None
        if speeds:
            avg_speed = round(sum(speeds) / len(speeds), 2)
            min_speed = round(min(speeds), 2)
            congestion = _speed_to_congestion(avg_speed)

        has_accident = int(
            any("사고" in inc["kind"] or "accident" in inc["kind"].lower()
                for inc in nearby_incs)
        )
        has_roadwork = int(
            any("공사" in inc["kind"] or "work" in inc["kind"].lower()
                for inc in nearby_incs)
        )

        rows.append((
            stat_id, observed_at,
            avg_speed, min_speed, congestion,
            len(nearby_incs), has_accident, has_roadwork, len(nearby_segs),
        ))
    return rows


def collect_once() -> None:
    now = datetime.now().replace(second=0, microsecond=0)
    # 5분 단위 정렬
    now = now.replace(minute=(now.minute // 5) * 5)

    stations = load_stations()
    if not stations:
        print(f"[{now}] 충전소 목록 없음")
        return

    segments = fetch_traffic_segments()
    incidents = fetch_incidents()
    print(
        f"[{now}] 소통구간={len(segments)} 돌발={len(incidents)} "
        f"충전소={len(stations)}"
    )

    rows = aggregate_for_stations(stations, segments, incidents, now)
    if not rows:
        return

    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        ensure_table(conn)
        with conn.cursor() as cur:
            cur.executemany(INSERT_SNAPSHOT_SQL, rows)
        conn.commit()
        print(f"[{now}] 교통 스냅샷 저장: {len(rows)}건")
    except Exception as exc:
        print(f"[{now}] DB 저장 오류: {exc}")
    finally:
        conn.close()


def main() -> None:
    import sys

    args = set(sys.argv[1:])
    interval = COLLECT_INTERVAL_SEC
    for arg in sys.argv[1:]:
        if arg.startswith("--interval="):
            try:
                interval = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    if "--once" in args:
        collect_once()
        return

    print(f"교통소통 수집 스케줄러 시작 (주기={interval}s)")
    print("  [주의] 최소 4주 이력 누적 후 모델 feature importance 검증 필요")

    collect_once()
    schedule.every(interval).seconds.do(collect_once)
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
