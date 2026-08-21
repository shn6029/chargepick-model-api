"""
소상공인시장진흥공단 상권정보 API → ev_charger_context 테이블
충전소 반경 500m 내 업종별 POI 수 집계 후 station_context_type 분류.

API: https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius
수집 주기: 월 1회 이하 (정적 데이터)
"""

from __future__ import annotations

import os
import time
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import pymysql
import requests
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

# 소상공인 상권정보 API 키 (환경변수 또는 공공데이터포털 발급 키)
CONTEXT_SERVICE_KEY = os.getenv(
    "CONTEXT_SERVICE_KEY",
    os.getenv("SERVICE_KEY", ""),
)
CONTEXT_API_BASE = (
    "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
)
RADIUS_M = 500  # 집계 반경(m)
REQUEST_DELAY_SEC = 0.5  # API 호출 간격

# 업종 대분류 코드 → 집계 컬럼 매핑
# 참고: 소상공인 상권정보 API 업종 대분류(uptaeNm) 기준
CATEGORY_MAP: dict[str, str] = {
    "음식": "restaurant_cnt_500m",
    "소매": "mart_cnt_500m",
    "생활서비스": "office_cnt_500m",
    "스포츠": "office_cnt_500m",
    "관광/여가/오락": "accommodation_cnt_500m",
    "숙박": "accommodation_cnt_500m",
    "의료": "hospital_cnt_500m",
    "교육": "office_cnt_500m",
    "부동산": "office_cnt_500m",
}

# 세부 업종명으로 카페/편의점 보정
CAFE_KEYWORDS = ("카페", "커피", "베이커리", "제과")
MART_KEYWORDS = ("마트", "편의점", "슈퍼", "할인점")

# station_context_type 분류 임계값
CONTEXT_TYPE_RULES: list[tuple[str, dict[str, int]]] = [
    # (context_type, {column: min_count})
    ("휴게소형", {"accommodation_cnt_500m": 3}),
    ("관광지형", {"accommodation_cnt_500m": 2}),
    ("병원형", {"hospital_cnt_500m": 2}),
    ("상업시설형", {"restaurant_cnt_500m": 5, "mart_cnt_500m": 2}),
    ("마트형", {"mart_cnt_500m": 3}),
    ("공공기관형", {"office_cnt_500m": 5}),
]
DEFAULT_CONTEXT_TYPE = "주거·혼합형"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_context (
    stat_id                VARCHAR(20)  NOT NULL PRIMARY KEY COMMENT '충전소 ID',
    restaurant_cnt_500m    SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 음식점 수',
    cafe_cnt_500m          SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 카페 수',
    mart_cnt_500m          SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 마트/편의점 수',
    hospital_cnt_500m      SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 병원 수',
    office_cnt_500m        SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 사무/공공기관 수',
    accommodation_cnt_500m SMALLINT     NOT NULL DEFAULT 0 COMMENT '반경500m 숙박/관광 수',
    commercial_poi_cnt_500m SMALLINT    NOT NULL DEFAULT 0 COMMENT '반경500m 총 상업 POI 수',
    context_type           VARCHAR(30)  NULL     COMMENT '충전소 시설 유형',
    api_success            TINYINT      NOT NULL DEFAULT 0 COMMENT 'API 호출 성공 여부',
    updated_at             DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                           ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전소 주변 상권 POI 집계 (소상공인 상권정보 API)'
"""

UPSERT_SQL = """
INSERT INTO ev_charger_context (
    stat_id,
    restaurant_cnt_500m, cafe_cnt_500m, mart_cnt_500m,
    hospital_cnt_500m, office_cnt_500m, accommodation_cnt_500m,
    commercial_poi_cnt_500m, context_type, api_success, updated_at
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    restaurant_cnt_500m    = VALUES(restaurant_cnt_500m),
    cafe_cnt_500m          = VALUES(cafe_cnt_500m),
    mart_cnt_500m          = VALUES(mart_cnt_500m),
    hospital_cnt_500m      = VALUES(hospital_cnt_500m),
    office_cnt_500m        = VALUES(office_cnt_500m),
    accommodation_cnt_500m = VALUES(accommodation_cnt_500m),
    commercial_poi_cnt_500m= VALUES(commercial_poi_cnt_500m),
    context_type           = VALUES(context_type),
    api_success            = VALUES(api_success),
    updated_at             = VALUES(updated_at)
"""


@dataclass
class PoiCounts:
    restaurant: int = 0
    cafe: int = 0
    mart: int = 0
    hospital: int = 0
    office: int = 0
    accommodation: int = 0

    def total(self) -> int:
        return (
            self.restaurant + self.cafe + self.mart
            + self.hospital + self.office + self.accommodation
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "restaurant_cnt_500m": self.restaurant,
            "cafe_cnt_500m": self.cafe,
            "mart_cnt_500m": self.mart,
            "hospital_cnt_500m": self.hospital,
            "office_cnt_500m": self.office,
            "accommodation_cnt_500m": self.accommodation,
        }


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    a = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(
        radians(lng2 - lng1) / 2
    ) ** 2
    return 2 * r * asin(sqrt(a))


def _classify_store(uptae_nm: str, indu_nm: str) -> Optional[str]:
    """업종 대분류·세부명으로 집계 컬럼명 반환. 미분류는 None."""
    indu_lower = indu_nm.lower()
    uptae_lower = uptae_nm.lower()

    if any(k in indu_nm for k in CAFE_KEYWORDS):
        return "cafe_cnt_500m"
    if any(k in indu_nm for k in MART_KEYWORDS):
        return "mart_cnt_500m"
    if "음식" in uptae_lower:
        return "restaurant_cnt_500m"
    if "의료" in uptae_lower:
        return "hospital_cnt_500m"
    if "숙박" in uptae_lower or "관광" in uptae_lower:
        return "accommodation_cnt_500m"
    for key, col in CATEGORY_MAP.items():
        if key in uptae_lower:
            return col
    return None


def _determine_context_type(counts: PoiCounts) -> str:
    d = counts.as_dict()
    for ctx_type, thresholds in CONTEXT_TYPE_RULES:
        if all(d.get(col, 0) >= min_cnt for col, min_cnt in thresholds.items()):
            return ctx_type
    return DEFAULT_CONTEXT_TYPE


def fetch_stores_in_radius(
    lat: float,
    lng: float,
    radius: int = RADIUS_M,
    page_size: int = 1000,
) -> tuple[PoiCounts, bool]:
    """API 호출 → PoiCounts. 성공 여부도 반환."""
    if not CONTEXT_SERVICE_KEY:
        print("  [경고] CONTEXT_SERVICE_KEY 미설정 → 상권 API 스킵")
        return PoiCounts(), False

    params = {
        "serviceKey": CONTEXT_SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": page_size,
        "radius": radius,
        "cx": lng,
        "cy": lat,
        "indsLclsCd": "",  # 전체 업종
        "type": "json",
    }
    try:
        resp = requests.get(CONTEXT_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body") or {}
        items = body.get("items") or []
        if isinstance(items, dict):
            items = items.get("item") or []
    except Exception as exc:
        print(f"  [오류] 상권 API 실패: {exc}")
        return PoiCounts(), False

    counts = PoiCounts()
    for item in items:
        if not isinstance(item, dict):
            continue
        uptae_nm = str(item.get("uptaeNm") or "")
        indu_nm = str(item.get("indsSclsNm") or item.get("indsNm") or "")
        col = _classify_store(uptae_nm, indu_nm)
        if col == "restaurant_cnt_500m":
            counts.restaurant += 1
        elif col == "cafe_cnt_500m":
            counts.cafe += 1
        elif col == "mart_cnt_500m":
            counts.mart += 1
        elif col == "hospital_cnt_500m":
            counts.hospital += 1
        elif col == "office_cnt_500m":
            counts.office += 1
        elif col == "accommodation_cnt_500m":
            counts.accommodation += 1
    return counts, True


def load_stations() -> list[tuple[str, float, float]]:
    """ev_charger_info에서 (stat_id, lat, lng) 목록 로드 (중복 제거)."""
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
                ORDER BY stat_id
                """
            )
            return [(r[0], float(r[1]), float(r[2])) for r in cur.fetchall()]
    finally:
        conn.close()


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def save_context_rows(rows: list[tuple]) -> int:
    if not rows:
        return 0
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        ensure_table(conn)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def run(stat_ids: list[str] | None = None, dry_run: bool = False) -> None:
    """상권 컨텍스트 수집 메인 루틴.

    stat_ids: 특정 충전소만 처리 (None이면 전체)
    dry_run: DB 저장 없이 콘솔 출력만
    """
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ev_charger_context 수집 시작")
    stations = load_stations()
    if stat_ids:
        stat_set = set(stat_ids)
        stations = [(s, lat, lng) for s, lat, lng in stations if s in stat_set]
    print(f"  처리 대상 충전소: {len(stations)}개 (반경 {RADIUS_M}m)")

    batch: list[tuple] = []
    ok_cnt = 0
    fail_cnt = 0

    for i, (stat_id, lat, lng) in enumerate(stations, 1):
        counts, success = fetch_stores_in_radius(lat, lng)
        ctx_type = _determine_context_type(counts) if success else None

        row = (
            stat_id,
            counts.restaurant,
            counts.cafe,
            counts.mart,
            counts.hospital,
            counts.office,
            counts.accommodation,
            counts.total(),
            ctx_type,
            1 if success else 0,
            datetime.now(),
        )

        if dry_run:
            print(
                f"  [{i}/{len(stations)}] {stat_id} "
                f"음식={counts.restaurant} 카페={counts.cafe} "
                f"마트={counts.mart} 병원={counts.hospital} "
                f"사무={counts.office} 숙박={counts.accommodation} "
                f"→ {ctx_type}"
            )
        else:
            batch.append(row)
            if success:
                ok_cnt += 1
            else:
                fail_cnt += 1

        if i % 50 == 0:
            print(
                f"  진행: {i}/{len(stations)} (성공={ok_cnt}, 실패={fail_cnt})"
            )
            if batch and not dry_run:
                save_context_rows(batch)
                batch = []

        time.sleep(REQUEST_DELAY_SEC)

    if batch and not dry_run:
        save_context_rows(batch)

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 완료: "
        f"총={len(stations)} 성공={ok_cnt} 실패={fail_cnt}"
    )


def main() -> None:
    import sys

    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    stat_ids: list[str] | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--stat-ids="):
            stat_ids = arg.split("=", 1)[1].split(",")

    if dry_run:
        print("[DRY RUN] DB 저장 없이 출력만 합니다.")

    run(stat_ids=stat_ids, dry_run=dry_run)


if __name__ == "__main__":
    main()
