"""
충전소 ↔ 주차장 매핑 (기존 DB 테이블 활용)

소스:
  - parking_lot_info          : 정적 주차장 정보 (이미 수집됨)
  - parking_realtime_status   : 실시간 잔여면·혼잡도 (이미 수집됨)

출력:
  - ev_charger_parking_map    : stat_id → pklt_id 매핑

매핑 우선순위:
  1. 충전소명 ↔ 주차장명 유사도
  2. 주소 토큰 겹침
  3. 좌표 거리 100m 이내
  4. 수동 검수(manual_verified)

사용법:
  py run_parking.py --map
  py run_parking.py --map --dry-run
"""

from __future__ import annotations

import os
from datetime import datetime
from difflib import SequenceMatcher
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import pymysql
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

MAX_COORD_DISTANCE_M = 100
MAX_FALLBACK_DISTANCE_M = 200
NAME_SIMILARITY_THRESHOLD = 0.6

CREATE_MAP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_parking_map (
    stat_id               VARCHAR(20)  NOT NULL COMMENT '충전소 ID',
    pklt_id               VARCHAR(40)  NOT NULL COMMENT 'parking_lot_info.pklt_id',
    parking_nm            VARCHAR(100) NULL,
    mapping_method        VARCHAR(20)  NOT NULL COMMENT 'name/addr/coord/manual',
    mapping_confidence    DECIMAL(4,2) NOT NULL DEFAULT 0.0,
    coord_distance_m      DECIMAL(8,1) NULL,
    parking_total_spaces  INT          NULL,
    parking_fee_type      VARCHAR(40)  NULL COMMENT '유료/무료 (crg_levy_se_nm)',
    parking_24h           TINYINT      NOT NULL DEFAULT 0,
    parking_lat           DECIMAL(12,8) NULL,
    parking_lng           DECIMAL(13,8) NULL,
    manual_verified       TINYINT      NOT NULL DEFAULT 0,
    updated_at            DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                          ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stat_id),
    KEY idx_pklt_id (pklt_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전소-주차장 매핑 (parking_lot_info 기준)'
"""

UPSERT_MAP_SQL = """
INSERT INTO ev_charger_parking_map (
    stat_id, pklt_id, parking_nm, mapping_method, mapping_confidence,
    coord_distance_m, parking_total_spaces, parking_fee_type,
    parking_24h, parking_lat, parking_lng, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    pklt_id              = VALUES(pklt_id),
    parking_nm           = VALUES(parking_nm),
    mapping_method       = VALUES(mapping_method),
    mapping_confidence   = VALUES(mapping_confidence),
    coord_distance_m     = VALUES(coord_distance_m),
    parking_total_spaces = VALUES(parking_total_spaces),
    parking_fee_type     = VALUES(parking_fee_type),
    parking_24h          = VALUES(parking_24h),
    parking_lat          = VALUES(parking_lat),
    parking_lng          = VALUES(parking_lng),
    updated_at           = VALUES(updated_at)
"""


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    a = (
        sin((p2 - p1) / 2) ** 2
        + cos(p1) * cos(p2) * sin(radians(lng2 - lng1) / 2) ** 2
    )
    return 2 * r * asin(sqrt(a))


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def get_conn():
    return pymysql.connect(**DB_CONFIG, connect_timeout=30)


def load_parking_lots() -> list[dict]:
    """parking_lot_info에서 좌표 있는 주차장 로드."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    pklt_id, pklt_nm, lotno_addr, road_nm_addr,
                    lat, lot, total_spaces, crg_levy_se_nm,
                    oper_hr_wkday_se_cd
                FROM parking_lot_info
                WHERE lat IS NOT NULL AND lot IS NOT NULL
                  AND (use_yn IS NULL OR use_yn = 'Y')
                """
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

    for r in rows:
        r["lat"] = float(r["lat"])
        r["lng"] = float(r["lot"])
        r["addr"] = str(r.get("lotno_addr") or r.get("road_nm_addr") or "")
        op = str(r.get("oper_hr_wkday_se_cd") or "")
        r["is_24h"] = "전일" in op or "24" in op
    print(f"[parking_lot_info] 로드: {len(rows)}개")
    return rows


def load_stations() -> list[tuple]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stat_id, stat_nm, addr, AVG(lat) AS lat, AVG(lng) AS lng
                FROM ev_charger_info
                WHERE (del_yn IS NULL OR del_yn <> 'Y')
                  AND lat IS NOT NULL AND lng IS NOT NULL
                GROUP BY stat_id, stat_nm, addr
                ORDER BY stat_id
                """
            )
            return cur.fetchall()
    finally:
        conn.close()


def match_parking(
    stat_nm: str,
    stat_addr: str,
    stat_lat: float,
    stat_lng: float,
    parking_list: list[dict],
) -> Optional[tuple[dict, str, float]]:
    candidates = [
        p for p in parking_list
        if haversine_m(stat_lat, stat_lng, p["lat"], p["lng"]) <= MAX_FALLBACK_DISTANCE_M
    ]
    if not candidates:
        return None

    best: Optional[tuple[dict, str, float]] = None

    for p in candidates:
        sim = name_similarity(stat_nm, str(p.get("pklt_nm") or ""))
        if sim >= NAME_SIMILARITY_THRESHOLD:
            if best is None or sim > best[2]:
                best = (p, "name", round(sim, 3))
    if best:
        return best

    def _tokens(addr: str) -> set[str]:
        return {t for t in addr.replace(",", " ").split() if len(t) >= 2}

    stat_tok = _tokens(stat_addr)
    for p in candidates:
        overlap = len(stat_tok & _tokens(p["addr"]))
        if overlap >= 2:
            sim = overlap / max(len(stat_tok | _tokens(p["addr"])), 1)
            if best is None or sim > best[2]:
                best = (p, "addr", round(sim, 3))
    if best:
        return best

    close = [
        (p, haversine_m(stat_lat, stat_lng, p["lat"], p["lng"]))
        for p in candidates
        if haversine_m(stat_lat, stat_lng, p["lat"], p["lng"]) <= MAX_COORD_DISTANCE_M
    ]
    if close:
        p, dist = min(close, key=lambda x: x[1])
        conf = max(0.0, 1.0 - dist / MAX_COORD_DISTANCE_M)
        return (p, "coord", round(conf, 3))

    return None


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_MAP_TABLE_SQL)
    conn.commit()


def run_mapping(dry_run: bool = False) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 주차장 매핑 시작 (DB parking_lot_info)")
    stations = load_stations()
    parking_list = load_parking_lots()
    print(f"  충전소: {len(stations)}개 | 주차장: {len(parking_list)}개")

    rows: list[tuple] = []
    matched = 0
    by_method: dict[str, int] = {}

    for stat_id, stat_nm, stat_addr, lat, lng in stations:
        result = match_parking(
            str(stat_nm or ""),
            str(stat_addr or ""),
            float(lat),
            float(lng),
            parking_list,
        )
        if result is None:
            continue

        p, method, confidence = result
        dist = haversine_m(float(lat), float(lng), p["lat"], p["lng"])
        by_method[method] = by_method.get(method, 0) + 1

        if dry_run:
            print(
                f"  {stat_id}({stat_nm}) → [{p['pklt_id']}]{p.get('pklt_nm')} "
                f"[{method}={confidence:.2f}, {dist:.0f}m]"
            )
        else:
            rows.append((
                str(stat_id),
                str(p["pklt_id"]),
                p.get("pklt_nm"),
                method,
                confidence,
                round(dist, 1),
                p.get("total_spaces"),
                p.get("crg_levy_se_nm"),
                1 if p.get("is_24h") else 0,
                p["lat"],
                p["lng"],
                datetime.now(),
            ))
        matched += 1

    if rows and not dry_run:
        conn = get_conn()
        try:
            ensure_table(conn)
            with conn.cursor() as cur:
                cur.executemany(UPSERT_MAP_SQL, rows)
            conn.commit()
            print(f"  DB 저장: {len(rows)}건 → ev_charger_parking_map")
        finally:
            conn.close()

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 매핑 완료: "
        f"총={len(stations)} 성공={matched} "
        f"방법={by_method}"
    )


def load_latest_occupancy_for_stations(
    stat_ids: list[str],
) -> dict[str, dict]:
    """매핑된 충전소의 최신 주차 혼잡도.

    Returns: {stat_id: {pklt_id, remaining_spaces, occupancy_rate, ...}}
    """
    if not stat_ids:
        return {}
    placeholders = ",".join(["%s"] * len(stat_ids))
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    m.stat_id, m.pklt_id, m.parking_nm, m.parking_total_spaces,
                    m.parking_fee_type, m.parking_24h,
                    r.remaining_spaces, r.occupancy_rate, r.congestion_status,
                    r.collected_at
                FROM ev_charger_parking_map m
                LEFT JOIN (
                    SELECT r1.*
                    FROM parking_realtime_status r1
                    INNER JOIN (
                        SELECT pklt_id, MAX(collected_at) AS max_at
                        FROM parking_realtime_status
                        GROUP BY pklt_id
                    ) latest
                      ON r1.pklt_id = latest.pklt_id
                     AND r1.collected_at = latest.max_at
                ) r ON m.pklt_id = r.pklt_id
                WHERE m.stat_id IN ({placeholders})
                """,
                tuple(stat_ids),
            )
            cols = [d[0] for d in cur.description]
            out: dict[str, dict] = {}
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                out[str(d["stat_id"])] = d
            return out
    except Exception as exc:
        print(f"[parking] occupancy join 실패 (매핑 테이블 없을 수 있음): {exc}")
        return {}
    finally:
        conn.close()


def parking_occupancy_coefficient(occupancy_rate: float | None) -> float:
    """주차 혼잡도 → 추천점수 접근성 배수 (확률 모델 피처 아님).

    None(정보없음) → 1.0
    <50% → 1.0 / 50~70 → 0.9 / 70~90 → 0.75 / ≥90 → 0.5
    remaining=0에 가까운 고혼잡은 진입 가능성만 반영
    """
    if occupancy_rate is None:
        return 1.0
    rate = float(occupancy_rate)
    if rate < 50:
        return 1.0
    if rate < 70:
        return 0.90
    if rate < 90:
        return 0.75
    return 0.50


def main() -> None:
    import sys

    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    if "--map" in args:
        run_mapping(dry_run=dry_run)
    else:
        print("사용법:")
        print("  py run_parking.py --map            # parking_lot_info ↔ 충전소 매핑")
        print("  py run_parking.py --map --dry-run  # 미리보기")
        print()
        print("실시간 혼잡도는 parking_realtime_status 를 그대로 조인합니다.")
        print("(별도 수집 불필요)")


if __name__ == "__main__":
    main()
