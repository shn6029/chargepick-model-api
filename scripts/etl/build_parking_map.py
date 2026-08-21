# -*- coding: utf-8 -*-
"""충전소 ↔ 주차장 매핑 테이블(ev_charger_parking_map) 생성·갱신.

왜 필요한가
-----------
recommend_api/parking.py 의 load_latest_occupancy_for_stations() 가 이 테이블을
조인한다. 없으면 함수가 except 로 빈 dict 를 돌려주고, 추천 응답의
parking_occupancy_coefficient 가 늘 기본값 1.0 이 된다. **에러가 안 나서 기능이
꺼진 걸 알아채기 어렵다** — 2026-08-04 까지 그 상태였다.

매칭 기준: 좌표 100m 이내
----------------------
100m 는 임의로 고른 값이 아니다. 실측하면 매칭된 충전소의 실제 거리가 대부분
1m 안팎이고 이름도 "범어공영주차장", "칠성공영주차장" 처럼 주차장 그 자체다.
즉 **주차장 부지 안에 설치된 충전기**라, 주차장 점유율이 곧 "충전기까지 진입
가능한가" 를 뜻한다. 300m 까지 넓히면 커버리지가 3.3% → 14.8% 로 오르지만
그 거리에서는 인과가 끊긴다.

실시간 상태(parking_realtime_status)가 들어오는 주차장만 대상으로 한다.
parking_lot_info 에는 1,767곳이 있지만 실시간이 오는 곳은 109곳뿐이다.

용도 — 표시용이지 피처용이 아니다
-------------------------------
모델 피처로는 기각됐다. 시간대 평균을 제거한 잔차 상관이 -0.073(R² 0.54%)로
사실상 없다(data/reports/parking_결합평가_20260804.md). 반면 표시용 가치는 있다.
점유율 90% 이상인 관측이 10.7% 라, 충전기가 비어 있어도 주차장이 만차라 못
들어가는 상황을 미리 알려줄 수 있다.

사용:
  py scripts/etl/build_parking_map.py --dry-run   # 매칭 결과만 출력
  py scripts/etl/build_parking_map.py             # 테이블 생성·갱신
  py scripts/etl/build_parking_map.py --radius 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

DEFAULT_RADIUS_M = 100

# 이 테이블은 **양쪽 세계에 조인**한다. 콜레이션을 컬럼별로 맞춰야 한다.
#   stat_id -> ev_charger_info / ev_charger_status  (utf8mb4_general_ci)
#   pklt_id -> parking_realtime_status / parking_lot_info (utf8mb4_unicode_ci, 팀원 소유)
# 하나로 통일하면 반대쪽 조인이 1267 "Illegal mix of collations" 로 죽는다.
# parking.py 는 그 예외를 except 로 삼켜 빈 dict 를 돌려주므로, 에러 없이
# 주차 정보만 사라진다(2026-08-04 에 실제로 겪었다).
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_parking_map (
    stat_id              VARCHAR(20)  NOT NULL COLLATE utf8mb4_general_ci,
    pklt_id              VARCHAR(40)  NOT NULL COLLATE utf8mb4_unicode_ci,
    parking_nm           VARCHAR(200),
    parking_total_spaces INT,
    parking_fee_type     VARCHAR(20),
    parking_24h          TINYINT(1) DEFAULT 0,
    distance_m           DECIMAL(8,2),
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                   ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stat_id),
    KEY idx_pklt (pklt_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

UPSERT_SQL = """
INSERT INTO ev_charger_parking_map
    (stat_id, pklt_id, parking_nm, parking_total_spaces,
     parking_fee_type, parking_24h, distance_m)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    pklt_id = VALUES(pklt_id),
    parking_nm = VALUES(parking_nm),
    parking_total_spaces = VALUES(parking_total_spaces),
    parking_fee_type = VALUES(parking_fee_type),
    parking_24h = VALUES(parking_24h),
    distance_m = VALUES(distance_m)
"""


def haversine_matrix(lat1, lng1, lat2, lng2) -> np.ndarray:
    """(N,) x (M,) → (N,M) 미터 거리."""
    la1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lo1 = np.radians(np.asarray(lng1, dtype=float))[:, None]
    la2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lo2 = np.radians(np.asarray(lng2, dtype=float))[None, :]
    return 2 * 6371000 * np.arcsin(np.sqrt(
        np.sin((la2 - la1) / 2) ** 2
        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    ))


def build(radius_m: int) -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        # 실시간 상태가 실제로 들어오는 주차장만. 좌표 없는 곳은 매칭 불가.
        lots = pd.read_sql(
            """
            SELECT i.pklt_id, i.pklt_nm, i.lat, i.lot AS lng, i.total_spaces,
                   i.crg_levy_se_nm, i.oper_hr_wkday_se_cd
            FROM parking_lot_info i
            WHERE i.lat IS NOT NULL AND i.lot IS NOT NULL AND i.lat > 0
              AND i.pklt_id IN (
                  SELECT DISTINCT pklt_id FROM parking_realtime_status
                  WHERE collected_at >= NOW() - INTERVAL 3 DAY
              )
            """,
            conn,
        ).drop_duplicates("pklt_id")

        stations = pd.read_sql(
            """
            SELECT stat_id, MAX(stat_nm) stat_nm, MAX(lat) lat, MAX(lng) lng
            FROM ev_charger_info
            WHERE (del_yn IS NULL OR del_yn <> 'Y')
              AND lat IS NOT NULL AND lng IS NOT NULL
            GROUP BY stat_id
            """,
            conn,
        )
    finally:
        conn.close()

    for df, cols in ((lots, ("lat", "lng")), (stations, ("lat", "lng"))):
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    lots = lots.dropna(subset=["lat", "lng"])
    stations = stations.dropna(subset=["lat", "lng"])
    print(f"실시간 주차장 {len(lots):,}곳 / 충전소 {len(stations):,}곳")

    d = haversine_matrix(stations.lat, stations.lng, lots.lat, lots.lng)
    idx = d.argmin(axis=1)
    stations["distance_m"] = d.min(axis=1)
    near = lots.iloc[idx].reset_index(drop=True)
    for c in ("pklt_id", "pklt_nm", "total_spaces", "crg_levy_se_nm",
              "oper_hr_wkday_se_cd"):
        stations[c] = near[c].values

    out = stations[stations.distance_m <= radius_m].copy()
    # '전일운영' 이 24시간 운영. 결측은 0(모름)으로 둔다.
    out["parking_24h"] = (out.oper_hr_wkday_se_cd == "전일운영").astype(int)
    out["parking_fee_type"] = out.crg_levy_se_nm.fillna("미상")
    out["parking_total_spaces"] = pd.to_numeric(
        out.total_spaces, errors="coerce"
    ).fillna(0).astype(int)
    return out.sort_values("distance_m")


def main() -> None:
    p = argparse.ArgumentParser(description="충전소-주차장 매핑 생성")
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M,
                   metavar="M", help=f"매칭 반경(m). 기본 {DEFAULT_RADIUS_M}")
    p.add_argument("--dry-run", action="store_true", help="저장하지 않고 출력만")
    args = p.parse_args()

    if not DB_CONFIG["password"]:
        raise SystemExit("DB_PASSWORD 미설정 (.env.example 참고)")

    m = build(args.radius)
    print(f"\n{args.radius}m 이내 매칭 {len(m):,}쌍")
    print(f"  거리 중앙값 {m.distance_m.median():.1f}m / 최대 {m.distance_m.max():.1f}m")
    print(f"  요금 구분: {dict(m.parking_fee_type.value_counts())}")
    print(f"  24시간 운영: {int(m.parking_24h.sum())}곳")
    print("\n가까운 순 상위 10:")
    print(m[["stat_nm", "pklt_nm", "distance_m"]].head(10).to_string(index=False))

    if args.dry_run:
        print("\n(--dry-run: 저장하지 않음)")
        return

    rows = [
        (r.stat_id, r.pklt_id, r.pklt_nm, int(r.parking_total_spaces),
         r.parking_fee_type, int(r.parking_24h), round(float(r.distance_m), 2))
        for r in m.itertuples()
    ]
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_SQL)
        conn.commit()
        # 매핑은 1,000행 미만이라 청크가 필요 없지만, 공용 DB 규칙을 따른다.
        with conn.cursor() as cur:
            for i in range(0, len(rows), 500):
                cur.executemany(UPSERT_SQL, rows[i:i + 500])
                conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ev_charger_parking_map")
            total = cur.fetchone()[0]
    finally:
        conn.close()
    print(f"\n저장 완료: {len(rows):,}쌍 upsert / 테이블 총 {total:,}행")


if __name__ == "__main__":
    main()
