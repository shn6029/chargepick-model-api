"""parking_* 테이블 보존 정책 (테이블별 차등)

왜 필요한가
-----------
2026-08-04 실측: parking 계열이 **105 MB/일** 로 늘고 있는데 보존 정책이 하나도
없었다. EV 파이프라인(status+features, 74 MB/일)보다 빠르고, 그쪽은 45일 롤링으로
3.3GB 에서 멈추는 반면 parking 은 무한히 늘어 1년이면 37.5GB 다(디스크 여유 45GB).

    테이블                     MB/일    우리 사용
    parking_api_raw            38.1    X  원본 JSON 아카이브
    parking_realtime_status    27.6    O  추천 API 주차 혼잡도 표시
    parking_realtime_area      34.7    X  면(구획) 단위
    parking_realtime_zone       3.8    X
    parking_realtime_floor      0.9    X
    parking_lot_info            0.0    O  매핑 생성(build_parking_map.py)

보존일을 왜 이렇게 잡았나
----------------------
- api_raw 7일: `normalize_latest_parking_realtime()` 이 **최신 raw 한 건만** 읽는다
  (normalize_realtime.py:73). 과거분은 파싱 로직을 고쳐 재처리할 때만 쓰는데 그건
  최근 것으로 충분하다. 행당 263KB 짜리 LONGTEXT 라 버퍼 풀도 크게 밀어낸다 —
  실측으로 이 컬럼 하나가 32초 만에 풀을 75%→97% 로 채웠다.
- realtime_status 45일: 추천 API 는 최신 1건만 조회하므로 이력이 길 이유는 없지만,
  EV 쪽 롤링(45일)과 맞춰 두면 함께 분석할 때 창이 어긋나지 않는다.
- area/zone/floor 7일: 우리 코드가 한 번도 참조하지 않는다. 백엔드에서 구획 단위를
  쓰기 시작하면 늘릴 것.

주의: 이 스크립트는 **팀 공용 DB** 를 지운다. 2GB 박스에 버퍼 풀 512MB 이므로
대량 DELETE 를 한 번에 던지면 안 된다. 청크(2만행) + 청크 사이 sleep 을 지킬 것.

사용:
  py scripts/etl/prune_parking.py --dry-run     # 삭제 대상만 집계
  py scripts/etl/prune_parking.py               # 정책 적용
  py scripts/etl/prune_parking.py --raw-days 3
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pymysql

try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
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

CHUNK = 20000
PAUSE_SEC = 0.3

# (테이블, 시각컬럼, 기본 보존일)
POLICY = [
    ("parking_api_raw", "collected_at", 7),
    ("parking_realtime_area", "collected_at", 7),
    ("parking_realtime_zone", "collected_at", 7),
    ("parking_realtime_floor", "collected_at", 7),
    ("parking_realtime_status", "collected_at", 45),
]


def report(conn, policy) -> None:
    with conn.cursor() as cur:
        for table, col, days in policy:
            cur.execute(
                f"SELECT COUNT(*), SUM({col} < NOW() - INTERVAL %s DAY), MIN({col}) "
                f"FROM {table}",
                (days,),
            )
            total, expired, oldest = cur.fetchone()
            print(f"  {table:<26} 보존 {days:>2}일 | 총 {total:>9,}행 "
                  f"| 삭제대상 {int(expired or 0):>9,}행 | 최古 {oldest}")


def prune_table(conn, table: str, col: str, days: int) -> int:
    """청크 단위 삭제. 한 번에 던지면 공용 DB 가 멈춘다."""
    deleted = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table} WHERE {col} < NOW() - INTERVAL %s DAY LIMIT %s",
                (days, CHUNK),
            )
            n = cur.rowcount
        conn.commit()
        deleted += n
        if n:
            print(f"    {table}: {deleted:,}행 삭제")
        if n < CHUNK:
            return deleted
        time.sleep(PAUSE_SEC)


def main() -> None:
    p = argparse.ArgumentParser(description="parking_* 보존 정책 적용")
    p.add_argument("--raw-days", type=int, default=None, metavar="N",
                   help="parking_api_raw 보존일 (기본 7)")
    p.add_argument("--detail-days", type=int, default=None, metavar="N",
                   help="area/zone/floor 보존일 (기본 7)")
    p.add_argument("--status-days", type=int, default=None, metavar="N",
                   help="parking_realtime_status 보존일 (기본 45)")
    p.add_argument("--dry-run", action="store_true", help="삭제하지 않고 집계만")
    args = p.parse_args()

    if not DB_CONFIG["password"]:
        raise SystemExit("환경변수 DB_PASSWORD 미설정 (.env.example 참고)")

    policy = []
    for table, col, days in POLICY:
        if table == "parking_api_raw" and args.raw_days:
            days = args.raw_days
        elif table == "parking_realtime_status" and args.status_days:
            days = args.status_days
        elif table.startswith("parking_realtime_") and args.detail_days:
            days = args.detail_days
        policy.append((table, col, days))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] parking 보존 정책")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        report(conn, policy)
        if args.dry_run:
            print("  (--dry-run: 삭제하지 않음)")
            return
        print()
        total = 0
        for table, col, days in policy:
            total += prune_table(conn, table, col, days)
        print(f"[{now}] 완료: 총 {total:,}행 삭제")
        print("  (테이블 파일 크기는 OPTIMIZE TABLE 전까지 즉시 줄지 않는다 —"
              " 빈 공간은 재사용되므로 보통 그대로 두면 된다)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
