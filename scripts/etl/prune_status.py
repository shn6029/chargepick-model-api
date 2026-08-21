"""
ev_charger_status / ev_charger_features 보존 정책 (21일 롤링)

왜 필요한가
-----------
run_status_snapshot.py 가 10분 주기로 전량(25,433대)을 적재하면 약 370만행/일,
월 수 GB 규모로 증가한다. 학습 창이 1~2주이므로 원본을 영구 보관할 이유가 없다.

정책
----
  원본 status/features   최근 RETENTION_DAYS(기본 21)일 유지
  초과분                 삭제 (필요하면 사전에 집계 아카이브를 떠 둘 것)

features 는 status.log_id 를 PK 로 참조하므로 status 보다 **먼저** 지운다.
(고아 features 행이 남으면 서빙 INNER JOIN 이 엉킨다)

긴 잠금을 피하려고 청크 단위로 지운다.

사용:
  py scripts/etl/prune_status.py --dry-run        # 삭제 대상만 집계
  py scripts/etl/prune_status.py                  # 21일 초과분 삭제
  py scripts/etl/prune_status.py --days 30        # 보존 기간 변경
"""

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

RETENTION_DAYS = 21
CHUNK = 20000


def report(conn, days: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(created_at <  NOW() - INTERVAL %s DAY) AS expired,
                   MIN(created_at) AS oldest
            FROM ev_charger_status
            """,
            (days,),
        )
        total, expired, oldest = cur.fetchone()
        print(f"  ev_charger_status   총 {total:,}행 / 삭제대상 {int(expired or 0):,}행 "
              f"/ 최古 {oldest}")

        cur.execute(
            """
            SELECT COUNT(*) FROM ev_charger_features
            WHERE created_at < NOW() - INTERVAL %s DAY
            """,
            (days,),
        )
        print(f"  ev_charger_features 삭제대상 {cur.fetchone()[0]:,}행")


def prune_table(conn, table: str, days: int) -> int:
    """청크 단위 삭제. 긴 트랜잭션/잠금을 피한다."""
    deleted = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table} "
                f"WHERE created_at < NOW() - INTERVAL %s DAY LIMIT %s",
                (days, CHUNK),
            )
            n = cur.rowcount
        conn.commit()
        deleted += n
        if n:
            print(f"    {table}: {deleted:,}행 삭제")
        if n < CHUNK:
            return deleted
        time.sleep(0.2)


def main() -> None:
    p = argparse.ArgumentParser(description="status/features 보존 정책 적용")
    p.add_argument("--days", type=int, default=RETENTION_DAYS,
                   metavar="N", help=f"보존 기간(일). 기본 {RETENTION_DAYS}")
    p.add_argument("--dry-run", action="store_true",
                   help="삭제하지 않고 대상 건수만 출력")
    args = p.parse_args()

    if not DB_CONFIG["password"]:
        raise SystemExit("환경변수 DB_PASSWORD 미설정 (.env.example 참고)")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 보존 정책: 최근 {args.days}일 유지")

    conn = pymysql.connect(**DB_CONFIG)
    try:
        report(conn, args.days)
        if args.dry_run:
            print("  (--dry-run: 삭제하지 않음)")
            return

        # features 를 먼저 (status.log_id 참조)
        f = prune_table(conn, "ev_charger_features", args.days)
        s = prune_table(conn, "ev_charger_status", args.days)
        print(f"[{now}] 완료: features {f:,}행, status {s:,}행 삭제")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
