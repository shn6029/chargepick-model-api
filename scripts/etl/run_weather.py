"""
기상청 ASOS 시간자료 수집 → weather 테이블 적재

- API: AsosHourlyInfoService/getWthrDataList
- 지점: 대구(143) 기본
- 갱신: 전일(D-1) 자료, 전일은 보통 11시 이후 조회 가능
- 스케줄: 매일 12:00 (전일 24시간 UPSERT) + 시작 시 최근 N일 백필
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

import pymysql
import requests
import schedule

load_dotenv()

# DB 접속정보는 환경변수에서 읽는다(저장소 루트 `.env`).
# 폴백값을 두지 않는다 — 예전에는 접속정보가 여기 박혀 있어서 `.env` 없이 돌리면
# 의도치 않게 운영 DB 에 조용히 붙었다. 지금은 비어 있으면 연결에서 실패한다.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

# 공공데이터포털 인증키 (ASOS API 활용신청 필요)
SERVICE_KEY = os.getenv("KMA_SERVICE_KEY", "")
API_BASE = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

STN_ID = "143"  # 대구
DAILY_AT = "12:00"  # 전일 자료 안정화 이후
BACKFILL_DAYS = 14  # 시작 시 최근 N일 백필

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS weather (
    observed_at   DATETIME     NOT NULL COMMENT '관측시각(시 단위)',
    stn_id        VARCHAR(10)  NOT NULL COMMENT '종관지점번호',
    stn_nm        VARCHAR(50)  NULL COMMENT '지점명',
    ta            DECIMAL(5,1) NULL COMMENT '기온(C)',
    rn            DECIMAL(6,1) NULL COMMENT '강수량(mm)',
    hm            DECIMAL(5,1) NULL COMMENT '습도(%)',
    ws            DECIMAL(5,1) NULL COMMENT '풍속(m/s)',
    dsnw          DECIMAL(5,1) NULL COMMENT '적설(cm)',
    collected_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                  ON UPDATE CURRENT_TIMESTAMP COMMENT '수집시각',
    PRIMARY KEY (observed_at, stn_id),
    KEY idx_observed_at (observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='기상청 ASOS 시간 기상(학습/분석용)'
"""

UPSERT_SQL = """
INSERT INTO weather (
    observed_at, stn_id, stn_nm, ta, rn, hm, ws, dsnw
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    stn_nm = VALUES(stn_nm),
    ta = VALUES(ta),
    rn = VALUES(rn),
    hm = VALUES(hm),
    ws = VALUES(ws),
    dsnw = VALUES(dsnw),
    collected_at = CURRENT_TIMESTAMP
"""


def text_or_none(element, tag: str):
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value if value else None


def parse_decimal(value: str | None):
    if value is None:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def parse_observed_at(value: str | None):
    """응답 tm 예: '2019-01-20 01' 또는 '2019-01-20 01:00'."""
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def fetch_hourly(start_dt: datetime, end_dt: datetime, stn_id: str = STN_ID) -> list[tuple]:
    """
    start_dt~end_dt (시 단위) ASOS 시간자료 조회.
    end는 전일까지만 유효할 수 있음.
    """
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": 900,
        "dataType": "XML",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_dt.strftime("%Y%m%d"),
        "startHh": start_dt.strftime("%H"),
        "endDt": end_dt.strftime("%Y%m%d"),
        "endHh": end_dt.strftime("%H"),
        "stnIds": stn_id,
    }

    last_error: Exception | None = None
    response = None
    for attempt in range(1, 4):
        try:
            response = requests.get(API_BASE, params=params, timeout=60)
            if response.status_code in (502, 503, 504):
                raise requests.HTTPError(
                    f"{response.status_code} gateway error", response=response
                )
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
            response = None
    if response is None:
        raise RuntimeError(
            f"기상 API 요청 실패(재시도 후). "
            f"공공데이터포털에서 'ASOS 시간자료' 활용신청/키 확인 필요. 원인: {last_error}"
        )

    root = ET.fromstring(response.content)
    result_code = root.findtext(".//header/resultCode") or root.findtext(".//resultCode")
    if result_code and result_code not in ("00", "0"):
        result_msg = root.findtext(".//header/resultMsg") or root.findtext(".//resultMsg")
        raise RuntimeError(f"API error: {result_code} - {result_msg}")

    rows = []
    items = root.findall(".//item") or root.findall(".//items/item")
    for item in items:
        observed_at = parse_observed_at(text_or_none(item, "tm"))
        stn = text_or_none(item, "stnId") or stn_id
        if not observed_at:
            continue
        rows.append(
            (
                observed_at,
                stn,
                text_or_none(item, "stnNm"),
                parse_decimal(text_or_none(item, "ta")),
                parse_decimal(text_or_none(item, "rn")),
                parse_decimal(text_or_none(item, "hm")),
                parse_decimal(text_or_none(item, "ws")),
                parse_decimal(text_or_none(item, "dsnw")),
            )
        )
    return rows


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def save_rows(rows: list[tuple]) -> int:
    if not rows:
        return 0
    conn = pymysql.connect(**DB_CONFIG)
    try:
        ensure_table(conn)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def collect_day(day: datetime, stn_id: str = STN_ID) -> int:
    """하루(00~23시) 자료 수집."""
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = day.replace(hour=23, minute=0, second=0, microsecond=0)
    rows = fetch_hourly(start, end, stn_id=stn_id)
    return save_rows(rows)


def collect_yesterday():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    yesterday = datetime.now().date() - timedelta(days=1)
    day = datetime.combine(yesterday, datetime.min.time())
    print(f"[{now}] 기상 수집 시작 (전일 {yesterday}, stn={STN_ID})")
    try:
        saved = collect_day(day)
        print(f"[{now}] 저장(UPSERT) 완료: {saved}건")
    except Exception as exc:
        print(f"[{now}] 수집 실패: {exc}")


def backfill_recent(days: int = BACKFILL_DAYS):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 기상 백필 시작 (최근 {days}일, 전일까지)")
    today = datetime.now().date()
    total = 0
    for i in range(1, days + 1):
        day = datetime.combine(today - timedelta(days=i), datetime.min.time())
        try:
            n = collect_day(day)
            total += n
            print(f"  {day.date()}: {n}건")
            time.sleep(0.3)
        except Exception as exc:
            print(f"  {day.date()}: 실패 - {exc}")
    print(f"[{now}] 백필 완료: 총 {total}건")


def ensure_table_only():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        ensure_table(conn)
        print("weather 테이블 준비 완료 (없으면 생성)")
    finally:
        conn.close()


def main():
    import sys

    print(f"기상(ASOS) 스케줄러 (매일 {DAILY_AT}, 지점={STN_ID})")
    print("참고: 전일 자료는 보통 11시 이후 조회 가능합니다.")
    ensure_table_only()

    # py run_weather.py --once  → 전일만 1회
    # py run_weather.py --backfill → 최근 N일만
    args = set(sys.argv[1:])
    if "--once" in args:
        collect_yesterday()
        return
    if "--backfill" in args:
        backfill_recent(BACKFILL_DAYS)
        return

    backfill_recent(BACKFILL_DAYS)
    schedule.every().day.at(DAILY_AT).do(collect_yesterday)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
