"""
충전기 상태 변경분 수집 (getChargerStatus) → ev_charger_status

period=5 는 최근 상태가 바뀐 충전기만 돌려준다(실측 상한 ~514대).
전량 스냅샷은 run_status_snapshot.py 가 getChargerInfo 로 따로 받는다.

주의: 이 파일은 서버(~/scheduler/run.py)에서 검증된 구현을 그대로 따른다.
필드를 튜플 순서로 매핑하지 말 것 — INSERT 컬럼과 어긋나면 조용히
stat_id/chger_id/busi_id 가 뒤섞여 테이블 전체가 오염된다.
"""

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import pymysql
import requests
import schedule

try:  # 저장소 루트의 .env (컨테이너에서는 환경변수로 주입)
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

# 환경변수 우선. 하드코딩 fallback 은 Phase 4(키 로테이션) 에서 제거 예정.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

SERVICE_KEY = os.getenv("EV_SERVICE_KEY", "")
ZCODE = os.getenv("EV_ZCODE", "27")  # 27=대구

API_URL = (
    "https://apis.data.go.kr/B552584/EvCharger/getChargerStatus"
    f"?serviceKey={SERVICE_KEY}"
    f"&pageNo=1&numOfRows=9999&period=5&zcode={ZCODE}"
)

INSERT_SQL = """
INSERT INTO ev_charger_status
    (stat_id, chger_id, busi_id, stat, stat_upd_dt, last_tsdt, last_tedt, now_tsdt)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def text_or_none(element, tag: str):
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value if value else None


def parse_api_datetime(value: str | None):
    """API 는 'YYYYMMDDHHMMSS' 문자열로 준다. DATETIME 으로 변환해서 넣는다."""
    if not value:
        return None
    return datetime.strptime(value, "%Y%m%d%H%M%S")


def parse_stat(value: str | None):
    if value is None:
        return None
    return int(value)


def fetch_charger_status() -> list[tuple]:
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    result_code = root.findtext(".//header/resultCode")
    if result_code != "00":
        result_msg = root.findtext(".//header/resultMsg")
        raise RuntimeError(f"API error: {result_code} - {result_msg}")

    rows = []
    for item in root.findall(".//body/items/item"):
        # INSERT_SQL 컬럼 순서와 1:1 로 대응시킨다(튜플 순서 매핑 금지).
        stat_id = text_or_none(item, "statId")
        chger_id = text_or_none(item, "chgerId")
        stat_upd_dt = parse_api_datetime(text_or_none(item, "statUpdDt"))
        if not stat_id or not chger_id or not stat_upd_dt:
            continue

        rows.append(
            (
                stat_id,
                chger_id,
                text_or_none(item, "busiId"),
                parse_stat(text_or_none(item, "stat")),
                stat_upd_dt,
                parse_api_datetime(text_or_none(item, "lastTsdt")),
                parse_api_datetime(text_or_none(item, "lastTedt")),
                parse_api_datetime(text_or_none(item, "nowTsdt")),
            )
        )
    return rows


def save_to_db(rows: list[tuple]) -> int:
    if not rows:
        return 0

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def collect_and_store():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 충전소 상태 수집 시작")
    try:
        rows = fetch_charger_status()
        saved = save_to_db(rows)
        print(f"[{now}] 저장 완료: {saved}건")
    except Exception as exc:
        print(f"[{now}] 수집 실패: {exc}")


def main():
    print("충전기 상태 스케줄러 시작 (5분 간격)")
    collect_and_store()
    schedule.every(5).minutes.do(collect_and_store)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
