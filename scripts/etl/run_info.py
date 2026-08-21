"""
전기차 충전소 정보(getChargerInfo) 일 1회 수집.
마스터 데이터이므로 하루 한 번이면 충분합니다.
"""
import os
from dotenv import load_dotenv

import time
import xml.etree.ElementTree as ET
from datetime import datetime
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

SERVICE_KEY = os.getenv("EV_SERVICE_KEY", "")
API_BASE = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
ZCODE = "27"  # 대구
NUM_OF_ROWS = 9999
DAILY_AT = "03:00"  # 매일 새벽 3시 (Asia/Seoul 권장)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_info (
    stat_id       VARCHAR(20)  NOT NULL COMMENT '충전소ID',
    chger_id      VARCHAR(10)  NOT NULL COMMENT '충전기ID',
    stat_nm       VARCHAR(100) NULL COMMENT '충전소명',
    chger_type    VARCHAR(10)  NULL COMMENT '충전기타입',
    addr          VARCHAR(200) NULL COMMENT '주소',
    addr_detail   VARCHAR(200) NULL COMMENT '주소상세',
    location      VARCHAR(200) NULL COMMENT '상세위치',
    lat           DECIMAL(12, 8) NULL COMMENT '위도',
    lng           DECIMAL(12, 8) NULL COMMENT '경도',
    use_time      VARCHAR(100) NULL COMMENT '이용가능시간',
    busi_id       VARCHAR(10)  NULL COMMENT '기관ID',
    bnm           VARCHAR(50)  NULL COMMENT '기관명',
    busi_nm       VARCHAR(100) NULL COMMENT '운영기관명',
    busi_call     VARCHAR(30)  NULL COMMENT '연락처',
    output        VARCHAR(20)  NULL COMMENT '충전용량(kW)',
    method        VARCHAR(20)  NULL COMMENT '충전방식',
    zcode         VARCHAR(10)  NULL COMMENT '시도코드',
    zscode        VARCHAR(10)  NULL COMMENT '시군구코드',
    kind          VARCHAR(10)  NULL COMMENT '충전소구분',
    kind_detail   VARCHAR(10)  NULL COMMENT '충전소구분명세',
    parking_free  CHAR(1)      NULL COMMENT '주차료무료 Y/N',
    note          VARCHAR(255) NULL COMMENT '안내',
    limit_yn      CHAR(1)      NULL COMMENT '이용제한 Y/N',
    limit_detail  VARCHAR(150) NULL COMMENT '이용제한 상세',
    del_yn        CHAR(1)      NULL COMMENT '삭제여부 Y/N',
    del_detail    VARCHAR(150) NULL COMMENT '삭제 상세',
    traffic_yn    CHAR(1)      NULL COMMENT '편의제공 Y/N',
    year          VARCHAR(10)  NULL COMMENT '설치년도',
    floor_num     VARCHAR(50)  NULL COMMENT '층수',
    floor_type    VARCHAR(10)  NULL COMMENT '지상/지하',
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                  ON UPDATE CURRENT_TIMESTAMP COMMENT '수집시각',
    PRIMARY KEY (stat_id, chger_id),
    KEY idx_lat_lng (lat, lng),
    KEY idx_zscode (zscode)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='전기차 충전기 마스터 정보'
"""

UPSERT_SQL = """
INSERT INTO ev_charger_info (
    stat_id, chger_id, stat_nm, chger_type, addr, addr_detail, location,
    lat, lng, use_time, busi_id, bnm, busi_nm, busi_call,
    output, method, zcode, zscode, kind, kind_detail,
    parking_free, note, limit_yn, limit_detail, del_yn, del_detail,
    traffic_yn, year, floor_num, floor_type
) VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    stat_nm = VALUES(stat_nm),
    chger_type = VALUES(chger_type),
    addr = VALUES(addr),
    addr_detail = VALUES(addr_detail),
    location = VALUES(location),
    lat = VALUES(lat),
    lng = VALUES(lng),
    use_time = VALUES(use_time),
    busi_id = VALUES(busi_id),
    bnm = VALUES(bnm),
    busi_nm = VALUES(busi_nm),
    busi_call = VALUES(busi_call),
    output = VALUES(output),
    method = VALUES(method),
    zcode = VALUES(zcode),
    zscode = VALUES(zscode),
    kind = VALUES(kind),
    kind_detail = VALUES(kind_detail),
    parking_free = VALUES(parking_free),
    note = VALUES(note),
    limit_yn = VALUES(limit_yn),
    limit_detail = VALUES(limit_detail),
    del_yn = VALUES(del_yn),
    del_detail = VALUES(del_detail),
    traffic_yn = VALUES(traffic_yn),
    year = VALUES(year),
    floor_num = VALUES(floor_num),
    floor_type = VALUES(floor_type),
    updated_at = CURRENT_TIMESTAMP
"""


def text_or_none(element, tag: str):
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value if value else None


def parse_decimal(value: str | None):
    if not value:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def parse_item(item) -> tuple | None:
    stat_id = text_or_none(item, "statId")
    chger_id = text_or_none(item, "chgerId")
    if not stat_id or not chger_id:
        return None

    return (
        stat_id,
        chger_id,
        text_or_none(item, "statNm"),
        text_or_none(item, "chgerType"),
        text_or_none(item, "addr"),
        text_or_none(item, "addrDetail"),
        text_or_none(item, "location"),
        parse_decimal(text_or_none(item, "lat")),
        parse_decimal(text_or_none(item, "lng")),
        text_or_none(item, "useTime"),
        text_or_none(item, "busiId"),
        text_or_none(item, "bnm"),
        text_or_none(item, "busiNm"),
        text_or_none(item, "busiCall"),
        text_or_none(item, "output"),
        text_or_none(item, "method"),
        text_or_none(item, "zcode"),
        text_or_none(item, "zscode"),
        text_or_none(item, "kind"),
        text_or_none(item, "kindDetail"),
        text_or_none(item, "parkingFree"),
        text_or_none(item, "note"),
        text_or_none(item, "limitYn"),
        text_or_none(item, "limitDetail"),
        text_or_none(item, "delYn"),
        text_or_none(item, "delDetail"),
        text_or_none(item, "trafficYn"),
        text_or_none(item, "year"),
        text_or_none(item, "floorNum"),
        text_or_none(item, "floorType"),
    )


def fetch_page(page_no: int) -> tuple[list[tuple], int]:
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": page_no,
        "numOfRows": NUM_OF_ROWS,
        "zcode": ZCODE,
    }
    response = requests.get(API_BASE, params=params, timeout=120)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    result_code = root.findtext(".//header/resultCode")
    if result_code != "00":
        result_msg = root.findtext(".//header/resultMsg")
        raise RuntimeError(f"API error: {result_code} - {result_msg}")

    total_count = int(root.findtext(".//header/totalCount") or "0")
    rows = []
    for item in root.findall(".//body/items/item"):
        row = parse_item(item)
        if row:
            rows.append(row)
    return rows, total_count


def fetch_charger_info() -> list[tuple]:
    all_rows: list[tuple] = []
    page_no = 1
    total_count = None

    while True:
        rows, total_count = fetch_page(page_no)
        all_rows.extend(rows)
        print(f"  page {page_no}: {len(rows)}건 (누적 {len(all_rows)}/{total_count})")
        if not rows or len(all_rows) >= total_count:
            break
        page_no += 1

    return all_rows


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def save_to_db(rows: list[tuple]) -> int:
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


def collect_and_store():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 충전소 정보 수집 시작 (zcode={ZCODE})")
    try:
        rows = fetch_charger_info()
        saved = save_to_db(rows)
        print(f"[{now}] 저장(UPSERT) 완료: {saved}건")
    except Exception as exc:
        print(f"[{now}] 수집 실패: {exc}")


def main():
    print(f"충전기 정보 스케줄러 시작 (매일 {DAILY_AT})")
    collect_and_store()  # 시작 시 1회
    schedule.every().day.at(DAILY_AT).do(collect_and_store)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
