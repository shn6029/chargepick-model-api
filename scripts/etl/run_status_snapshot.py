"""
전체 충전기 상태 스냅샷 수집 → ev_charger_status

왜 필요한가
-----------
`run.py` 가 쓰는 getChargerStatus 는 **최근 상태가 바뀐 충전기만** 돌려준다.
period 를 키워도 상한이 있다(실측: period=5 → ~326대, period=10/30/60/1440 → ~514대).
그 결과 충전 중(stat=3)인 충전기는 세션이 끝날 때까지 재보고되지 않아
학습 라벨링(merge_asof)에서 대량 탈락한다. 실측 h30 매칭률:

    stat=2(대기)   82.8%
    stat=3(충전중)   9.1%   <-- 9배 차이
    stat=4(운영중지)  0.0%

이 편향 때문에 학습셋 positive_rate 가 원본 가용률(≈0.74) 대비 0.88 로 부풀고,
h30/45/60 의 unavailable-recall 이 0.25~0.30 으로 붕괴한다.

해법
----
getChargerInfo 는 마스터 정보이면서 `stat`, `statUpdDt`, `lastTsdt`, `lastTedt`,
`nowTsdt` 를 **전 충전기에 대해** 함께 준다(실측 zcode=27 기준 25,433대, 3페이지).
이걸 주기적으로 받아 ev_charger_status 에 스냅샷으로 적재하면, 상태와 무관하게
모든 충전기가 일정 주기로 관측되어 라벨 매칭률이 균등해진다.

기존 5분 변경분 수집(run.py)은 그대로 두고 **병행**한다.
- run.py                  5분 주기, 변경분  → 짧은 horizon(5~20분) 해상도 유지
- run_status_snapshot.py  6시간 주기, 전량  → 패널 앵커링 · 커버리지 · 드리프트 보정

주기를 6시간으로 잡은 이유 (2026-08-03 개정)
--------------------------------------------
당초 10분으로 잡았던 근거는 "merge_asof(±6분) 매칭 창에 t+h 가 들어갈 확률"
이었다. 이 전제 자체가 틀렸다. delta 로그를 LOCF(마지막 관측 상태 유지)로
복원하면 물리적 행이 없어도 임의 시각의 상태를 알 수 있다. 라벨링은 수집
주기가 아니라 **학습 코드**에서 풀어야 할 문제였다(model_store.py 참조).

그리고 10분 주기는 실제로 서버를 눕혔다. 2026-07-31 12:50/13:00 두 배치
직후 DB 가 13:05~13:35, 13:45~14:15 멈췄고 15:13 에 재시작됐다.
적재량이 25,433대 × 6회/시간 × 24시간 ≈ **370만행/일 = 739MB/일**,
21일 보존이면 15.2GB 인데 이 서버는 RAM 2GB(Lightsail)다. 불가능하다.

그럼 얼마나 자주 찍어야 하는가. 2026-07-31 스냅샷을 정답지로 delta LOCF
복원 정확도를 측정하면 무너지는 지점이 명확하다:

    상태 나이 30분~8시간   정확도 97.7%   가용률 편차 -0.5%p
    상태 나이  8~12시간    정확도 91.6%   가용률 편차 -7.3%p
    상태 나이 12~24시간    정확도 73.2%   가용률 편차 -23.9%p  <-- 붕괴

8시간까지 평평하다가 그 뒤로 무너진다(주 원인: delta 가 충전 종료 3→2
이벤트를 놓쳐 충전기가 '충전중'에 갇힌다). 즉 **모든 충전기의 staleness 가
8시간을 넘지 않게 앵커링**하면 충분하고, 6시간이면 여유까지 있다.

    주기      행/일        21일 보존
    10분     3,655,440     15.2 GB   <-- 서버 사망
     1시간     609,240      2.5 GB
     6시간     101,540      0.4 GB   <-- 채택 (현재 delta 78,000행/일의 1.3배)
     1일        25,385      0.1 GB   <-- staleness 24시간, 정확도 73% 구간

부수 효과로 커버리지 문제도 같이 풀린다. delta 에만 의존하면 12일 동안
한 번도 상태가 안 바뀐 충전기가 등장하지 않아 25,433대 중 **76.8%** 만
관측된다. 6시간 스냅샷은 이걸 100% 로 만든다.

사용 (명령 실행 시 1회만 수집·저장 후 종료):
  py scripts/etl/run_status_snapshot.py

주기 반복은 스케줄러(cron / Windows 작업 스케줄러)에서 6시간마다 호출한다.
(이 스크립트 자체는 while 루프로 상주하지 않는다.)
실측 소요: 25,385행 기준 약 260~320초(API 3페이지 + 51청크 커밋).

적재량(6시간 주기): 25,433대 × 4회/일 ≈ **10만행/일**.
보존은 scripts/etl/prune_status.py 로 21일 롤링(기본)한다. 반드시 함께 운영할 것.

source 컬럼
-----------
스냅샷/변경분을 구분하기 위해 ev_charger_status.source 를 쓴다
('snapshot' / 'delta'). 컬럼이 없으면 최초 실행 시 자동 추가하며,
기존 행과 run.py 의 INSERT 는 DEFAULT 'delta' 로 채워진다.
"""

import os
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime

import pymysql
import requests

try:  # 저장소 루트의 .env 를 읽는다 (없으면 순수 환경변수만 사용)
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

# 접속 정보는 환경변수 우선 (docs/팀원_DB_접속_가이드.md: 코드에 비밀번호 커밋 금지).
# 기존 scripts/etl/*.py 는 아직 하드코딩 상태이며 일괄 정리 대상이다.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

# run.py 와 동일하게 하드코딩 fallback 을 둔다(환경변수 미설정 시 배포가 조용히
# 죽는 것보다 낫다). Phase 4 에서 키 로테이션 후 양쪽 fallback 을 함께 제거한다.
SERVICE_KEY = os.getenv("EV_SERVICE_KEY", "")
ZCODE = os.getenv("EV_ZCODE", "27")  # 27=대구
API_BASE = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
NUM_OF_ROWS = 9999
# 작업 스케줄러에 등록할 때의 권장 주기(분). 이 스크립트는 상주하지 않는다.
RECOMMENDED_INTERVAL_MIN = 10

# source/observed_at/batch_id 는 API 필드가 아니라 수집 메타라 뒤에 붙인다.
INSERT_SQL = """
INSERT INTO ev_charger_status
    (stat_id, chger_id, busi_id, stat, stat_upd_dt, last_tsdt, last_tedt,
     now_tsdt, source, observed_at, snapshot_batch_id)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, 'snapshot', %s, %s)
"""

# 기존 행/run.py 의 INSERT 는 DEFAULT 로 'delta' 가 들어간다.
# observed_at 은 "언제 관측했는가"(폴링 시각)로, created_at(DB insert 시각)과 다르다.
# 3페이지 fetch + 배치 insert 에 수십 초가 걸리므로 배치 전체가 같은 값을 갖게 한다.
ADD_COLUMN_SQLS = {
    "source": """
        ALTER TABLE ev_charger_status
            ADD COLUMN source VARCHAR(10) NOT NULL DEFAULT 'delta'
            COMMENT '수집 경로: delta=getChargerStatus 변경분, snapshot=getChargerInfo 전량'
    """,
    "observed_at": """
        ALTER TABLE ev_charger_status
            ADD COLUMN observed_at DATETIME NULL
            COMMENT '실제 관측(폴링) 시각. NULL 이면 created_at 사용'
    """,
    "snapshot_batch_id": """
        ALTER TABLE ev_charger_status
            ADD COLUMN snapshot_batch_id VARCHAR(36) NULL
            COMMENT '스냅샷 배치 식별자 (ev_status_snapshot_batch 참조)'
    """,
}

# 배치가 전 페이지를 다 받았는지(부분 스냅샷 아닌지) 기록.
# 미완료 배치를 학습에서 걸러내려면 이 표가 필요하다.
CREATE_BATCH_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_status_snapshot_batch (
    batch_id       VARCHAR(36)  NOT NULL COMMENT '배치 UUID',
    observed_at    DATETIME     NOT NULL COMMENT '폴링 시작 시각',
    expected_count INT          NULL     COMMENT 'API totalCount',
    fetched_count  INT          NULL     COMMENT '실제 파싱 건수',
    saved_count    INT          NULL     COMMENT 'DB 저장 건수',
    is_complete    TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '전 페이지 수신 여부',
    error          VARCHAR(500) NULL,
    finished_at    TIMESTAMP    NULL,
    PRIMARY KEY (batch_id),
    KEY idx_observed_at (observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='전량 스냅샷 배치 이력'
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
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def parse_stat(value: str | None):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_item(item) -> tuple | None:
    """INSERT_SQL 컬럼 순서와 1:1 로 대응시킨다(튜플 순서 매핑 금지).

    statId/chgerId/stat/statUpdDt 중 하나라도 없으면 버린다.
    stat_upd_dt 는 테이블이 NOT NULL 이라 넣으면 배치 전체가 실패한다
    (전량 스냅샷 25,433건 중 실측 ~49건이 statUpdDt 결측).
    run.py 의 동일 가드와 맞춘 것이다.
    """
    stat_id = text_or_none(item, "statId")
    chger_id = text_or_none(item, "chgerId")
    stat = parse_stat(text_or_none(item, "stat"))
    stat_upd_dt = parse_api_datetime(text_or_none(item, "statUpdDt"))
    if not stat_id or not chger_id or stat is None or not stat_upd_dt:
        return None

    return (
        stat_id,
        chger_id,
        text_or_none(item, "busiId"),
        stat,
        stat_upd_dt,
        parse_api_datetime(text_or_none(item, "lastTsdt")),
        parse_api_datetime(text_or_none(item, "lastTedt")),
        parse_api_datetime(text_or_none(item, "nowTsdt")),
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
    items = root.findall(".//body/items/item")
    rows = []
    for item in items:
        row = parse_item(item)
        if row:
            rows.append(row)
    # 수신 건수(len(items))와 파싱 생존 건수(len(rows))는 다르다.
    # 완결성 판정은 수신 건수로 해야 한다.
    return rows, total_count, len(items)


def fetch_snapshot() -> tuple[list[tuple], int, int]:
    """전 페이지 순회. 단일 페이지로는 9999건에서 잘린다(실측 총 25,433건).

    반환: (행 목록, API totalCount, 수신 item 총계)
    """
    all_rows: list[tuple] = []
    received = 0
    page_no = 1
    total_count = 0

    while True:
        rows, total_count, n_items = fetch_page(page_no)
        all_rows.extend(rows)
        received += n_items
        print(
            f"  page {page_no}: 수신 {n_items}건 / 유효 {len(rows)}건 "
            f"(누적 {received}/{total_count})"
        )
        if not n_items or received >= total_count:
            break
        page_no += 1
        time.sleep(0.3)

    skipped = received - len(all_rows)
    if skipped:
        print(f"  필수 필드 결측으로 제외: {skipped}건")
    return all_rows, total_count, received


def ensure_schema(conn) -> None:
    """필요한 컬럼/테이블이 없으면 추가. MySQL 은 ADD COLUMN IF NOT EXISTS 가 없다."""
    with conn.cursor() as cur:
        for col, sql in ADD_COLUMN_SQLS.items():
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = %s AND table_name = 'ev_charger_status'
                  AND column_name = %s
                """,
                (DB_CONFIG["database"], col),
            )
            if not cur.fetchone()[0]:
                print(f"  ev_charger_status.{col} 컬럼 추가")
                cur.execute(sql)
        cur.execute(CREATE_BATCH_TABLE_SQL)
    conn.commit()


def record_batch(conn, batch: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ev_status_snapshot_batch
                (batch_id, observed_at, expected_count, fetched_count,
                 saved_count, is_complete, error, finished_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                expected_count = VALUES(expected_count),
                fetched_count  = VALUES(fetched_count),
                saved_count    = VALUES(saved_count),
                is_complete    = VALUES(is_complete),
                error          = VALUES(error),
                finished_at    = NOW()
            """,
            (
                batch["batch_id"],
                batch["observed_at"],
                batch.get("expected_count"),
                batch.get("fetched_count"),
                batch.get("saved_count"),
                1 if batch.get("is_complete") else 0,
                batch.get("error"),
            ),
        )
    conn.commit()


DEFAULT_BATCH_SIZE = 500
DEFAULT_PAUSE_SEC = 0.5


def save_to_db(
    rows: list[tuple],
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_sec: float = DEFAULT_PAUSE_SEC,
    on_progress=None,
) -> int:
    """청크마다 커밋한다. 배치 전체를 단일 트랜잭션으로 묶지 않는다.

    왜 청크 커밋인가
    ---------------
    2026-07-31 12:50/13:00 스냅샷 2배치(각 25,385행) 직후 DB 가 13:05~13:35,
    13:45~14:15 두 차례 멈췄고(그 사이 run.py 의 delta 수집도 0건), 15:13 에
    서버가 재시작됐다. 원인은 인서트 자체가 아니라 **한 트랜잭션의 크기**다.

        innodb_buffer_pool_size = 128MB   (MariaDB 기본값)
        ev_charger_status       = 208MB   (이미 버퍼 풀의 1.6배)
        인덱스                   = 5개    (PRIMARY + secondary 4)

    행마다 B-tree 5개를 갱신하는데 secondary 가 stat_id 선두라 페이지 접근이
    랜덤이다. 25,385행을 단일 트랜잭션으로 묶으면 커밋 전까지 더티 페이지와
    undo 로그가 통째로 쌓여 버퍼 풀을 밀어내고, 같은 DB 에 쓰던 run.py 까지
    함께 굶는다. 500행마다 끊어서 커밋하면 더티 페이지가 그때그때 flush 되어
    체크포인트 폭주가 사라진다.

    pause_sec 는 청크 사이에 쉬어 주는 시간이다. 이 스크립트는 팀 공용 DB 를
    쓰므로 5분 주기 delta 수집이 끼어들 틈을 남겨야 한다. 25,385행 기준
    51청크 x 0.5초 = 약 25초가 더 걸리는데, 수집 주기가 6시간이라 문제없다.

    부분 저장에 대하여
    -----------------
    청크 커밋이므로 중간에 실패하면 **이미 커밋된 행은 롤백되지 않는다**.
    단일 트랜잭션 시절의 all-or-nothing 이 깨지는 것이라, 소비하는 쪽은
    반드시 ev_status_snapshot_batch.is_complete = 1 인 batch_id 만 써야 한다
    (기존 부분 스냅샷 판정과 동일한 가드다). on_progress 로 진행 건수를
    올려 주므로, 예외가 나도 호출부가 마지막 저장 건수를 배치 행에 남긴다.
    """
    if not rows:
        return 0

    conn = pymysql.connect(**DB_CONFIG)
    try:
        saved = 0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, chunk)
            conn.commit()
            saved += len(chunk)
            if on_progress:
                on_progress(saved)
            if pause_sec and i + batch_size < len(rows):
                time.sleep(pause_sec)
        return saved
    finally:
        conn.close()


def collect_and_store(
    batch_size: int = DEFAULT_BATCH_SIZE, pause_sec: float = DEFAULT_PAUSE_SEC
):
    observed_at = datetime.now().replace(microsecond=0)
    batch_id = str(uuid.uuid4())
    stamp = observed_at.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] 전체 상태 스냅샷 수집 시작 (zcode={ZCODE}, batch={batch_id[:8]})")

    conn = pymysql.connect(**DB_CONFIG)
    try:
        ensure_schema(conn)
    finally:
        conn.close()

    batch = {"batch_id": batch_id, "observed_at": observed_at}
    try:
        rows, total_count, received = fetch_snapshot()
        # 저장 중 죽어도 배치 행에 남도록 fetch 결과를 먼저 채운다.
        batch.update(expected_count=total_count, fetched_count=received,
                     saved_count=0, is_complete=False)
        # 관측 시각·배치 ID 를 전 행에 동일하게 부여
        stamped = [r + (observed_at, batch_id) for r in rows]
        saved = save_to_db(
            stamped, batch_size=batch_size, pause_sec=pause_sec,
            # 청크 커밋이라 중간 실패 시 부분 저장이 남는다. 마지막 커밋
            # 지점을 배치 행에 계속 갱신해 둬야 어디까지 들어갔는지 안다.
            on_progress=lambda n: batch.__setitem__("saved_count", n),
        )
        batch.update(
            expected_count=total_count,
            fetched_count=received,
            saved_count=saved,
            # 완결성은 '전 페이지를 다 받았는가'로 본다.
            # 필수 필드 결측으로 걸러진 행이 있어도 스냅샷 자체는 완전하다.
            is_complete=bool(total_count) and received >= total_count,
        )
        note = "" if batch["is_complete"] else "  ** 부분 스냅샷 (미완료) **"
        print(
            f"[{stamp}] 저장 {saved:,}건 / 수신 {received:,}건 "
            f"/ API {total_count:,}건{note}"
        )
    except Exception as exc:
        batch.update(is_complete=False, error=str(exc)[:500])
        print(f"[{stamp}] 수집 실패: {exc}")
    finally:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            record_batch(conn, batch)
        finally:
            conn.close()


def main():
    """CMD에서 실행할 때마다 1회 수집 후 종료. 상주 스케줄러 없음."""
    if not DB_CONFIG["password"]:
        raise SystemExit("DB_PASSWORD 미설정 (.env.example 참고)")
    if not SERVICE_KEY:
        raise SystemExit("EV_SERVICE_KEY 미설정 (.env.example 참고)")
    collect_and_store()


if __name__ == "__main__":
    main()
