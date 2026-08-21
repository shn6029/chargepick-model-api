"""
ev_charger_status → ev_charger_features 파생 피처 생성
테이블 생성 후 전체(또는 최근) 데이터를 계산해 UPSERT

  전체 재빌드:  py scripts/etl/build_features.py
  증분(5분 주기): py scripts/etl/build_features.py --lookback-hours 6

서빙(recommend_api/service.py)은 최신 상태 행에 features를 INNER JOIN 하고
60분 이내 신선도를 요구하므로(config.VERY_STALE_EXCLUDE_MIN), 증분 실행이
5분 주기로 돌지 않으면 추천 후보가 0건이 된다.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

try:
    import holidays
except ImportError:  # pragma: no cover
    holidays = None

try:  # 저장소 루트의 .env (컨테이너에서는 환경변수로 주입)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

# DB 접속정보는 전부 환경변수에서 온다. 폴백값을 두지 않는 이유는
# `.env` 없이 돌렸을 때 운영 DB 로 조용히 붙는 사고를 막기 위해서다.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

# 5분 슬롯 기본 (15분으로 바꾸려면 15)
MINUTE_SLOT_SIZE = 5

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_features (
    log_id                   BIGINT       NOT NULL COMMENT 'ev_charger_status.log_id',
    stat_id                  VARCHAR(20)  NOT NULL,
    chger_id                 VARCHAR(10)  NOT NULL,
    created_at               DATETIME     NOT NULL,
    hour                     TINYINT      NOT NULL,
    minute_slot              TINYINT      NOT NULL COMMENT '분 구간 시작(0,5,10,...)',
    day_of_week              TINYINT      NOT NULL COMMENT '0=월 ... 6=일',
    is_weekend               TINYINT      NOT NULL,
    is_holiday               TINYINT      NOT NULL,
    is_holiday_eve           TINYINT      NOT NULL DEFAULT 0 COMMENT '다음날 공휴일',
    is_after_holiday         TINYINT      NOT NULL DEFAULT 0 COMMENT '전날 공휴일',
    consecutive_holiday_days TINYINT      NOT NULL DEFAULT 0 COMMENT '연속 공휴일/주말 일수',
    is_long_weekend          TINYINT      NOT NULL DEFAULT 0 COMMENT '3일 이상 연휴',
    current_state_duration   INT          NULL COMMENT '현재 상태 유지(분)',
    changes_30m              INT          NULL COMMENT '최근 30분 상태 변경 횟수',
    avail_ratio_15m          DECIMAL(6,4) NULL COMMENT '최근 15분 사용가능 비율',
    avail_ratio_30m          DECIMAL(6,4) NULL COMMENT '최근 30분 사용가능 비율',
    avail_ratio_60m          DECIMAL(6,4) NULL COMMENT '최근 60분 사용가능 비율',
    time_since_available     INT          NULL COMMENT '마지막 사용가능 이후(분)',
    time_since_charge_started INT         NULL COMMENT '충전 시작 후 경과(분)',
    time_since_charge_ended  INT          NULL COMMENT '충전 종료 후 경과(분)',
    built_at                 TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (log_id),
    KEY idx_charger_time (stat_id, chger_id, created_at),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전기 상태 파생 피처'
"""

# 신규 컬럼 추가 (기존 테이블 마이그레이션용)
ALTER_TABLE_SQLS = [
    "ALTER TABLE ev_charger_features ADD COLUMN IF NOT EXISTS is_holiday_eve TINYINT NOT NULL DEFAULT 0 COMMENT '다음날 공휴일' AFTER is_holiday",
    "ALTER TABLE ev_charger_features ADD COLUMN IF NOT EXISTS is_after_holiday TINYINT NOT NULL DEFAULT 0 COMMENT '전날 공휴일' AFTER is_holiday_eve",
    "ALTER TABLE ev_charger_features ADD COLUMN IF NOT EXISTS consecutive_holiday_days TINYINT NOT NULL DEFAULT 0 COMMENT '연속 공휴일/주말 일수' AFTER is_after_holiday",
    "ALTER TABLE ev_charger_features ADD COLUMN IF NOT EXISTS is_long_weekend TINYINT NOT NULL DEFAULT 0 COMMENT '3일 이상 연휴' AFTER consecutive_holiday_days",
]

UPSERT_SQL = """
INSERT INTO ev_charger_features (
    log_id, stat_id, chger_id, created_at,
    hour, minute_slot, day_of_week, is_weekend, is_holiday,
    is_holiday_eve, is_after_holiday, consecutive_holiday_days, is_long_weekend,
    current_state_duration, changes_30m,
    avail_ratio_15m, avail_ratio_30m, avail_ratio_60m,
    time_since_available, time_since_charge_started, time_since_charge_ended
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s,
    %s, %s, %s,
    %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    hour = VALUES(hour),
    minute_slot = VALUES(minute_slot),
    day_of_week = VALUES(day_of_week),
    is_weekend = VALUES(is_weekend),
    is_holiday = VALUES(is_holiday),
    is_holiday_eve = VALUES(is_holiday_eve),
    is_after_holiday = VALUES(is_after_holiday),
    consecutive_holiday_days = VALUES(consecutive_holiday_days),
    is_long_weekend = VALUES(is_long_weekend),
    current_state_duration = VALUES(current_state_duration),
    changes_30m = VALUES(changes_30m),
    avail_ratio_15m = VALUES(avail_ratio_15m),
    avail_ratio_30m = VALUES(avail_ratio_30m),
    avail_ratio_60m = VALUES(avail_ratio_60m),
    time_since_available = VALUES(time_since_available),
    time_since_charge_started = VALUES(time_since_charge_started),
    time_since_charge_ended = VALUES(time_since_charge_ended),
    built_at = CURRENT_TIMESTAMP
"""


LOCK_PATH = Path(tempfile.gettempdir()) / "scheduler_build_features.lock"
LOCK_STALE_SEC = 1800  # 30분 넘은 잠금은 죽은 프로세스로 보고 회수


@contextmanager
def feature_lock(wait_seconds: int = 0):
    """전체 재빌드와 증분 실행의 동시 수행을 막는다.

    주간 전체 재빌드는 70만행을 UPSERT 하느라 오래 걸리는데, 그동안 5분 주기
    증분이 계속 끼어들면 같은 테이블을 두 프로세스가 함께 쓰며 경합한다.
    잠금을 못 잡으면 예외 대신 (False) 를 넘겨 호출부가 조용히 건너뛰게 한다.
    """
    acquired = False
    deadline = time.time() + wait_seconds
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:  # 죽은 프로세스가 남긴 잠금 회수
                if time.time() - LOCK_PATH.stat().st_mtime > LOCK_STALE_SEC:
                    print(f"경고: 오래된 잠금 회수 ({LOCK_PATH})")
                    LOCK_PATH.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() >= deadline:
                break
            time.sleep(2)
    try:
        yield acquired
    finally:
        if acquired:
            LOCK_PATH.unlink(missing_ok=True)


def get_kr_holidays(years: set[int]):
    if holidays is None:
        return set()
    kr = holidays.country_holidays("KR", years=sorted(years))
    return set(kr.keys())


def load_status(lookback_hours: int | None = None) -> pd.DataFrame:
    """상태 로그 로드.

    lookback_hours=None 이면 전체 재빌드, 정수면 최근 N시간만 로드(증분).
    증분 시에도 롤링 윈도우(최대 60분)와 상태 유지 시간(최대 360분 캡)을
    정확히 계산하려면 저장 대상 구간보다 넉넉한 이력이 필요하다.
    """
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        base = """
            SELECT log_id, stat_id, chger_id, stat,
                   stat_upd_dt, last_tsdt, last_tedt, now_tsdt, created_at
            FROM ev_charger_status
        """
        order = " ORDER BY stat_id, chger_id, created_at, log_id"
        if lookback_hours is None:
            df = pd.read_sql(base + order, conn)
        else:
            query = base + " WHERE created_at >= NOW() - INTERVAL %s HOUR" + order
            df = pd.read_sql(query, conn, params=(int(lookback_hours),))
    finally:
        conn.close()

    for col in ("created_at", "stat_upd_dt", "last_tsdt", "last_tedt", "now_tsdt"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_anchors(lookback_hours: int, horizon_days: int = 1) -> pd.DataFrame:
    """증분 창 시작 이전 시점의 충전기별 상태 앵커.

    증분 계산 시 창의 첫 행은 이전 상태를 몰라 무조건 '상태 변경'으로 처리되어
    current_state_duration / time_since_available 이 크게 틀어진다(실측 49% 불일치).
    창 밖 이력에서 아래 3가지를 미리 뽑아 시드로 넣는다.

      anchor_stat       창 직전 마지막 관측 상태
      state_start_at    그 상태가 시작된 시각(마지막 상태 변경 시각)
      last_available_at 창 직전 마지막으로 stat=2 였던 시각

    horizon_days 안에 상태 변경이 없으면 그 구간 시작으로 포화된다.
    모델 입력은 360분에서 캡되므로(config.DURATION_CAP_MIN) 실용상 충분하다.

    horizon 은 서버 CPU 비용을 크게 좌우한다(실측: 7일 14.7초 vs 1일 2.3초).
    DB 가 vCPU 2개짜리 버스터블 인스턴스에 MySQL 과 함께 있어 지속 20% 를 넘기면
    버스트 크레딧이 소진되고 인스턴스 전체가 스로틀된다. 기본 1일을 유지할 것.
    """
    sql = """
    WITH hist AS (
        SELECT stat_id, chger_id, stat, created_at,
               LAG(stat) OVER (PARTITION BY stat_id, chger_id
                               ORDER BY created_at, log_id) AS prev_stat
        FROM ev_charger_status
        WHERE created_at <  NOW() - INTERVAL %s HOUR
          AND created_at >= NOW() - INTERVAL %s HOUR - INTERVAL %s DAY
    ),
    last_row AS (
        SELECT stat_id, chger_id, stat AS anchor_stat
        FROM (SELECT stat_id, chger_id, stat,
                     ROW_NUMBER() OVER (PARTITION BY stat_id, chger_id
                                        ORDER BY created_at DESC) AS rn
              FROM hist) t
        WHERE rn = 1
    ),
    last_change AS (
        SELECT stat_id, chger_id, MAX(created_at) AS state_start_at
        FROM hist
        WHERE prev_stat IS NULL OR stat <> prev_stat
        GROUP BY stat_id, chger_id
    ),
    last_avail AS (
        SELECT stat_id, chger_id, MAX(created_at) AS last_available_at
        FROM hist WHERE stat = 2 GROUP BY stat_id, chger_id
    )
    SELECT r.stat_id, r.chger_id, r.anchor_stat,
           c.state_start_at, a.last_available_at
    FROM last_row r
    LEFT JOIN last_change c ON r.stat_id=c.stat_id AND r.chger_id=c.chger_id
    LEFT JOIN last_avail  a ON r.stat_id=a.stat_id AND r.chger_id=a.chger_id
    """
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=60)
    try:
        df = pd.read_sql(sql, conn, params=(lookback_hours, lookback_hours, horizon_days))
    finally:
        conn.close()
    for col in ("state_start_at", "last_available_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _window_avail_ratio(times: pd.Series, avail: pd.Series, window: str) -> pd.Series:
    """충전기 시계열에서 시간 윈도우 사용가능 비율 (현재 행 포함)."""
    tmp = pd.DataFrame({"avail": avail.astype(float)}).set_index(times)
    # 동일 시각 중복 대비 평균 후 rolling
    tmp = tmp.groupby(level=0)["avail"].mean().to_frame()
    rolled = tmp["avail"].rolling(window, min_periods=1).mean()
    # 원래 인덱스에 매핑
    return times.map(rolled)


def _window_changes(times: pd.Series, changed: pd.Series, window: str) -> pd.Series:
    tmp = pd.DataFrame({"changed": changed.astype(float)}).set_index(times)
    tmp = tmp.groupby(level=0)["changed"].sum().to_frame()
    rolled = tmp["changed"].rolling(window, min_periods=1).sum()
    return times.map(rolled)


def compute_features(
    df: pd.DataFrame,
    lookback_hours: int | None = None,
    anchors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["charger_key"] = work["stat_id"].astype(str) + "_" + work["chger_id"].astype(str)
    work = work.sort_values(["charger_key", "created_at", "log_id"]).reset_index(drop=True)

    # 증분 창 밖 이력에서 뽑은 앵커 (load_anchors 참고)
    anchor_stat = anchor_start = anchor_avail = None
    if anchors is not None and not anchors.empty:
        a = anchors.copy()
        a["charger_key"] = a["stat_id"].astype(str) + "_" + a["chger_id"].astype(str)
        a = a.drop_duplicates("charger_key").set_index("charger_key")
        anchor_stat = a["anchor_stat"]
        anchor_start = a["state_start_at"]
        anchor_avail = a["last_available_at"]

    work["hour"] = work["created_at"].dt.hour.astype("int16")
    work["minute_slot"] = (
        (work["created_at"].dt.minute // MINUTE_SLOT_SIZE) * MINUTE_SLOT_SIZE
    ).astype("int16")
    work["day_of_week"] = work["created_at"].dt.weekday.astype("int16")  # Mon=0
    work["is_weekend"] = (work["day_of_week"] >= 5).astype("int8")

    years = set(work["created_at"].dt.year.dropna().astype(int).unique())
    # 전후 연도까지 포함해 연휴 경계 계산 오류 방지
    holiday_set = get_kr_holidays(years | {y - 1 for y in years} | {y + 1 for y in years})
    dates = work["created_at"].dt.date
    work["is_holiday"] = dates.map(
        lambda d: 1 if d in holiday_set else 0
    ).astype("int8")

    # is_holiday_eve: 다음날이 공휴일 또는 주말
    from datetime import date as _date, timedelta as _td
    next_dates = (work["created_at"] + pd.Timedelta(days=1)).dt.date
    prev_dates = (work["created_at"] - pd.Timedelta(days=1)).dt.date

    def _is_off(d: _date) -> bool:
        return d in holiday_set or d.weekday() >= 5

    work["is_holiday_eve"] = next_dates.map(lambda d: 1 if _is_off(d) else 0).astype("int8")
    work["is_after_holiday"] = prev_dates.map(lambda d: 1 if _is_off(d) else 0).astype("int8")

    # consecutive_holiday_days: 현재 날짜가 포함된 연속 휴일 구간의 길이
    # 날짜별 계산 후 매핑
    all_dates = set(dates.unique())
    _consec_cache: dict[_date, int] = {}

    def _consec(d: _date) -> int:
        if d in _consec_cache:
            return _consec_cache[d]
        if not _is_off(d):
            _consec_cache[d] = 0
            return 0
        # 구간 시작 찾기
        start = d
        while _is_off(start - _td(days=1)):
            start -= _td(days=1)
        end = d
        while _is_off(end + _td(days=1)):
            end += _td(days=1)
        length = (end - start).days + 1
        cur = start
        while cur <= end:
            _consec_cache[cur] = length
            cur += _td(days=1)
        return length

    work["consecutive_holiday_days"] = dates.map(_consec).astype("int8")
    work["is_long_weekend"] = (work["consecutive_holiday_days"] >= 3).astype("int8")

    # 상태 변경 / 유지 시작 시각
    prev_stat = work.groupby("charger_key")["stat"].shift(1)
    # 충전기별 창 첫 행: 앵커 상태를 이전 상태로 사용 (없으면 종전대로 변경 취급)
    first_idx = work.groupby("charger_key", sort=False).head(1).index
    if anchor_stat is not None:
        seeded = work.loc[first_idx, "charger_key"].map(anchor_stat)
        prev_stat.loc[first_idx] = prev_stat.loc[first_idx].fillna(seeded)
    work["stat_changed"] = (
        prev_stat.isna() | (work["stat"] != prev_stat)
    ).astype(int)
    state_start = work["created_at"].where(work["stat_changed"] == 1)
    work["state_start_at"] = state_start
    if anchor_start is not None:
        # 첫 행이 앵커와 같은 상태면 상태 시작은 창 밖에 있다
        seeded = work.loc[first_idx, "charger_key"].map(anchor_start)
        work.loc[first_idx, "state_start_at"] = work.loc[
            first_idx, "state_start_at"
        ].fillna(seeded)
    work["state_start_at"] = work.groupby("charger_key")["state_start_at"].ffill()
    work["state_start_at"] = work["state_start_at"].fillna(work["stat_upd_dt"]).fillna(
        work["created_at"]
    )

    duration = (work["created_at"] - work["state_start_at"]).dt.total_seconds() / 60.0
    work["current_state_duration"] = duration.clip(lower=0).round().astype("Int64")

    work["is_available"] = (work["stat"] == 2).astype(int)

    # 충전기별 rolling 피처
    changes_list = []
    ratio_15, ratio_30, ratio_60 = [], [], []
    for _, g in work.groupby("charger_key", sort=False):
        times = g["created_at"]
        changes_list.append(_window_changes(times, g["stat_changed"], "30min"))
        ratio_15.append(_window_avail_ratio(times, g["is_available"], "15min"))
        ratio_30.append(_window_avail_ratio(times, g["is_available"], "30min"))
        ratio_60.append(_window_avail_ratio(times, g["is_available"], "60min"))

    work["changes_30m"] = pd.concat(changes_list).sort_index().round().astype("Int64")
    work["avail_ratio_15m"] = pd.concat(ratio_15).sort_index().astype(float)
    work["avail_ratio_30m"] = pd.concat(ratio_30).sort_index().astype(float)
    work["avail_ratio_60m"] = pd.concat(ratio_60).sort_index().astype(float)

    # 마지막 사용가능 시각
    work["last_available_at"] = work["created_at"].where(work["is_available"] == 1)
    if anchor_avail is not None:
        # 창 안에 stat=2 이력이 없는 충전기는 앵커 값에서 이어받는다
        seeded = work.loc[first_idx, "charger_key"].map(anchor_avail)
        work.loc[first_idx, "last_available_at"] = work.loc[
            first_idx, "last_available_at"
        ].fillna(seeded)
    work["last_available_at"] = work.groupby("charger_key")["last_available_at"].ffill()
    since_avail = (work["created_at"] - work["last_available_at"]).dt.total_seconds() / 60.0
    work["time_since_available"] = since_avail.clip(lower=0).round().astype("Int64")

    started = (work["created_at"] - work["now_tsdt"]).dt.total_seconds() / 60.0
    work["time_since_charge_started"] = started.clip(lower=0).round().astype("Int64")

    ended = (work["created_at"] - work["last_tedt"]).dt.total_seconds() / 60.0
    work["time_since_charge_ended"] = ended.clip(lower=0).round().astype("Int64")

    cols = [
        "log_id",
        "stat_id",
        "chger_id",
        "created_at",
        "hour",
        "minute_slot",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "is_holiday_eve",
        "is_after_holiday",
        "consecutive_holiday_days",
        "is_long_weekend",
        "current_state_duration",
        "changes_30m",
        "avail_ratio_15m",
        "avail_ratio_30m",
        "avail_ratio_60m",
        "time_since_available",
        "time_since_charge_started",
        "time_since_charge_ended",
    ]
    out = work[cols].copy()
    # MySQL용 NaN → None
    out = out.where(pd.notnull(out), None)
    return out


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        # 기존 테이블에 신규 컬럼 추가 (이미 있으면 무시)
        for sql in ALTER_TABLE_SQLS:
            try:
                cur.execute(sql)
            except Exception:
                pass
    conn.commit()


def _to_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return int(value)


def _to_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return float(value)


def save_features(features: pd.DataFrame, batch_size: int = 2000) -> int:
    if features.empty:
        return 0

    rows = []
    for r in features.itertuples(index=False):
        created = r.created_at
        if isinstance(created, pd.Timestamp):
            created = created.to_pydatetime()
        rows.append(
            (
                int(r.log_id),
                r.stat_id,
                r.chger_id,
                created,
                int(r.hour),
                int(r.minute_slot),
                int(r.day_of_week),
                int(r.is_weekend),
                int(r.is_holiday),
                int(r.is_holiday_eve),
                int(r.is_after_holiday),
                int(r.consecutive_holiday_days),
                int(r.is_long_weekend),
                _to_int(r.current_state_duration),
                _to_int(r.changes_30m),
                _to_float(r.avail_ratio_15m),
                _to_float(r.avail_ratio_30m),
                _to_float(r.avail_ratio_60m),
                _to_int(r.time_since_available),
                _to_int(r.time_since_charge_started),
                _to_int(r.time_since_charge_ended),
            )
        )

    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        ensure_table(conn)
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                cur.executemany(UPSERT_SQL, rows[i : i + batch_size])
                conn.commit()
                print(f"  upsert {min(i + batch_size, len(rows))}/{len(rows)}")
        return len(rows)
    finally:
        conn.close()


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="ev_charger_status → ev_charger_features 파생 피처 생성"
    )
    p.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        metavar="N",
        help=(
            "최근 N시간만 계산(증분). 미지정 시 전체 재빌드. "
            "서빙(VERY_STALE_EXCLUDE_MIN=60)을 만족시키려면 5분 주기로 "
            "--lookback-hours 6 실행을 권장."
        ),
    )
    p.add_argument(
        "--anchor-horizon-days",
        type=int,
        default=1,
        metavar="D",
        help=(
            "증분 시 창 밖 앵커를 찾아볼 최대 기간(일). 기본 1. "
            "늘리면 서버 CPU 비용이 급증한다(7일=14.7초 vs 1일=2.3초)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lookback = args.lookback_hours

    # 전체 재빌드는 오래 걸리므로 증분이 끝나기를 잠시 기다린다.
    # 증분은 5분 뒤 어차피 다시 도니 기다리지 않고 건너뛴다.
    wait = 0 if lookback is not None else 180
    with feature_lock(wait_seconds=wait) as acquired:
        if not acquired:
            print(
                "다른 build_features 실행 중 → 건너뜀 "
                f"(잠금: {LOCK_PATH})"
            )
            return
        _run(args, lookback)


def _run(args, lookback: int | None) -> None:
    mode = "전체 재빌드" if lookback is None else f"증분(최근 {lookback}시간)"
    print(f"상태 데이터 로드 중... [{mode}]")
    status = load_status(lookback_hours=lookback)
    print(f"로드: {len(status):,}행")

    if status.empty:
        print("상태 데이터가 없습니다.")
        return

    if holidays is None:
        print("경고: holidays 미설치 → is_holiday=0 고정. pip install holidays")

    anchors = None
    if lookback is not None:
        print("창 밖 앵커 로드 중...")
        anchors = load_anchors(lookback, horizon_days=args.anchor_horizon_days)
        print(f"앵커: {len(anchors):,}대")

    print("피처 계산 중...")
    features = compute_features(status, lookback_hours=lookback, anchors=anchors)
    print(f"피처 행: {len(features):,}")
    print(features.head(3).to_string(index=False))

    print("DB 저장 중...")
    saved = save_features(features)
    print(f"완료: ev_charger_features UPSERT {saved:,}건")
    print(f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
