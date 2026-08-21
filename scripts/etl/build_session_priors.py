"""
충전 세션 이력(charge_history) → 요일별 사전확률 피처

  ev_charger_session_map    : (stat_id, chger_id) ↔ 세션이력 (station_name, charger_unit_id)
  ev_charger_session_prior  : 충전기 × 요일 요일별 점유율·세션수·kWh prior

  py scripts/etl/build_session_priors.py            # 매핑 + prior
  py scripts/etl/build_session_priors.py --map      # 매핑만
  py scripts/etl/build_session_priors.py --priors   # prior만 (기존 매핑 사용)
  py scripts/etl/build_session_priors.py --dry-run  # DB 쓰기 없이 요약만

왜 status 가 아니라 세션 이력인가
--------------------------------
ev_charger_status 는 2026-07-22 부터라 13일치뿐이다. 충전기 × 요일로 쪼개면
요일당 관측이 1~2회라 개별 충전기의 요일 패턴을 만들 수 없다.
반면 charge_history 세션 이력은 2025-01-01~2026-06-30(545일 전후)이며
충전기 × 요일 셀당 샘플이 충분한 편이라 그대로 쓸 수 있다.

두 데이터는 기간이 겹치지 않는다(세션 ~2026-06, status 2026-07~).
따라서 여기서 만든 prior 를 status 기반 가용확률 모델의 피처로 넣어도 누수가 없다.

주의: 세션 이력은 '수요' 지표고 horizon_hgb 의 라벨은 도착 시점 stat==2 인
'가용성'이다. 같은 양이 아니므로 라벨 근사가 아니라 사전확률 피처로만 쓸 것.

커버리지 한계
-------------
세션 이력은 기후에너지환경부 급속기에만 존재한다. 대구 전체 충전기 24,659 대 중
285 대(1.2%), 급속(50kW+) 1,802 대 중 264 대(14.7%)만 개별 이력이 있다.
이건 시간이 지나도 해소되지 않는 구조적 결손이므로 has_charger_history=0 은
'아직 안 쌓임'이 아니라 '앞으로도 없음'으로 해석해야 하고,
지역 × 속도군 그룹 prior 가 임시방편이 아니라 상시 주력 경로다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pymysql

# 저장소 전반이 pymysql 커넥션을 pd.read_sql 에 그대로 넘긴다(service.py, model_store.py 등).
# 동작에는 문제가 없고 요약 출력만 가리므로 이 경고만 끈다.
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

try:  # 저장소 루트의 .env (컨테이너에서는 환경변수로 주입)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS = ROOT / "charge_history" / "cache" / "daegu_sessions_all.parquet"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

# 세션 길이 정제
#   duration_min p99 = 46분인데 최대 20,216분(14일)까지 있다. 미종료 세션이
#   섞인 것으로 보이므로 하루로 캡한다. 0 이하는 버린다.
DURATION_CAP_MIN = 1440
MINUTES_PER_DAY = 1440

# 계층 shrinkage 강도. w = n / (n + SHRINK_K).
#   충전기 × 요일 셀당 중앙값이 105 건이므로 k=30 이면 커버된 충전기는
#   개별 패턴이 지배하고(w≈0.78), 표본이 얇은 셀만 상위 계층으로 끌려간다.
#   홀드아웃으로 재조정할 것.
SHRINK_K = 30.0

# 세션 이력 power_kw ↔ speed_class 실측 대응
#   급속 50/100 · 고출력급속 200 · 초급속 350/400
#   DB output 의 300·320kW 는 세션 이력에 없어 고출력급속으로 버킷팅한다.
#   50kW 미만(완속)은 세션 이력 자체가 없어 speed_class = None → prior 없음.
SPEED_BUCKETS = [(350.0, "초급속"), (200.0, "고출력급속"), (50.0, "급속")]

CREATE_MAP_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_session_map (
    stat_id            VARCHAR(20)  NOT NULL COMMENT 'ev_charger_info.stat_id',
    chger_id           VARCHAR(10)  NOT NULL COMMENT 'ev_charger_info.chger_id',
    station_name       VARCHAR(200) NOT NULL COMMENT '세션이력 station_name',
    charger_unit_id    VARCHAR(10)  NOT NULL COMMENT '세션이력 charger_unit_id',
    district           VARCHAR(40)  NULL,
    speed_class        VARCHAR(20)  NULL COMMENT '급속/고출력급속/초급속',
    mapping_method     VARCHAR(20)  NOT NULL COMMENT 'name_unit',
    mapping_confidence DECIMAL(4,2) NOT NULL DEFAULT 0.0 COMMENT 'unit 집합 겹침 비율',
    session_count      INT          NOT NULL DEFAULT 0,
    updated_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                       ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stat_id, chger_id),
    KEY idx_station_name (station_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전기 ↔ 충전세션이력 매핑'
"""

UPSERT_MAP_SQL = """
INSERT INTO ev_charger_session_map (
    stat_id, chger_id, station_name, charger_unit_id,
    district, speed_class, mapping_method, mapping_confidence, session_count
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    station_name       = VALUES(station_name),
    charger_unit_id    = VALUES(charger_unit_id),
    district           = VALUES(district),
    speed_class        = VALUES(speed_class),
    mapping_method     = VALUES(mapping_method),
    mapping_confidence = VALUES(mapping_confidence),
    session_count      = VALUES(session_count)
"""

CREATE_PRIOR_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_session_prior (
    stat_id                  VARCHAR(20)  NOT NULL,
    chger_id                 VARCHAR(10)  NOT NULL,
    day_of_week              TINYINT      NOT NULL COMMENT '0=월 ... 6=일',
    charger_occupancy        DECIMAL(6,4) NULL COMMENT '충전기 요일 점유율(충전중 시간/1440)',
    charger_sessions_per_day DECIMAL(8,3) NULL,
    charger_kwh_per_day      DECIMAL(10,3) NULL,
    charger_n                INT          NOT NULL DEFAULT 0 COMMENT '해당 요일 세션 수',
    charger_days             INT          NOT NULL DEFAULT 0 COMMENT '해당 요일 관측 일수',
    station_occupancy        DECIMAL(6,4) NULL COMMENT '충전소 요일 점유율(충전기 1대 환산)',
    station_n                INT          NOT NULL DEFAULT 0,
    group_occupancy          DECIMAL(6,4) NULL COMMENT '지역x속도군 요일 점유율',
    group_n                  INT          NOT NULL DEFAULT 0,
    group_key                VARCHAR(80)  NULL COMMENT 'district|speed_class',
    occupancy_prior          DECIMAL(6,4) NULL COMMENT '계층 shrinkage 최종값',
    kwh_per_day_prior        DECIMAL(10,3) NULL,
    shrink_weight            DECIMAL(6,4) NULL COMMENT '최종값에서 개별 충전기 비중',
    has_charger_history      TINYINT      NOT NULL DEFAULT 0,
    source_from              DATE         NULL,
    source_to                DATE         NULL,
    built_at                 TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stat_id, chger_id, day_of_week),
    KEY idx_dow (day_of_week)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전기 x 요일 세션이력 prior (수요 사전확률)'
"""

UPSERT_PRIOR_SQL = """
INSERT INTO ev_charger_session_prior (
    stat_id, chger_id, day_of_week,
    charger_occupancy, charger_sessions_per_day, charger_kwh_per_day,
    charger_n, charger_days,
    station_occupancy, station_n,
    group_occupancy, group_n, group_key,
    occupancy_prior, kwh_per_day_prior, shrink_weight,
    has_charger_history, source_from, source_to
) VALUES (
    %s, %s, %s,
    %s, %s, %s,
    %s, %s,
    %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    charger_occupancy        = VALUES(charger_occupancy),
    charger_sessions_per_day = VALUES(charger_sessions_per_day),
    charger_kwh_per_day      = VALUES(charger_kwh_per_day),
    charger_n                = VALUES(charger_n),
    charger_days             = VALUES(charger_days),
    station_occupancy        = VALUES(station_occupancy),
    station_n                = VALUES(station_n),
    group_occupancy          = VALUES(group_occupancy),
    group_n                  = VALUES(group_n),
    group_key                = VALUES(group_key),
    occupancy_prior          = VALUES(occupancy_prior),
    kwh_per_day_prior        = VALUES(kwh_per_day_prior),
    shrink_weight            = VALUES(shrink_weight),
    has_charger_history      = VALUES(has_charger_history),
    source_from              = VALUES(source_from),
    source_to                = VALUES(source_to)
"""


def get_conn():
    return pymysql.connect(**DB_CONFIG, connect_timeout=30)


def norm_name(s: str) -> str:
    """충전소명 정규화. 세션이력 station_name ↔ info.stat_nm 대조용."""
    return re.sub(r"[\s()\-_,.]", "", str(s)).lower()


def speed_class_of(kw: float | None) -> str | None:
    if kw is None or pd.isna(kw):
        return None
    for threshold, label in SPEED_BUCKETS:
        if kw >= threshold:
            return label
    return None


def weekday_day_count(start: date, end: date, weekday: int) -> int:
    """[start, end] 안에 해당 요일이 몇 번 오는지."""
    if end < start:
        return 0
    first = start + timedelta(days=(weekday - start.weekday()) % 7)
    if first > end:
        return 0
    return (end - first).days // 7 + 1


# ---------------------------------------------------------------- load


def load_sessions(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"세션 캐시 없음: {path}\n"
            "  py -m charge_history ingest  로 먼저 만들 것"
        )
    df = pd.read_parquet(path)

    before = len(df)
    df = df[df["duration_min"] > 0].copy()
    over_cap = int((df["duration_min"] > DURATION_CAP_MIN).sum())
    df["duration_min"] = df["duration_min"].clip(upper=DURATION_CAP_MIN)
    print(
        f"[sessions] {before:,} → {len(df):,} 건 "
        f"(비정상 {before - len(df)}건 제외, {DURATION_CAP_MIN}분 캡 {over_cap}건)"
    )

    df["k"] = df["station_name"].map(norm_name)
    df["unit"] = df["charger_unit_id"].astype(str).str.zfill(2)
    df["sdate"] = pd.to_datetime(df["start_date"]).dt.date
    # start_weekday 는 이미 0=월 (ev_charger_features.day_of_week 와 동일 규약)
    df["dow"] = df["start_weekday"].astype(int)
    return df


def load_info(conn) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT stat_id, chger_id, stat_nm, busi_nm, addr, output
        FROM ev_charger_info
        WHERE del_yn IS NULL OR del_yn <> 'Y'
        """,
        conn,
    )
    df = df.drop_duplicates(subset=["stat_id", "chger_id"]).copy()
    df["k"] = df["stat_nm"].map(norm_name)
    df["unit"] = df["chger_id"].astype(str).str.zfill(2)
    df["kw"] = pd.to_numeric(df["output"], errors="coerce")
    df["speed_class"] = df["kw"].map(speed_class_of)
    # addr 은 '대구광역시 수성구 …' 와 '대구 수성구 …' 두 형태가 섞여 있다.
    # 두 경우 모두 두 번째 토큰이 구·군이다.
    df["district"] = (
        df["addr"].astype(str).str.split().str[1].where(lambda s: s.str.len() > 0)
    )
    print(f"[info] 충전기 {len(df):,}대 / 충전소 {df['stat_id'].nunique():,}곳")
    return df


# ---------------------------------------------------------------- map


def build_map(sess: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """충전소명 정규화 일치 + unit 집합 겹침으로 (stat_id, chger_id) 결정.

    같은 stat_nm 이 여러 stat_id 로 존재한다(대구 3,853개 중 244개).
    한 장소에 환경부 충전기와 민간 충전기가 같이 있는 경우다. 세션 이력은
    환경부 것이므로 unit 겹침 → 환경부 사업자 → ME 접두사 순으로 해소한다.
    """
    sess_units = sess.groupby("k")["unit"].apply(lambda s: set(s.unique()))
    sess_counts = sess.groupby(["k", "unit"]).size()
    meta = (
        sess.groupby(["k", "unit"])[["district", "speed_class"]]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    )
    info_by_k = {k: g for k, g in info.groupby("k")}

    rows: list[dict] = []
    unmatched_station: list[str] = []
    partial: list[tuple[str, str, int, int]] = []

    for k, units in sess_units.items():
        cand = info_by_k.get(k)
        if cand is None:
            unmatched_station.append(k)
            continue

        best_sid, best_score, best_units = None, None, set()
        for sid, g in cand.groupby("stat_id"):
            db_units = set(g["unit"])
            overlap = units & db_units
            score = (
                len(overlap) / len(units),
                int(g["busi_nm"].astype(str).str.contains("환경부").any()),
                int(str(sid).startswith("ME")),
            )
            if best_score is None or score > best_score:
                best_sid, best_score, best_units = sid, score, overlap

        if not best_units:
            unmatched_station.append(k)
            continue
        if len(best_units) < len(units):
            partial.append((k, best_sid, len(best_units), len(units)))

        station_name = sess.loc[sess["k"] == k, "station_name"].iat[0]
        # 동명 충전소가 여러 stat_id 로 갈라지므로 chger_id 조회를 best_sid 로 좁힌다.
        # 좁히지 않으면 같은 unit 번호를 쓰는 다른 사업자 행을 집을 수 있다.
        best_rows = cand[cand["stat_id"] == best_sid]
        for unit in sorted(best_units):
            d, sp = meta.loc[(k, unit)] if (k, unit) in meta.index else (None, None)
            rows.append(
                {
                    "stat_id": best_sid,
                    "chger_id": best_rows.loc[best_rows["unit"] == unit, "chger_id"].iat[0],
                    "station_name": station_name,
                    "charger_unit_id": unit,
                    "district": d,
                    "speed_class": sp,
                    "mapping_method": "name_unit",
                    "mapping_confidence": round(best_score[0], 2),
                    "session_count": int(sess_counts.get((k, unit), 0)),
                    "k": k,
                    "unit": unit,
                }
            )

    mapping = pd.DataFrame(rows)
    print(
        f"[map] 세션 충전소 {len(sess_units)}곳 중 {len(sess_units) - len(unmatched_station)}곳 매칭 "
        f"→ 충전기 {len(mapping)}대"
    )
    if partial:
        print("  unit 부분 매칭 (세션에는 있으나 현재 info 에 없는 충전기):")
        for k, sid, got, want in partial:
            print(f"    {k} ({sid}): {got}/{want}")
    if unmatched_station:
        print(f"  미매칭 충전소: {unmatched_station}")
    return mapping


# ---------------------------------------------------------------- priors


def build_priors(
    sess: pd.DataFrame, info: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """충전기 x 요일 prior. 대구 전체 충전기에 대해 7행씩 생성한다.

    이력이 없는 충전기(98.8%)도 지역 x 속도군 그룹 prior 를 받도록
    LEFT JOIN 없이 여기서 미리 채운다. 서빙은 (stat_id, chger_id, dow) 단순 조회.
    """
    data_start = sess["sdate"].min()
    data_end = sess["sdate"].max()
    print(f"[prior] 세션 기간 {data_start} ~ {data_end}")

    # --- 관측 시작일: 충전기가 중간에 신설된 경우 분모가 과대해지지 않게
    charger_first = sess.groupby(["k", "unit"])["sdate"].min()
    station_first = sess.groupby("k")["sdate"].min()

    # --- 충전기 x 요일
    cw = (
        sess.groupby(["k", "unit", "dow"])
        .agg(n=("duration_min", "size"), dur=("duration_min", "sum"), kwh=("kwh", "sum"))
        .reset_index()
    )
    cw["days"] = [
        weekday_day_count(charger_first.loc[(k, u)], data_end, d)
        for k, u, d in zip(cw["k"], cw["unit"], cw["dow"])
    ]
    cw = cw[cw["days"] > 0]
    cw["occ"] = cw["dur"] / (cw["days"] * MINUTES_PER_DAY)
    cw["sess_per_day"] = cw["n"] / cw["days"]
    cw["kwh_per_day"] = cw["kwh"] / cw["days"]
    cw_idx = cw.set_index(["k", "unit", "dow"])

    # --- 충전소 x 요일 (충전기 1대 환산 — 충전기 수준 값과 단위를 맞춘다)
    station_chargers = sess.groupby("k")["unit"].nunique()
    sw = (
        sess.groupby(["k", "dow"])
        .agg(n=("duration_min", "size"), dur=("duration_min", "sum"), kwh=("kwh", "sum"))
        .reset_index()
    )
    sw["days"] = [
        weekday_day_count(station_first.loc[k], data_end, d)
        for k, d in zip(sw["k"], sw["dow"])
    ]
    sw = sw[sw["days"] > 0]
    sw["ncharger"] = sw["k"].map(station_chargers)
    sw["occ"] = sw["dur"] / (sw["days"] * MINUTES_PER_DAY * sw["ncharger"])
    sw["kwh_per_day"] = sw["kwh"] / (sw["days"] * sw["ncharger"])
    sw_idx = sw.set_index(["k", "dow"])

    # --- 지역 x 속도군 x 요일 (충전기 1대 환산)
    sess_g = sess.assign(gk=sess["district"].astype(str) + "|" + sess["speed_class"].astype(str))
    group_chargers = sess_g.groupby("gk").apply(
        lambda g: g.groupby(["k", "unit"]).ngroups, include_groups=False
    )
    gw = (
        sess_g.groupby(["gk", "dow"])
        .agg(n=("duration_min", "size"), dur=("duration_min", "sum"), kwh=("kwh", "sum"))
        .reset_index()
    )
    gw["days"] = [weekday_day_count(data_start, data_end, d) for d in gw["dow"]]
    gw["ncharger"] = gw["gk"].map(group_chargers)
    gw["occ"] = gw["dur"] / (gw["days"] * MINUTES_PER_DAY * gw["ncharger"])
    gw["kwh_per_day"] = gw["kwh"] / (gw["days"] * gw["ncharger"])
    gw_idx = gw.set_index(["gk", "dow"])

    # --- 매핑을 (stat_id, chger_id) → (k, unit) 조회표로
    key_of = {}
    if not mapping.empty:
        for r in mapping.itertuples():
            key_of[(r.stat_id, r.chger_id)] = (r.k, r.unit)

    out: list[dict] = []
    for r in info.itertuples():
        gk = f"{r.district}|{r.speed_class}" if r.speed_class else None
        ku = key_of.get((r.stat_id, r.chger_id))

        for dow in range(7):
            g = gw_idx.loc[(gk, dow)] if gk is not None and (gk, dow) in gw_idx.index else None
            g_occ = float(g["occ"]) if g is not None else None
            g_kwh = float(g["kwh_per_day"]) if g is not None else None
            g_n = int(g["n"]) if g is not None else 0

            c_occ = c_sess = c_kwh = None
            c_n = c_days = 0
            s_occ = s_kwh = None
            s_n = 0

            if ku is not None:
                k, unit = ku
                if (k, unit, dow) in cw_idx.index:
                    c = cw_idx.loc[(k, unit, dow)]
                    c_occ, c_n, c_days = float(c["occ"]), int(c["n"]), int(c["days"])
                    c_sess, c_kwh = float(c["sess_per_day"]), float(c["kwh_per_day"])
                if (k, dow) in sw_idx.index:
                    s = sw_idx.loc[(k, dow)]
                    s_occ, s_n, s_kwh = float(s["occ"]), int(s["n"]), float(s["kwh_per_day"])

            # 2단계 shrinkage: 충전기 → 충전소 → 그룹
            occ_prior, kwh_prior, w_charger = _shrink(
                (c_occ, c_kwh, c_n), (s_occ, s_kwh, s_n), (g_occ, g_kwh, g_n)
            )

            out.append(
                {
                    "stat_id": r.stat_id,
                    "chger_id": r.chger_id,
                    "day_of_week": dow,
                    "charger_occupancy": c_occ,
                    "charger_sessions_per_day": c_sess,
                    "charger_kwh_per_day": c_kwh,
                    "charger_n": c_n,
                    "charger_days": c_days,
                    "station_occupancy": s_occ,
                    "station_n": s_n,
                    "group_occupancy": g_occ,
                    "group_n": g_n,
                    "group_key": gk,
                    "occupancy_prior": occ_prior,
                    "kwh_per_day_prior": kwh_prior,
                    "shrink_weight": w_charger,
                    "has_charger_history": int(c_n > 0),
                    "source_from": data_start,
                    "source_to": data_end,
                }
            )

    df = pd.DataFrame(out)
    covered = int(df.loc[df["day_of_week"] == 0, "has_charger_history"].sum())
    n_chargers = len(df) // 7
    with_prior = int(df.loc[df["day_of_week"] == 0, "occupancy_prior"].notna().sum())
    print(
        f"[prior] {len(df):,}행 (충전기 {n_chargers:,}대 x 7요일)\n"
        f"  개별 이력 보유 : {covered:,}대 ({100 * covered / n_chargers:.1f}%)\n"
        f"  prior 산출 가능: {with_prior:,}대 ({100 * with_prior / n_chargers:.1f}%) "
        f"- 나머지는 완속이라 세션 이력 자체가 없음"
    )
    return df


def _shrink(charger, station, group):
    """(occ, kwh, n) 3계층을 n/(n+k) 가중으로 접는다. 아래에서 위로."""
    c_occ, c_kwh, c_n = charger
    s_occ, s_kwh, s_n = station
    g_occ, g_kwh, g_n = group

    base_occ, base_kwh = g_occ, g_kwh
    if s_occ is not None:
        w = s_n / (s_n + SHRINK_K)
        base_occ = w * s_occ + (1 - w) * g_occ if g_occ is not None else s_occ
        base_kwh = w * s_kwh + (1 - w) * g_kwh if g_kwh is not None else s_kwh

    if c_occ is None:
        return _r(base_occ, 4), _r(base_kwh, 3), 0.0
    if base_occ is None:
        return _r(c_occ, 4), _r(c_kwh, 3), 1.0

    w = c_n / (c_n + SHRINK_K)
    return (
        _r(w * c_occ + (1 - w) * base_occ, 4),
        _r(w * c_kwh + (1 - w) * base_kwh, 3),
        round(w, 4),
    )


def _r(v, nd):
    return None if v is None or pd.isna(v) else round(float(v), nd)


# ---------------------------------------------------------------- write


def _executemany(conn, sql: str, rows: list[tuple], chunk: int = 2000) -> int:
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), chunk):
            cur.executemany(sql, rows[i : i + chunk])
            total += cur.rowcount
        conn.commit()
    return total


def write_map(conn, mapping: pd.DataFrame) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_MAP_SQL)
    conn.commit()
    rows = [
        (
            r.stat_id,
            r.chger_id,
            r.station_name,
            r.charger_unit_id,
            r.district,
            r.speed_class,
            r.mapping_method,
            r.mapping_confidence,
            r.session_count,
        )
        for r in mapping.itertuples()
    ]
    _executemany(conn, UPSERT_MAP_SQL, rows)
    print(f"[map] ev_charger_session_map UPSERT {len(rows):,}건")


def write_priors(conn, priors: pd.DataFrame) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_PRIOR_SQL)
    conn.commit()
    cols = [
        "stat_id", "chger_id", "day_of_week",
        "charger_occupancy", "charger_sessions_per_day", "charger_kwh_per_day",
        "charger_n", "charger_days",
        "station_occupancy", "station_n",
        "group_occupancy", "group_n", "group_key",
        "occupancy_prior", "kwh_per_day_prior", "shrink_weight",
        "has_charger_history", "source_from", "source_to",
    ]
    rows = [
        tuple(None if pd.isna(v) else v for v in rec)
        for rec in priors[cols].itertuples(index=False, name=None)
    ]
    _executemany(conn, UPSERT_PRIOR_SQL, rows)
    print(f"[prior] ev_charger_session_prior UPSERT {len(rows):,}건")


def read_map(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT stat_id, chger_id, station_name, charger_unit_id FROM ev_charger_session_map",
        conn,
    )
    if df.empty:
        sys.exit("ev_charger_session_map 이 비어 있다. --map 을 먼저 실행할 것")
    df["k"] = df["station_name"].map(norm_name)
    df["unit"] = df["charger_unit_id"].astype(str).str.zfill(2)
    return df


def preview(priors: pd.DataFrame) -> None:
    """요일별 요약 — 패턴이 실제로 있는지 눈으로 확인."""
    cov = priors[priors["has_charger_history"] == 1]
    if cov.empty:
        return
    names = ["월", "화", "수", "목", "금", "토", "일"]
    t = cov.groupby("day_of_week").agg(
        점유율=("charger_occupancy", "mean"),
        세션수=("charger_sessions_per_day", "mean"),
        kWh=("charger_kwh_per_day", "mean"),
    )
    t.index = [names[i] for i in t.index]
    print("\n이력 보유 충전기 요일 평균 (충전기 1대 기준)")
    print(t.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="세션 이력 → 요일 prior")
    ap.add_argument("--map", action="store_true", help="매핑 테이블만 빌드")
    ap.add_argument("--priors", action="store_true", help="prior 테이블만 빌드")
    ap.add_argument("--dry-run", action="store_true", help="DB 쓰기 없이 요약만")
    ap.add_argument(
        "--sessions",
        type=Path,
        default=DEFAULT_SESSIONS,
        help=f"세션 parquet 경로 (기본 {DEFAULT_SESSIONS.name})",
    )
    args = ap.parse_args()

    do_map = args.map or not args.priors
    do_priors = args.priors or not args.map

    conn = get_conn()
    try:
        sess = load_sessions(args.sessions)
        info = load_info(conn)

        mapping = pd.DataFrame()
        if do_map:
            mapping = build_map(sess, info)
            if args.dry_run:
                print("[dry-run] 매핑 DB 쓰기 생략")
            else:
                write_map(conn, mapping)

        if do_priors:
            if mapping.empty:
                mapping = read_map(conn)
            priors = build_priors(sess, info, mapping)
            preview(priors)
            if args.dry_run:
                print("[dry-run] prior DB 쓰기 생략")
            else:
                write_priors(conn, priors)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
