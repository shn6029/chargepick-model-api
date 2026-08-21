from __future__ import annotations

import hashlib
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pymysql
from sklearn.ensemble import HistGradientBoostingClassifier

_SCHEDULER_ROOT = Path(__file__).resolve().parent.parent
if str(_SCHEDULER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHEDULER_ROOT))

from .config import (
    ARTIFACTS_DIR,
    AVAILABLE_FUTURE_STAT,
    DURATION_CAP_MIN,
    FEATURE_COLS,
    HORIZONS,
    LABELABLE_FUTURE_STATS,
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    LONG_STATE_DURATION_MIN,
    MATCH_TOLERANCE_MIN,
    METRICS_PATH,
    MODEL_PATH,
    STATUS_UPDATE_AGE_CAP_MIN,
    STALE_UPDATE_AGE_MIN,
    db_config,
)
from .access import access_coefficient, classify_access
from .holdout_eval import run_full_evaluation, save_metrics
from .use_time import operating_at, operating_at_per_row


def get_connection():
    return pymysql.connect(**db_config(), connect_timeout=40)


def table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
            LIMIT 1
            """,
            (table_name,),
        )
        return cur.fetchone() is not None


def column_exists(conn, table_name: str, column_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = %s AND column_name = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        return cur.fetchone() is not None


def _holiday_feature_select(conn) -> str:
    """공휴일 파생 컬럼이 아직 없으면 0으로 대체 (build_features 마이그레이션 전)."""
    cols = (
        "is_holiday_eve",
        "is_after_holiday",
        "consecutive_holiday_days",
        "is_long_weekend",
    )
    parts = []
    for c in cols:
        if column_exists(conn, "ev_charger_features", c):
            parts.append(f"f.{c}")
        else:
            parts.append(f"0 AS {c}")
    return ",\n                ".join(parts)


def _context_select_and_join(conn) -> tuple[str, str]:
    """ev_charger_context 없으면 NULL 컬럼 + JOIN 생략."""
    context_cols = (
        "restaurant_cnt_500m",
        "cafe_cnt_500m",
        "mart_cnt_500m",
        "hospital_cnt_500m",
        "office_cnt_500m",
        "accommodation_cnt_500m",
        "commercial_poi_cnt_500m",
        "context_type",
    )
    if table_exists(conn, "ev_charger_context"):
        select = ",\n                ".join(f"c.{c}" for c in context_cols)
        join = "LEFT JOIN ev_charger_context c ON s.stat_id = c.stat_id"
        return select, join
    select = ",\n                ".join(f"NULL AS {c}" for c in context_cols)
    return select, ""


def get_git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_SCHEDULER_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        return None
    return None


def compute_feature_schema_hash(
    base_cols: list[str], feature_columns: list[str]
) -> str:
    payload = "|".join(base_cols) + "||" + "|".join(feature_columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_artifact_features(
    artifact: dict[str, Any], *, strict: bool = False
) -> list[str]:
    """base_feature_cols / feature_columns 앞부분이 FEATURE_COLS와 일치하는지 검사."""
    msgs: list[str] = []
    base = artifact.get("base_feature_cols")
    cols = list(artifact.get("feature_columns") or [])
    expected = list(FEATURE_COLS)

    if base is not None:
        if list(base) != expected:
            msgs.append(
                f"base_feature_cols 불일치: artifact={list(base)!r} config={expected!r}"
            )
    else:
        n = len(expected)
        if cols[:n] != expected:
            msgs.append(
                "feature_columns 앞부분이 FEATURE_COLS와 불일치 "
                f"(artifact[:{n}]={cols[:n]!r})"
            )

    for m in msgs:
        if strict:
            raise RuntimeError(m)
        warnings.warn(m, stacklevel=2)
    return msgs


def apply_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    """장시간 동일상태 vs 실제 갱신지연(stale) 분리.

    - is_long_state_duration: current_state_duration >= 360
    - is_stale_status: (created_at - stat_upd_dt) >= 360  (갱신 시각 기준)
    - is_invalid_status_update_time: 품질 플래그(모델 피처 아님)
    """
    if df.empty:
        return df
    out = df.copy()
    dur = pd.to_numeric(out.get("current_state_duration"), errors="coerce").fillna(0)
    out["current_state_duration"] = dur
    out["is_long_state_duration"] = (dur >= LONG_STATE_DURATION_MIN).astype(int)
    out["capped_state_duration"] = dur.clip(lower=0, upper=DURATION_CAP_MIN)

    if "stat_upd_dt" in out.columns:
        out["stat_upd_dt"] = pd.to_datetime(out["stat_upd_dt"], errors="coerce")
        invalid = out["stat_upd_dt"].isna() | (out["stat_upd_dt"] > out["created_at"])
        age = (out["created_at"] - out["stat_upd_dt"]).dt.total_seconds() / 60.0
        age = age.fillna(0).clip(lower=0)
    else:
        invalid = pd.Series(True, index=out.index)
        age = pd.Series(0.0, index=out.index)
    out["is_invalid_status_update_time"] = invalid.astype(int)
    out["status_update_age_min"] = age
    out["capped_status_update_age_min"] = age.clip(upper=STATUS_UPDATE_AGE_CAP_MIN)
    out["is_stale_status"] = (age >= STALE_UPDATE_AGE_MIN).astype(int)
    return out


# 하위 호환 별칭
def apply_stale_features(df: pd.DataFrame) -> pd.DataFrame:
    return apply_status_flags(df)


def enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out["output_kw"] = pd.to_numeric(out.get("output"), errors="coerce").fillna(0)
    out["is_fast"] = (out["output_kw"] >= 50).astype(int)
    out["parking_free_yn"] = (out.get("parking_free") == "Y").astype(int)
    out["limit_yn_flag"] = (out.get("limit_yn") == "Y").astype(int)
    out["traffic_yn_flag"] = (out.get("traffic_yn") == "Y").astype(int)
    # 공휴일 파생변수: 컬럼이 없으면 0으로 채움
    for col in ("is_holiday_eve", "is_after_holiday", "consecutive_holiday_days", "is_long_weekend"):
        if col not in out.columns:
            out[col] = 0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # 상권 컨텍스트 컬럼: 없으면 0 / -1 기본값
    context_int_cols = [
        "restaurant_cnt_500m", "cafe_cnt_500m", "mart_cnt_500m",
        "hospital_cnt_500m", "office_cnt_500m", "accommodation_cnt_500m",
        "commercial_poi_cnt_500m",
    ]
    for col in context_int_cols:
        if col not in out.columns:
            out[col] = 0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # context_type → 순서형 인코딩
    _CONTEXT_TYPE_ORDER = {
        "주거·혼합형": 0,
        "공공기관형": 1,
        "상업시설형": 2,
        "마트형": 3,
        "병원형": 4,
        "관광지형": 5,
        "휴게소형": 6,
    }
    if "context_type" not in out.columns:
        out["context_type_encoded"] = -1
    else:
        out["context_type_encoded"] = (
            out["context_type"].map(_CONTEXT_TYPE_ORDER).fillna(-1).astype(int)
        )
    out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["lng"] = pd.to_numeric(out.get("lng"), errors="coerce")
    return apply_status_flags(out)


def load_joined(limit: int | None = None, include_snapshot: bool = False) -> pd.DataFrame:
    """학습용 status+features+info 조인.

    include_snapshot=False (기본): source='snapshot' 행을 제외한다.
    전량 스냅샷(getChargerInfo, 회당 25,433행)은 상시 수집이 아니라 특정 시점에만
    돌았기 때문에, 그 몇 분에 전체의 수 %가 몰린다. 그대로 학습하면
    hour/minute_slot/day_of_week 가 "그 시각엔 모든 충전기가 관측된다"를 외운다.
    스냅샷이 상시 주기로 돌게 되면 이 기본값을 재검토할 것.

    limit 은 무작위 표본이 아니다 — 빠른 테스트 외에는 쓰지 말 것
    -----------------------------------------------------------
    LIMIT 이 ORDER BY stat_id, chger_id, created_at 뒤에 붙으므로, 상한을 걸면
    stat_id 알파벳 앞쪽 충전기만 남고 뒤쪽은 통째로 사라진다. 시간 절단이 아니라
    **충전기 절단**이라 학습 기간은 그대로인 채 모집단만 바뀐다.

    2026-08-03 `train --max-rows 400000` 실측 (전량 977,732행 대비):
        충전기   20,013 → 9,730 (48.6%)
        사업자   61 → 40   ← 21곳이 학습에서 완전 소멸
        급속 행  63.5% → 53.7%
        기후부 9.7%→0% · GS차지비 4.6%→0% · 플러그링크 3.6%→0% (실트래픽 약 22%)
        채비 7.5%→18.2% · LG유플러스볼트업 4.8%→11.8% (2.5배 과대대표)

    결과로 busi_ 더미가 39개만 생성됐다(전량 60개). align_features 는 없는 컬럼을
    0으로 채우므로, 서빙에서 그 21개 사업자 충전기는 **모든 사업자 더미가 0인
    '본 적 없는 집단'**으로 들어간다. positive_rate 도 0.685 로 전량(0.745)과
    6%p 어긋나 available_prob 캘리브레이션과 warn_threshold 가 함께 틀어진다.

    그런데 지표로는 안 잡힌다. 홀드아웃도 같은 400,000행에서 뽑히니 date_holdout
    AUC 0.9716 / 불가R 0.875 로 멀쩡해 보인다. 편향 여부는 지표가 아니라
    n_features · busi_ 더미 개수 · positive_rate 로 확인할 것.
    """
    conn = get_connection()
    try:
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        holiday_sel = _holiday_feature_select(conn)
        context_sel, context_join = _context_select_and_join(conn)
        source_filter = ""
        if not include_snapshot and column_exists(conn, "ev_charger_status", "source"):
            source_filter = "AND s.source <> 'snapshot'"
        query = f"""
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                s.stat_upd_dt,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                {holiday_sel},
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.kind_detail,
                i.parking_free, i.limit_yn, i.limit_detail, i.traffic_yn,
                i.use_time,
                i.stat_nm, i.addr, i.lat, i.lng,
                {context_sel}
            FROM ev_charger_status s
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            {context_join}
            WHERE (i.del_yn IS NULL OR i.del_yn <> 'Y')
              {source_filter}
            ORDER BY s.stat_id, s.chger_id, s.created_at, s.log_id
            {limit_sql}
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return enrich_frame(df)

def load_label_series(days: int | None = None) -> pd.DataFrame:
    """라벨 전용 상태 시계열. delta + snapshot 을 **모두** 포함한다.

    학습 표본(load_joined)과 라벨 원천을 분리하는 이유
    ------------------------------------------------
    전량 스냅샷은 6시간마다 25,433행이 1분 안에 몰린다. 이걸 학습 표본으로
    쓰면 hour/minute_slot 이 "그 시각엔 모든 충전기가 관측된다"를 외운다
    (load_joined 이 기본으로 snapshot 을 빼는 이유가 이것이다).

    그런데 **라벨** 쪽에서는 정반대다. 스냅샷은 상태와 무관하게 전 충전기를
    찍으므로 LOCF 사슬을 끊어 주는 앵커다. 이게 없으면 delta 로 한 번도
    보고되지 않은 충전기(12일 기준 25,433대 중 23.2%)는 라벨이 아예 안 붙고,
    나머지도 staleness 가 무한정 늘어난다.

    그래서 좌변(표본)은 delta 만, 우변(라벨)은 delta+snapshot 을 쓴다.
    features/info 조인이 필요 없으므로 최소 컬럼만 읽는다.
    """
    conn = get_connection()
    try:
        where = ""
        if days:
            where = f"WHERE created_at >= NOW() - INTERVAL {int(days)} DAY"
        query = f"""
            SELECT stat_id, chger_id, created_at, stat
            FROM ev_charger_status
            {where}
            ORDER BY stat_id, chger_id, created_at
        """
        out = pd.read_sql(query, conn)
    finally:
        conn.close()
    out["created_at"] = pd.to_datetime(out["created_at"])
    return out


def build_horizon_dataset(
    df: pd.DataFrame,
    horizons: list[int],
    label_df: pd.DataFrame | None = None,
    method: str | None = None,
    max_staleness_min: int | None = None,
) -> pd.DataFrame:
    """기준 시점 × ETA → 도착 시점 가용 여부(y) 샘플 생성.

    미래 상태가 LABELABLE_FUTURE_STATS(2,3,4,5)인 경우만 학습에 포함.
    stat=1(통신이상)·9(상태미확인)은 사용불가와 혼동되지 않도록 제외.

    라벨링 방식 (method)
    -------------------
    "locf" (기본, config.LABEL_METHOD)
        t+h 이전 마지막 관측 상태를 라벨로 쓴다. delta 수집이 '변경분만'
        보고하므로 관측이 없다 = 상태가 그대로였다 는 뜻이기 때문이다.
        max_staleness_min 보다 오래된 관측만 있으면 표본을 버린다.

    "asof_nearest" (구버전)
        t+h 의 ±MATCH_TOLERANCE_MIN 안에 물리적 행이 있어야 라벨을 붙인다.
        stat=3(충전중)은 세션이 끝날 때까지 행이 안 생겨 h30 매칭률이 9.1%
        까지 떨어졌다. 비교·재현용으로만 남긴다.

    label_df
        None 이면 df 자기 자신에서 미래 상태를 찾는다(구 동작).
        load_label_series() 결과를 넘기면 snapshot 행까지 라벨 원천으로 쓴다.

    LOCF 라벨의 정확도는 100%가 아니다. 2026-07-31 전량 스냅샷 대비 실측으로
    staleness 8시간 이내에서 97.7%(가용률 편차 -0.5%p)이고, 그 밖에서는
    급격히 무너진다. 상세 근거는 config.LABEL_MAX_STALENESS_MIN 주석 참고.
    """
    method = method or LABEL_METHOD
    if method not in ("locf", "asof_nearest"):
        raise ValueError(f"알 수 없는 라벨링 방식: {method}")
    max_staleness_min = (
        LABEL_MAX_STALENESS_MIN if max_staleness_min is None else max_staleness_min
    )

    # 충전기별 파이썬 루프 대신 merge_asof(by=...) 로 horizon 당 1회씩만 매칭한다.
    # 예전 구조는 (충전기 2만 대 x horizon 7개) = 14만 번의 merge_asof + 그만큼의
    # 와이드 DataFrame 복사가 일어나 96만 행에서 한 시간이 넘어도 안 끝났다.
    # by= 를 쓰면 pandas 가 그룹 매칭을 내부에서 처리하므로 호출이 7번으로 준다.
    KEY = ["stat_id", "chger_id"]
    src = label_df if label_df is not None else df
    future_all = (
        src[KEY + ["created_at", "stat"]]
        .rename(columns={"created_at": "future_at", "stat": "future_stat"})
        .sort_values("future_at", kind="mergesort")
        .reset_index(drop=True)
    )

    # 관측이 1건뿐인 충전기 제외 (구 동작 유지)
    size = df.groupby(KEY, sort=False)["created_at"].transform("size")
    base = df.loc[size >= 2]

    # 라벨 원천에 아예 등장하지 않는 충전기 제외
    lbl_keys = pd.MultiIndex.from_frame(future_all[KEY]).unique()
    has_label = pd.MultiIndex.from_frame(base[KEY]).isin(lbl_keys)
    n_no_label_row = int((~has_label).sum()) * len(horizons)
    base = base.loc[has_label]

    if method == "locf":
        # backward = t+h 이하의 가장 가까운 관측 = 마지막 상태 유지
        direction = "backward"
        tolerance = pd.Timedelta(minutes=max_staleness_min)
    else:
        direction = "nearest"
        tolerance = pd.Timedelta(minutes=MATCH_TOLERANCE_MIN)

    parts: list[pd.DataFrame] = []
    n_matched = 0
    n_excluded_unlabelable = 0
    n_dropped_stale = 0

    for h in horizons:
        left = base.copy()
        left["eta_minutes"] = h
        left["target_at"] = left["created_at"] + pd.Timedelta(minutes=h)
        left = left.sort_values("target_at", kind="mergesort")

        merged = pd.merge_asof(
            left,
            future_all,
            left_on="target_at",
            right_on="future_at",
            by=KEY,
            direction=direction,
            tolerance=tolerance,
        )
        del left

        n_dropped_stale += int(merged["future_stat"].isna().sum())
        merged = merged.dropna(subset=["future_stat"])
        if merged.empty:
            continue

        # 라벨이 얼마나 오래된 관측에서 왔는지. 평가·가중치용으로 남긴다.
        merged["label_staleness_min"] = (
            (merged["target_at"] - merged["future_at"]).dt.total_seconds() / 60.0
        )

        n_matched += len(merged)
        labelable = merged["future_stat"].isin(LABELABLE_FUTURE_STATS)
        n_excluded_unlabelable += int((~labelable).sum())
        merged = merged.loc[labelable]
        if merged.empty:
            continue

        merged["y"] = (merged["future_stat"] == AVAILABLE_FUTURE_STAT).astype(int)
        merged["arrival_hour"] = merged["target_at"].dt.hour
        merged["arrival_weekday"] = merged["target_at"].dt.weekday
        # 도착 시점 운영 여부 (Y=1 / N=0 / UNKNOWN=NaN)
        if "use_time" in merged.columns:
            merged["is_operating_at_arrival"] = operating_at_per_row(
                merged["use_time"], merged["target_at"]
            )
        else:
            merged["is_operating_at_arrival"] = np.nan
        # 공휴일 파생변수가 없을 때 기본값 보장 (DB 컬럼 추가 전 학습 호환)
        for hcol in ("is_holiday_eve", "is_after_holiday", "consecutive_holiday_days", "is_long_weekend"):
            if hcol not in merged.columns:
                merged[hcol] = 0
        parts.append(merged)

    if not parts:
        out = pd.DataFrame()
    else:
        out = pd.concat(parts, ignore_index=True)

    out.attrs["label_method"] = method
    out.attrs["match_tolerance_min"] = MATCH_TOLERANCE_MIN
    out.attrs["label_max_staleness_min"] = max_staleness_min
    out.attrs["label_source_includes_snapshot"] = label_df is not None
    out.attrs["n_matched_before_label_filter"] = n_matched
    out.attrs["n_excluded_unlabelable_stat"] = n_excluded_unlabelable
    out.attrs["n_dropped_no_match"] = n_dropped_stale
    out.attrs["n_dropped_no_label_series"] = n_no_label_row
    out.attrs["labelable_future_stats"] = sorted(LABELABLE_FUTURE_STATS)
    if not out.empty and "label_staleness_min" in out.columns:
        st = out["label_staleness_min"]
        out.attrs["label_staleness_median_min"] = float(st.median())
        out.attrs["label_staleness_p90_min"] = float(st.quantile(0.9))
    if n_matched:
        print(
            f"  라벨링({method}"
            + (f", staleness<={max_staleness_min}분" if method == "locf" else "")
            + f"): 매칭 {n_matched:,} -> 학습 {len(out):,} "
            f"(stat not in {sorted(LABELABLE_FUTURE_STATS)} 제외 "
            f"{n_excluded_unlabelable:,}, 미매칭 {n_dropped_stale:,})"
        )
        if "label_staleness_median_min" in out.attrs:
            print(
                f"  라벨 나이: 중앙값 {out.attrs['label_staleness_median_min']:.1f}분 / "
                f"p90 {out.attrs['label_staleness_p90_min']:.1f}분"
            )
    return out


def _servable_mask(horizon_df: pd.DataFrame, is_fast: pd.Series) -> pd.Series:
    """서빙에서 실제로 추천 후보가 되는 행 = 급속 ∧ access 하드제외 아님.

    access 분류에는 limit_detail 이 필수다. 없으면 RESIDENT/RESTRICTED 를
    못 잡아 전부 통과해버리므로, 그 경우 판정을 포기하고 NA 를 돌려준다
    (조용히 틀린 값을 내는 것보다 낫다).
    """
    if "limit_detail" not in horizon_df.columns:
        warnings.warn(
            "limit_detail 없음 → is_servable 판정 불가. "
            "load_joined 의 SELECT 를 확인하세요.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.Series(pd.NA, index=horizon_df.index, dtype="boolean")

    cols = ["stat_id", "chger_id", "limit_yn", "limit_detail", "kind"]
    opt = [c for c in ("kind_detail", "stat_nm", "addr") if c in horizon_df.columns]
    uniq = horizon_df[[c for c in cols if c in horizon_df.columns] + opt].drop_duplicates(
        subset=[c for c in ("stat_id", "chger_id") if c in horizon_df.columns]
    )

    rows = []
    for r in uniq.to_dict("records"):
        cls = classify_access(
            limit_yn=r.get("limit_yn"),
            limit_detail=r.get("limit_detail"),
            kind=r.get("kind"),
            kind_detail=r.get("kind_detail"),
            stat_nm=r.get("stat_nm"),
            addr=r.get("addr"),
        )
        rows.append({
            "stat_id": r.get("stat_id"),
            "chger_id": r.get("chger_id"),
            "_not_excluded": access_coefficient(
                cls["access_type"], cls["is_housing"]
            ) is not None,
        })

    flags = horizon_df[["stat_id", "chger_id"]].merge(
        pd.DataFrame(rows), on=["stat_id", "chger_id"], how="left"
    )["_not_excluded"]
    flags.index = horizon_df.index
    return (flags.fillna(False) & (is_fast.astype(int) == 1)).astype("boolean")


def make_xy(horizon_df: pd.DataFrame, fill_median: bool = True):
    """X, y, medians, meta 반환.

    meta: created_at, eta_minutes, is_fast, stat_id, chger_id,
          is_stale_status, status_update_age_min (있을 때).
    fill_median=False 이면 홀드아웃 시 train-only median을 쓰기 위해 NaN 유지.
    is_fast: output_kw >= 50 (enrich_frame과 동일).
    """
    type_dummies = pd.get_dummies(
        horizon_df["chger_type"].fillna("UNK"), prefix="ctype"
    )
    kind_dummies = pd.get_dummies(horizon_df["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(horizon_df["busi_id"].fillna("UNK"), prefix="busi")

    X = pd.concat(
        [horizon_df[FEATURE_COLS], type_dummies, kind_dummies, busi_dummies],
        axis=1,
    )
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    if fill_median:
        X = X.fillna(med)
    y = horizon_df["y"].astype(int)
    meta = horizon_df[["created_at", "eta_minutes"]].copy()
    if "is_fast" in horizon_df.columns:
        meta["is_fast"] = (
            pd.to_numeric(horizon_df["is_fast"], errors="coerce").fillna(0).astype(int)
        )
    elif "is_fast" in X.columns:
        meta["is_fast"] = (
            pd.to_numeric(X["is_fast"], errors="coerce").fillna(0).astype(int)
        )
    else:
        meta["is_fast"] = 0
    for col in ("stat_id", "chger_id"):
        if col in horizon_df.columns:
            meta[col] = horizon_df[col].astype(str)
    # 추천 후보로 실제 나갈 수 있는 행인가 (승격 기준 모집단).
    # access.py 의 RESIDENT/RESTRICTED 는 서빙에서 하드 제외되므로,
    # 이들이 섞인 전체 기준 지표는 실제 서빙 성능보다 낙관적이다.
    meta["is_servable"] = _servable_mask(horizon_df, meta["is_fast"])
    if "is_stale_status" in horizon_df.columns:
        meta["is_stale_status"] = (
            pd.to_numeric(horizon_df["is_stale_status"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    elif "is_stale_status" in X.columns:
        meta["is_stale_status"] = (
            pd.to_numeric(X["is_stale_status"], errors="coerce").fillna(0).astype(int)
        )
    if "status_update_age_min" in horizon_df.columns:
        meta["status_update_age_min"] = pd.to_numeric(
            horizon_df["status_update_age_min"], errors="coerce"
        )
    elif "capped_status_update_age_min" in horizon_df.columns:
        meta["status_update_age_min"] = pd.to_numeric(
            horizon_df["capped_status_update_age_min"], errors="coerce"
        )
    if "output_kw" in horizon_df.columns:
        meta["output_kw"] = pd.to_numeric(horizon_df["output_kw"], errors="coerce")
    elif "output_kw" in X.columns:
        meta["output_kw"] = pd.to_numeric(X["output_kw"], errors="coerce")
    return X, y, med, meta


def align_features(
    X: pd.DataFrame, feature_columns: list[str], medians: dict
) -> pd.DataFrame:
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    X = X.replace([np.inf, -np.inf], np.nan)
    for col, value in medians.items():
        if col in X.columns:
            X[col] = X[col].fillna(value)
    return X.fillna(0)


def train_and_save(
    max_rows: int | None = None,
    skip_eval: bool = False,
) -> dict[str, Any]:
    print("JOIN 로드 중...")
    df = load_joined(limit=max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise RuntimeError("학습 데이터가 없습니다. features/status를 먼저 적재하세요.")

    # 라벨 원천은 표본과 분리한다. snapshot 행은 학습 표본에서는 빼되(시각 편향)
    # 라벨 쪽에서는 LOCF 사슬을 끊는 앵커로 쓴다. load_label_series docstring 참고.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()
    print(f"라벨 원천: {len(label_df):,}행 / 충전기 {label_df.groupby(['stat_id','chger_id']).ngroups:,}대")

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon 샘플: {len(hz):,} | 사용가능 비율 {hz['y'].mean():.1%}")
    if hz.empty:
        raise RuntimeError("horizon 샘플이 없습니다. 수집 기간을 늘리세요.")

    if "future_stat" in hz.columns:
        bad = ~hz["future_stat"].isin(LABELABLE_FUTURE_STATS)
        if bad.any():
            raise RuntimeError(
                f"라벨 필터 실패: future_stat not in "
                f"{sorted(LABELABLE_FUTURE_STATS)} {int(bad.sum())}건"
            )

    X_raw, y, _med_all, meta = make_xy(hz, fill_median=False)

    eval_block: dict[str, Any] | None = None
    thresholds_rec: dict[str, Any] | None = None
    if not skip_eval:
        eval_block = run_full_evaluation(X_raw, y, meta, ARTIFACTS_DIR)
        if eval_block.get("thresholds"):
            thresholds_rec = eval_block["thresholds"]["recommended"]
            print(
                "임계값 권고: "
                f"rank={thresholds_rec['rank_threshold']}  "
                f"warn={thresholds_rec['warn_threshold']}  "
                f"(warn 사용불가R={thresholds_rec['warn_unavailable_recall']:.3f})"
            )

    # 운영 모델: 전체 데이터 final fit
    med = X_raw.median(numeric_only=True)
    X = X_raw.fillna(med)
    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )
    print(f"HistGradientBoosting 최종 학습(전체)... features={X.shape[1]}")
    model.fit(X, y)

    now_utc = datetime.now(timezone.utc)
    trained_at = now_utc.isoformat()
    model_version = now_utc.strftime("%Y%m%dT%H%M%SZ")
    feature_columns = X.columns.tolist()
    base_cols = list(FEATURE_COLS)
    schema_hash = compute_feature_schema_hash(base_cols, feature_columns)
    git_commit = get_git_commit()
    data_start = None
    data_end = None
    if "created_at" in df.columns:
        ca = pd.to_datetime(df["created_at"], errors="coerce")
        if ca.notna().any():
            data_start = ca.min().isoformat()
            data_end = ca.max().isoformat()

    metrics_payload = {
        "model_version": model_version,
        "trained_at": trained_at,
        "git_commit": git_commit,
        "training_data_start": data_start,
        "training_data_end": data_end,
        "training_row_count": int(len(df)),
        "n_rows_raw": int(len(df)),
        "n_horizon_samples": int(len(hz)),
        "n_samples": int(len(X)),
        "positive_rate": float(y.mean()),
        "n_features": int(X.shape[1]),
        "feature_count": int(X.shape[1]),
        "base_feature_cols": base_cols,
        "feature_schema_hash": schema_hash,
        "stale_threshold_min": STALE_UPDATE_AGE_MIN,
        "duration_cap_min": DURATION_CAP_MIN,
        "long_state_duration_min": LONG_STATE_DURATION_MIN,
        "horizons": HORIZONS,
        "match_tolerance_min": MATCH_TOLERANCE_MIN,
        "label_method": hz.attrs.get("label_method", LABEL_METHOD),
        "label_max_staleness_min": hz.attrs.get(
            "label_max_staleness_min", LABEL_MAX_STALENESS_MIN
        ),
        "label_source_includes_snapshot": bool(
            hz.attrs.get("label_source_includes_snapshot", False)
        ),
        "label_staleness_median_min": hz.attrs.get("label_staleness_median_min"),
        "label_staleness_p90_min": hz.attrs.get("label_staleness_p90_min"),
        "labelable_future_stats": sorted(LABELABLE_FUTURE_STATS),
        "n_matched_before_label_filter": int(
            hz.attrs.get("n_matched_before_label_filter", len(hz))
        ),
        "n_excluded_unlabelable_stat": int(
            hz.attrs.get("n_excluded_unlabelable_stat", 0)
        ),
        "n_dropped_no_match": int(hz.attrs.get("n_dropped_no_match", 0)),
        "model_name": "HistGradientBoosting",
        "canonical_model_path": str(MODEL_PATH),
        "status": "prototype_validated",
        "status_note": (
            "서비스 적용 가능한 1차 검증 모델. "
            "장거리 ETA 성능 저하와 사용불가 Recall 개선을 위해 "
            "추가 데이터 기반 운영 검증이 필요합니다. 최종 모델이 아닙니다."
        ),
        "feature_notes": {
            "long_state_duration_min": LONG_STATE_DURATION_MIN,
            "stale_update_age_min": STALE_UPDATE_AGE_MIN,
            "duration_cap_min": DURATION_CAP_MIN,
            "uses_capped_state_duration": True,
            "is_long_state_duration": "same status held >= long_state_duration_min",
            "is_stale_status": "created_at - stat_upd_dt >= stale_update_age_min",
            "dropped_raw_current_state_duration": True,
            "label_rule": (
                "future_stat==2 → y=1; future_stat in {3,4,5} → y=0; "
                "future_stat in {1,9} excluded from training"
            ),
            "match_tolerance_min": MATCH_TOLERANCE_MIN,
        },
    }
    if eval_block is not None:
        metrics_payload.update(eval_block)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics_payload, METRICS_PATH)
    print(f"지표 저장: {METRICS_PATH}")

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "base_feature_cols": base_cols,
        "feature_schema_hash": schema_hash,
        "feature_count": len(feature_columns),
        "medians": med.to_dict(),
        "trained_at": trained_at,
        "model_version": model_version,
        "git_commit": git_commit,
        "training_data_start": data_start,
        "training_data_end": data_end,
        "training_row_count": int(len(df)),
        "n_samples": int(len(X)),
        "positive_rate": float(y.mean()),
        "horizons": HORIZONS,
        "model_name": "HistGradientBoosting",
        "metrics_path": str(METRICS_PATH),
        "thresholds": thresholds_rec
        or {"rank_threshold": 0.5, "warn_threshold": 0.5},
        "long_state_duration_min": LONG_STATE_DURATION_MIN,
        "stale_update_age_min": STALE_UPDATE_AGE_MIN,
        "stale_threshold_min": STALE_UPDATE_AGE_MIN,
        "duration_cap_min": DURATION_CAP_MIN,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}  version={model_version} hash={schema_hash}")
    return {
        "path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "model_version": model_version,
        "n_samples": artifact["n_samples"],
        "positive_rate": artifact["positive_rate"],
        "n_features": len(artifact["feature_columns"]),
        "feature_schema_hash": schema_hash,
        "trained_at": artifact["trained_at"],
        "skip_eval": skip_eval,
        "thresholds": artifact["thresholds"],
    }


def load_artifact(*, strict_features: bool = False) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 `py -m recommend_api.train` 을 실행하세요."
        )
    artifact = joblib.load(MODEL_PATH)
    validate_artifact_features(artifact, strict=strict_features)
    return artifact
