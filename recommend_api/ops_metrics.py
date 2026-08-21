"""예측 로그 기반 운영 지표.

사용:
  py -m recommend_api.ops_metrics
  py -m recommend_api.ops_metrics --days 14
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymysql

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import (
    ARTIFACTS_DIR,
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    LABELABLE_FUTURE_STATS,
    RAPID_SHADOW_WARN_THRESHOLD_BY_ETA,
)
from recommend_api.model_store import get_connection
from recommend_api.prediction_log import ensure_table

OUT_PATH = ARTIFACTS_DIR / "ops_metrics.json"


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, str | None]:
    try:
        from sklearn.metrics import roc_auc_score

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            note = (
                "테스트 데이터에 사용불가 표본이 없음"
                if n_neg == 0
                else "테스트 데이터에 사용가능(stat=2) 표본이 없음"
            )
            return None, note
        return float(roc_auc_score(y_true, y_score)), None
    except Exception as exc:
        return None, str(exc)


def _safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import brier_score_loss

        return float(brier_score_loss(y_true, y_score))
    except Exception:
        return None


def _group_metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    if df.empty:
        return {
            "label": label,
            "n": 0,
            "n_samples": 0,
            "n_available": 0,
            "n_unavailable": 0,
            "roc_auc": None,
            "brier": None,
            "unavailable_recall_thr0.5": None,
            "metric_note": "표본 없음",
        }
    y = (df["actual_stat"].astype(int) == 2).astype(int).to_numpy()
    p = df["available_prob"].astype(float).to_numpy()
    n_available = int((y == 1).sum())
    n_unavailable = int((y == 0).sum())
    unavail = df["actual_stat"].astype(int) != 2
    pred_unavail = p < 0.5
    recall_unavail = (
        float((unavail & pred_unavail).sum() / unavail.sum()) if unavail.any() else None
    )
    auc, auc_note = _safe_auc(y, p)
    note = auc_note
    if recall_unavail is None and not unavail.any():
        note = note or "사용불가 표본이 없어 unavailable_recall 미계산"
    return {
        "label": label,
        "n": int(len(df)),
        "n_samples": int(len(df)),
        "n_available": n_available,
        "n_unavailable": n_unavailable,
        "positive_rate_actual_stat2": float(y.mean()) if len(y) else None,
        "roc_auc": auc,
        "brier": _safe_brier(y, p),
        "unavailable_recall_thr0.5": recall_unavail,
        "metric_note": note,
    }


def load_matched_logs(days: int | None) -> pd.DataFrame:
    ensure_table()
    conn = get_connection()
    try:
        # actual_stat 은 백필이 라벨 가능한 상태(2·3·4·5)일 때만 채운다
        # (prediction_log.backfill_outcomes). 구 스키마로 채워진 1·9 행이 남아
        # 있을 수 있으므로 여기서도 한 번 더 거른다 — 학습과 같은 모집단이어야 한다.
        where = (
            "WHERE actual_stat IS NOT NULL AND available_prob IS NOT NULL"
            f" AND actual_stat IN ({','.join(str(s) for s in sorted(LABELABLE_FUTURE_STATS))})"
        )
        params: list[Any] = []
        if days is not None:
            where += " AND created_at >= NOW() - INTERVAL %s DAY"
            params.append(int(days))
        sql = f"""
            SELECT
                request_id, created_at, model_version, eta_minutes,
                stat_id, chger_id, station_rank, available_prob, current_stat, actual_stat,
                is_long_state_duration, is_stale_status, status_update_age_min,
                vehicle_model_id, min_output_kw,
                outcome_match_direction, outcome_match_status,
                outcome_match_diff_seconds, outcome_label_method,
                shadow_warn, shadow_warn_threshold, is_fast
            FROM ev_recommend_prediction_log
            {where}
        """
        try:
            return pd.read_sql(sql, conn, params=params or None)
        except Exception:
            # 구 스키마(shadow/is_fast 컬럼 없음)
            sql_legacy = f"""
                SELECT
                    request_id, created_at, model_version, eta_minutes,
                    stat_id, chger_id, station_rank, available_prob,
                    current_stat, actual_stat,
                    is_long_state_duration, is_stale_status, status_update_age_min,
                    outcome_match_direction, outcome_match_status,
                    outcome_match_diff_seconds
                FROM ev_recommend_prediction_log
                {where}
            """
            return pd.read_sql(sql_legacy, conn, params=params or None)
    finally:
        conn.close()


def _top1_station_failure(
    df: pd.DataFrame,
    *,
    label: str = "all",
) -> dict[str, Any] | None:
    """request_id × rank=1 충전소: 충전기 중 하나라도 actual_stat==2 이면 성공."""
    top1 = df[df["station_rank"] == 1].copy()
    if top1.empty or "request_id" not in top1.columns:
        return {
            "label": label,
            "n_requests": 0,
            "n_failed": 0,
            "rate": None,
            "note": "표본 없음",
        }
    if "stat_id" not in top1.columns:
        return {
            "label": label,
            "n_requests": 0,
            "rate": None,
            "note": "stat_id 컬럼 없음",
        }
    top1["ok"] = top1["actual_stat"].astype(int) == 2
    by_req = (
        top1.groupby(["request_id", "stat_id"], as_index=False)["ok"]
        .any()
        .rename(columns={"ok": "station_success"})
    )
    first_station = top1.groupby("request_id", as_index=False).agg(
        stat_id=("stat_id", "first")
    )
    merged = first_station.merge(by_req, on=["request_id", "stat_id"], how="left")
    merged["station_success"] = merged["station_success"].fillna(False)
    n = int(len(merged))
    fail_rate = float((~merged["station_success"]).mean()) if n else None
    return {
        "label": label,
        "n_requests": n,
        "n_failed": int((~merged["station_success"]).sum()) if n else 0,
        "rate": fail_rate,
        "success_definition": (
            "rank=1 충전소의 필터된 충전기 중 하나라도 actual_stat==2 이면 성공"
        ),
    }


def _enrich_with_charger_info(df: pd.DataFrame) -> pd.DataFrame:
    """output_kw / is_fast를 info 테이블에서 붙임."""
    if df.empty:
        return df
    conn = get_connection()
    try:
        info = pd.read_sql(
            """
            SELECT stat_id, chger_id, output
            FROM ev_charger_info
            WHERE del_yn IS NULL OR del_yn <> 'Y'
            """,
            conn,
        )
    finally:
        conn.close()
    info["output_kw"] = pd.to_numeric(info["output"], errors="coerce").fillna(0)
    info["is_fast"] = (info["output_kw"] >= 50).astype(int)
    info["stat_id"] = info["stat_id"].astype(str)
    info["chger_id"] = info["chger_id"].astype(str)
    out = df.copy()
    out["stat_id"] = out["stat_id"].astype(str)
    out["chger_id"] = out["chger_id"].astype(str)
    return out.merge(
        info[["stat_id", "chger_id", "output_kw", "is_fast"]],
        on=["stat_id", "chger_id"],
        how="left",
    )


def _filter_rapid_fresh(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "is_fast" in out.columns:
        out = out[out["is_fast"].fillna(0).astype(int) == 1]
    if "is_stale_status" in out.columns:
        out = out[out["is_stale_status"].fillna(0).astype(int) == 0]
    if "status_update_age_min" in out.columns:
        age = pd.to_numeric(out["status_update_age_min"], errors="coerce")
        out = out[age.isna() | (age < 60)]
    # 요청 시점 호환 kW가 있으면 해당 이상만
    if "min_output_kw" in out.columns and "output_kw" in out.columns:
        need = pd.to_numeric(out["min_output_kw"], errors="coerce")
        have = pd.to_numeric(out["output_kw"], errors="coerce").fillna(0)
        keep = need.isna() | (have >= need)
        out = out.loc[keep]
    return out.reset_index(drop=True)


def _shadow_warn_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """급속 shadow warn 경고율·오경고율·miss율.

    - shadow_warn_rate: 급속 matched 행 중 warn=1 비율
    - shadow_false_warn_rate: warn=1 & actual_stat=2 / warn=1 (오경고 — 실제 가용인데 경고)
    - shadow_miss_rate: warn=0 & actual_stat!=2 / actual_stat!=2 (불가 놓침)
    """
    if "shadow_warn" not in df.columns or "is_fast" not in df.columns:
        return {
            "note": "shadow_warn/is_fast 컬럼 없음 — 신규 스키마 로그 축적 후 재계산",
            "shadow_warn_rate": None,
            "shadow_false_warn_rate": None,
            "shadow_miss_rate": None,
        }

    rapid = df[df["is_fast"].fillna(0).astype(int) == 1].copy()
    if rapid.empty:
        return {
            "note": "급속 matched 표본 없음",
            "shadow_warn_rate": None,
            "shadow_false_warn_rate": None,
            "shadow_miss_rate": None,
        }

    has_warn = rapid["shadow_warn"].notna()
    if not has_warn.any():
        return {
            "note": "shadow_warn 값이 모두 NULL — 신규 요청 로그 축적 후 재계산",
            "shadow_warn_rate": None,
            "shadow_false_warn_rate": None,
            "shadow_miss_rate": None,
        }

    rapid = rapid[has_warn]
    warn = rapid["shadow_warn"].astype(int) == 1
    actual_avail = rapid["actual_stat"].astype(int) == 2

    n_rapid = int(len(rapid))
    n_warn = int(warn.sum())
    warn_rate = float(warn.mean()) if n_rapid else None

    # 오경고: warn=1 중 실제 가용 비율
    false_warn_rate: float | None = None
    if n_warn > 0:
        false_warn_rate = float((warn & actual_avail).sum() / n_warn)

    # miss: 실제 불가(actual!=2) 중 warn=0 비율
    actual_unavail = ~actual_avail
    n_unavail = int(actual_unavail.sum())
    miss_rate: float | None = None
    if n_unavail > 0:
        miss_rate = float((actual_unavail & ~warn).sum() / n_unavail)

    return {
        "n_rapid_matched": n_rapid,
        "n_warn": n_warn,
        "shadow_warn_rate": warn_rate,
        "shadow_false_warn_rate": false_warn_rate,
        "shadow_miss_rate": miss_rate,
        "note": (
            "warn=1: available_prob < 요청 ETA별 임계값"
            f"(config 곡선 {RAPID_SHADOW_WARN_THRESHOLD_BY_ETA}, 사이는 선형 보간). "
            "행마다 값이 다르므로 실제 적용값은 로그의 shadow_warn_threshold 를 볼 것. "
            "오경고=warn & 실제가용. miss=불가인데warn=0."
        ),
    }


def _match_diff_buckets() -> dict[str, Any]:
    """matched 행의 라벨 나이 버킷.

    LOCF 라벨에서 |diff| 는 "관측이 얼마나 오래된 것인가"(staleness)다. 이 나이가
    곧 라벨 오차다 — 2026-08-05 신규 2일 실측(delta vs delta+snapshot 라벨 불일치):

        <5분 0.06% · 5~15분 1.14% · 15~30분 3.95% · 30~60분 6.44% · 60~120분 7.11%

    그래서 6분 단일 버킷 대신 이 구간으로 쪼갠다. 구 asof_nearest 로 매칭된 행은
    부호가 반대(미래 관측)라 legacy_ 로 따로 센다.
    """
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT outcome_match_diff_seconds, outcome_label_method
                FROM ev_recommend_prediction_log
                WHERE outcome_match_status = 'matched'
                  AND outcome_match_diff_seconds IS NOT NULL
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    edges = [(5, "0_5"), (15, "5_15"), (30, "15_30"), (60, "30_60"), (120, "60_120")]
    buckets = {name: 0 for _, name in edges}
    buckets["over_120"] = 0
    legacy = {"0_6": 0, "6_11": 0, "over_11": 0}
    n_locf = 0
    for row in rows:
        abs_min = abs(int(row["outcome_match_diff_seconds"])) / 60.0
        if (row.get("outcome_label_method") or "") == "locf":
            n_locf += 1
            for edge, name in edges:
                if abs_min < edge:
                    buckets[name] += 1
                    break
            else:
                buckets["over_120"] += 1
        else:
            if abs_min <= 6:
                legacy["0_6"] += 1
            elif abs_min <= 11:
                legacy["6_11"] += 1
            else:
                legacy["over_11"] += 1
    return {
        "label_staleness_min_buckets": buckets,
        "diff_abs_min_buckets": legacy,
        "n_with_diff": int(len(rows)),
        "n_locf": n_locf,
        "note": (
            f"label_staleness_min_buckets = LOCF 라벨 나이(상한 {LABEL_MAX_STALENESS_MIN}분). "
            "나이가 곧 라벨 오차이므로 30분 초과 비중이 커지면 수집 누락을 의심할 것. "
            "diff_abs_min_buckets 는 구 asof_nearest(±6분)로 매칭된 잔존 행."
        ),
    }


def compute_ops_metrics(df: pd.DataFrame) -> dict[str, Any]:
    ensure_table()
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN target_at <= NOW() THEN 1 ELSE 0 END)
                        AS eligible_for_backfill,
                    SUM(CASE WHEN outcome_match_status = 'matched' THEN 1 ELSE 0 END)
                        AS matched_outcomes,
                    SUM(CASE WHEN outcome_match_status = 'no_observation' THEN 1 ELSE 0 END)
                        AS no_observation,
                    SUM(CASE WHEN outcome_match_status = 'unlabelable_stat' THEN 1 ELSE 0 END)
                        AS unlabelable_stat,
                    SUM(CASE WHEN COALESCE(outcome_label_method, '') = %s THEN 1 ELSE 0 END)
                        AS labeled_current_method,
                    SUM(
                        CASE
                            WHEN target_at <= NOW()
                             AND (outcome_match_status IS NULL
                                  OR outcome_match_status NOT IN
                                     ('matched', 'no_observation', 'unlabelable_stat'))
                            THEN 1 ELSE 0
                        END
                    ) AS pending_eligible
                FROM ev_recommend_prediction_log
                """,
                (LABEL_METHOD,),
            )
            bf = cur.fetchone() or {}
    finally:
        conn.close()

    total = int(bf.get("total_predictions") or 0)
    eligible = int(bf.get("eligible_for_backfill") or 0)
    matched = int(bf.get("matched_outcomes") or 0)
    no_obs = int(bf.get("no_observation") or 0)
    unlabelable = int(bf.get("unlabelable_stat") or 0)
    labeled_current = int(bf.get("labeled_current_method") or 0)
    pending = int(bf.get("pending_eligible") or 0)
    attempted = matched + no_obs + unlabelable
    backfill_rate = (matched / attempted) if attempted else None

    diff_info = _match_diff_buckets()

    backfill = {
        "total_predictions": total,
        "eligible_for_backfill": eligible,
        "matched_outcomes": matched,
        "no_observation": no_obs,
        "unlabelable_stat": unlabelable,
        "pending_eligible": pending,
        "backfill_success_rate": backfill_rate,
        "label_method": LABEL_METHOD,
        "label_max_staleness_min": LABEL_MAX_STALENESS_MIN,
        "labeled_with_current_method": labeled_current,
        "label_method_note": (
            "학습 라벨(model_store.build_horizon_dataset)과 같은 규칙이어야 오프라인 "
            "지표와 같은 눈금이 된다. labeled_with_current_method 가 total 보다 작으면 "
            "`py -m recommend_api.backfill_outcomes --relabel` 로 맞출 것."
        ),
        # aliases
        "matched": matched,
        "success_rate": backfill_rate,
        "with_actual": matched,
        "total_past_target": eligible,
    }

    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_matched_rows": int(len(df)),
        "backfill": backfill,
        "label_staleness_min_buckets": diff_info["label_staleness_min_buckets"],
        "diff_abs_min_buckets": diff_info["diff_abs_min_buckets"],
        "diff_abs_min_buckets_meta": {
            "n_with_diff": diff_info["n_with_diff"],
            "n_locf": diff_info["n_locf"],
            "note": diff_info["note"],
        },
        # ── 주지표는 headline(= 신선 라벨 구간)이다 ──────────────────────────
        # overall 을 헤드라인으로 쓰면 오해를 부른다. 2026-08-10 실측(n=1,260):
        # 라벨 나이 60분 초과 구간은 positive 0.9189 로 신선 구간(0.8097)보다 높다.
        # 조용한 충전기는 대개 대기(stat=2)이고 LOCF 가 그 상태를 유지하기 때문이다.
        # 즉 오래된 라벨은 잡음일 뿐 아니라 **가용 쪽으로 편향**돼 있고, 그 구간
        # ROC 는 0.632 로 신선 구간(0.747)보다 낮다.
        # 기존 키는 지우지 않는다(소비자 호환). 순서와 note 로 우선순위를 드러낸다.
        "headline": {
            "label": "label_staleness<=60min",
            "n": 0,
            "source": "overall_fresh_label",
            "why": (
                "오래된 LOCF 라벨은 가용 쪽으로 편향된다. overall 은 참고용이며 "
                "모델 판단 근거로 쓰지 말 것."
            ),
        },
        "overall_fresh_label": {"label": "label_staleness<=60min", "n": 0},
        "overall_stale_label": {"label": "label_staleness>60min", "n": 0},
        "overall": _group_metrics(df, "overall"),
        "shadow_warn": _shadow_warn_metrics(df),
        "by_eta": [],
        "by_long_state": [],
        "by_stale": [],
        "by_model_version": [],
        "by_label_method": [],
        "top1_station_failure_rate": {
            "label": "all",
            "n_requests": 0,
            "n_failed": 0,
            "rate": None,
            "note": "matched 표본 없음",
        },
        "top1_failure_rate": None,
        "top1_station_failure_rapid": {
            "label": "rapid_fresh_compatible",
            "n_requests": 0,
            "n_failed": 0,
            "rate": None,
            "note": "matched 표본 없음 - 오프라인 station_top1_metrics.json 참고",
        },
    }

    if df.empty:
        out["shadow_warn"] = _shadow_warn_metrics(df)
        return out

    # 라벨 나이가 오래된 행은 LOCF 오차가 커진다(30~60분 6.4%, 60~120분 7.1% —
    # 2026-08-05 실측). 추천으로 나가는 충전기는 대개 유휴라 delta 행이 드물어서
    # 운영 로그의 라벨 나이는 학습 표본보다 훨씬 길다. 신선한 라벨만 따로 본다.
    if "outcome_match_diff_seconds" in df.columns:
        age_min = pd.to_numeric(df["outcome_match_diff_seconds"], errors="coerce").abs() / 60.0
        fresh = df.loc[age_min <= 60]
        stale = df.loc[age_min > 60]
        out["overall_fresh_label"] = _group_metrics(fresh, "label_staleness<=60min")
        out["overall_fresh_label"]["note"] = (
            "라벨 나이 60분 이내만. 이것이 주지표다(headline). "
            "전체(overall)는 오래된 LOCF 라벨의 잡음과 편향을 포함한다."
        )
        out["overall_stale_label"] = _group_metrics(stale, "label_staleness>60min")
        out["overall_stale_label"]["note"] = (
            "라벨 나이 60분 초과. 진단용이다 — 이 구간의 positive 가 신선 구간보다 "
            "높으면 LOCF 가 '조용한 충전기 = 대기'로 굳히고 있다는 신호다."
        )
        out["headline"] = {
            **out["overall_fresh_label"],
            "source": "overall_fresh_label",
            "why": out["headline"]["why"],
            "n_excluded_stale": int(len(stale)),
            "stale_share": (round(len(stale) / len(df), 4) if len(df) else None),
        }
        out["overall"]["note"] = (
            "참고용. 신선/오래된 라벨이 섞여 있어 모델 판단에 쓰지 말 것 → headline 사용."
        )

    eta = df["eta_minutes"].astype(float)
    bins = [0, 7.5, 12.5, 17.5, 25, 37.5, 52.5, 1000]
    labels = ["~5", "~10", "~15", "~20", "~30", "~45", "~60+"]
    df = df.copy()
    df["eta_bucket"] = pd.cut(eta, bins=bins, labels=labels, include_lowest=True)
    for lab, g in df.groupby("eta_bucket", observed=False):
        out["by_eta"].append(_group_metrics(g, str(lab)))

    for flag, key in [
        ("is_long_state_duration", "by_long_state"),
        ("is_stale_status", "by_stale"),
    ]:
        if flag not in df.columns:
            continue
        for val, g in df.groupby(flag):
            out[key].append(_group_metrics(g, f"{flag}={int(val)}"))

    for ver, g in df.groupby(df["model_version"].fillna("unknown")):
        out["by_model_version"].append(_group_metrics(g, str(ver)))

    # 라벨 방식이 섞여 있으면 지표를 하나로 읽으면 안 된다(눈금이 다르다).
    if "outcome_label_method" in df.columns:
        for meth, g in df.groupby(df["outcome_label_method"].fillna("legacy_asof_nearest")):
            out["by_label_method"].append(_group_metrics(g, str(meth)))

    station_top1 = _top1_station_failure(df, label="all")
    out["top1_station_failure_rate"] = station_top1
    top1 = df[df["station_rank"] == 1]
    if not top1.empty:
        fail = (top1["actual_stat"].astype(int) != 2).mean()
        out["top1_failure_rate"] = {
            "n": int(len(top1)),
            "rate_actual_not_stat2": float(fail),
            "note": "충전기 행 기준(중복 가능). 운영 해석은 top1_station_failure_rate 사용",
        }

    try:
        enriched = _enrich_with_charger_info(df)
        rapid_fresh = _filter_rapid_fresh(enriched)
        out["top1_station_failure_rapid"] = _top1_station_failure(
            rapid_fresh, label="rapid_fresh_compatible"
        )
        out["top1_station_failure_rapid"]["filters"] = {
            "is_fast": True,
            "is_stale_status": 0,
            "status_update_age_min_lt": 60,
            "respect_min_output_kw_when_logged": True,
        }
    except Exception as exc:  # noqa: BLE001
        out["top1_station_failure_rapid"] = {
            "label": "rapid_fresh_compatible",
            "n_requests": 0,
            "rate": None,
            "error": str(exc),
        }

    if "outcome_match_direction" in df.columns:
        out["match_direction_counts"] = (
            df["outcome_match_direction"].fillna("unknown").value_counts().to_dict()
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ops metrics from prediction log")
    parser.add_argument("--days", type=int, default=None, help="최근 N일만")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df = load_matched_logs(args.days)
    metrics = compute_ops_metrics(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    bf = metrics["backfill"]
    print(f"저장: {args.out}")
    print(
        f"label_method={bf['label_method']}(<= {bf['label_max_staleness_min']}분) "
        f"total={bf['total_predictions']} eligible={bf['eligible_for_backfill']} "
        f"matched={bf['matched_outcomes']} no_obs={bf['no_observation']} "
        f"unlabelable={bf['unlabelable_stat']} pending={bf['pending_eligible']} "
        f"backfill_success_rate={bf['backfill_success_rate']}"
    )
    # 미라벨 행은 두 종류이고 처방이 다르다. 2026-08-10 이전에는 둘을 뭉뚱그려
    # "다른 방식으로 라벨돼 있다 → --relabel" 로 안내했는데, 실제 잔여 31행은
    # 방식이 다른 게 아니라 **아예 백필이 안 돈 것**이었다. --relabel 은 방식 변경
    # 직후 1회용이므로, 정기 백필이 안 도는 상황에 이걸 권하면 원인을 가린다.
    n_unlabeled = bf["total_predictions"] - bf["labeled_with_current_method"]
    if n_unlabeled:
        pending = bf.get("pending_eligible") or 0
        other_method = max(0, n_unlabeled - pending)
        if pending:
            print(
                f"  경고: {pending}행이 아직 백필되지 않았다 → backfill_outcomes "
                "(정기 실행되고 있는지 확인할 것: compose 서비스 backfill)"
            )
        if other_method:
            print(
                f"  경고: {other_method}행이 다른 방식으로 라벨돼 있다 "
                "→ backfill_outcomes --relabel (방식 변경 직후 1회용)"
            )
    # 주지표를 먼저 찍는다. 이전에는 overall / fresh 둘 다 콘솔에 안 나와서
    # JSON 키 순서가 사실상 헤드라인 역할을 했다.
    hl = metrics.get("headline") or {}
    ov = metrics.get("overall") or {}
    st = metrics.get("overall_stale_label") or {}
    if hl.get("n"):
        print(
            f"[주지표] {hl.get('label')}  n={hl.get('n')}  "
            f"ROC={hl.get('roc_auc')}  Brier={hl.get('brier')}  "
            f"불가recall={hl.get('unavailable_recall_thr0.5')}  "
            f"positive={round(hl['n_available'] / hl['n'], 4) if hl.get('n') else None}"
        )
        print(
            f"  (참고) overall n={ov.get('n')} ROC={ov.get('roc_auc')} · "
            f"제외한 오래된 라벨 {hl.get('n_excluded_stale')}건"
            f"({hl.get('stale_share')})"
        )
        if st.get("n") and st.get("n_available") is not None:
            p_st = round(st["n_available"] / st["n"], 4)
            p_hl = round(hl["n_available"] / hl["n"], 4)
            if p_st > p_hl:
                print(
                    f"  경고: 오래된 라벨 구간 positive {p_st} > 신선 {p_hl} — "
                    "LOCF 가 '조용한 충전기 = 대기'로 굳히는 중"
                )
    else:
        print("[주지표] 신선 라벨(<=60분) 표본 없음 — headline 미계산")
    print("label_staleness_min_buckets:", metrics.get("label_staleness_min_buckets"))
    if any(metrics.get("diff_abs_min_buckets", {}).values()):
        print("  (구 asof_nearest 잔존:", metrics["diff_abs_min_buckets"], ")")
    if metrics.get("top1_station_failure_rate"):
        print("top1_station_failure:", metrics["top1_station_failure_rate"])
    if metrics.get("top1_station_failure_rapid"):
        print("top1_station_failure_rapid:", metrics["top1_station_failure_rapid"])
    if metrics.get("shadow_warn"):
        sw = metrics["shadow_warn"]
        print(
            f"shadow_warn: rate={sw.get('shadow_warn_rate')} "
            f"false_warn={sw.get('shadow_false_warn_rate')} "
            f"miss={sw.get('shadow_miss_rate')}"
        )


if __name__ == "__main__":
    main()
