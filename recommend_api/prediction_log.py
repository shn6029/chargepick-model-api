"""추천 예측 로그 테이블 및 best-effort 기록 / 백필."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import pymysql

from .config import (
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    LABELABLE_FUTURE_STATS,
    OUTCOME_MATCH_TOLERANCE_MIN,
)
from .model_store import get_connection

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ev_recommend_prediction_log (
    id                      BIGINT       NOT NULL AUTO_INCREMENT,
    request_id              CHAR(36)     NOT NULL,
    created_at              DATETIME     NOT NULL,
    model_version           VARCHAR(32)  NULL,
    dest_lat                DOUBLE       NOT NULL,
    dest_lng                DOUBLE       NOT NULL,
    eta_minutes             DOUBLE       NOT NULL,
    radius_km               DOUBLE       NULL,
    top_k                   INT          NULL,
    stat_id                 VARCHAR(20)  NOT NULL,
    chger_id                VARCHAR(10)  NOT NULL,
    station_rank            INT          NULL,
    available_prob          DOUBLE       NULL,
    current_stat            TINYINT      NULL,
    current_state_duration  INT          NULL,
    status_update_age_min   DOUBLE       NULL,
    is_long_state_duration  TINYINT      NULL,
    is_stale_status         TINYINT      NULL,
    confidence_level        VARCHAR(16)  NULL,
    vehicle_model_id        VARCHAR(64)  NULL,
    min_output_kw           DOUBLE       NULL,
    n_candidate_stations    INT          NULL,
    detour_minutes          DOUBLE       NULL,
    n_compatible_fast       INT          NULL,
    shadow_warn             TINYINT      NULL,
    shadow_warn_threshold   DOUBLE       NULL,
    is_fast                 TINYINT      NULL,
    target_at               DATETIME     NOT NULL,
    actual_stat             TINYINT      NULL,
    actual_observed_at      DATETIME     NULL,
    matched_at              DATETIME     NULL,
    outcome_match_diff_seconds INT       NULL,
    outcome_match_direction VARCHAR(24)  NULL,
    outcome_match_status    VARCHAR(24)  NULL,
    outcome_label_method    VARCHAR(16)  NULL,
    PRIMARY KEY (id),
    KEY idx_req (request_id),
    KEY idx_pending (actual_stat, target_at),
    KEY idx_charger_target (stat_id, chger_id, target_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 기존 테이블에 컬럼이 없을 수 있음
_ALTER_COLUMNS = [
    ("status_update_age_min", "DOUBLE NULL"),
    ("is_long_state_duration", "TINYINT NULL"),
    ("outcome_match_diff_seconds", "INT NULL"),
    ("outcome_match_direction", "VARCHAR(24) NULL"),
    ("outcome_match_status", "VARCHAR(24) NULL"),
    ("outcome_label_method", "VARCHAR(16) NULL"),
    ("vehicle_model_id", "VARCHAR(64) NULL"),
    ("min_output_kw", "DOUBLE NULL"),
    ("n_candidate_stations", "INT NULL"),
    ("detour_minutes", "DOUBLE NULL"),
    ("n_compatible_fast", "INT NULL"),
    ("shadow_warn", "TINYINT NULL"),
    ("shadow_warn_threshold", "DOUBLE NULL"),
    ("is_fast", "TINYINT NULL"),
]

INSERT_SQL = """
INSERT INTO ev_recommend_prediction_log (
    request_id, created_at, model_version,
    dest_lat, dest_lng, eta_minutes, radius_km, top_k,
    stat_id, chger_id, station_rank,
    available_prob, current_stat, current_state_duration,
    status_update_age_min, is_long_state_duration, is_stale_status,
    confidence_level, vehicle_model_id, min_output_kw,
    n_candidate_stations, detour_minutes, n_compatible_fast,
    shadow_warn, shadow_warn_threshold, is_fast, target_at
) VALUES (
    %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s
)
"""


def ensure_table(conn=None) -> None:
    own = conn is None
    if own:
        conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            for col, typedef in _ALTER_COLUMNS:
                try:
                    cur.execute(
                        f"ALTER TABLE ev_recommend_prediction_log "
                        f"ADD COLUMN {col} {typedef}"
                    )
                except pymysql.err.OperationalError as exc:
                    # Duplicate column name
                    if getattr(exc, "args", [None])[0] != 1060:
                        raise
        conn.commit()
    finally:
        if own:
            conn.close()


def log_recommendations(
    response: dict[str, Any],
    *,
    dest_lat: float,
    dest_lng: float,
    eta_minutes: float,
    radius_km: float,
    top_k: int,
    frame_by_key: dict[tuple[str, str], dict[str, Any]] | None = None,
    vehicle_model_id: str | None = None,
    min_output_kw: float | None = None,
    n_candidate_stations: int | None = None,
) -> str | None:
    """응답의 추천 충전기 행을 best-effort로 기록. 실패 시 None."""
    request_id = str(uuid.uuid4())
    created_at = datetime.now().replace(microsecond=0)
    target_at = created_at + timedelta(minutes=float(eta_minutes))
    meta = response.get("meta") or {}
    model_version = meta.get("model_version")
    confidence = meta.get("confidence_level", "high")
    shadow_warn_thr = meta.get("shadow_warn_threshold")
    rows = []

    for station in response.get("recommendations") or []:
        rank = station.get("rank")
        stat_id = str(station.get("stat_id", ""))
        detour_min = station.get("detour_minutes")
        # 해당 충전소 내 급속 충전기 수
        charger_list = station.get("chargers") or []
        n_fast = sum(
            1 for ch in charger_list
            if ch.get("output_kw") is not None and float(ch["output_kw"]) >= 50
        )
        for ch in charger_list:
            chger_id = str(ch.get("chger_id", ""))
            key = (stat_id, chger_id)
            extra = (frame_by_key or {}).get(key, {})
            duration = ch.get("current_state_duration", extra.get("current_state_duration"))
            age = ch.get("status_update_age_min", extra.get("status_update_age_min"))
            long_state = ch.get(
                "is_long_state_duration", extra.get("is_long_state_duration", 0)
            )
            stale = ch.get("is_stale_status", extra.get("is_stale_status", 0))
            output_kw = ch.get("output_kw")
            is_fast_ch = int(output_kw is not None and float(output_kw) >= 50)
            shadow = ch.get("shadow_unavailable_warn")
            rows.append(
                (
                    request_id,
                    created_at,
                    model_version,
                    float(dest_lat),
                    float(dest_lng),
                    float(eta_minutes),
                    float(radius_km),
                    int(top_k),
                    stat_id,
                    chger_id,
                    int(rank) if rank is not None else None,
                    float(ch["available_prob"]) if ch.get("available_prob") is not None else None,
                    int(ch["current_stat"]) if ch.get("current_stat") is not None else None,
                    int(duration) if duration is not None and str(duration) != "nan" else None,
                    float(age) if age is not None and str(age) != "nan" else None,
                    int(bool(long_state)),
                    int(bool(stale)),
                    confidence,
                    str(vehicle_model_id) if vehicle_model_id else None,
                    float(min_output_kw) if min_output_kw is not None else None,
                    int(n_candidate_stations) if n_candidate_stations is not None else None,
                    float(detour_min) if detour_min is not None else None,
                    int(n_fast),
                    int(bool(shadow)) if shadow is not None else None,
                    float(shadow_warn_thr) if shadow_warn_thr is not None else None,
                    is_fast_ch,
                    target_at,
                )
            )

    if not rows:
        return request_id

    try:
        conn = get_connection()
        try:
            # 서빙 경로에서는 스키마 보장을 하지 않는다(기본값).
            #
            # ensure_table 은 CREATE TABLE IF NOT EXISTS + ALTER TABLE 을 실행하는데,
            # 테이블이 이미 있어도 MariaDB 는 CREATE 권한 자체를 검사한다. 그래서
            # 최소 권한 계정(SELECT + 이 테이블 INSERT)으로 서빙하면 여기서 1142 로
            # 막히고, 아래 except 에 먹혀 **추천 응답은 200 인데 로그만 조용히
            # 사라진다**. 2026-08-04 에 실제로 겪었다.
            #
            # 스키마 생성·변경은 배치의 일이다 — ops_metrics.py, verify_ops.py,
            # backfill_outcomes.py 가 각자 ensure_table() 을 직접 부른다.
            # 빈 DB 로 새로 띄우는 경우에만 이 환경변수를 1 로 켜면 된다.
            if os.getenv("PREDICTION_LOG_ENSURE_TABLE", "0") == "1":
                ensure_table(conn)
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, rows)
            conn.commit()
        finally:
            conn.close()
        return request_id
    except Exception as exc:  # noqa: BLE001
        print(f"[prediction_log] 기록 실패(무시): {exc}")
        return None


def _pick_observation_locf(
    cur,
    *,
    stat_id: str,
    chger_id: str,
    target_at: datetime,
    max_staleness_min: int,
) -> tuple[dict[str, Any] | None, str]:
    """target_at 이전 마지막 관측 = LOCF. (hit, direction).

    학습 라벨(`model_store.build_horizon_dataset`, merge_asof direction="backward",
    tolerance=LABEL_MAX_STALENESS_MIN)과 **같은 규칙**이다. 운영 지표와 오프라인
    지표를 같은 눈금으로 보려면 여기서 갈리면 안 된다.

    forward 매칭이 아니라 backward 인 이유
    -------------------------------------
    delta 수집은 상태가 바뀐 충전기만 행을 만든다. target_at 직후(+2분)에 행이
    생겼다는 건 "그 시점에 상태가 바뀌었다"는 뜻이므로, target_at 시점의 상태는
    **바뀌기 전** 값이다. 구 forward 매칭은 이걸 바뀐 뒤 값으로 라벨링했다.

    인덱스: idx_charger_created(stat_id, chger_id, created_at) 로 range 1행.
    """
    cur.execute(
        """
        SELECT stat, created_at
        FROM ev_charger_status
        WHERE stat_id = %s AND chger_id = %s
          AND created_at <= %s
          AND created_at >= DATE_SUB(%s, INTERVAL %s MINUTE)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (stat_id, chger_id, target_at, target_at, int(max_staleness_min)),
    )
    hit = cur.fetchone()
    if hit:
        return hit, "locf"
    return None, "none"


def _pick_observation_nearest(
    cur,
    *,
    stat_id: str,
    chger_id: str,
    target_at: datetime,
    tolerance_min: int,
) -> tuple[dict[str, Any] | None, str]:
    """구 방식: forward 우선, 없으면 backward. (hit, direction).

    2026-08-05 까지의 기본값이었다. ±6분 안에 **물리적 관측 행**을 요구하므로
    delta 가 침묵한 충전기(=상태가 그대로인 충전기)는 통째로 버려진다.
    실측 백필률 34.8%(430/1,236)에 남은 표본도 "그 시각 상태가 움직인 충전기"로
    치우쳤다. 학습은 LOCF 라벨이라 눈금 자체가 달랐다.

    비교·재현용으로만 남긴다(`--method asof_nearest`).
    """
    # exact
    cur.execute(
        """
        SELECT stat, created_at
        FROM ev_charger_status
        WHERE stat_id = %s AND chger_id = %s AND created_at = %s
        LIMIT 1
        """,
        (stat_id, chger_id, target_at),
    )
    hit = cur.fetchone()
    if hit:
        return hit, "exact"

    # forward: target_at 이후 최초
    cur.execute(
        """
        SELECT stat, created_at
        FROM ev_charger_status
        WHERE stat_id = %s AND chger_id = %s
          AND created_at > %s
          AND created_at <= DATE_ADD(%s, INTERVAL %s MINUTE)
        ORDER BY created_at ASC
        LIMIT 1
        """,
        (stat_id, chger_id, target_at, target_at, int(tolerance_min)),
    )
    hit = cur.fetchone()
    if hit:
        return hit, "forward"

    # backward fallback
    cur.execute(
        """
        SELECT stat, created_at
        FROM ev_charger_status
        WHERE stat_id = %s AND chger_id = %s
          AND created_at < %s
          AND created_at >= DATE_SUB(%s, INTERVAL %s MINUTE)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (stat_id, chger_id, target_at, target_at, int(tolerance_min)),
    )
    hit = cur.fetchone()
    if hit:
        return hit, "backward_fallback"

    return None, "none"


def backfill_outcomes(
    limit: int = 5000,
    *,
    method: str | None = None,
    max_staleness_min: int | None = None,
    tolerance_min: int = OUTCOME_MATCH_TOLERANCE_MIN,
    settle_min: int = OUTCOME_MATCH_TOLERANCE_MIN,
    relabel: bool = False,
    chunk_size: int = 200,
) -> dict[str, Any]:
    """target_at 이 지난 예측 로그에 실제 상태를 채운다. 기본은 학습과 같은 LOCF.

    라벨 규칙을 학습(`config.LABEL_METHOD` = "locf", staleness <= 480분)과 맞춘다.
    구 방식(asof_nearest ±6분)은 백필률 34.8% 에 표본까지 편향돼 있어
    ops_metrics 수치를 오프라인 지표와 비교할 수 없었다. 2026-08-05 전환.

    settle_min
        target_at 직후 구간은 아직 수집이 안 들어왔을 수 있다(delta 5분 주기).
        NOW() - settle_min 이전 target 만 처리해 "수집 지연을 상태 유지로
        오해하는" 라벨을 막는다.

    stat 1(통신이상)·9(상태미확인)
        학습에서 제외하는 상태다(`config.LABELABLE_FUTURE_STATS`). 여기서도
        actual_stat 을 채우지 않고 `unlabelable_stat` 으로 표시만 한다 —
        ops_metrics 가 사용불가로 오해하면 안 되기 때문이다.

    relabel=True
        현재 method 로 라벨링되지 않은 행을 **전부** 다시 매긴다(구 매칭분·
        no_observation 포함). 라벨 방식을 바꾼 직후 한 번 돌리는 용도다.

    커밋은 chunk_size 행마다 나눠서 한다. 같은 DB 에 5분 수집이 함께 쓰므로
    긴 트랜잭션을 잡으면 안 된다(CLAUDE.md §DB 제약).
    """
    method = method or LABEL_METHOD
    if method not in ("locf", "asof_nearest"):
        raise ValueError(f"알 수 없는 라벨링 방식: {method}")
    max_staleness_min = (
        LABEL_MAX_STALENESS_MIN if max_staleness_min is None else max_staleness_min
    )

    ensure_table()
    conn = get_connection()
    updated = 0
    scanned = 0
    no_obs = 0
    unlabelable = 0
    staleness: list[float] = []
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            if relabel:
                target_filter = "COALESCE(outcome_label_method, '') <> %s"
                params: tuple[Any, ...] = (method, int(settle_min), int(limit))
            else:
                target_filter = (
                    "(actual_stat IS NULL "
                    " AND COALESCE(outcome_match_status, '') "
                    "     NOT IN ('no_observation', 'unlabelable_stat'))"
                )
                params = (int(settle_min), int(limit))
            cur.execute(
                f"""
                SELECT id, stat_id, chger_id, target_at
                FROM ev_recommend_prediction_log
                WHERE {target_filter}
                  AND target_at <= NOW() - INTERVAL %s MINUTE
                ORDER BY target_at
                LIMIT %s
                """,
                params,
            )
            pending = cur.fetchall()
            scanned = len(pending)

            for i, row in enumerate(pending, 1):
                if method == "locf":
                    hit, direction = _pick_observation_locf(
                        cur,
                        stat_id=row["stat_id"],
                        chger_id=row["chger_id"],
                        target_at=row["target_at"],
                        max_staleness_min=max_staleness_min,
                    )
                else:
                    hit, direction = _pick_observation_nearest(
                        cur,
                        stat_id=row["stat_id"],
                        chger_id=row["chger_id"],
                        target_at=row["target_at"],
                        tolerance_min=tolerance_min,
                    )

                if not hit:
                    cur.execute(
                        """
                        UPDATE ev_recommend_prediction_log
                        SET actual_stat = NULL,
                            actual_observed_at = NULL,
                            outcome_match_status = 'no_observation',
                            outcome_match_direction = 'none',
                            outcome_match_diff_seconds = NULL,
                            outcome_label_method = %s,
                            matched_at = NOW()
                        WHERE id = %s
                        """,
                        (method, row["id"]),
                    )
                    no_obs += 1
                else:
                    obs_at = hit["created_at"]
                    diff_sec = int((obs_at - row["target_at"]).total_seconds())
                    stat = int(hit["stat"])
                    if stat in LABELABLE_FUTURE_STATS:
                        cur.execute(
                            """
                            UPDATE ev_recommend_prediction_log
                            SET actual_stat = %s,
                                actual_observed_at = %s,
                                matched_at = NOW(),
                                outcome_match_diff_seconds = %s,
                                outcome_match_direction = %s,
                                outcome_match_status = 'matched',
                                outcome_label_method = %s
                            WHERE id = %s
                            """,
                            (stat, obs_at, diff_sec, direction, method, row["id"]),
                        )
                        staleness.append(abs(diff_sec) / 60.0)
                        updated += 1
                    else:
                        # stat 1·9 는 사용불가와 구분이 안 된다 — 라벨을 붙이지 않는다.
                        cur.execute(
                            """
                            UPDATE ev_recommend_prediction_log
                            SET actual_stat = NULL,
                                actual_observed_at = %s,
                                matched_at = NOW(),
                                outcome_match_diff_seconds = %s,
                                outcome_match_direction = %s,
                                outcome_match_status = 'unlabelable_stat',
                                outcome_label_method = %s
                            WHERE id = %s
                            """,
                            (obs_at, diff_sec, direction, method, row["id"]),
                        )
                        unlabelable += 1

                if i % chunk_size == 0:
                    conn.commit()
                    time.sleep(0.05)
        conn.commit()
    finally:
        conn.close()

    result: dict[str, Any] = {
        "method": method,
        "scanned": scanned,
        "updated": updated,
        "no_observation": no_obs,
        "unlabelable_stat": unlabelable,
        "relabel": relabel,
    }
    if method == "locf":
        result["max_staleness_min"] = int(max_staleness_min)
    if staleness:
        arr = sorted(staleness)
        result["label_staleness_median_min"] = round(arr[len(arr) // 2], 2)
        result["label_staleness_p90_min"] = round(arr[int(len(arr) * 0.9) - 1], 2)
    return result
