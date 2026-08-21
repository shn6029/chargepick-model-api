"""운영 흐름 통합 검증.

사용:
  py -m recommend_api.verify_ops
  py -m recommend_api.verify_ops --skip-backfill
  py -m recommend_api.verify_ops --lat 35.84217 --lng 128.68043
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import (
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_PATH,
    VERY_STALE_EXCLUDE_MIN,
)
from recommend_api.model_store import (
    get_connection,
    load_artifact,
    validate_artifact_features,
)
from recommend_api.prediction_log import backfill_outcomes, ensure_table
from recommend_api.service import recommend


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def check_versions() -> str:
    if not MODEL_PATH.exists():
        _fail(f"모델 없음: {MODEL_PATH}")
    artifact = load_artifact(strict_features=False)
    joblib_ver = artifact.get("model_version")
    if not joblib_ver:
        _fail("joblib에 model_version 없음")

    if not METRICS_PATH.exists():
        _fail(f"metrics 없음: {METRICS_PATH}")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics_ver = metrics.get("model_version")
    if joblib_ver != metrics_ver:
        _fail(f"버전 불일치 joblib={joblib_ver} metrics={metrics_ver}")

    _ok(f"model_version={joblib_ver} (joblib=metrics)")
    return str(joblib_ver)


def check_feature_schema() -> None:
    artifact = load_artifact(strict_features=False)
    msgs = validate_artifact_features(artifact, strict=False)
    if msgs:
        _fail("; ".join(msgs))

    cols = list(artifact.get("feature_columns") or [])
    n_base = len(FEATURE_COLS)
    if cols[:n_base] != list(FEATURE_COLS):
        _fail("artifact feature_columns 앞부분 != FEATURE_COLS")

    base = artifact.get("base_feature_cols")
    if base is not None and list(base) != list(FEATURE_COLS):
        _fail("artifact base_feature_cols != FEATURE_COLS")

    fcount = artifact.get("feature_count") or len(cols)
    if int(fcount) != len(cols):
        _fail(f"feature_count={fcount} != len(feature_columns)={len(cols)}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    ah = artifact.get("feature_schema_hash")
    mh = metrics.get("feature_schema_hash")
    if ah and mh and ah != mh:
        _fail(f"feature_schema_hash 불일치 artifact={ah} metrics={mh}")
    if metrics.get("n_features") is not None and int(metrics["n_features"]) != len(cols):
        _fail(
            f"metrics n_features={metrics['n_features']} != artifact cols={len(cols)}"
        )

    _ok(
        f"feature schema ok base={n_base} encoded={len(cols)} "
        f"hash={ah or '(unstamped)'}"
    )


def check_feature_lag() -> None:
    """피처 배치 지연 점검.

    서빙은 최신 status 행에 ev_charger_features 를 INNER JOIN 하고
    VERY_STALE_EXCLUDE_MIN 이내 신선도를 요구하므로, 피처가 밀리면
    추천이 조용히 0건이 된다. scripts/incremental_features.ps1(5분 주기) 확인.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT TIMESTAMPDIFF(MINUTE, MAX(created_at), NOW()) "
                "FROM ev_charger_features"
            )
            row = cur.fetchone()
    finally:
        conn.close()

    lag = None if row is None or row[0] is None else int(row[0])
    if lag is None:
        _fail("ev_charger_features 가 비어 있음 → 추천 0건")
    if lag > VERY_STALE_EXCLUDE_MIN:
        _fail(
            f"피처 지연 {lag}분 > 임계 {VERY_STALE_EXCLUDE_MIN}분 → 추천 0건 위험. "
            "scripts/incremental_features.ps1 (5분 주기) 등록/실행 확인"
        )
    _ok(f"feature lag {lag}분 (임계 {VERY_STALE_EXCLUDE_MIN}분)")


def check_recommend(
    *,
    lat: float,
    lng: float,
    eta: float,
    expect_confidence: str,
    expect_horizon_substr: str | None,
    forbid_horizon_substr: str | None = None,
    model_version: str,
) -> str:
    resp = recommend(
        dest_lat=lat,
        dest_lng=lng,
        eta_minutes=eta,
        radius_km=2.0,
        top_k=5,
        log_predictions=True,
    )
    meta = resp.get("meta") or {}
    conf = meta.get("confidence_level")
    if conf != expect_confidence:
        _fail(f"ETA {eta}: confidence_level={conf!r} (expected {expect_confidence})")
    note = meta.get("horizon_note") or ""
    if expect_horizon_substr and expect_horizon_substr not in note:
        _fail(f"ETA {eta}: horizon_note에 '{expect_horizon_substr}' 없음: {note!r}")
    if forbid_horizon_substr and forbid_horizon_substr in note:
        _fail(f"ETA {eta}: horizon_note에 '{forbid_horizon_substr}' 있음: {note!r}")
    resp_ver = meta.get("model_version")
    if resp_ver != model_version:
        _fail(f"응답 model_version={resp_ver!r} != {model_version!r}")
    req_id = meta.get("request_id")
    if not req_id:
        _fail(f"ETA {eta}: request_id 없음 (로그 실패?)")
    _ok(
        f"ETA {eta}: confidence={conf}, request_id={req_id}, "
        f"n_recs={len(resp.get('recommendations') or [])}"
    )
    return str(req_id)


def check_log_rows(request_ids: list[str], model_version: str) -> None:
    ensure_table()
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            for rid in request_ids:
                cur.execute(
                    """
                    SELECT COUNT(*) AS n,
                           MIN(model_version) AS mv,
                           MIN(target_at) AS target_at,
                           MIN(eta_minutes) AS eta
                    FROM ev_recommend_prediction_log
                    WHERE request_id = %s
                    """,
                    (rid,),
                )
                row = cur.fetchone()
                n = int(row["n"] or 0)
                if n < 1:
                    _fail(f"로그 없음 request_id={rid}")
                if row["mv"] != model_version:
                    _fail(
                        f"로그 model_version={row['mv']!r} != {model_version!r} "
                        f"(request_id={rid})"
                    )
                if row["target_at"] is None:
                    _fail(f"target_at NULL request_id={rid}")
                _ok(
                    f"log request_id={rid}: rows={n}, eta={row['eta']}, "
                    f"target_at={row['target_at']}"
                )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify recommend ops flow")
    parser.add_argument("--lat", type=float, default=35.84217)
    parser.add_argument("--lng", type=float, default=128.68043)
    parser.add_argument("--skip-backfill", action="store_true")
    args = parser.parse_args()

    print("=== verify_ops ===")
    model_version = check_versions()
    check_feature_schema()
    check_feature_lag()

    # service._confidence_level: ≤15 high · 16–20 medium · ≥21 low
    # service._horizon_note:     16–20 변동 · 21–30 중거리 · >30 장거리
    # 경계값 15/16/20/21 을 직접 짚어 구간 끝을 고정한다(45는 장거리 문구 확인).
    rid15 = check_recommend(
        lat=args.lat,
        lng=args.lng,
        eta=15,
        expect_confidence="high",
        expect_horizon_substr=None,
        forbid_horizon_substr="중거리",
        model_version=model_version,
    )
    rid16 = check_recommend(
        lat=args.lat,
        lng=args.lng,
        eta=16,
        expect_confidence="medium",
        expect_horizon_substr="도착 시각이 멀어 변동 가능성이 있습니다",
        model_version=model_version,
    )
    rid20 = check_recommend(
        lat=args.lat,
        lng=args.lng,
        eta=20,
        expect_confidence="medium",  # medium 상한
        expect_horizon_substr="도착 시각이 멀어 변동 가능성이 있습니다",
        model_version=model_version,
    )
    rid21 = check_recommend(
        lat=args.lat,
        lng=args.lng,
        eta=21,
        expect_confidence="low",  # low 하한
        expect_horizon_substr="중거리: 예측 점수 반영 비중을 낮춰 해석하세요",
        model_version=model_version,
    )
    # >30 장거리 문구까지 한 번 더 (구간 3개가 모두 다른 문구를 낸다)
    rid45 = check_recommend(
        lat=args.lat,
        lng=args.lng,
        eta=45,
        expect_confidence="low",
        expect_horizon_substr="장거리: 현재 혼잡도 참고 수준으로 보세요",
        model_version=model_version,
    )
    check_log_rows([rid15, rid16, rid20, rid21, rid45], model_version)

    if not args.skip_backfill:
        result = backfill_outcomes(limit=200)
        _ok(f"backfill sample: {result}")
        # match 컬럼 샘플
        conn = get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT outcome_match_status, outcome_match_direction,
                           outcome_match_diff_seconds, COUNT(*) AS n
                    FROM ev_recommend_prediction_log
                    WHERE outcome_match_status IS NOT NULL
                    GROUP BY outcome_match_status, outcome_match_direction,
                             outcome_match_diff_seconds IS NULL
                    ORDER BY n DESC
                    LIMIT 10
                    """
                )
                samples = cur.fetchall()
                if samples:
                    print("[OK] outcome_match sample groups:")
                    for s in samples:
                        print("   ", dict(s))
                else:
                    print("[OK] outcome_match 샘플 없음 (target_at 미도래일 수 있음)")
        finally:
            conn.close()

    print("=== verify_ops PASSED ===")


if __name__ == "__main__":
    main()
