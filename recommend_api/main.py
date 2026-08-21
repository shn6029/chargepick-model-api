from __future__ import annotations

import secrets
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# `py -m recommend_api.main` 및 `uvicorn recommend_api.main:app` 모두 지원
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import (
    FEATURE_COLS,
    MODEL_PATH,
    RECOMMEND_API_KEY,
    VERY_STALE_EXCLUDE_MIN,
)
from recommend_api.code_stamp import code_commit, code_file_count, code_fingerprint
from recommend_api.model_store import get_connection, load_artifact, validate_artifact_features
from recommend_api.response_schema import RecommendResponse
from recommend_api.service import recommend

ChargeMode = Literal["external", "home"]

app = FastAPI(
    title="EV Charger Recommend API",
    version="0.4.0",
    description=(
        "DB + HistGradientBoosting 기반 도착 ETA 충전소 추천 "
        "(후보 제외 → 100점 기본점수 → 접근성 계수)"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    dest_lat: float
    dest_lng: float
    eta_minutes: float = Field(..., ge=1)
    radius_km: float = Field(2.0, ge=0.1)
    top_k: int = Field(10, ge=1, le=50)
    arrival_at: Optional[str] = None
    mode: ChargeMode = "external"
    registered_stat_ids: Optional[list[str]] = None
    origin_lat: Optional[float] = None
    origin_lng: Optional[float] = None
    current_soc: Optional[float] = Field(None, ge=0, le=100)
    vehicle_model_id: Optional[str] = None
    min_output_kw: Optional[float] = Field(None, ge=0)
    include_slow: bool = Field(
        False,
        description="False(기본)=급속만 추천. True면 완속 포함",
    )


def feature_lag() -> dict:
    """ev_charger_features 지연(분) + 커버리지 점검.

    서빙은 최신 status 행에 features 를 INNER JOIN 하고 신선도
    VERY_STALE_EXCLUDE_MIN 이내를 요구하므로, 피처 배치가 이 값을 넘겨
    밀리면 추천 결과가 조용히 0건이 된다. 그 상태를 노출한다.
    (scripts/incremental_features.ps1 을 5분 주기로 등록해야 함)

    lag(최신 시각)만으로는 부분 누락을 못 잡는다. 배치가 최신 행 몇 개만
    처리하고 나머지를 빠뜨려도 lag 는 0으로 보이므로, 신선도 창 안의 관측 행
    중 실제로 조인 가능한 비율(커버리지)을 함께 잰다.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT TIMESTAMPDIFF(MINUTE, MAX(created_at), NOW()) "
                    "FROM ev_charger_features"
                )
                row = cur.fetchone()
                # 신선도 창 안의 관측 행 기준 커버리지
                cur.execute(
                    """
                    SELECT COUNT(*),
                           SUM(CASE WHEN f.log_id IS NOT NULL THEN 1 ELSE 0 END)
                    FROM ev_charger_status s
                    LEFT JOIN ev_charger_features f ON s.log_id = f.log_id
                    WHERE s.created_at >= NOW() - INTERVAL %s MINUTE
                    """,
                    (VERY_STALE_EXCLUDE_MIN,),
                )
                cov = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return {"feature_lag_min": None, "feature_lag_ok": None,
                "feature_lag_error": str(exc)[:200]}

    lag = None if row is None or row[0] is None else int(row[0])
    if lag is None:
        return {"feature_lag_min": None, "feature_lag_ok": False,
                "feature_lag_note": "ev_charger_features 가 비어 있음 → 추천 0건"}

    recent = int(cov[0] or 0)
    joined = int(cov[1] or 0)
    coverage = round(joined / recent, 4) if recent else None

    ok = lag <= VERY_STALE_EXCLUDE_MIN and (coverage is None or coverage >= 0.9)
    out = {
        "feature_lag_min": lag,
        "feature_lag_ok": ok,
        "feature_lag_threshold_min": VERY_STALE_EXCLUDE_MIN,
        "feature_coverage": coverage,
        "feature_coverage_window_min": VERY_STALE_EXCLUDE_MIN,
        "recent_status_rows": recent,
        "recent_rows_with_features": joined,
    }
    if not ok:
        out["feature_lag_note"] = (
            f"피처 지연 {lag}분 / 커버리지 {coverage} "
            f"(임계 {VERY_STALE_EXCLUDE_MIN}분, 0.9) → 추천 결과가 0건이거나 "
            "후보가 크게 줄 수 있음. scripts/incremental_features.ps1 "
            "(5분 주기) 등록/실행 상태를 확인하세요."
        )
    return out


@app.get("/health")
def health():
    model_version = None
    feature_schema_hash = None
    feature_ok = None
    git_commit = None
    if MODEL_PATH.exists():
        try:
            art = load_artifact(strict_features=False)
            model_version = art.get("model_version")
            feature_schema_hash = art.get("feature_schema_hash")
            git_commit = art.get("git_commit")
            msgs = validate_artifact_features(art, strict=False)
            feature_ok = len(msgs) == 0
        except Exception:
            model_version = None
            feature_ok = False
    lag = feature_lag()
    degraded = lag.get("feature_lag_ok") is False
    return {
        "ok": True,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_version": model_version,
        "feature_schema_hash": feature_schema_hash,
        "feature_schema_ok": feature_ok,
        "base_feature_count": len(FEATURE_COLS),
        # git_commit 은 **모델 학습 시점** 커밋이다(아티팩트에 박혀 있다).
        # 배포된 코드와는 무관하니 드리프트 확인에 쓰지 말 것 — 아래 두 개를 쓴다.
        "git_commit": git_commit,
        "code_commit": code_commit(),
        "code_fingerprint": code_fingerprint(),
        "code_file_count": code_file_count(),
        "status": "degraded" if degraded else "prototype_validated",
        "ranking": "exclude_then_100pt_then_access_coef",
        # 키를 안 넣고 배포하면 인증이 없는 상태이므로, 밖에서 확인 가능하게 노출한다.
        "auth_required": bool(RECOMMEND_API_KEY),
        **lag,
    }


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    """RECOMMEND_API_KEY 가 설정돼 있을 때만 검사한다.

    비어 있으면 통과시키는 이유: 키를 넣기 전에 배포해도 기존 호출이 안 깨지게 하려는
    것이다. 대신 운영에서 키를 안 넣으면 인증이 없는 것과 같으므로, /health 응답의
    auth_required 로 현재 상태를 노출한다.

    compare_digest 를 쓰는 건 타이밍 공격 때문이다. `!=` 는 앞에서부터 비교하다
    다른 문자가 나오면 즉시 반환해서, 응답 시간 차이로 키를 한 글자씩 맞출 수 있다.
    """
    if not RECOMMEND_API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, RECOMMEND_API_KEY):
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


@app.post(
    "/api/v1/chargers/recommend",
    dependencies=[Depends(require_api_key)],
    response_model=RecommendResponse,
    # 현재 와이어 포맷 보존: 입력 dict 에 있던 키만 내보낸다. 값이 None 이어도
    # 키가 있었으면 남고(`addr: null`), 없었으면 빠진다(`parking`). 이 구분은
    # 연동가이드가 "null 체크가 아니라 키 존재 여부로 분기"라고 명시한 계약이다.
    # exclude_none 을 쓰면 `addr: null` 까지 사라져 와이어가 바뀐다.
    response_model_exclude_unset=True,
)
def recommend_chargers(body: RecommendRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="모델이 없습니다. scheduler에서 `py -m recommend_api.train` 을 먼저 실행하세요.",
        )
    if body.mode == "home" and not body.registered_stat_ids:
        raise HTTPException(
            status_code=400,
            detail="home 모드에서는 registered_stat_ids가 필요합니다.",
        )
    try:
        art = load_artifact(strict_features=False)
        msgs = validate_artifact_features(art, strict=False)
        if msgs:
            raise HTTPException(
                status_code=503,
                detail="학습 모델과 API 피처 구성이 다릅니다: " + "; ".join(msgs),
            )
        return recommend(
            dest_lat=body.dest_lat,
            dest_lng=body.dest_lng,
            eta_minutes=body.eta_minutes,
            radius_km=body.radius_km,
            top_k=body.top_k,
            arrival_at=body.arrival_at,
            mode=body.mode,
            registered_stat_ids=body.registered_stat_ids,
            origin_lat=body.origin_lat,
            origin_lng=body.origin_lng,
            current_soc=body.current_soc,
            vehicle_model_id=body.vehicle_model_id,
            min_output_kw=body.min_output_kw,
            include_slow=body.include_slow,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recommend_api.main:app", host="0.0.0.0", port=8000, reload=False)
