"""추천 응답 스키마 — `/openapi.json` 이 응답까지 스스로 말하게 하는 모델.

## 왜 만들었나 (2026-08-19)

`api_contract/` 폴더가 2026-08-03 `8090f93` "first commit" 이후 **한 번도 갱신되지
않았다.** 그 사이 `service.py` · `scoring.py` 는 08-18 까지 움직였고, 감사 결과
필드 10개 누락 + 의미가 뒤집힌 서술 3건 + `confidence_level` enum 오류가 나왔다.

| 대조 | 결과 |
|---|---|
| 팀 저장소(`ev-daegue_eta_api/web`)의 `api_contract` 참조 | **0건** — 쓴 적이 없다 |
| 팀 백엔드 `RecommendParking` 의 8개 키 | 계약이 아니라 **실물 응답**을 보고 맞췄다 |
| 라이브 `/openapi.json` 요청 스키마 | 자동 생성 · `include_slow` 포함 (항상 정확) |
| `api_contract/openapi.yaml` 요청 스키마 | 손으로 베낀 사본 · `include_slow` 없음 |
| `api_contract/openapi.yaml` `confidence_level` | `enum: [high, low]` — 서버는 `medium` 도 낸다 |

요청 스키마는 FastAPI 가 `RecommendRequest` 에서 이미 정확하게 뽑아낸다. 손으로
쓴 사본만 틀렸다. 못 뽑던 건 **응답뿐이었다** — 엔드포인트가 `response_model`
없이 raw dict 을 반환해서 `/openapi.json` 의 200 응답이 `"schema": {}` 였다.

그래서 사본을 다시 쓰는 대신 **모델이 곧 응답**이 되게 한다. 이러면 표류가
구조적으로 불가능하다. 응답을 바꾸려면 이 파일을 지나야 하기 때문이다.

## 왜 `extra="allow"` 인가 — 조용한 필드 소실을 막는다

`response_model` 을 붙이면 FastAPI 가 응답을 모델로 필터링한다. 모델에 없는 키는
**예외 없이 잘려나간다.** 즉 `service.py` 에 필드를 추가하고 여기를 깜빡하면,
에러 없이 응답에서만 사라진다 — 이 저장소가 반복해서 당한 "조용히 틀림" 패턴
그대로다(`parking.py` 콜레이션 1267, `prediction_log` 권한 1142 참고).

`extra="allow"` 면 모델에 없는 키도 그대로 통과하므로 **소실이 원천 봉쇄된다.**
대신 스펙이 그만큼 뒤처지는데, 그건 `test_response_schema.py` 의 키 집합 회귀
테스트가 **시끄럽게** 잡는다. 조용한 데이터 손실을 요란한 테스트 실패로 바꾼 것이다.

## 왜 `exclude_unset` 인가 — 현재 와이어 포맷을 그대로 보존한다

지금 응답은 "키가 아예 없는 것"과 "키는 있고 값이 `null` 인 것"을 **의도적으로**
섞어 쓴다.

- 키 자체가 없음: `parking`(주차 매칭 있을 때만), `arrival_soc_pct`,
  `long_state_note` · `stale_note`, `pred_remaining_min` 3종(충전중일 때만)
- 키는 있고 `null`: `addr`, `chger_type`, `output_kw`, `current_state_duration`,
  `status_update_age_min`, `arrival_at`, `radius_note` …

연동가이드가 `parking` 에 대해 "`null` 체크가 아니라 **키 존재 여부**로 분기하라"고
명시했으므로 이 구분은 계약의 일부다. `response_model_exclude_none=True` 를 쓰면
`addr: null` 같은 것까지 전부 키가 사라져 와이어가 바뀐다.

`exclude_unset=True` 는 dict 검증 시 **입력에 있던 키만** set 으로 표시되는
Pydantic v2 동작을 그대로 쓴다. 값이 `None` 이어도 키가 있었으면 남고, 없었으면
빠진다. 현재 포맷이 키 단위로 보존된다.

## 타입을 Literal 로 좁히지 않은 이유

`confidence_level` · `recommendation_label` · `mode` · `remaining_source` 는 값
집합이 정해져 있지만 `str` 로 뒀다. Literal 로 좁히면 값이 하나 늘어나는 순간
응답 검증이 실패해 **추천 엔드포인트가 500 으로 죽는다.** 서빙 경로에서 문서
정확도를 위해 가용성을 거는 건 손해다. 값 목록은 각 필드 description 에 적었다.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Passthrough(BaseModel):
    """모델에 없는 키도 응답에 그대로 통과시킨다. 상단 독스트링 참고."""

    model_config = ConfigDict(extra="allow")


class ScoreBreakdown(_Passthrough):
    """기본점수 100점의 항목별 내역.

    배분은 2026-08-18 기준 가용확률 50 + 경로 20(시간 15 + 거리 5) +
    충전기수 15 + 속도 10 + 최신성 5 = 100 이다. 7월 배분(45/20/10/10/10/5)에서
    바뀌었다. 옛 표를 싣고 있던 `api_contract/README.md` 는 폐기했다.
    """

    availability: Optional[float] = Field(None, description="가용확률 점수. 최대 50")
    route_time: Optional[float] = Field(None, description="우회 시간 점수. 최대 15")
    route_distance: Optional[float] = Field(None, description="추가 거리 점수. 최대 5")
    battery: Optional[float] = Field(
        None,
        description=(
            "**항상 0.0.** 배점에서 제거했고 계약 호환용으로 키만 남겼다. "
            "항목별 그래프를 그린다면 이 항목은 빼야 한다."
        ),
    )
    charger_count: Optional[float] = Field(None, description="호환 충전기 수 점수. 최대 15")
    speed_fit: Optional[float] = Field(None, description="속도 적합 점수. 최대 10")
    freshness: Optional[float] = Field(None, description="상태 최신성 점수. 최대 5")
    base_total: Optional[float] = Field(None, description="접근성 계수 적용 전 합계. 0~100")
    access_coefficient: Optional[float] = Field(None, description="접근성 배수")
    parking_occupancy_coefficient: Optional[float] = Field(
        None,
        description=(
            "**항상 1.0 이고 점수에 곱해지지 않는다.** 주차 점유율과 충전기 가용률의 "
            "시간대 제거 후 잔차 상관이 -0.073(R² 0.54%)이라 2026-08-04 에 순위 "
            "반영을 껐다. 표시 호환용 필드다."
        ),
    )
    parking_occupancy_rate: Optional[float] = Field(
        None, description="주차 점유율 원자료(%). 매핑 없으면 null"
    )


class ChargerBadges(_Passthrough):
    """충전소 단위 요약 플래그. 충전기 목록을 안 쓰더라도 이것만으로 경고를 띄울 수 있다."""

    has_fast: Optional[bool] = Field(None, description="급속 충전기 보유")
    parking_free: Optional[bool] = Field(None, description="주차 무료 충전기 보유")
    has_stale_charger: Optional[bool] = Field(
        None,
        description=(
            "상태 갱신이 오래된(≥360분) 충전기가 섞여 있다. "
            "수집 지연을 사용자에게 알릴 수 있는 유일한 신호다."
        ),
    )
    has_long_state_charger: Optional[bool] = Field(
        None, description="동일 상태를 장시간(≥360분) 유지 중인 충전기가 있다(지연과는 다름)"
    )


class ChargerPrediction(_Passthrough):
    """충전기 한 대의 도착시점 예측."""

    chger_id: Optional[str] = None
    chger_type: Optional[str] = None
    output_kw: Optional[float] = None
    current_stat: Optional[int] = Field(
        None, description="1 통신이상 · 2 대기 · 3 충전중 · 4 운영중지 · 5 점검 · 9 미확인"
    )
    available_prob: Optional[float] = Field(
        None,
        description=(
            "도착(ETA) 시점 사용가능 확률 0~1. 현재 상태가 대기든 충전중이든 의미는 같다."
        ),
    )
    hgb_available_prob: Optional[float] = Field(
        None, description="블렌드 전 HGB 단독 확률. REMAINING_BLEND_WEIGHT=0 이라 현재 동일"
    )
    current_state_duration: Optional[int] = Field(None, description="현재 상태 유지 시간(분)")
    status_update_age_min: Optional[float] = Field(
        None, description="상태 갱신 경과(분). created_at - stat_upd_dt"
    )
    is_long_state_duration: Optional[bool] = None
    is_stale_status: Optional[bool] = None
    is_invalid_status_update_time: Optional[bool] = Field(
        None, description="stat_upd_dt 가 미래이거나 파싱 불가"
    )
    shadow_unavailable_warn: Optional[bool] = Field(
        None, description="급속인데 확률이 ETA별 경고 임계 미만(섀도 경고)"
    )
    # --- 아래는 조건부. 해당 없으면 키 자체가 없다 ---
    long_state_note: Optional[str] = Field(None, description="장시간 동일 상태일 때만")
    stale_note: Optional[str] = Field(None, description="상태 갱신 지연일 때만")
    pred_remaining_min: Optional[float] = Field(None, description="충전중일 때만. 예상 잔여(분)")
    free_by_eta_score: Optional[float] = Field(
        None,
        description=(
            "충전중일 때만. **점수에는 반영되지 않는다**(REMAINING_BLEND_WEIGHT=0, "
            "2026-08-12). ETA30 에서 거의 전부 1.0 으로 포화하니 표시에 주의."
        ),
    )
    remaining_source: Optional[str] = Field(None, description="ml | power_lookup | none")


class StationParking(_Passthrough):
    """주차장 정보. **매칭된 충전소에만 붙는 선택 필드다.**

    전체 충전소 대비 매핑률이 낮고 지역 편중이 심해 top-10 에서 보통 0~3건에만
    붙는다. 없으면 키 자체가 없으므로 `null` 체크가 아니라 **키 존재 여부**로
    분기해야 한다. "정보 없음"으로 표시하면 화면 대부분이 그 문구로 찬다.

    값어치는 만차 경고다 — 충전기가 비어 있어도 주차장이 만차면 진입을 못 한다.
    모델이 예측하지 못하는 축이라(잔차 R² 0.54%) UI 안내로만 막을 수 있다.
    """

    pklt_id: Optional[str] = None
    parking_nm: Optional[str] = None
    total_spaces: Optional[int] = None
    remaining_spaces: Optional[int] = None
    occupancy_rate: Optional[float] = Field(
        None, description="점유율(%). 원본이 비면 1 - remaining/total 로 채운다"
    )
    congestion_status: Optional[str] = Field(
        None, description="원본 그대로의 한글 문구. 값 집합이 보장되지 않고 null 가능"
    )
    fee_type: Optional[str] = Field(None, description='"유료" / "무료" / "유료+무료" / "미상"')
    is_24h: Optional[bool] = None


class StationRecommendation(_Passthrough):
    """추천된 충전소 한 곳."""

    rank: Optional[int] = None
    stat_id: Optional[str] = None
    stat_nm: Optional[str] = None
    addr: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    distance_m: Optional[float] = Field(None, description="목적지로부터의 직선거리(m)")
    score: Optional[float] = Field(None, description="호환용. recommendation_score / 100")
    recommendation_score: Optional[float] = Field(None, description="0~100 최종 추천점수")
    recommendation_label: Optional[str] = Field(
        None,
        description=(
            "매우 추천(≥80) / 추천(65~79) / 조건부 추천(50~64) / "
            "주의·대안 부족(35~49) / 추천 어려움(<35)"
        ),
    )
    access_coefficient: Optional[float] = Field(
        None, description="PUBLIC 1.0 · 아파트외부가능 0.9 · UNKNOWN 0.6 · 등록아파트 1.0"
    )
    score_breakdown: Optional[ScoreBreakdown] = None
    avg_available_prob: Optional[float] = Field(
        None, description="충전소 평균 모델 확률. 추천점수와는 별개 축이다"
    )
    pred_available: Optional[int] = Field(
        None, description="도착시점 사용가능으로 예측된 충전기 수(확률 ≥ rank_threshold)"
    )
    total_chargers: Optional[int] = Field(None, description="호환 충전기 수")
    availability_rate: Optional[float] = Field(None, description="pred_available / total_chargers")
    detour_minutes: Optional[float] = Field(
        None,
        description=(
            "**origin_lat/origin_lng 를 보내지 않으면 우회가 아니다.** 그 경우 "
            "목적지↔충전소 직선거리를 30km/h로 환산한 값이라 distance_m 의 재표현이다. "
            "실제 우회((출발→충전소 + 충전소→목적지) − 출발→목적지)를 받으려면 origin 을 보낼 것."
        ),
    )
    extra_distance_km: Optional[float] = Field(
        None,
        description=(
            "**origin 없이 호출하면 `distance_m / 1000` 과 완전히 같은 값이다.** "
            "'추가 거리'로 표시하면 거짓이 된다. detour_minutes 설명 참고."
        ),
    )
    access_type: Optional[str] = Field(None, description="PUBLIC / RESIDENT / UNKNOWN 등")
    is_housing: Optional[bool] = None
    access_warning: Optional[str] = None
    badges: Optional[ChargerBadges] = None
    chargers: Optional[list[ChargerPrediction]] = None
    # --- 아래는 조건부. 해당 없으면 키 자체가 없다 ---
    arrival_soc_pct: Optional[float] = Field(
        None,
        description=(
            "**현재 구현에서는 나가지 않는다.** 도착 예상 SOC 를 추천점수·후보 제외에 "
            "쓰지 않기로 하면서 내부적으로 항상 None 이고, 응답 조립 때 키가 빠진다. "
            "소비자 스키마에 남겨두면 '언젠가 오는 값'으로 오해된다."
        ),
    )
    parking: Optional[StationParking] = None


class RecommendMeta(_Passthrough):
    """응답 메타. 요청 에코 + 모델 신원 + 이번 호출에 쓰인 임계값."""

    dest_lat: Optional[float] = None
    dest_lng: Optional[float] = None
    eta_minutes: Optional[float] = None
    arrival_at: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = Field(
        None, description="어느 모델이 답했는지. 로그에 남기면 나중에 성능 비교에 쓸 수 있다"
    )
    confidence_level: Optional[str] = Field(
        None, description="high(ETA ≤15분) / medium(16~20) / low(≥21)"
    )
    include_slow: Optional[bool] = Field(None, description="false 면 급속만 추천")
    min_output_kw: Optional[float] = Field(None, description="실제 적용된 최소 출력 하드 필터")
    remaining_model: Optional[bool] = Field(
        None,
        description=(
            "잔여시간 모델 로드 여부. **2026-08-12 부터 점수에는 영향이 없다** — "
            "false 면 pred_remaining_min · free_by_eta_score 표시값만 폴백으로 바뀐다."
        ),
    )
    rank_threshold: Optional[float] = Field(
        None, description="pred_available 을 세는 확률 임계. ETA 별로 다르다"
    )
    shadow_warn_threshold: Optional[float] = Field(
        None, description="shadow_unavailable_warn 판정 임계. ETA 별로 다르다"
    )
    horizon_note: Optional[str] = Field(None, description="UI 안내 문구")
    mode: Optional[str] = Field(None, description="external | home")
    radius_km: Optional[float] = Field(None, description="실제 사용한 반경(확장 반영)")
    radius_expanded: Optional[bool] = Field(
        None, description="최고점 <50 이라 반경을 넓혔는지(+1km, 최대 +2, 상한 5km)"
    )
    radius_note: Optional[str] = None
    request_id: Optional[str] = Field(
        None, description="예측 로그와 대조 가능한 UUID. 로깅이 꺼져 있으면 없을 수 있다"
    )


class RecommendResponse(_Passthrough):
    """`POST /api/v1/chargers/recommend` 200 응답.

    `recommendations` 가 빈 배열일 수 있다 — 반경 내 후보가 전부 하드 제외
    (호환 충전기 0 · 상태 지연 · 접근 제한)된 상황이다. **에러가 아니므로**
    UI 에서 "조건에 맞는 충전소 없음"으로 처리해야 한다.
    """

    meta: Optional[RecommendMeta] = None
    recommendations: list[StationRecommendation] = Field(default_factory=list)


def response_key_shape(payload: dict[str, Any]) -> dict[str, set[str]]:
    """응답 dict 에서 객체별 키 집합을 뽑는다 — 회귀 테스트·스키마 대조용.

    `service.recommend()` 의 실제 산출물과 이 모듈의 모델을 견주는 데 쓴다.
    조건부 키(`parking` 등)는 표본에 따라 없을 수 있으므로 **합집합**으로 모은다.
    """
    shape: dict[str, set[str]] = {
        "response": set(payload),
        "meta": set(payload.get("meta") or {}),
        "station": set(),
        "score_breakdown": set(),
        "badges": set(),
        "charger": set(),
        "parking": set(),
    }
    for station in payload.get("recommendations") or []:
        shape["station"] |= set(station)
        shape["score_breakdown"] |= set(station.get("score_breakdown") or {})
        shape["badges"] |= set(station.get("badges") or {})
        shape["parking"] |= set(station.get("parking") or {})
        for charger in station.get("chargers") or []:
            shape["charger"] |= set(charger)
    return shape


#: `response_key_shape` 의 키 → 그 객체를 기술하는 모델
SHAPE_MODELS: dict[str, type[BaseModel]] = {
    "response": RecommendResponse,
    "meta": RecommendMeta,
    "station": StationRecommendation,
    "score_breakdown": ScoreBreakdown,
    "badges": ChargerBadges,
    "charger": ChargerPrediction,
    "parking": StationParking,
}
