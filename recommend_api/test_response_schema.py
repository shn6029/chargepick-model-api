"""응답 스키마 회귀 테스트 — 모델이 구현보다 뒤처지면 시끄럽게 실패한다 (DB/모델 불필요).

## 이 테스트가 막으려는 사고

`api_contract/` 가 2026-08-03 이후 갱신되지 않은 채 `service.py` 만 08-18 까지
움직여 필드 10개가 계약에서 빠졌다. 아무도 몰랐다 — **계약 파일은 아무것도
실행하지 않기 때문이다.** 팀 백엔드는 결국 계약이 아니라 실물 응답을 보고
스키마를 맞췄다.

`response_schema.py` 로 옮긴 뒤에도 같은 일이 날 수 있다. `response_model` 은
붙였지만 `extra="allow"` 라 모델에 없는 키도 그냥 통과하므로, 모델이 뒤처져도
**응답은 멀쩡하다**(그게 조용한 소실을 막는 의도된 설계다). 그러면 뒤처짐을
알아챌 방법이 없다. 그 역할이 이 테스트다.

## 왜 소스를 AST 로 읽는가

응답 전체를 만들려면 DB 와 모델 아티팩트가 필요하다. 손으로 쓴 고정 픽스처를
두면 그 픽스처가 `service.py` 와 같이 낡는다 — 지금 고치고 있는 문제를 한 겹
아래에 다시 만드는 셈이다.

그래서 **구현 소스의 dict 리터럴 키를 직접 파싱**한다. 누군가
`service.py` 의 station dict 에 `"foo": ...` 를 추가하면, 이 테스트가 그 자리에서
"모델에 foo 가 없다"고 실패한다. 응답을 한 번도 만들지 않고도 잡힌다.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pytest
from pydantic import BaseModel

from recommend_api.response_schema import (
    ChargerBadges,
    ChargerPrediction,
    RecommendMeta,
    RecommendResponse,
    ScoreBreakdown,
    StationParking,
    StationRecommendation,
    response_key_shape,
)

_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 소스에서 dict 리터럴 키 뽑기
# --------------------------------------------------------------------------


def _parse(filename: str) -> ast.Module:
    return ast.parse((_HERE / filename).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"함수를 찾지 못했다: {name} — 이름이 바뀌었으면 테스트도 고칠 것")


def _literal_keys(node: ast.AST) -> set[str]:
    """dict 리터럴의 문자열 키. `**other` 전개는 무시한다(키를 정적으로 모른다)."""
    if not isinstance(node, ast.Dict):
        return set()
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _dict_assigned_to(fn: ast.FunctionDef, var: str) -> ast.Dict | None:
    """`var = {...}` 또는 `var: T = {...}` 의 dict 리터럴."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var:
                    return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Dict)
            and isinstance(node.target, ast.Name)
            and node.target.id == var
        ):
            return node.value
    return None


def _subscript_keys(fn: ast.FunctionDef, var: str) -> set[str]:
    """`var["key"] = ...` 형태로 나중에 붙는 조건부 키."""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == var
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def _nested_dict(container: ast.Dict | None, key: str) -> ast.Dict | None:
    """dict 리터럴 안에서 `key` 의 값이 다시 dict 리터럴이면 그것."""
    if container is None:
        return None
    for k, v in zip(container.keys, container.values):
        if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Dict):
            return v
    return None


def _nested_subscript_keys(fn: ast.FunctionDef, var: str, first: str) -> set[str]:
    """`var["first"]["key"] = ...` 형태 (meta 의 request_id)."""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)):
                continue
            inner = target.value
            if (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == var
                and isinstance(inner.slice, ast.Constant)
                and inner.slice.value == first
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def implementation_keys() -> dict[str, set[str]]:
    """`service.py` · `scoring.py` 가 실제로 내보내는 객체별 키 집합."""
    service = _parse("service.py")
    scoring = _parse("scoring.py")

    build_chargers = _function(service, "_build_charger_items")
    rank_stations = _function(service, "_rank_stations_from_frame")
    recommend_fn = _function(service, "recommend")
    score_fn = _function(scoring, "score_station")

    station_dict = _dict_assigned_to(rank_stations, "station")
    payload_dict = _dict_assigned_to(recommend_fn, "payload")

    charger = _literal_keys(_dict_assigned_to(build_chargers, "item")) | _subscript_keys(
        build_chargers, "item"
    )
    station = _literal_keys(station_dict) | _subscript_keys(rank_stations, "station")
    # rank 는 payload 조립 직전에 `{"rank": i, **station}` 으로 덧붙는다
    station |= {"rank"}
    meta = _literal_keys(_nested_dict(payload_dict, "meta")) | _nested_subscript_keys(
        recommend_fn, "payload", "meta"
    )

    return {
        "response": _literal_keys(payload_dict),
        "meta": meta,
        "station": station,
        "score_breakdown": _literal_keys(_dict_assigned_to(score_fn, "breakdown")),
        "badges": _literal_keys(_nested_dict(station_dict, "badges")),
        "charger": charger,
        "parking": _literal_keys(_nested_dict(station_dict, "parking"))
        | _literal_keys(
            next(
                (
                    node.value
                    for node in ast.walk(rank_stations)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Dict)
                    and any(
                        isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "parking"
                        for t in node.targets
                    )
                ),
                ast.Dict(keys=[], values=[]),
            )
        ),
    }


def _model_fields(model: type[BaseModel]) -> set[str]:
    return set(model.model_fields)


# --------------------------------------------------------------------------
# 1) 파서 자체가 살아 있는지 — 이게 죽으면 아래 테스트가 전부 공허하게 통과한다
# --------------------------------------------------------------------------

_MIN_KEYS = {
    "response": 2,
    "meta": 15,
    "station": 20,
    "score_breakdown": 10,
    "badges": 4,
    "charger": 15,
    "parking": 8,
}


def test_ast_parser_still_finds_the_dicts():
    """구조가 바뀌어 파싱이 빈 집합을 돌려주면 여기서 먼저 실패한다.

    이 가드가 없으면 함수 이름이나 변수명이 바뀌었을 때 `implementation_keys()`
    가 조용히 빈 집합을 내고, "모델에 누락 없음" 이 자동으로 참이 되어 나머지
    테스트가 통째로 무력화된다.
    """
    keys = implementation_keys()
    for name, minimum in _MIN_KEYS.items():
        assert len(keys[name]) >= minimum, (
            f"{name} 키를 {len(keys[name])}개밖에 못 찾았다(최소 {minimum}). "
            "service.py/scoring.py 구조가 바뀌었으면 이 파일의 AST 헬퍼도 고칠 것."
        )


# --------------------------------------------------------------------------
# 2) 본 검사 — 구현이 내보내는 키가 전부 모델에 있는가
# --------------------------------------------------------------------------

_TARGETS: list[tuple[str, type[BaseModel]]] = [
    ("response", RecommendResponse),
    ("meta", RecommendMeta),
    ("station", StationRecommendation),
    ("score_breakdown", ScoreBreakdown),
    ("badges", ChargerBadges),
    ("charger", ChargerPrediction),
    ("parking", StationParking),
]


@pytest.mark.parametrize("shape_name,model", _TARGETS, ids=[t[0] for t in _TARGETS])
def test_model_covers_every_implementation_key(shape_name: str, model: type[BaseModel]):
    """구현에 있고 모델에 없는 키 = 스펙 뒤처짐.

    `extra="allow"` 라 응답 자체는 멀쩡하다. 하지만 `/openapi.json` 이 그 필드를
    문서화하지 못하므로, 소비자는 또다시 실물 응답을 리버스 엔지니어링해야 한다.
    그 상태를 여기서 끊는다.
    """
    missing = implementation_keys()[shape_name] - _model_fields(model)
    assert not missing, (
        f"{model.__name__} 에 없는 구현 키: {sorted(missing)}\n"
        f"→ recommend_api/response_schema.py 의 {model.__name__} 에 필드를 추가할 것. "
        "설명에는 '무엇'이 아니라 '왜/언제 나오는지'를 적는다."
    )


@pytest.mark.parametrize("shape_name,model", _TARGETS, ids=[t[0] for t in _TARGETS])
def test_model_has_no_phantom_fields(shape_name: str, model: type[BaseModel]):
    """모델에 있고 구현에 없는 키 = 유령 필드.

    `arrival_soc_pct` 가 정확히 이 사고였다. 계약에는 "도착 예상 SOC" 로 남아
    있는데 구현은 SOC 점수·제외를 없애면서 내보내지 않게 됐고, 팀 백엔드·프론트
    양쪽 스키마에 죽은 필드가 남았다. 소비자는 '언젠가 오는 값'으로 오해한다.

    구현에서 조건부로만 나가는 키(`parking`, `stale_note` …)는 AST 가 잡으므로
    여기서 걸리지 않는다. 걸린다면 정말로 아무 데서도 안 만드는 필드다.
    """
    phantom = _model_fields(model) - implementation_keys()[shape_name]
    assert not phantom, (
        f"{model.__name__} 의 유령 필드(구현이 만들지 않음): {sorted(phantom)}\n"
        "→ 구현에서 제거된 필드라면 모델에서도 지울 것. 소비자에게 죽은 필드를 광고하게 된다."
    )


# --------------------------------------------------------------------------
# 3) 와이어 포맷 보존 — exclude_unset 이 키 존재 여부를 지키는가
# --------------------------------------------------------------------------


def _sample_payload() -> dict:
    """조건부 키가 '있는 쪽'과 '없는 쪽'을 한 응답에 섞은 표본.

    - 1번 충전소: parking 있음 / 충전중 충전기(잔여시간 3종 + 지연 note) +
      장시간 동일상태 충전기(long_state_note) / arrival_soc_pct 있음
    - 2번 충전소: parking 없음 / 대기 충전기(조건부 키 전부 없음, addr 은 null)

    `arrival_soc_pct` 는 현재 런타임에서는 나가지 않지만(내부 값이 항상 None),
    `station["arrival_soc_pct"] = ...` 대입 자체는 코드에 남아 있다. 모델 필드를
    훑기 위해 표본에는 넣는다 — 이 표본은 '실제 캡처'가 아니라 '모델 훑기'용이다.
    """
    return {
        "meta": {
            "dest_lat": 35.84217,
            "dest_lng": 128.68043,
            "eta_minutes": 15.0,
            "arrival_at": None,
            "model": "HistGradientBoosting",
            "model_version": "20260818T022438Z",
            "confidence_level": "high",
            "include_slow": False,
            "min_output_kw": None,
            "remaining_model": True,
            "rank_threshold": 0.5,
            "shadow_warn_threshold": 0.27,
            "horizon_note": "도착 15분 기준입니다.",
            "mode": "external",
            "radius_km": 2.0,
            "radius_expanded": False,
            "radius_note": None,
            "request_id": "0e4c1b2a-0000-4000-8000-000000000000",
        },
        "recommendations": [
            {
                "rank": 1,
                "stat_id": "ME184067",
                "stat_nm": "대구시청",
                "addr": "대구광역시 중구",
                "lat": 35.87,
                "lng": 128.60,
                "distance_m": 412.5,
                "score": 0.831,
                "recommendation_score": 83.1,
                "recommendation_label": "매우 추천",
                "access_coefficient": 1.0,
                "score_breakdown": {
                    "availability": 40.2,
                    "route_time": 15.0,
                    "route_distance": 5.0,
                    "battery": 0.0,
                    "charger_count": 12.0,
                    "speed_fit": 9.0,
                    "freshness": 5.0,
                    "base_total": 86.2,
                    "access_coefficient": 1.0,
                    "parking_occupancy_coefficient": 1.0,
                    "parking_occupancy_rate": 80.43,
                },
                "avg_available_prob": 0.804,
                "pred_available": 1,
                "total_chargers": 2,
                "availability_rate": 0.5,
                "detour_minutes": 0.8,
                "extra_distance_km": 0.412,
                "arrival_soc_pct": 55.3,
                "access_type": "PUBLIC",
                "is_housing": False,
                "access_warning": None,
                "badges": {
                    "has_fast": True,
                    "parking_free": False,
                    "has_stale_charger": True,
                    "has_long_state_charger": True,
                },
                "chargers": [
                    {
                        "chger_id": "01",
                        "chger_type": "04",
                        "output_kw": 100.0,
                        "current_stat": 3,
                        "available_prob": 0.71,
                        "hgb_available_prob": 0.71,
                        "current_state_duration": 22,
                        "status_update_age_min": 4.2,
                        "is_long_state_duration": False,
                        "is_stale_status": True,
                        "is_invalid_status_update_time": False,
                        "shadow_unavailable_warn": False,
                        "stale_note": "상태 갱신이 오래된 충전기입니다",
                        "pred_remaining_min": 18.4,
                        "free_by_eta_score": 0.62,
                        "remaining_source": "ml",
                    },
                    {
                        "chger_id": "02",
                        "chger_type": "04",
                        "output_kw": 100.0,
                        "current_stat": 2,
                        "available_prob": 0.898,
                        "hgb_available_prob": 0.898,
                        "current_state_duration": 512,
                        "status_update_age_min": 2.1,
                        "is_long_state_duration": True,
                        "is_stale_status": False,
                        "is_invalid_status_update_time": False,
                        "shadow_unavailable_warn": False,
                        "long_state_note": "동일한 상태가 장시간 유지되고 있습니다",
                    },
                ],
                "parking": {
                    "pklt_id": "155-3-000006",
                    "parking_nm": "수성구청 주차장",
                    "total_spaces": 142,
                    "remaining_spaces": 27,
                    "occupancy_rate": 80.43,
                    "congestion_status": "혼잡(점유 90%미만)",
                    "fee_type": "미상",
                    "is_24h": False,
                },
            },
            {
                "rank": 2,
                "stat_id": "CV003571",
                "stat_nm": "대구시_대구시청 지상1층",
                "addr": None,
                "lat": 35.871,
                "lng": 128.602,
                "distance_m": 980.0,
                "score": 0.612,
                "recommendation_score": 61.2,
                "recommendation_label": "조건부 추천",
                "access_coefficient": 0.9,
                "score_breakdown": {
                    "availability": 25.0,
                    "route_time": 12.0,
                    "route_distance": 4.0,
                    "battery": 0.0,
                    "charger_count": 9.0,
                    "speed_fit": 9.0,
                    "freshness": 5.0,
                    "base_total": 68.0,
                    "access_coefficient": 0.9,
                    "parking_occupancy_coefficient": 1.0,
                    "parking_occupancy_rate": None,
                },
                "avg_available_prob": 0.5,
                "pred_available": 1,
                "total_chargers": 2,
                "availability_rate": 0.5,
                "detour_minutes": 1.96,
                "extra_distance_km": 0.98,
                "access_type": "PUBLIC",
                "is_housing": False,
                "access_warning": None,
                "badges": {
                    "has_fast": True,
                    "parking_free": True,
                    "has_stale_charger": False,
                    "has_long_state_charger": False,
                },
                "chargers": [
                    {
                        "chger_id": "01",
                        "chger_type": None,
                        "output_kw": None,
                        "current_stat": 2,
                        "available_prob": 0.5,
                        "hgb_available_prob": 0.5,
                        "current_state_duration": None,
                        "status_update_age_min": None,
                        "is_long_state_duration": False,
                        "is_stale_status": False,
                        "is_invalid_status_update_time": False,
                        "shadow_unavailable_warn": False,
                    }
                ],
            },
        ],
    }


def _dump(payload: dict) -> dict:
    """엔드포인트와 같은 설정으로 직렬화 (`response_model_exclude_unset=True`)."""
    return RecommendResponse.model_validate(payload).model_dump(exclude_unset=True)


def test_exclude_unset_preserves_key_presence_exactly():
    """검증을 통과시켜도 키가 늘거나 줄지 않아야 한다.

    `response_model` 도입의 유일한 실패 모드가 "조용한 키 손실"이라 여기가 본진이다.
    `parking` 이 사라지면 CLAUDE.md 가 경고한 '주차 정보만 조용히 사라짐'이 재현된다.
    """
    payload = _sample_payload()
    assert response_key_shape(_dump(payload)) == response_key_shape(payload)


def test_absent_optional_keys_do_not_reappear_as_null():
    """`exclude_unset` 이라 없던 키가 `null` 로 되살아나면 안 된다.

    되살아나면 "키 존재 여부로 분기하라"는 `parking` 계약이 깨진다 —
    모든 충전소에 `parking: null` 이 붙어 조건 분기가 항상 참이 된다.
    """
    dumped = _dump(_sample_payload())
    second = dumped["recommendations"][1]

    assert "parking" not in second
    assert "arrival_soc_pct" not in second
    assert "stale_note" not in second["chargers"][0]
    assert "pred_remaining_min" not in second["chargers"][0]


def test_explicit_nulls_survive():
    """반대 방향 — 값이 `None` 이어도 키가 있었으면 남아야 한다.

    `exclude_none` 을 잘못 쓰면 여기가 깨진다. `addr: null` 이 사라지면
    소비자 입장에서 '주소가 없는 충전소'와 '필드가 빠진 응답'을 구별할 수 없다.
    """
    dumped = _dump(_sample_payload())

    assert dumped["recommendations"][1]["addr"] is None
    assert dumped["meta"]["arrival_at"] is None
    assert dumped["meta"]["radius_note"] is None
    assert dumped["recommendations"][1]["chargers"][0]["output_kw"] is None


def test_unmodeled_key_passes_through_instead_of_vanishing():
    """모델에 없는 키가 응답에서 사라지지 않는다 (`extra="allow"` 의 목적).

    스펙 뒤처짐은 위의 `test_model_covers_every_implementation_key` 가 잡는다.
    여기서 지키는 건 **그때도 사용자 데이터는 안 잃는다**는 쪽이다.
    """
    payload = _sample_payload()
    payload["recommendations"][0]["brand_new_field"] = "미래에 추가될 값"
    payload["meta"]["brand_new_meta"] = 42

    dumped = _dump(payload)

    assert dumped["recommendations"][0]["brand_new_field"] == "미래에 추가될 값"
    assert dumped["meta"]["brand_new_meta"] == 42


def test_empty_recommendations_is_not_an_error_shape():
    """후보가 전부 하드 제외되면 빈 배열이 온다. meta 는 그대로 있어야 한다."""
    payload = _sample_payload()
    payload["recommendations"] = []
    payload["meta"]["request_id"] = "0e4c1b2a-0000-4000-8000-000000000000"

    dumped = _dump(payload)

    assert dumped["recommendations"] == []
    assert dumped["meta"]["request_id"] == "0e4c1b2a-0000-4000-8000-000000000000"


def test_sample_payload_exercises_every_modeled_key():
    """표본이 모델 필드를 전부 훑는지 — 안 훑으면 위 보존 테스트에 사각이 생긴다."""
    shape = response_key_shape(_sample_payload())
    gaps: dict[str, Iterable[str]] = {}
    for name, model in _TARGETS:
        if name == "response":
            continue
        missing = _model_fields(model) - shape[name]
        if missing:
            gaps[model.__name__] = sorted(missing)

    assert not gaps, f"_sample_payload() 가 다루지 않는 모델 필드: {gaps}"
