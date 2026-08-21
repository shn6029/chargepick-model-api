"""충전소 출입 가능 유형 분류 + 접근성 계수."""

from __future__ import annotations

from typing import Any

AccessType = str  # PUBLIC | RESIDENT | RESTRICTED | UNKNOWN

HOUSING_KINDS = {"H0", "H"}

RESIDENT_KEYS = (
    "거주자",
    "입주민",
    "입주자",
    "단지주민",
    "아파트주민",
)

RESTRICTED_KEYS = (
    "외부인",
    "관계자",
    "회원",
    "비공용",
    "비회원",
    "전용충전기",
    "택시",
    "버스",
)

SOFT_KEYS = (
    "수 있음",
    "수있음",
    "수 있습니다",
    "수있습니다",
    "될 수 있음",
)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return str(value).strip()


def is_housing_facility(
    kind: Any = None,
    kind_detail: Any = None,
    stat_nm: Any = None,
    addr: Any = None,
) -> bool:
    k = _norm(kind).upper()
    if k.startswith("H"):
        return True
    blob = " ".join([_norm(kind_detail), _norm(stat_nm), _norm(addr)])
    return any(token in blob for token in ("아파트", "공동주택", "오피스텔", "주상복합"))


def classify_access(
    *,
    limit_yn: Any = None,
    limit_detail: Any = None,
    kind: Any = None,
    kind_detail: Any = None,
    stat_nm: Any = None,
    addr: Any = None,
) -> dict[str, Any]:
    """access_type + UI용 배지/경고 문구."""
    detail = _norm(limit_detail)
    limit = _norm(limit_yn).upper() or "N"
    housing = is_housing_facility(kind, kind_detail, stat_nm, addr)

    resident_hit = any(k in detail for k in RESIDENT_KEYS)
    restricted_hit = any(k in detail for k in RESTRICTED_KEYS)
    soft = any(k in detail for k in SOFT_KEYS) or ("제한될 수" in detail)

    access: AccessType
    if resident_hit:
        access = "RESIDENT"
    elif restricted_hit and not soft:
        access = "RESTRICTED"
    elif limit == "Y" and detail and not soft:
        if any(x in detail for x in ("불가", "전용", "제한", "출입")):
            access = "RESTRICTED" if not resident_hit else "RESIDENT"
        else:
            access = "UNKNOWN"
    elif limit == "Y" or soft:
        access = "UNKNOWN"
    elif housing and limit == "N":
        access = "PUBLIC"
    elif housing:
        access = "UNKNOWN"
    else:
        access = "PUBLIC"

    warning = None
    if access == "UNKNOWN" and housing:
        warning = "공동주택 내 충전소 · 외부 차량은 출입이 제한될 수 있습니다."
    elif access == "UNKNOWN":
        warning = "출입·이용 제한이 있을 수 있습니다."
    elif access == "RESIDENT":
        warning = "입주민·거주자 전용일 수 있습니다."
    elif access == "RESTRICTED":
        warning = "관계자·회원 등 이용이 제한될 수 있습니다."

    return {
        "access_type": access,
        "is_housing": housing,
        "access_warning": warning,
    }


def access_coefficient(
    access_type: str,
    is_housing: bool = False,
    *,
    registered: bool = False,
) -> float | None:
    """최종점수 배수. None이면 하드 제외 대상.

    PUBLIC ×1.00 / PUBLIC+housing ×0.90 / UNKNOWN ×0.60 /
    RESIDENT·RESTRICTED 비등록 → None / 등록 아파트 ×1.00
    """
    if registered:
        return 1.0
    if access_type in ("RESIDENT", "RESTRICTED"):
        return None
    if access_type == "UNKNOWN":
        return 0.60
    if access_type == "PUBLIC" and is_housing:
        return 0.90
    if access_type == "PUBLIC":
        return 1.0
    return 0.60


def parking_occupancy_coefficient(occupancy_rate: float | None) -> float:
    """주차 혼잡도 → 추천점수 접근성 추가 배수 (확률 모델 피처 아님).

    None(정보없음) → 1.0
    <50% → 1.0 / 50~70 → 0.9 / 70~90 → 0.75 / ≥90 → 0.5
    """
    if occupancy_rate is None:
        return 1.0
    try:
        rate = float(occupancy_rate)
    except (TypeError, ValueError):
        return 1.0
    if rate != rate:  # NaN
        return 1.0
    if rate < 50:
        return 1.0
    if rate < 70:
        return 0.90
    if rate < 90:
        return 0.75
    return 0.50
