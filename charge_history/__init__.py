# -*- coding: utf-8 -*-
"""충전이력 월별 확장 검증 (세션 수요 ≠ ETA 가용확률 모델)."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEDULER_ROOT = PACKAGE_ROOT.parent
CACHE_DIR = PACKAGE_ROOT / "cache"
ARTIFACTS_DIR = SCHEDULER_ROOT / "recommend_api" / "artifacts"
DEFAULT_DOWNLOADS = Path(r"f:\Users\상현\Downloads")
DEFAULT_DATA_RAW = SCHEDULER_ROOT / "data" / "raw"

# 월 키 YYYY-MM → 파일 상대 이름 (Downloads / data/raw 기준)
MONTH_FILES: dict[str, str] = {
    "2025-01": "충전기 활용 현황 상세정보 (2025.01).xlsx",
    "2025-02": "충전기 활용 현황 상세정보 (2025.02).xlsx",
    "2025-03": "충전기 활용 현황 상세정보 (2025.03).xlsx",
    "2025-04": "충전기 활용 현황 상세정보 (2025.04).xlsx",
    "2025-05": "충전기 활용 현황 상세정보 (2025.05).xlsx",
    "2025-06": "충전기 활용 현황 상세정보 (2025.06).xlsx",
    "2025-07": "충전기 활용 현황 상세정보 (2025.07).xlsx",
    "2025-08": "충전기 활용 현황 상세정보 (2025.08).xlsx",
    "2025-09": "충전기 활용 현황 상세정보 (2025.09).xlsx",
    "2025-10": "20251104 2025년10월 환경부 충전기 충전이력(상세내역).xlsx",
    "2025-11": "20251202 2025년11월 환경부 충전기 충전이력(상세내역).xlsx",
    "2025-12": "20260105 2025년12월 환경부 충전기 충전이력(상세내역).xlsx",
    "2026-01": "20260203 2026년1월 환경부 충전기 충전이력(상세내역).xlsx",
    "2026-02": "20260306 2026년2월 환경부 충전기 충전이력(상세내역).xlsx",
    "2026-03": "20260402 2026년3월 기후부 충전기 충전이력(상세내역).xlsx",
    "2026-04": "20260504 2026년4월 기후부 충전기 충전이력(상세내역).xlsx",
    "2026-05": "20260601 2026년5월 기후부 충전기 충전이력(상세내역).xlsx",
    "2026-06": "20260727 2026년6월 기후부 충전기 충전이력(상세내역).xlsx",
}

MONTHS = list(MONTH_FILES.keys())

SESSION_COLS = [
    "station_name",
    "charger_unit_id",
    "facility_type_major",
    "facility_type_minor",
    "region",
    "district",
    "address",
    "connector_type",
    "power_label",
    "power_kw",
    "speed_class",
    "start_at",
    "end_at",
    "duration_min",
    "duration_text",
    "kwh",
    "start_date",
    "start_hour",
    "start_weekday",
    "year_month",
]


def scope_prefix(scope: str) -> str:
    """캐시/산출 파일명 접두어. daegu | nationwide"""
    s = (scope or "daegu").strip().lower()
    if s in ("nationwide", "national", "all", "korea"):
        return "nationwide"
    return "daegu"


def scope_label(scope: str) -> str:
    return "Nationwide" if scope_prefix(scope) == "nationwide" else "Daegu only"
