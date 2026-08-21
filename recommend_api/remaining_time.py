"""충전중(stat=3) 잔여시간 예측 초안.

가정:
  동일/유사 출력(kW)이면 공공·민간 모두 비슷한 세션 길이를 가진다.
  → 충전소 ID 매칭 없이 power_kw(+시간·기상)로 잔여시간을 추정한다.

모델 출처(옵션):
  EVCharger-model-test/models/daegu_remaining_time_hgb.joblib
  없으면 용량별 평균 세션길이 lookup으로 폴백.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REMAINING_MODEL_PATH, WEATHER_DEFAULTS

# 학습 데이터에 자주 나온 용량 라벨에 가깝게 스냅 (민간도 같은 kW면 동일 버킷)
_CAPACITY_BUCKETS: list[tuple[float, str]] = [
    (50.0, "급속(50kW)"),
    (100.0, "급속(100kW단독)"),
    (200.0, "급속(200kW동시)"),
    (350.0, "초급속(350kW단독)"),
    (400.0, "급속(400kW동시)"),
]

# 리포트 Train 평균 충전시간(분) — 모델 없을 때 폴백
_AVG_DURATION_BY_KW: dict[float, float] = {
    50.0: 30.27,
    100.0: 28.30,
    200.0: 27.45,
    350.0: 22.15,
    400.0: 25.75,
}

NUMERIC_FEATURES = [
    "elapsed_min",
    "power_kw",
    "hour",
    "dow",
    "is_weekend",
    "ta",
    "rn",
    "hm",
    "ws",
    "dsnw",
    "dc10Tca",
]
CATEGORICAL_FEATURES = [
    "capacity_raw",
    "charger_type",
    "speed_class",
    "capacity_mode",
    "facility_major",
    "district",
]


def snap_power_kw(power_kw: float) -> float:
    """급속(≥50kW)만 학습 버킷에 스냅. 완속은 원값 유지."""
    if not np.isfinite(power_kw) or power_kw <= 0:
        return 100.0
    if power_kw < 50:
        return float(power_kw)
    keys = list(_AVG_DURATION_BY_KW.keys())
    return float(min(keys, key=lambda k: abs(k - power_kw)))


def capacity_label(power_kw: float) -> str:
    if power_kw < 50:
        return f"완속({int(round(power_kw))}kW단독)"
    snapped = snap_power_kw(power_kw)
    for kw, label in _CAPACITY_BUCKETS:
        if abs(kw - snapped) < 1e-6:
            return label
    return f"급속({int(snapped)}kW단독)"


def district_from_addr(addr: Any) -> str:
    if addr is None or (isinstance(addr, float) and np.isnan(addr)):
        return "unknown"
    text = str(addr)
    m = re.search(r"(달성군|군위군|[가-힣]+구)", text)
    return m.group(1) if m else "unknown"


def _elapsed_min(row: pd.Series) -> float:
    # scheduler features: 분 단위
    v = row.get("time_since_charge_started")
    if pd.notna(v):
        return max(float(v), 0.0)
    dur = row.get("current_state_duration")
    if pd.notna(dur):
        return max(float(dur), 0.0)
    return 0.0


@lru_cache(maxsize=1)
def _load_remaining_artifact() -> dict[str, Any] | None:
    path = Path(REMAINING_MODEL_PATH)
    if not path.exists():
        return None
    import joblib

    blob = joblib.load(path)
    if "pipeline" not in blob:
        return None
    return blob


def model_available() -> bool:
    return _load_remaining_artifact() is not None


def build_remaining_feature_frame(
    charging: pd.DataFrame,
    now: pd.Timestamp | None = None,
    weather: dict[str, float] | None = None,
) -> pd.DataFrame:
    """민간/공공 구분 없이 power_kw 중심으로 피처 구성."""
    now = now or pd.Timestamp.now()
    weather = {**WEATHER_DEFAULTS, **(weather or {})}

    out = charging.copy()
    out["elapsed_min"] = out.apply(_elapsed_min, axis=1)
    out["power_kw"] = pd.to_numeric(out.get("output_kw"), errors="coerce").fillna(100.0)
    out["hour"] = int(now.hour)
    out["dow"] = int(now.dayofweek)
    out["is_weekend"] = int(now.dayofweek >= 5)
    for k, v in weather.items():
        out[k] = float(v)

    out["capacity_raw"] = out["power_kw"].map(capacity_label)
    out["speed_class"] = np.where(out["power_kw"] >= 300, "초급속", np.where(out["power_kw"] >= 50, "급속", "완속"))
    out["capacity_mode"] = "unknown"  # 민간은 단독/동시 구분 어려움 → unknown
    out["charger_type"] = out.get("chger_type", "unknown").astype(str).fillna("unknown")
    out["facility_major"] = "unknown"
    out["district"] = out.get("addr", pd.Series(index=out.index)).map(district_from_addr)
    return out


def _lookup_remaining(power_kw: float, elapsed_min: float) -> float:
    if power_kw < 50:
        avg = 180.0  # 완속은 이력 표본이 거의 없어 보수적 기본값
    else:
        snapped = snap_power_kw(power_kw)
        avg = _AVG_DURATION_BY_KW.get(snapped, 28.0)
    return max(avg - max(elapsed_min, 0.0), 0.0)


def predict_remaining_minutes(charging: pd.DataFrame) -> pd.Series:
    """충전중 행에 대해 잔여 분 예측. index 유지."""
    if charging.empty:
        return pd.Series(dtype=float)

    feat = build_remaining_feature_frame(charging)
    artifact = _load_remaining_artifact()

    if artifact is None:
        return feat.apply(
            lambda r: _lookup_remaining(float(r["power_kw"]), float(r["elapsed_min"])),
            axis=1,
        )

    pipe = artifact["pipeline"]
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).fillna("unknown")
    for col in NUMERIC_FEATURES:
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)

    pred = np.clip(pipe.predict(feat[cols]), 0, None)
    return pd.Series(pred, index=charging.index, name="pred_remaining_min")


def free_by_eta_score(remaining_min: float, eta_minutes: float, soft_minutes: float = 10.0) -> float:
    """ETA 전에 빌 가능성 점수 [0,1]. remaining << eta 이면 1에 가깝다."""
    eta = max(float(eta_minutes), 1.0)
    rem = max(float(remaining_min), 0.0)
    # rem <= eta - soft → 1.0 / rem >= eta + soft → 0.0
    return float(np.clip((eta + soft_minutes - rem) / (2.0 * soft_minutes), 0.0, 1.0))
