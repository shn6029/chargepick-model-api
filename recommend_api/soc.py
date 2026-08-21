"""도착 SOC(State of Charge) 추정."""

from __future__ import annotations

from typing import Any

VEHICLE_CATALOG: dict[str, dict[str, Any]] = {
    "ioniq5": {
        "id": "ioniq5",
        "name": "아이오닉 5",
        "battery_kwh": 72.6,
        "efficiency_km_per_kwh": 5.1,
        "min_output_kw": 50,
    },
    "ev6": {
        "id": "ev6",
        "name": "EV6",
        "battery_kwh": 77.4,
        "efficiency_km_per_kwh": 4.8,
        "min_output_kw": 50,
    },
    "kona_ev": {
        "id": "kona_ev",
        "name": "코나 일렉트릭",
        "battery_kwh": 64.8,
        "efficiency_km_per_kwh": 5.5,
        "min_output_kw": 50,
    },
    "niro_ev": {
        "id": "niro_ev",
        "name": "니로 EV",
        "battery_kwh": 64.8,
        "efficiency_km_per_kwh": 5.3,
        "min_output_kw": 50,
    },
    "gv60": {
        "id": "gv60",
        "name": "GV60",
        "battery_kwh": 77.4,
        "efficiency_km_per_kwh": 4.5,
        "min_output_kw": 50,
    },
    "model3": {
        "id": "model3",
        "name": "Model 3",
        "battery_kwh": 60.0,
        "efficiency_km_per_kwh": 6.0,
        "min_output_kw": 50,
    },
    "model_y": {
        "id": "model_y",
        "name": "Model Y",
        "battery_kwh": 75.0,
        "efficiency_km_per_kwh": 5.5,
        "min_output_kw": 50,
    },
    "generic": {
        "id": "generic",
        "name": "일반 EV (기본)",
        "battery_kwh": 64.0,
        "efficiency_km_per_kwh": 5.0,
        "min_output_kw": None,
    },
}

DEFAULT_VEHICLE_ID = "generic"

RISK_DANGER_SOC = 10.0
RISK_CAUTION_SOC = 20.0


def list_vehicles() -> list[dict[str, Any]]:
    return [dict(v) for v in VEHICLE_CATALOG.values()]


def get_vehicle(model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    spec = VEHICLE_CATALOG.get(str(model_id))
    return dict(spec) if spec else None


def classify_discharge_risk(arrival_soc: float) -> str:
    if arrival_soc <= 0:
        return "unreachable"
    if arrival_soc <= RISK_DANGER_SOC:
        return "danger"
    if arrival_soc <= RISK_CAUTION_SOC:
        return "caution"
    return "safe"


def estimate_arrival_soc(
    current_soc: float,
    distance_km: float,
    *,
    efficiency_km_per_kwh: float,
    battery_kwh: float,
) -> dict[str, Any]:
    current = float(max(0.0, min(100.0, current_soc)))
    dist = max(0.0, float(distance_km))
    eff = max(1e-6, float(efficiency_km_per_kwh))
    cap = max(1e-6, float(battery_kwh))

    energy_kwh = dist / eff
    soc_drop = (energy_kwh / cap) * 100.0
    arrival_soc = round(current - soc_drop, 1)
    risk = classify_discharge_risk(arrival_soc)

    return {
        "arrival_soc_pct": arrival_soc,
        "energy_used_kwh": round(energy_kwh, 2),
        "soc_drop_pct": round(soc_drop, 1),
        "travel_distance_km": round(dist, 2),
        "discharge_risk": risk,
        "reachable": risk != "unreachable",
    }


def compute_station_soc(
    *,
    current_soc: float | None,
    model_id: str | None,
    travel_distance_m: float,
) -> dict[str, Any] | None:
    if current_soc is None:
        return None
    vehicle = get_vehicle(model_id) or get_vehicle(DEFAULT_VEHICLE_ID)
    if vehicle is None:
        return None

    result = estimate_arrival_soc(
        float(current_soc),
        float(travel_distance_m) / 1000.0,
        efficiency_km_per_kwh=float(vehicle["efficiency_km_per_kwh"]),
        battery_kwh=float(vehicle["battery_kwh"]),
    )
    result["vehicle_id"] = vehicle["id"]
    result["vehicle_name"] = vehicle["name"]
    return result


def resolve_min_output_kw(
    min_output_kw: float | None,
    vehicle_model_id: str | None,
) -> float | None:
    if min_output_kw is not None:
        return float(min_output_kw)
    vehicle = get_vehicle(vehicle_model_id)
    if vehicle and vehicle.get("min_output_kw") is not None:
        return float(vehicle["min_output_kw"])
    return None
