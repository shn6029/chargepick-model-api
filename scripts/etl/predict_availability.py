"""
저장된 사용가능 확률 모델로 충전소 추천/예측

예시:
  py scripts/etl/predict_availability.py --eta 15 --lat 35.84217 --lng 128.68043 --radius-km 2 --top 10
  py scripts/etl/predict_availability.py --eta 10 --stat-id ME19X405
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pymysql

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]

load_dotenv()

# DB 접속정보는 환경변수에서 읽는다(저장소 루트 `.env`).
# 폴백값을 두지 않는다 — 예전에는 접속정보가 여기 박혀 있어서 `.env` 없이 돌리면
# 의도치 않게 운영 DB 에 조용히 붙었다. 지금은 비어 있으면 연결에서 실패한다.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

MODEL_PATH = _ROOT / "recommend_api" / "artifacts" / "horizon_hgb.joblib"

BASE_FEATURE_COLS = [
    "eta_minutes",
    "arrival_hour",
    "arrival_weekday",
    "hour",
    "minute_slot",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "capped_state_duration",
    "is_long_state_duration",
    "is_stale_status",
    "capped_status_update_age_min",
    "changes_30m",
    "avail_ratio_15m",
    "avail_ratio_30m",
    "avail_ratio_60m",
    "time_since_available",
    "time_since_charge_started",
    "time_since_charge_ended",
    "stat",
    "output_kw",
    "is_fast",
    "parking_free_yn",
    "limit_yn_flag",
    "traffic_yn_flag",
]


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_artifact(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            "먼저 실행: py -m recommend_api.train"
        )
    return joblib.load(path)


def _artifact_medians(artifact: dict) -> pd.Series:
    """API(medians dict) / 구 CLI(median Series) 호환."""
    if "medians" in artifact:
        return pd.Series(artifact["medians"])
    if "median" in artifact:
        med = artifact["median"]
        return med if isinstance(med, pd.Series) else pd.Series(med)
    raise KeyError("artifact에 medians/median 이 없습니다")


def _rank_threshold(artifact: dict) -> float:
    thr = artifact.get("threshold", 0.5)
    nested = artifact.get("thresholds")
    if isinstance(nested, dict) and "rank_threshold" in nested:
        return float(nested["rank_threshold"])
    return float(thr)



def load_latest_chargers() -> pd.DataFrame:
    """충전기별 최신 status+features+info."""
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=40)
    try:
        sql = """
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.parking_free, i.limit_yn, i.traffic_yn,
                i.stat_nm, i.addr, i.lat, i.lng,
                s.stat_upd_dt
            FROM ev_charger_status s
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            INNER JOIN (
                SELECT stat_id, chger_id, MAX(created_at) AS max_created
                FROM ev_charger_status
                GROUP BY stat_id, chger_id
            ) latest
              ON s.stat_id = latest.stat_id
             AND s.chger_id = latest.chger_id
             AND s.created_at = latest.max_created
            WHERE (i.del_yn IS NULL OR i.del_yn <> 'Y')
              AND i.lat IS NOT NULL AND i.lng IS NOT NULL
        """
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["stat_upd_dt"] = pd.to_datetime(df["stat_upd_dt"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["output_kw"] = pd.to_numeric(df["output"], errors="coerce").fillna(0)
    df["is_fast"] = (df["output_kw"] >= 50).astype(int)
    df["parking_free_yn"] = (df["parking_free"] == "Y").astype(int)
    df["limit_yn_flag"] = (df["limit_yn"] == "Y").astype(int)
    df["traffic_yn_flag"] = (df["traffic_yn"] == "Y").astype(int)
    dur = pd.to_numeric(df["current_state_duration"], errors="coerce").fillna(0)
    df["current_state_duration"] = dur
    df["is_long_state_duration"] = (dur >= 360).astype(int)
    df["capped_state_duration"] = dur.clip(lower=0, upper=360)
    age = (df["created_at"] - df["stat_upd_dt"]).dt.total_seconds() / 60.0
    age = age.fillna(0).clip(lower=0)
    df["status_update_age_min"] = age
    df["capped_status_update_age_min"] = age.clip(upper=360)
    df["is_stale_status"] = (age >= 360).astype(int)
    # 동일 시각 중복 시 마지막 log
    df = df.sort_values("log_id").drop_duplicates(["stat_id", "chger_id"], keep="last")
    return df


def nearest_horizon(eta_minutes: float, horizons: list[int]) -> int:
    return int(min(horizons, key=lambda h: abs(h - eta_minutes)))


def build_features_for_infer(df: pd.DataFrame, eta_minutes: float) -> pd.DataFrame:
    work = df.copy()
    now = pd.Timestamp.now()
    # created_at 기준 도착 시각 근사
    arrival = work["created_at"] + pd.Timedelta(minutes=float(eta_minutes))
    # 데이터가 오래됐으면 현재+ETA 사용
    arrival = arrival.where(
        (now - work["created_at"]) <= pd.Timedelta(hours=6),
        now + pd.Timedelta(minutes=float(eta_minutes)),
    )
    work["eta_minutes"] = float(eta_minutes)
    work["arrival_hour"] = arrival.dt.hour
    work["arrival_weekday"] = arrival.dt.weekday

    type_dummies = pd.get_dummies(work["chger_type"].fillna("UNK"), prefix="ctype")
    kind_dummies = pd.get_dummies(work["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(work["busi_id"].fillna("UNK"), prefix="busi")
    X = pd.concat([work[BASE_FEATURE_COLS], type_dummies, kind_dummies, busi_dummies], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def align_features(X: pd.DataFrame, feature_columns: list[str], median: pd.Series) -> pd.DataFrame:
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]
    return X.fillna(median)


def predict_chargers(artifact, df: pd.DataFrame, eta_minutes: float) -> pd.DataFrame:
    horizons = artifact.get("horizons", [5, 10, 15, 20, 30, 45, 60])
    h = nearest_horizon(eta_minutes, horizons)
    X = build_features_for_infer(df, h)
    X = align_features(X, artifact["feature_columns"], _artifact_medians(artifact))
    proba = artifact["model"].predict_proba(X)[:, 1]
    out = df[
        [
            "stat_id",
            "chger_id",
            "stat_nm",
            "addr",
            "lat",
            "lng",
            "stat",
            "chger_type",
            "output_kw",
            "parking_free",
        ]
    ].copy()
    out["eta_minutes_used"] = h
    out["available_prob"] = proba
    out["pred_available"] = (proba >= _rank_threshold(artifact)).astype(int)
    return out


def aggregate_stations(
    charger_pred: pd.DataFrame,
    dest_lat: float | None = None,
    dest_lng: float | None = None,
    radius_km: float | None = None,
) -> pd.DataFrame:
    work = charger_pred.copy()
    if dest_lat is not None and dest_lng is not None:
        work["distance_m"] = [
            haversine_m(dest_lat, dest_lng, la, ln)
            for la, ln in zip(work["lat"], work["lng"])
        ]
        if radius_km is not None:
            work = work[work["distance_m"] <= radius_km * 1000]
    else:
        work["distance_m"] = np.nan

    if work.empty:
        return work

    rows = []
    r_m = (radius_km or 2.0) * 1000
    for stat_id, g in work.groupby("stat_id"):
        total = len(g)
        pred_available = int(g["pred_available"].sum())
        avg_prob = float(g["available_prob"].mean())
        rate = pred_available / total if total else 0.0
        dist = float(g["distance_m"].min()) if g["distance_m"].notna().any() else None
        if dist is not None and not math.isnan(dist):
            distance_score = 1.0 - min(dist, r_m) / r_m
            score = 0.7 * rate + 0.3 * distance_score
        else:
            score = rate
        row0 = g.iloc[0]
        rows.append(
            {
                "stat_id": stat_id,
                "stat_nm": row0["stat_nm"],
                "addr": row0["addr"],
                "lat": float(row0["lat"]),
                "lng": float(row0["lng"]),
                "distance_m": None if dist is None or math.isnan(dist) else round(dist, 1),
                "score": round(score, 4),
                "avg_available_prob": round(avg_prob, 4),
                "pred_available": pred_available,
                "total_chargers": total,
                "availability_rate": round(rate, 4),
                "has_fast": bool((g["output_kw"] >= 50).any()),
                "parking_free": bool((g["parking_free"] == "Y").any()),
                "eta_minutes_used": int(g["eta_minutes_used"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def to_api_response(stations: pd.DataFrame, eta_minutes: float, dest_lat, dest_lng) -> dict:
    recs = []
    for _, r in stations.iterrows():
        recs.append(
            {
                "rank": int(r["rank"]),
                "stat_id": r["stat_id"],
                "stat_nm": r["stat_nm"],
                "addr": r["addr"],
                "lat": r["lat"],
                "lng": r["lng"],
                "distance_m": r["distance_m"],
                "score": r["score"],
                "avg_available_prob": r["avg_available_prob"],
                "pred_available": int(r["pred_available"]),
                "total_chargers": int(r["total_chargers"]),
                "availability_rate": r["availability_rate"],
                "badges": {
                    "has_fast": bool(r["has_fast"]),
                    "parking_free": bool(r["parking_free"]),
                },
            }
        )
    return {
        "meta": {
            "dest_lat": dest_lat,
            "dest_lng": dest_lng,
            "eta_minutes": eta_minutes,
            "model": "HistGradientBoosting",
            "horizon_note": f"약 {int(eta_minutes)}분 뒤 도착 기준 예상",
        },
        "recommendations": recs,
    }


def main():
    parser = argparse.ArgumentParser(description="충전소 사용가능 확률 예측")
    parser.add_argument("--eta", type=float, required=True, help="도착까지 분(ETA)")
    parser.add_argument("--lat", type=float, default=None, help="도착 위도")
    parser.add_argument("--lng", type=float, default=None, help="도착 경도")
    parser.add_argument("--radius-km", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--stat-id", type=str, default=None, help="특정 충전소만")
    parser.add_argument("--json", action="store_true", help="추천 API 응답 형식 JSON 출력")
    args = parser.parse_args()

    artifact = load_artifact()
    print("최신 충전기 상태 로드 중...")
    latest = load_latest_chargers()
    print(f"최신 관측 충전기: {len(latest):,}")

    if args.stat_id:
        latest = latest[latest["stat_id"] == args.stat_id]
        if latest.empty:
            raise SystemExit(f"충전소 없음: {args.stat_id}")

    charger_pred = predict_chargers(artifact, latest, args.eta)
    stations = aggregate_stations(
        charger_pred,
        dest_lat=args.lat,
        dest_lng=args.lng,
        radius_km=args.radius_km if args.lat is not None else None,
    )
    stations = stations.head(args.top)

    if args.json:
        payload = to_api_response(stations, args.eta, args.lat, args.lng)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(
        f"\nETA={args.eta}분 (모델 사용 H={stations['eta_minutes_used'].iloc[0] if len(stations) else '-'})"
    )
    if stations.empty:
        print("조건에 맞는 충전소가 없습니다.")
        return
    cols = [
        "rank",
        "stat_nm",
        "distance_m",
        "score",
        "avg_available_prob",
        "pred_available",
        "total_chargers",
    ]
    print(stations[cols].to_string(index=False))


if __name__ == "__main__":
    main()
