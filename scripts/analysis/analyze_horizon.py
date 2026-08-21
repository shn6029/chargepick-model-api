"""
H분 뒤(ETA horizon) 사용가능 확률 학습/평가

- 입력: status + info + features + eta_minutes(H)
- 라벨: 현재 시각으로부터 약 H분 뒤 관측에서 stat==2 여부
- 모델: HistGradientBoosting (주력), RandomForest 비교

서비스 가정:
  지도 ETA(분) → eta_minutes 로 넣어 도착 시점 사용가능 확률 예측
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recommend_api.eval_metrics import (
    always_available_baseline,
    evaluate_classifier,
    plot_calibration_curves,
    split_by_date,
)

warnings.filterwarnings("ignore")

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CALIBRATION_PNG = ARTIFACT_DIR / "calibration_curve.png"
METRICS_JSON = ARTIFACT_DIR / "analyze_horizon_metrics.json"

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

# 학습에 사용할 도착 예정 시간(분)
HORIZONS = [5, 10, 15, 20, 30, 45, 60]

# 목표 시각과의 허용 오차(분): 이 안이면 해당 관측을 H분 뒤 상태로 인정
MATCH_TOLERANCE_MIN = 4

FEATURE_COLS = [
    "eta_minutes",
    "arrival_hour",
    "arrival_weekday",
    "hour",
    "minute_slot",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "current_state_duration",
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


def load_joined() -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=40)
    try:
        query = """
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.parking_free, i.limit_yn, i.traffic_yn,
                i.stat_nm, i.addr
            FROM ev_charger_status s
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            WHERE i.del_yn IS NULL OR i.del_yn <> 'Y'
            ORDER BY s.stat_id, s.chger_id, s.created_at, s.log_id
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["output_kw"] = pd.to_numeric(df["output"], errors="coerce").fillna(0)
    df["is_fast"] = (df["output_kw"] >= 50).astype(int)
    df["parking_free_yn"] = (df["parking_free"] == "Y").astype(int)
    df["limit_yn_flag"] = (df["limit_yn"] == "Y").astype(int)
    df["traffic_yn_flag"] = (df["traffic_yn"] == "Y").astype(int)
    return df


def build_horizon_dataset(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    충전기별로 merge_asof 하여 H분 뒤 상태에 가장 가까운 관측을 라벨로 사용.
    """
    parts: list[pd.DataFrame] = []
    grouped = df.groupby(["stat_id", "chger_id"], sort=False)

    for (stat_id, chger_id), g in grouped:
        g = g.sort_values("created_at").reset_index(drop=True)
        if len(g) < 2:
            continue

        future = g[["created_at", "stat"]].rename(
            columns={"created_at": "future_at", "stat": "future_stat"}
        )

        for h in horizons:
            left = g.copy()
            left["eta_minutes"] = h
            left["target_at"] = left["created_at"] + pd.Timedelta(minutes=h)

            merged = pd.merge_asof(
                left.sort_values("target_at"),
                future.sort_values("future_at"),
                left_on="target_at",
                right_on="future_at",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=MATCH_TOLERANCE_MIN),
            )
            merged = merged.dropna(subset=["future_stat"])
            if merged.empty:
                continue

            merged["y"] = (merged["future_stat"] == 2).astype(int)
            merged["arrival_hour"] = merged["target_at"].dt.hour
            merged["arrival_weekday"] = merged["target_at"].dt.weekday
            parts.append(merged)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    return out


def make_xy(horizon_df: pd.DataFrame):
    type_dummies = pd.get_dummies(
        horizon_df["chger_type"].fillna("UNK"), prefix="ctype"
    )
    kind_dummies = pd.get_dummies(horizon_df["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(horizon_df["busi_id"].fillna("UNK"), prefix="busi")

    X = pd.concat(
        [horizon_df[FEATURE_COLS], type_dummies, kind_dummies, busi_dummies],
        axis=1,
    )
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    y = horizon_df["y"].astype(int)
    meta = horizon_df[
        ["stat_id", "chger_id", "created_at", "eta_minutes", "stat_nm", "addr"]
    ].copy()
    return X, y, meta


def time_split(X, y, meta, ratio: float = 0.8):
    order = meta["created_at"].argsort(kind="mergesort")
    X, y, meta = X.iloc[order], y.iloc[order], meta.iloc[order]
    split = int(len(X) * ratio)
    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:],
        meta.iloc[split:],
    )


def eval_model(name, model, X_train, X_test, y_train, y_test, meta_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    overall = evaluate_classifier(y_test, proba)
    overall["model"] = name

    per_h = []
    tmp = meta_test.copy()
    tmp["y"] = y_test.values
    tmp["proba"] = proba
    for h, g in tmp.groupby("eta_minutes"):
        m = evaluate_classifier(g["y"], g["proba"])
        per_h.append(
            {
                "model": name,
                "eta_minutes": int(h),
                "n": len(g),
                "avail_rate": g["y"].mean(),
                "accuracy": m["accuracy"],
                "f1": m["f1_available"],
                "roc_auc": m["roc_auc"] if m["roc_auc"] is not None else np.nan,
                "balanced_accuracy": m["balanced_accuracy"],
                "unavailable_recall": m["unavailable_recall"],
            }
        )

    return overall, pd.DataFrame(per_h), proba


def _flatten_overall(overall: dict) -> dict:
    row = {k: v for k, v in overall.items() if k != "confusion_matrix"}
    cm = overall["confusion_matrix"]
    row.update({f"cm_{k}": v for k, v in cm.items()})
    return row


def demo_eta_predictions(model, feature_columns, horizon_df: pd.DataFrame, X: pd.DataFrame):
    """최신 관측 일부에 대해 ETA별 확률 예시."""
    print("\n" + "=" * 64)
    print("ETA(도착 예정분)별 사용가능 확률 예시")
    print("=" * 64)

    # 충전기 수가 있는 충전소 하나 고르기
    latest_time = horizon_df["created_at"].max()
    recent = horizon_df[horizon_df["created_at"] >= latest_time - pd.Timedelta(minutes=30)]
    if recent.empty:
        recent = horizon_df.tail(5000)

    station_counts = (
        recent.groupby(["stat_id", "stat_nm"])["chger_id"].nunique().sort_values(ascending=False)
    )
    if station_counts.empty:
        print("예시용 충전소 없음")
        return

    stat_id = station_counts.index[0][0]
    stat_nm = station_counts.index[0][1]
    sample = recent[recent["stat_id"] == stat_id].copy()
    # 각 충전기의 가장 최근 행 + 각 horizon
    idx = sample.groupby(["chger_id", "eta_minutes"])["created_at"].idxmax()
    sample = sample.loc[idx]

    # X는 horizon_df와 같은 순서라고 가정할 수 없으므로 다시 피처 생성
    Xs, _, meta = make_xy(sample)
    for col in feature_columns:
        if col not in Xs.columns:
            Xs[col] = 0
    Xs = Xs[feature_columns]
    proba = model.predict_proba(Xs)[:, 1]
    meta = meta.copy()
    meta["available_prob"] = proba

    by_eta = (
        meta.groupby("eta_minutes")
        .agg(
            n_chargers=("chger_id", "nunique"),
            avg_prob=("available_prob", "mean"),
            pred_available=("available_prob", lambda s: int((s >= 0.5).sum())),
        )
        .reset_index()
        .sort_values("eta_minutes")
    )

    print(f"충전소: [{stat_id}] {stat_nm}")
    print(by_eta.to_string(index=False))
    print("(avg_prob = 충전기 평균 사용가능 확률, pred_available = 확률>=0.5 대수)")


def main() -> None:
    print("=" * 64)
    print("H분 뒤 사용가능 학습/평가 (eta_minutes horizon)")
    print(f"horizons={HORIZONS}, match_tolerance=±{MATCH_TOLERANCE_MIN}분")
    print("=" * 64)

    print("JOIN 로드 중...")
    df = load_joined()
    print(
        f"JOIN: {len(df):,}행 | 충전소 {df['stat_id'].nunique():,} | "
        f"기간 {df['created_at'].min()} ~ {df['created_at'].max()}"
    )
    if df.empty:
        return

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS)
    print(
        f"horizon 샘플: {len(hz):,} | "
        f"사용가능 비율 {hz['y'].mean():.1%} | "
        f"ETA 분포:\n{hz['eta_minutes'].value_counts().sort_index().to_string()}"
    )
    if hz.empty:
        print("horizon 샘플이 없습니다. 수집 기간을 더 늘리세요.")
        return

    X, y, meta = make_xy(hz)
    X_train, X_test, y_train, y_test, meta_test = time_split(X, y, meta)
    print(f"학습 {len(X_train):,} / 테스트 {len(X_test):,} (시간순 80/20)")

    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    overall_rows = []
    per_h_frames = []
    curves: dict[str, tuple[pd.Series, np.ndarray]] = {}
    best_model = None
    best_auc = -1.0
    best_name = ""

    for name, model in models.items():
        print(f"\n학습/평가: {name} ...")
        overall, per_h, proba = eval_model(
            name, model, X_train, X_test, y_train, y_test, meta_test
        )
        overall_rows.append(overall)
        per_h_frames.append(per_h)
        curves[name] = (y_test, proba)
        print(
            f"  Accuracy={overall['accuracy']:.3f}  F1={overall['f1_available']:.3f}  "
            f"ROC-AUC={overall['roc_auc']:.3f}  BalancedAcc={overall['balanced_accuracy']:.3f}  "
            f"사용불가Recall={overall['unavailable_recall']:.3f}"
        )
        if overall["roc_auc"] > best_auc:
            best_auc = overall["roc_auc"]
            best_model = model
            best_name = name

    baseline = always_available_baseline(y_test)
    baseline["model"] = "항상 사용 가능(기준모델)"
    overall_rows.append(baseline)
    curves["항상 사용 가능(기준모델)"] = (y_test, np.full(len(y_test), 0.999))

    print("\n" + "=" * 64)
    print("전체 성능 비교 (기준모델 포함)")
    print("=" * 64)
    compare_cols = [
        "model", "accuracy", "f1_available", "roc_auc", "balanced_accuracy",
        "unavailable_recall", "unavailable_precision", "pr_auc_unavailable",
        "brier_score", "log_loss", "positive_rate",
    ]
    print(pd.DataFrame(overall_rows)[compare_cols].round(4).to_string(index=False))

    print("\n" + "=" * 64)
    print(f"ETA별 성능 ({best_name})")
    print("=" * 64)
    best_per_h = pd.concat(per_h_frames, ignore_index=True)
    best_per_h = best_per_h[best_per_h["model"] == best_name].sort_values("eta_minutes")
    print(best_per_h.round(4).to_string(index=False))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration_curves(curves, CALIBRATION_PNG)
    print(f"\ncalibration curve 저장: {CALIBRATION_PNG}")

    metrics_payload = {
        "overall": [_flatten_overall(row) for row in overall_rows],
        "per_horizon_best_model": best_per_h.round(6).to_dict(orient="records"),
        "best_model": best_name,
    }

    # 날짜 단위 분할 검증 (base_timestamp 그룹 단위) — 80/20 row-split과 별도로
    # 날짜가 다른 미래 구간에서도 성능이 유지되는지 확인
    print("\n" + "=" * 64)
    print("날짜 단위 분할 검증 (row 80/20과 별개, 날짜 집합 기준)")
    print("=" * 64)
    is_train_date, train_dates, test_dates = split_by_date(meta)
    if len(test_dates) == 0 or len(train_dates) == 0:
        print("수집 기간이 짧아 날짜 단위 분할을 수행할 수 없습니다(하루 미만 데이터).")
        metrics_payload["date_split"] = None
    else:
        X_dtr, X_dte = X[is_train_date], X[~is_train_date]
        y_dtr, y_dte = y[is_train_date], y[~is_train_date]
        date_model = HistGradientBoostingClassifier(
            max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
        )
        date_model.fit(X_dtr, y_dtr)
        date_proba = date_model.predict_proba(X_dte)[:, 1]
        date_metrics = evaluate_classifier(y_dte, date_proba)
        print(
            f"train 날짜 {len(train_dates)}일 / test 날짜 {len(test_dates)}일 "
            f"(test: {[str(d) for d in test_dates]})"
        )
        print(
            f"  Accuracy={date_metrics['accuracy']:.3f}  F1={date_metrics['f1_available']:.3f}  "
            f"ROC-AUC={date_metrics['roc_auc']}  BalancedAcc={date_metrics['balanced_accuracy']:.3f}  "
            f"사용불가Recall={date_metrics['unavailable_recall']:.3f}"
        )
        metrics_payload["date_split"] = {
            "train_days": len(train_dates),
            "test_days": len(test_dates),
            "test_dates": [str(d) for d in test_dates],
            "metrics": date_metrics,
        }

    METRICS_JSON.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n지표 저장: {METRICS_JSON}")

    # 재학습된 best_model 사용 (이미 fit 상태)
    demo_eta_predictions(best_model, X.columns.tolist(), hz, X)

    print("\n" + "=" * 64)
    print("사용 방법(서비스):")
    print("  1) 경로 API로 ETA(분) 계산")
    print("  2) 현재 충전기 상태/피처 + eta_minutes=ETA 로 모델 예측")
    print("  3) 충전소 단위로 확률 평균/대수 합산 표시")
    print("실행: py analyze_horizon.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
