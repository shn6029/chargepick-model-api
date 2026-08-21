"""
세 테이블 JOIN 기반 모델 비교 평가
- ev_charger_status  : 상태 시계열
- ev_charger_info    : 충전소/충전기 마스터
- ev_charger_features: 파생 피처

추천 모델 (표형식 + 사용가능 분류):
1) HistGradientBoosting  - 표 데이터에서 보통 가장 강함
2) RandomForest          - 안정적인 앙상블 베이스라인
3) LogisticRegression    - 해석 쉬운 선형 베이스라인
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import warnings

import numpy as np
import pandas as pd
import pymysql
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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

FEATURE_COLS = [
    # features 테이블
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
    # status
    "stat",
    # info 파생
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


def prepare_xy(df: pd.DataFrame):
    work = df.copy()
    work["charger_key"] = work["stat_id"].astype(str) + "_" + work["chger_id"].astype(str)
    work["next_stat"] = work.groupby("charger_key")["stat"].shift(-1)
    work = work.dropna(subset=["next_stat"])
    work["y"] = (work["next_stat"] == 2).astype(int)

    type_dummies = pd.get_dummies(work["chger_type"].fillna("UNK"), prefix="ctype")
    kind_dummies = pd.get_dummies(work["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(work["busi_id"].fillna("UNK"), prefix="busi")

    X = pd.concat([work[FEATURE_COLS], type_dummies, kind_dummies, busi_dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    # 트리/부스팅은 NaN 허용, 로지스틱용 중앙값 대체는 파이프라인에서 처리
    meta = work[["stat_id", "chger_id", "created_at", "stat_nm"]].copy()
    return X, work["y"], meta


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


def build_models():
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        ),
    }


def fillna_median(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test, meta_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-proba))
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    # 충전소×시점 잔여 대수 MAE
    tmp = meta_test.copy()
    tmp["y"] = y_test.values
    tmp["pred"] = pred
    grouped = (
        tmp.groupby(["stat_id", "created_at"], as_index=False)
        .agg(actual=("y", "sum"), predicted=("pred", "sum"), n=("y", "size"))
    )
    grouped = grouped[grouped["n"] >= 2]
    station_mae = (
        mean_absolute_error(grouped["actual"], grouped["predicted"])
        if not grouped.empty
        else np.nan
    )

    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
        "station_count_mae": station_mae,
        "proba": proba,
        "pred": pred,
    }


def main() -> None:
    print("=" * 64)
    print("모델 추천 (3테이블 JOIN / 다음 시점 사용가능 분류)")
    print("=" * 64)
    print("1) HistGradientBoosting : 표 데이터·결측 혼합에 강함 (1순위)")
    print("2) RandomForest         : 해석·안정성 좋은 베이스라인")
    print("3) LogisticRegression   : 단순·해석용 베이스라인")
    print()

    print("데이터 로드(JOIN) 중...")
    df = load_joined()
    print(
        f"JOIN: {len(df):,}행 | 충전소 {df['stat_id'].nunique():,} | "
        f"스냅샷 {df['created_at'].nunique()}"
    )
    if df.empty:
        print("JOIN 결과 없음")
        return

    X, y, meta = prepare_xy(df)
    X_train, X_test, y_train, y_test, meta_test = time_split(X, y, meta)
    print(
        f"학습 {len(X_train):,} / 테스트 {len(X_test):,} | "
        f"사용가능 비율 {y.mean():.1%} (시간순 80/20)"
    )

    # 로지스틱용 결측 채움 복사본
    Xtr_f, Xte_f = fillna_median(X_train, X_test)

    results = []
    models = build_models()
    for name, model in models.items():
        print(f"\n학습/평가: {name} ...")
        if name == "LogisticRegression":
            res = evaluate_classifier(
                name, model, Xtr_f, Xte_f, y_train, y_test, meta_test
            )
        else:
            # RF는 NaN 비허용 → median fill, HGB는 NaN 허용이지만 동일 처리로 공정 비교
            res = evaluate_classifier(
                name, model, Xtr_f, Xte_f, y_train, y_test, meta_test
            )
        results.append(res)
        print(
            f"  Accuracy={res['accuracy']:.3f}  F1={res['f1']:.3f}  "
            f"ROC-AUC={res['roc_auc']:.3f}  충전소대수MAE={res['station_count_mae']:.3f}"
        )

    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "accuracy": round(r["accuracy"], 4),
                "f1": round(r["f1"], 4),
                "roc_auc": round(r["roc_auc"], 4),
                "station_count_mae": round(r["station_count_mae"], 4),
            }
            for r in results
        ]
    ).sort_values("roc_auc", ascending=False)

    print("\n" + "=" * 64)
    print("비교 결과 (ROC-AUC 내림차순)")
    print("=" * 64)
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print("\n추천 채택:")
    print(
        f"  → {best['model']} "
        f"(ROC-AUC={best['roc_auc']}, 충전소 잔여대수 MAE={best['station_count_mae']})"
    )
    print("목표: 충전소 선택 시 다음 시점 사용가능 대수/확률 예측")
    print("=" * 64)


if __name__ == "__main__":
    main()
