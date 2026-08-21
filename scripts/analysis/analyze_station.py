"""
충전소 정보(ev_charger_info) + 상태 로그(ev_charger_status) JOIN 분석
1) 충전기별 다음 스냅샷 사용가능 확률 (정보 피처 포함)
2) 충전소 선택 → 예상 잔여(사용가능) 대수
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import warnings

import numpy as np
import pandas as pd
import pymysql
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

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

AVAIL_THRESHOLD = 0.5

BASE_FEATURE_COLS = [
    "hour",
    "minute",
    "weekday",
    "stat",
    "prev_stat",
    "is_charging",
    "is_available",
    "is_error",
    "output_kw",
    "is_fast",
    "parking_free_yn",
    "limit_yn_flag",
    "traffic_yn_flag",
    "mins_since_last_end",
    "has_now_tsdt",
    "avail_roll3",
]


def load_joined() -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        query = """
            SELECT
                s.stat_id, s.chger_id, s.busi_id, s.stat,
                s.last_tsdt, s.last_tedt, s.now_tsdt, s.created_at,
                i.stat_nm, i.chger_type, i.addr, i.lat, i.lng,
                i.output, i.method, i.zscode, i.kind, i.kind_detail,
                i.parking_free, i.limit_yn, i.traffic_yn, i.use_time
            FROM ev_charger_status s
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            WHERE i.del_yn IS NULL OR i.del_yn <> 'Y'
            ORDER BY s.created_at, s.stat_id, s.chger_id
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    for col in ("created_at", "last_tsdt", "last_tedt", "now_tsdt"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["output_kw"] = pd.to_numeric(df["output"], errors="coerce").fillna(0)
    return df


def add_engineered_columns(work: pd.DataFrame) -> pd.DataFrame:
    work = work.copy()
    work["charger_key"] = work["stat_id"].astype(str) + "_" + work["chger_id"].astype(str)
    work = work.sort_values(["charger_key", "created_at"])

    work["hour"] = work["created_at"].dt.hour
    work["minute"] = work["created_at"].dt.minute
    work["weekday"] = work["created_at"].dt.weekday
    work["is_charging"] = (work["stat"] == 3).astype(int)
    work["is_available"] = (work["stat"] == 2).astype(int)
    work["is_error"] = (work["stat"] == 1).astype(int)
    work["prev_stat"] = (
        work.groupby("charger_key")["stat"].shift(1).fillna(work["stat"])
    )
    work["parking_free_yn"] = (work["parking_free"] == "Y").astype(int)
    work["limit_yn_flag"] = (work["limit_yn"] == "Y").astype(int)
    work["traffic_yn_flag"] = (work["traffic_yn"] == "Y").astype(int)
    work["is_fast"] = (work["output_kw"] >= 50).astype(int)

    since_end = (work["created_at"] - work["last_tedt"]).dt.total_seconds() / 60.0
    work["mins_since_last_end"] = since_end.fillna(24 * 60).clip(0, 24 * 60)
    work["has_now_tsdt"] = work["now_tsdt"].notna().astype(int)
    work["avail_roll3"] = (
        work.groupby("charger_key")["is_available"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(work["is_available"])
    )
    return work


def build_feature_matrix(work: pd.DataFrame) -> pd.DataFrame:
    type_dummies = pd.get_dummies(work["chger_type"].fillna("UNK"), prefix="ctype")
    kind_dummies = pd.get_dummies(work["kind"].fillna("UNK"), prefix="kind")
    busi_dummies = pd.get_dummies(work["busi_id"].fillna("UNK"), prefix="busi")
    return pd.concat(
        [work[BASE_FEATURE_COLS], type_dummies, kind_dummies, busi_dummies],
        axis=1,
    )


def align_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    return X[columns]


def prepare_train_set(df: pd.DataFrame):
    work = add_engineered_columns(df)
    work["next_stat"] = work.groupby("charger_key")["stat"].shift(-1)
    work = work.dropna(subset=["next_stat"])
    work["y"] = (work["next_stat"] == 2).astype(int)

    X = build_feature_matrix(work)
    meta = work[
        [
            "stat_id",
            "chger_id",
            "stat_nm",
            "addr",
            "lat",
            "lng",
            "created_at",
            "stat",
            "output_kw",
            "chger_type",
        ]
    ].copy()
    return X, work["y"], meta


def train_model(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    order = meta["created_at"].argsort(kind="mergesort")
    X, y, meta = X.iloc[order], y.iloc[order], meta.iloc[order]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    meta_test = meta.iloc[split:]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= AVAIL_THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print("1) JOIN 피처 RF: 다음 스냅샷 사용가능 확률")
    print("=" * 60)
    print(f"샘플 {len(X):,} | 학습 {len(X_train):,} / 테스트 {len(X_test):,}")
    print(f"사용가능 비율(전체 라벨): {y.mean():.1%}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, target_names=["사용불가", "사용가능"]))

    importance = (
        pd.Series(clf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("상위 피처 중요도:")
    for name, val in importance.items():
        print(f"  {name:24s} {val:.3f}")

    return clf, X.columns.tolist(), y_test, meta_test, proba


def build_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["stat_id", "chger_id"])["created_at"].idxmax()
    return df.loc[idx].copy()


def predict_station_availability(
    clf,
    feature_columns: list[str],
    history_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    stat_id: str,
    threshold: float = AVAIL_THRESHOLD,
) -> dict:
    """충전소 선택 → 다음 시점 예상 사용가능 대수."""
    # roll/prev 계산을 위해 해당 충전소 히스토리 + 최신 행 사용
    hist = history_df[history_df["stat_id"] == stat_id].copy()
    latest = latest_df[latest_df["stat_id"] == stat_id].copy()
    if latest.empty:
        return {"stat_id": stat_id, "error": "충전소 데이터 없음"}

    work = add_engineered_columns(pd.concat([hist, latest], ignore_index=True))
    # 충전기별 마지막 행만 추론
    last_idx = work.groupby("charger_key")["created_at"].idxmax()
    work_last = work.loc[last_idx].copy()

    X = align_columns(build_feature_matrix(work_last), feature_columns)
    proba = clf.predict_proba(X)[:, 1]
    work_last["available_prob"] = proba
    work_last["pred_available"] = (proba >= threshold).astype(int)

    row0 = work_last.iloc[0]
    return {
        "stat_id": stat_id,
        "stat_nm": row0.get("stat_nm"),
        "addr": row0.get("addr"),
        "lat": float(row0["lat"]) if pd.notna(row0.get("lat")) else None,
        "lng": float(row0["lng"]) if pd.notna(row0.get("lng")) else None,
        "total_chargers": len(work_last),
        "now_available": int((work_last["stat"] == 2).sum()),
        "pred_available_next": int(work_last["pred_available"].sum()),
        "avg_available_prob": float(work_last["available_prob"].mean()),
    }


def evaluate_station_count(
    meta_test: pd.DataFrame, y_test: pd.Series, proba: np.ndarray
) -> None:
    print("\n" + "=" * 60)
    print("2) 충전소별 잔여 대수 예측 평가 (테스트 구간)")
    print("=" * 60)

    eval_df = meta_test.copy()
    eval_df["y"] = y_test.values
    eval_df["pred"] = (proba >= AVAIL_THRESHOLD).astype(int)

    grouped = (
        eval_df.groupby(["stat_id", "created_at"], as_index=False)
        .agg(
            actual_available=("y", "sum"),
            pred_available=("pred", "sum"),
            n=("y", "size"),
        )
    )
    grouped = grouped[grouped["n"] >= 2]
    if grouped.empty:
        print("평가할 충전소 스냅샷이 부족합니다.")
        return

    mae = (grouped["actual_available"] - grouped["pred_available"]).abs().mean()
    exact = (grouped["actual_available"] == grouped["pred_available"]).mean()
    within1 = (
        (grouped["actual_available"] - grouped["pred_available"]).abs() <= 1
    ).mean()

    print(f"평가 건수(충전소×시점): {len(grouped):,}")
    print(f"잔여 대수 MAE: {mae:.3f}")
    print(f"정확히 일치: {exact:.1%}")
    print(f"±1대 이내:   {within1:.1%}")

    sample = grouped.sort_values("n", ascending=False).head(8)
    print("\n예시 (충전기 많은 충전소):")
    print(
        sample[["stat_id", "created_at", "n", "actual_available", "pred_available"]]
        .to_string(index=False)
    )


def demo_station_predictions(clf, feature_columns, df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3) 충전소 선택 → 예상 잔여 대수 (최신 스냅샷)")
    print("=" * 60)

    latest = build_latest_snapshot(df)
    station_sizes = (
        latest.groupby(["stat_id", "stat_nm", "addr"], dropna=False)
        .size()
        .reset_index(name="n_chargers")
        .sort_values("n_chargers", ascending=False)
    )
    print(f"JOIN된 충전소 수: {station_sizes.shape[0]:,}")
    print("\n충전기 많은 충전소 TOP 5 예측:")

    for _, row in station_sizes.head(5).iterrows():
        result = predict_station_availability(
            clf, feature_columns, df, latest, row["stat_id"]
        )
        if result.get("error"):
            print(f"  - {row['stat_id']}: {result['error']}")
            continue
        print(
            f"  - [{result['stat_id']}] {result['stat_nm']}\n"
            f"      주소: {result['addr']}\n"
            f"      충전기 {result['total_chargers']}대 | "
            f"지금 사용가능 {result['now_available']}대 → "
            f"다음 예상 {result['pred_available_next']}대 "
            f"(평균확률 {result['avg_available_prob']:.0%})"
        )


def main() -> None:
    print("정보+상태 JOIN 데이터 로드 중...")
    df = load_joined()
    print(
        f"JOIN 완료: {len(df):,}행 | "
        f"충전소 {df['stat_id'].nunique():,} | "
        f"스냅샷 {df['created_at'].nunique()}"
    )
    if df.empty:
        print("JOIN 결과가 비었습니다. run_info.py / run.py 수집을 확인하세요.")
        return

    X, y, meta = prepare_train_set(df)
    clf, feature_columns, y_test, meta_test, proba = train_model(X, y, meta)
    evaluate_station_count(meta_test, y_test, proba)
    demo_station_predictions(clf, feature_columns, df)

    print("\n" + "=" * 60)
    print("요약: 충전소(stat_id) 선택 → 충전기별 사용가능 확률 → 잔여 대수 합산")
    print("실행: py analyze_station.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
