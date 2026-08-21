"""
ev_charger_status 데이터로
1) Random Forest 분류: 다음 스냅샷 사용가능(stat=2) 확률
2) 시계열 예측: 스냅샷별 사용가능 대수
테스트 스크립트
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import warnings

import pymysql
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

STAT_LABELS = {
    1: "통신이상",
    2: "충전대기(사용가능)",
    3: "충전중",
    4: "운영중지",
    5: "점검중",
    9: "상태미확인",
}


def load_data() -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=20)
    try:
        query = """
            SELECT stat_id, chger_id, busi_id, stat,
                   last_tsdt, last_tedt, now_tsdt, created_at
            FROM ev_charger_status
            ORDER BY created_at, stat_id, chger_id
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    for col in ("created_at", "last_tsdt", "last_tedt", "now_tsdt"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def prepare_rf_classification(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """충전기별로 다음 스냅샷의 사용가능(stat==2) 여부를 라벨로 만듦."""
    work = df.copy()
    work["charger_key"] = work["stat_id"] + "_" + work["chger_id"]
    work = work.sort_values(["charger_key", "created_at"])

    work["next_stat"] = work.groupby("charger_key")["stat"].shift(-1)
    work = work.dropna(subset=["next_stat"])
    work["y"] = (work["next_stat"] == 2).astype(int)

    work["hour"] = work["created_at"].dt.hour
    work["minute"] = work["created_at"].dt.minute
    work["weekday"] = work["created_at"].dt.weekday
    work["is_charging"] = (work["stat"] == 3).astype(int)
    work["is_available"] = (work["stat"] == 2).astype(int)
    work["is_error"] = (work["stat"] == 1).astype(int)
    work["prev_stat"] = work.groupby("charger_key")["stat"].shift(1).fillna(work["stat"])

    # 추가 시간 피처 (최근 충전 이력)
    last_dur = (work["last_tedt"] - work["last_tsdt"]).dt.total_seconds() / 60.0
    work["last_charge_min"] = last_dur.fillna(0).clip(lower=0, upper=24 * 60)
    since_end = (work["created_at"] - work["last_tedt"]).dt.total_seconds() / 60.0
    work["mins_since_last_end"] = since_end.fillna(24 * 60).clip(lower=0, upper=24 * 60)
    work["has_now_tsdt"] = work["now_tsdt"].notna().astype(int)
    charging_for = (work["created_at"] - work["now_tsdt"]).dt.total_seconds() / 60.0
    work["charging_for_min"] = charging_for.fillna(0).clip(lower=0, upper=24 * 60)

    # 같은 충전기 최근 사용가능 비율 (직전 3스냅샷)
    work["avail_roll3"] = (
        work.groupby("charger_key")["is_available"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(work["is_available"])
    )

    busi_dummies = pd.get_dummies(work["busi_id"].fillna("UNK"), prefix="busi")
    features = pd.concat(
        [
            work[
                [
                    "hour",
                    "minute",
                    "weekday",
                    "stat",
                    "prev_stat",
                    "is_charging",
                    "is_available",
                    "is_error",
                    "last_charge_min",
                    "mins_since_last_end",
                    "has_now_tsdt",
                    "charging_for_min",
                    "avail_roll3",
                ]
            ],
            busi_dummies,
        ],
        axis=1,
    )
    meta = work[["stat_id", "chger_id", "busi_id", "created_at", "stat"]].copy()
    return features, work["y"], meta


def run_random_forest(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("1) Random Forest: 다음 스냅샷 사용가능(stat=2) 확률")
    print("=" * 60)

    X, y, meta = prepare_rf_classification(df)
    print(f"샘플 수: {len(X):,}  |  사용가능 비율: {y.mean():.1%}")
    print(
        f"기간: {meta['created_at'].min()} ~ {meta['created_at'].max()} "
        f"(스냅샷 {meta['created_at'].nunique()}개)"
    )

    # 시간순 분할 (미래 시점 예측에 가깝게)
    order = meta["created_at"].argsort(kind="mergesort")
    X, y, meta = X.iloc[order], y.iloc[order], meta.iloc[order]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    meta_test = meta.iloc[split:]
    print(f"학습 {len(X_train):,} / 테스트 {len(X_test):,} (시간순 80/20)")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]  # 사용가능 확률

    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")
    print(
        classification_report(
            y_test,
            pred,
            target_names=["사용불가", "사용가능(대기)"],
        )
    )

    importance = (
        pd.Series(clf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(8)
    )
    print("상위 피처 중요도:")
    for name, val in importance.items():
        print(f"  {name:20s} {val:.3f}")

    sample = meta_test.copy()
    sample["actual_available"] = y_test.values
    sample["pred_available"] = pred
    sample["available_prob"] = proba
    sample = sample.sort_values("available_prob", ascending=False)

    print("\n사용가능 확률 상위 5건 예시:")
    print(
        sample.head(5)[
            ["stat_id", "chger_id", "stat", "available_prob", "actual_available"]
        ]
        .assign(available_prob=lambda d: (d["available_prob"] * 100).round(1).astype(str) + "%")
        .to_string(index=False)
    )

    print("\n사용가능 확률 하위 5건 예시:")
    print(
        sample.tail(5)[
            ["stat_id", "chger_id", "stat", "available_prob", "actual_available"]
        ]
        .assign(available_prob=lambda d: (d["available_prob"] * 100).round(1).astype(str) + "%")
        .to_string(index=False)
    )


def build_available_series(df: pd.DataFrame) -> pd.Series:
    """스냅샷(created_at)별 사용가능(stat=2) 충전기 대수."""
    snap = (
        df.assign(is_available=(df["stat"] == 2).astype(int))
        .groupby("created_at")["is_available"]
        .sum()
        .sort_index()
    )
    return snap.astype(float)


def run_time_series(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2) 시계열 예측: 스냅샷별 사용가능(대기) 대수")
    print("=" * 60)

    series = build_available_series(df)
    print(f"스냅샷 수: {len(series)}")
    print(
        f"사용가능 대수 범위: {series.min():.0f} ~ {series.max():.0f} "
        f"(평균 {series.mean():.1f})"
    )

    if len(series) < 12:
        print("스냅샷이 부족해 시계열 테스트를 건너뜁니다.")
        return

    test_n = max(8, int(len(series) * 0.2))
    split = len(series) - test_n
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"학습 {len(train)} / 테스트 {len(test)}")

    hw = ExponentialSmoothing(
        train.values,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_pred = pd.Series(hw.forecast(len(test)), index=test.index)

    try:
        arima = ARIMA(train.values, order=(1, 1, 1)).fit()
        arima_pred = pd.Series(arima.forecast(len(test)), index=test.index)
    except Exception as exc:
        print(f"ARIMA 실패: {exc}")
        arima_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)

    hist = series.to_frame("y")
    for lag in (1, 2, 3):
        hist[f"lag_{lag}"] = hist["y"].shift(lag)
    hist["roll_mean_3"] = hist["y"].shift(1).rolling(3).mean()
    hist = hist.dropna()

    hist_train = hist.loc[hist.index <= train.index[-1]]
    hist_test = hist.loc[hist.index.isin(test.index)]

    rf_ok = len(hist_train) >= 8 and len(hist_test) >= 1
    if rf_ok:
        feat_cols = [c for c in hist.columns if c != "y"]
        rfr = RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )
        rfr.fit(hist_train[feat_cols], hist_train["y"])
        rf_pred = pd.Series(rfr.predict(hist_test[feat_cols]), index=hist_test.index)
    else:
        rf_pred = None

    def metrics(name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
        aligned = pd.concat([y_true, y_pred], axis=1, join="inner").dropna()
        if aligned.empty:
            print(f"{name}: 비교 불가")
            return
        yt, yp = aligned.iloc[:, 0], aligned.iloc[:, 1]
        mae = mean_absolute_error(yt, yp)
        rmse = mean_squared_error(yt, yp) ** 0.5
        print(f"{name:22s}  MAE={mae:6.2f}  RMSE={rmse:6.2f}")

    print("\n모델 성능 (테스트 구간, 사용가능 대수):")
    metrics("Holt-Winters", test, hw_pred)
    metrics("ARIMA(1,1,1)", test, arima_pred)
    if rf_pred is not None:
        metrics("RF Regressor(lags)", test, rf_pred)

    print("\n테스트 구간 실제 vs 예측:")
    compare = pd.DataFrame({"actual": test})
    compare["holt_winters"] = hw_pred
    compare["arima"] = arima_pred
    if rf_pred is not None:
        compare["rf_lag"] = rf_pred
    print(compare.round(1).to_string())

    naive = test.shift(1)
    naive.iloc[0] = train.iloc[-1]
    metrics("Naive(직전값)", test, naive)


def main() -> None:
    print("DB에서 데이터 로드 중...")
    df = load_data()
    print(f"로드 완료: {len(df):,}행, 스냅샷 {df['created_at'].nunique()}개")

    print("\n상태(stat) 분포:")
    vc = df["stat"].value_counts().sort_index()
    for s, cnt in vc.items():
        print(f"  {s} ({STAT_LABELS.get(s, '?')}): {cnt:,}")

    run_random_forest(df)
    run_time_series(df)

    print("\n" + "=" * 60)
    print("참고: 데이터가 늘수록 시간대 패턴 학습이 안정됩니다.")
    print("약 2주 모이면 요일·출퇴근 혼잡까지 반영한 예측이 가능합니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
