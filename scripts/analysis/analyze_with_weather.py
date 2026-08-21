"""
weather + status/features/info JOIN 분석
- 날씨 피처 포함 vs 미포함 모델 비교
- 기온/강수와 사용가능 비율 간단 상관
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

import warnings

import numpy as np
import pandas as pd
import pymysql
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

BASE_COLS = [
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
]

WEATHER_COLS = ["ta", "rn", "hm", "ws", "dsnw", "is_rain"]


def load_joined() -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=40)
    try:
        status_q = """
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.parking_free, i.limit_yn
            FROM ev_charger_status s
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            WHERE i.del_yn IS NULL OR i.del_yn <> 'Y'
            ORDER BY s.created_at, s.stat_id, s.chger_id
        """
        weather_q = """
            SELECT observed_at, stn_id, stn_nm, ta, rn, hm, ws, dsnw
            FROM weather
            WHERE stn_id = '143'
        """
        df = pd.read_sql(status_q, conn)
        weather = pd.read_sql(weather_q, conn)
    finally:
        conn.close()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    weather["observed_at"] = pd.to_datetime(weather["observed_at"], errors="coerce")
    df["weather_hour"] = df["created_at"].dt.floor("h")
    weather["weather_hour"] = weather["observed_at"].dt.floor("h")
    weather = weather.drop_duplicates("weather_hour")

    df = df.merge(
        weather[["weather_hour", "ta", "rn", "hm", "ws", "dsnw", "observed_at"]],
        on="weather_hour",
        how="left",
    )
    df = df.rename(columns={"observed_at": "weather_at"})

    for col in ("ta", "rn", "hm", "ws", "dsnw", "output"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["output_kw"] = df["output"].fillna(0)
    df["is_fast"] = (df["output_kw"] >= 50).astype(int)
    df["parking_free_yn"] = (df["parking_free"] == "Y").astype(int)
    df["limit_yn_flag"] = (df["limit_yn"] == "Y").astype(int)
    df["is_rain"] = (df["rn"].fillna(0) > 0).astype(int)
    return df


def prepare(df: pd.DataFrame, with_weather: bool):
    work = df.copy()
    work["charger_key"] = work["stat_id"].astype(str) + "_" + work["chger_id"].astype(str)
    work["next_stat"] = work.groupby("charger_key")["stat"].shift(-1)
    work = work.dropna(subset=["next_stat"])
    work["y"] = (work["next_stat"] == 2).astype(int)

    cols = list(BASE_COLS)
    if with_weather:
        # 날씨 있는 행만 (공정 비교용 교집합은 호출측에서 처리 가능)
        cols = cols + WEATHER_COLS

    type_dummies = pd.get_dummies(work["chger_type"].fillna("UNK"), prefix="ctype")
    busi_dummies = pd.get_dummies(work["busi_id"].fillna("UNK"), prefix="busi")
    X = pd.concat([work[cols], type_dummies, busi_dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    meta = work[["created_at", "stat_id", "ta", "rn", "hm", "is_rain"]].copy()
    return X, work["y"], meta


def time_split(X, y, meta, ratio=0.8):
    order = meta["created_at"].argsort(kind="mergesort")
    X, y, meta = X.iloc[order], y.iloc[order], meta.iloc[order]
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], meta.iloc[split:]


def eval_hgb(name, X_train, X_test, y_train, y_test):
    # 분산 없는 컬럼 제거 (rn 전부 NaN/상수면 HGB binning 오류)
    nunique = X_train.nunique(dropna=True)
    keep = nunique[nunique >= 2].index.tolist()
    X_train = X_train[keep]
    X_test = X_test.reindex(columns=keep)

    med = X_train.median(numeric_only=True)
    Xtr, Xte = X_train.fillna(med), X_test.fillna(med)
    clf = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=200, random_state=42
    )
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(keep),
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def weather_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("날씨 EDA (시간대별 사용가능 비율)")
    print("=" * 60)
    tmp = df.dropna(subset=["ta"]).copy()
    if tmp.empty:
        print("날씨와 JOIN된 행이 없습니다.")
        return

    tmp["is_available"] = (tmp["stat"] == 2).astype(int)
    by_hour = (
        tmp.groupby(tmp["created_at"].dt.floor("h"))
        .agg(
            n=("stat", "size"),
            avail_rate=("is_available", "mean"),
            ta=("ta", "mean"),
            rn=("rn", "mean"),
            hm=("hm", "mean"),
        )
        .reset_index()
    )
    print(by_hour.round(3).to_string(index=False))

    # 간단 상관 (시간 집계)
    corr = by_hour[["avail_rate", "ta", "rn", "hm"]].corr()
    print("\n시간 집계 상관 (avail_rate vs 날씨):")
    print(corr["avail_rate"].round(3).to_string())

    rain = tmp.groupby("is_rain")["is_available"].mean()
    print("\n강수 여부별 사용가능 비율:")
    for k, v in rain.items():
        label = "비옴" if k == 1 else "비안옴"
        print(f"  {label}: {v:.1%} (n={int((tmp['is_rain']==k).sum()):,})")


def main() -> None:
    print("날씨 JOIN 데이터 로드 중...")
    df = load_joined()
    joined = df["ta"].notna().sum()
    print(
        f"전체 {len(df):,}행 | 날씨 JOIN {joined:,}행 "
        f"({joined / max(len(df), 1):.1%}) | "
        f"기간 {df['created_at'].min()} ~ {df['created_at'].max()}"
    )

    weather_eda(df)

    # 공정 비교: 날씨가 있는 행만
    df_w = df.dropna(subset=["ta"]).copy()
    if len(df_w) < 200:
        print("\n날씨 JOIN 표본이 너무 적어 모델 비교를 건너뜁니다.")
        print("weather에 상태 수집 기간과 겹치는 날짜를 더 넣으면 됩니다.")
        print("실행: py analyze_with_weather.py")
        return

    print("\n" + "=" * 60)
    print("모델 비교 (같은 표본: 날씨 JOIN된 행만)")
    print("=" * 60)

    X0, y0, m0 = prepare(df_w, with_weather=False)
    X1, y1, m1 = prepare(df_w, with_weather=True)

    Xtr0, Xte0, ytr0, yte0, _ = time_split(X0, y0, m0)
    Xtr1, Xte1, ytr1, yte1, _ = time_split(X1, y1, m1)

    if len(Xtr0) < 50 or len(Xte0) < 20:
        print(f"학습/테스트 표본 부족: train={len(Xtr0)}, test={len(Xte0)}")
        return

    r0 = eval_hgb("HGB (날씨 없음)", Xtr0, Xte0, ytr0, yte0)
    r1 = eval_hgb("HGB (날씨 포함)", Xtr1, Xte1, ytr1, yte1)

    summary = pd.DataFrame([r0, r1]).round(4)
    print(summary.to_string(index=False))

    print("\n참고: weather는 현재 짧은 기간만 있어 효과는 제한적일 수 있습니다.")
    print("며칠~2주 쌓이면 기온/강수 피처 기여가 더 분명해집니다.")
    print("실행: py analyze_with_weather.py")


if __name__ == "__main__":
    main()
