"""실험: 학습 모집단 ≠ 서빙 모집단 확인.

recommend_api/access.py 는 RESIDENT/RESTRICTED 를 추천에서 **하드 제외**한다
(access_coefficient → None). 그런데 학습·평가에는 그대로 들어간다.

실측(2026-07-31): 전 충전기의 57.0% 가 하드 제외 대상이고,
최근 3일 관측 236,359건 중 22.6% 가 그 충전기들 것이다.

아파트(RESIDENT) 충전기는 야간 장시간 충전 등 패턴이 규칙적이라 맞히기 쉽다.
그래서 전체 기준 AUC 가 **실제 서빙 모집단보다 낙관적**일 수 있다.
이 스크립트는 같은 모델을 세그먼트별로 평가해 그 격차를 잰다.

운영 아티팩트는 건드리지 않는다.

사용:
  py -m recommend_api.eval_by_access
  py -m recommend_api.eval_by_access --max-rows 300000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .access import access_coefficient
from .config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS
from .eval_metrics import evaluate_classifier, split_by_date
from .holdout_eval import _new_model, _prepare_fold
from .model_store import build_horizon_dataset, load_joined, load_label_series
from .scoring import station_access_payload_from_group

OUT_DIR = ARTIFACTS_DIR / "experiments"


def annotate_access(df: pd.DataFrame) -> pd.DataFrame:
    """충전소별 access_type / 하드제외 여부를 모든 충전기 행에 붙인다.

    limit_detail 이 반드시 필요하다. 이게 없으면 RESIDENT/RESTRICTED 판정
    키워드(거주자·입주민·외부인 등)를 못 잡아 전부 PUBLIC/UNKNOWN 으로 떨어진다.
    같은 충전소에서 제한 정보가 섞이면 서빙과 동일하게 가장 보수적인 유형을 쓴다.
    """
    required = ["stat_id", "chger_id", "limit_yn", "limit_detail", "kind"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"필수 컬럼 누락: {missing} — load_joined 의 SELECT 를 확인하세요. "
            "특히 limit_detail 없이는 access 분류가 무의미합니다."
        )

    opt = [c for c in ("kind_detail", "stat_nm", "addr") if c in df.columns]
    uniq = df[required + opt].drop_duplicates(subset=["stat_id", "chger_id"])

    recs = []
    for stat_id, group in uniq.groupby("stat_id", sort=False):
        c = station_access_payload_from_group(group)
        recs.append({
            "stat_id": stat_id,
            "access_type": c["access_type"],
            "hard_excluded": access_coefficient(c["access_type"], c["is_housing"]) is None,
        })
    return df.merge(pd.DataFrame(recs), on="stat_id", how="left")


def main() -> None:
    p = argparse.ArgumentParser(description="access 세그먼트별 성능 격차 측정")
    p.add_argument(
        "--max-rows", type=int, default=None, metavar="N",
        help=(
            "load_joined 상한. 기본 전량. 무작위 표본이 아니라 stat_id 앞쪽만 "
            "남으므로 결론용 실행에는 쓰지 말 것 (load_joined docstring 참고)."
        ),
    )
    args = p.parse_args()

    if args.max_rows:
        print(f"주의: --max-rows {args.max_rows:,} 는 stat_id 정렬 절단이라 "
              "사업자가 통째로 빠진다. 배선 확인용으로만 쓸 것.")

    print("데이터 로드...")
    base = load_joined(limit=args.max_rows)
    print(f"  {len(base):,}행")

    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). model_store.load_label_series 참고.
    print("라벨 시계열 로드 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋...")
    df = build_horizon_dataset(base, HORIZONS, label_df=label_df).reset_index(drop=True)
    df = annotate_access(df)
    print(f"  {len(df):,}샘플")
    print("\naccess_type 별 샘플 비중:")
    vc = df["access_type"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<12}{v:>9,}  ({100*v/len(df):5.1f}%)")

    dummies = [
        pd.get_dummies(df["chger_type"].fillna("UNK"), prefix="ctype"),
        pd.get_dummies(df["kind"].fillna("UNK"), prefix="kind"),
        pd.get_dummies(df["busi_id"].fillna("UNK"), prefix="busi"),
    ]
    X = pd.concat([df[FEATURE_COLS], *dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    y = df["y"]

    train_mask, train_dates, test_dates = split_by_date(df)
    X_tr, X_te, y_tr, y_te = _prepare_fold(X, y, train_mask)
    print(f"\n홀드아웃: train {len(train_dates)}일 / test {len(test_dates)}일")

    print("학습 중...")
    model = _new_model()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    te = df.loc[X_te.index]
    is_fast = te["is_fast"].fillna(0).astype(int) == 1
    operating = pd.to_numeric(
        te.get("is_operating_at_arrival", pd.Series(np.nan, index=te.index)),
        errors="coerce",
    )
    open_or_unknown = operating.isna() | operating.ne(0)
    servable = (~te["hard_excluded"].fillna(False)) & is_fast & open_or_unknown
    arrival_hour = pd.to_numeric(te["arrival_hour"], errors="coerce")
    night = arrival_hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
    day = ~night
    serving_name = "★ 실제 서빙 모집단 (급속 & 접근가능 & 운영중/미상)"

    segments = {
        "ALL (현재 보고 기준)": pd.Series(True, index=te.index),
        "급속만": is_fast,
        "추천가능(하드제외 아님)": ~te["hard_excluded"].fillna(False),
        serving_name: servable,
        "★ 실제 서빙 × 주간(06~21)": servable & day,
        "★ 실제 서빙 × 야간(22~05)": servable & night,
        "-- PUBLIC × 야간": (te["access_type"] == "PUBLIC") & is_fast & open_or_unknown & night,
        "-- UNKNOWN × 야간": (te["access_type"] == "UNKNOWN") & is_fast & open_or_unknown & night,
        "-- RESIDENT (서빙 안 됨)": te["access_type"] == "RESIDENT",
        "-- PUBLIC": te["access_type"] == "PUBLIC",
    }

    print(f"\n{'세그먼트':<48}{'n':>10}{'가용률':>9}{'AUC':>9}{'recall':>9}{'PR-AUC':>9}{'Brier':>9}")
    print("-" * 103)
    out = {}
    for name, mask in segments.items():
        m = mask.values
        if m.sum() < 500:
            print(f"{name:<48}{int(m.sum()):>10}   (표본 부족)")
            continue
        r = evaluate_classifier(y_te[m], proba[m])
        out[name] = r
        print(f"{name:<48}{r['n']:>10,}{r['positive_rate']:>9.3f}{r['roc_auc']:>9.4f}"
              f"{r['unavailable_recall']:>9.3f}{r['pr_auc_unavailable']:>9.4f}"
              f"{r['brier_score']:>9.4f}")

    if "ALL (현재 보고 기준)" in out and serving_name in out:
        a = out["ALL (현재 보고 기준)"]; s = out[serving_name]
        print("\n" + "=" * 66)
        print("전체 기준 대비 실제 서빙 모집단 격차")
        print("=" * 66)
        for k in ("roc_auc", "unavailable_recall", "pr_auc_unavailable"):
            print(f"  {k:<24}{a[k]:>8.4f} -> {s[k]:>8.4f}   ({s[k]-a[k]:+.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "access_segment_eval.json"
    path.write_text(json.dumps({
        "experiment": "access_segment_gap",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "운영 아티팩트 미변경. 지표 해석용.",
        "n_samples": int(len(df)),
        "access_mix": {k: int(v) for k, v in vc.items()},
        "segments": out,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
