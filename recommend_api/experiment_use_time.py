"""실험: is_operating_at_arrival(도착 시점 운영 여부) 피처 효과 측정.

운영 아티팩트(horizon_hgb.joblib)는 건드리지 않는다. 승격 판단용이 아니라
"이 피처가 값을 하는가"만 본다.

설계: 같은 데이터·같은 날짜 홀드아웃에서 **피처 집합만** 바꿔 비교한다.
  A(baseline) = config.FEATURE_COLS (25) + chger_type/kind/busi 더미
  B(+optime)  = A + is_operating_at_arrival

왜 이 피처인가 (2026-07-31 실측):
  문 닫힌 충전기의 stat=2 비율 89.0% > 운영 중 72.0% (새벽엔 95~97%).
  물리적으로 못 쓰는데 stat 은 유휴라 사용가능으로 보고된다.
  hour 만으로는 24시간 충전소와 야간 폐쇄 충전소를 구분할 수 없다.

2026-08-03 재실행 이력
---------------------
최초 실행(2026-07-31)은 두 가지가 잘못돼 있었다.
  1. --max-rows 기본값이 400000 이라 stat_id 앞쪽 절반만 학습했다.
     사업자 61곳 중 21곳이 통째로 빠진 표본이다(load_joined docstring 참고).
  2. label_df 없이 build_horizon_dataset 을 불러 snapshot 앵커가 빠졌다.
     정본 학습(train_and_save)은 delta+snapshot 을 라벨 원천으로 쓴다.
둘 다 고쳐서 재실행했다. 기각 결론이 바뀌는지 확인이 목적이었다.

사용:
  py -m recommend_api.experiment_use_time
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS
from .eval_metrics import evaluate_classifier, split_by_date
from .holdout_eval import _new_model, _per_eta_metrics, _prepare_fold
from .model_store import build_horizon_dataset, load_joined, load_label_series

EXTRA_COL = "is_operating_at_arrival"
OUT_DIR = ARTIFACTS_DIR / "experiments"


def build_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """make_xy 와 동일한 더미 구성. 피처 목록만 바꿔 끼운다."""
    dummies = [
        pd.get_dummies(df["chger_type"].fillna("UNK"), prefix="ctype"),
        pd.get_dummies(df["kind"].fillna("UNK"), prefix="kind"),
        pd.get_dummies(df["busi_id"].fillna("UNK"), prefix="busi"),
    ]
    X = pd.concat([df[cols], *dummies], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def run_arm(
    df: pd.DataFrame, cols: list[str], train_mask: pd.Series, label: str
) -> dict[str, Any]:
    X = build_matrix(df, cols)
    y = df["y"]
    X_tr, X_te, y_tr, y_te = _prepare_fold(X, y, train_mask)

    model = _new_model()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    overall = evaluate_classifier(y_te, proba)
    per_eta = _per_eta_metrics(y_te, proba, df.loc[X_te.index, "eta_minutes"])

    # 야간(운영시간 효과가 가장 클 구간) 별도 측정
    hour = df.loc[X_te.index, "arrival_hour"]
    night = hour.isin([0, 1, 2, 3, 4, 5, 22, 23])
    night_m = (
        evaluate_classifier(y_te[night.values], proba[night.values])
        if night.sum() > 100
        else None
    )

    print(f"\n[{label}]  피처 {X.shape[1]}개")
    print(f"  overall  AUC {overall['roc_auc']:.4f} | "
          f"unavail-recall {overall['unavailable_recall']:.3f} | "
          f"PR-AUC {overall['pr_auc_unavailable']:.4f} | "
          f"Brier {overall['brier_score']:.4f}")
    if night_m:
        print(f"  야간(22-05) AUC {night_m['roc_auc']:.4f} | "
              f"unavail-recall {night_m['unavailable_recall']:.3f} | n={night_m['n']:,}")

    return {
        "label": label,
        "n_features": int(X.shape[1]),
        "overall": overall,
        "per_eta": per_eta,
        "night": night_m,
        "_proba": proba,
        "_y": y_te,
        "_idx": X_te.index,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="운영시간 피처 효과 실험")
    p.add_argument("--max-rows", type=int, default=None,
                   metavar="N",
                   help=(
                       "load_joined 상한. 기본 전량. "
                       "무작위 표본이 아니라 stat_id 앞쪽만 남으므로 결론용 실행에는 "
                       "쓰지 말 것 (load_joined docstring 참고)."
                   ))
    args = p.parse_args()

    if args.max_rows:
        print(f"주의: --max-rows {args.max_rows:,} 는 stat_id 정렬 절단이라 "
              "사업자가 통째로 빠진다. 배선 확인용으로만 쓸 것.")

    print("데이터 로드..." if not args.max_rows else f"데이터 로드 (최대 {args.max_rows:,}행)...")
    base = load_joined(limit=args.max_rows)
    print(f"  {len(base):,}행")

    # 라벨 원천은 정본 학습(train_and_save)과 맞춘다. label_df 없이 부르면
    # df 자기 자신(delta only)에서 미래 상태를 찾는데, snapshot 앵커가 빠져
    # LOCF 사슬이 끊기고 표본이 크게 줄어든다. 2026-07-31 최초 실행은 이 경로였다.
    print("라벨 시계열 로드 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성...")
    df = build_horizon_dataset(base, HORIZONS, label_df=label_df).reset_index(drop=True)
    print(f"  {len(df):,}샘플 / positive_rate {df['y'].mean():.4f}")

    if EXTRA_COL not in df.columns:
        raise SystemExit(f"{EXTRA_COL} 컬럼 없음 — model_store 배선 확인")
    known = df[EXTRA_COL].notna()
    print(f"  {EXTRA_COL}: 판정률 {100*known.mean():.1f}% "
          f"(Y {int((df[EXTRA_COL]==1).sum()):,} / N {int((df[EXTRA_COL]==0).sum()):,})")

    train_mask, train_dates, test_dates = split_by_date(df)
    info = {
        "n_train_dates": len(train_dates),
        "n_test_dates": len(test_dates),
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
    }
    print(f"\n날짜 홀드아웃: train {len(train_dates)}일 / test {len(test_dates)}일"
          f"  ({int(train_mask.sum()):,} vs {int((~train_mask).sum()):,}샘플)")

    a = run_arm(df, FEATURE_COLS, train_mask, "A baseline")
    b = run_arm(df, FEATURE_COLS + [EXTRA_COL], train_mask, "B +optime")

    print("\n" + "=" * 66)
    print("차이 (B - A)")
    print("=" * 66)
    for k, fmt in (("roc_auc", "+.4f"), ("unavailable_recall", "+.4f"),
                   ("pr_auc_unavailable", "+.4f"), ("brier_score", "+.4f")):
        d = b["overall"][k] - a["overall"][k]
        note = " (낮을수록 좋음)" if k == "brier_score" else ""
        print(f"  {k:<24} {d:{fmt}}{note}")
    if a["night"] and b["night"]:
        print(f"  {'night roc_auc':<24} {b['night']['roc_auc']-a['night']['roc_auc']:+.4f}")
        print(f"  {'night unavail_recall':<24} "
              f"{b['night']['unavailable_recall']-a['night']['unavailable_recall']:+.4f}")

    print(f"\n{'ETA':>5}{'A AUC':>10}{'B AUC':>10}{'차이':>9}"
          f"{'A recall':>11}{'B recall':>11}{'차이':>9}")
    print("-" * 66)
    bm = {r["eta_minutes"]: r for r in b["per_eta"]}
    for ra in a["per_eta"]:
        h = ra["eta_minutes"]; rb = bm[h]
        print(f"{h:>5}{ra['roc_auc']:>10.4f}{rb['roc_auc']:>10.4f}"
              f"{rb['roc_auc']-ra['roc_auc']:>+9.4f}"
              f"{ra['unavailable_recall']:>11.3f}{rb['unavailable_recall']:>11.3f}"
              f"{rb['unavailable_recall']-ra['unavailable_recall']:>+9.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "use_time_is_operating_at_arrival",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "운영 아티팩트 미변경. 승격 판단 아님.",
        "n_samples": int(len(df)),
        "positive_rate": float(df["y"].mean()),
        "split": info,
        "coverage": {
            "known_pct": float(100 * known.mean()),
            "n_open": int((df[EXTRA_COL] == 1).sum()),
            "n_closed": int((df[EXTRA_COL] == 0).sum()),
        },
        "arms": {
            k: {kk: vv for kk, vv in arm.items() if not kk.startswith("_")}
            for k, arm in (("A_baseline", a), ("B_plus_optime", b))
        },
    }
    path = OUT_DIR / "use_time_experiment.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
