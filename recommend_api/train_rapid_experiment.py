"""급속 전용 HGB 실험 (서비스 정본 미승격).

동일 날짜 holdout에서:
  - 통합 모델: train 전체 fit → 급속 test 점수
  - 급속 전용: train 급속만 fit → 동일 급속 test 점수

사용:
  cd f:\\dev\\scheduler
  py -m recommend_api.train_rapid_experiment
  py -m recommend_api.train_rapid_experiment --max-rows 50000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gc

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS, MATCH_TOLERANCE_MIN
from recommend_api.eval_metrics import evaluate_classifier, split_by_date
from recommend_api.holdout_eval import (
    _new_model,
    _per_eta_metrics,
    _prepare_fold,
    save_metrics,
)
from recommend_api.model_store import (
    build_horizon_dataset,
    compute_feature_schema_hash,
    load_joined,
    load_label_series,
    make_xy,
)

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
RAPID_MODEL_PATH = EXPERIMENTS_DIR / "rapid_horizon_hgb.joblib"
COMPARE_PATH = EXPERIMENTS_DIR / "rapid_vs_unified_metrics.json"


def _eta_recall(per_eta: list[dict[str, Any]], eta: int) -> float | None:
    row = next((r for r in per_eta if r["eta_minutes"] == eta), None)
    return None if row is None else row.get("unavailable_recall")


def _score_block(y_true: pd.Series, proba: np.ndarray, eta: pd.Series) -> dict[str, Any]:
    overall = evaluate_classifier(y_true, proba)
    per_eta = _per_eta_metrics(y_true, proba, eta)
    return {
        "n": int(len(y_true)),
        "unavailable_rate": float(1.0 - overall["positive_rate"]),
        "overall": {
            "roc_auc": overall["roc_auc"],
            "pr_auc_unavailable": overall["pr_auc_unavailable"],
            "brier_score": overall["brier_score"],
            "unavailable_recall": overall["unavailable_recall"],
            "unavailable_precision": overall["unavailable_precision"],
            "unavailable_f1": overall["unavailable_f1"],
            "accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
        },
        "per_eta": per_eta,
        "eta_20_unavailable_recall": _eta_recall(per_eta, 20),
        "eta_30_unavailable_recall": _eta_recall(per_eta, 30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train experimental rapid-only HGB vs unified (no service promote)"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    print("JOIN 로드 중...")
    df = load_joined(limit=args.max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise SystemExit("학습 데이터가 없습니다.")

    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). 통합 모델과 급속 전용 모델을
    # 비교하는 스크립트이므로 라벨 경로가 train_and_save 와 달라지면 비교가 무의미하다.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성 중...")
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon 샘플: {len(hz):,} | 사용가능 {hz['y'].mean():.1%}")
    if hz.empty:
        raise SystemExit("horizon 샘플이 없습니다.")

    X, y, _med, meta = make_xy(hz, fill_median=False)

    # hz/df/label_df 는 여기서부터 안 쓴다. 692만 행 x 전체 컬럼(stat_id 등 object,
    # datetime 포함)이라 수 GB를 잡고 있어, 그대로 두면 아래 unified.fit 에서
    # HGB 내부 bool 마스크(5.1M x 104) 할당이 실패한다(2026-08-03 OOM).
    del hz, df, label_df
    gc.collect()

    is_fast = pd.to_numeric(meta["is_fast"], errors="coerce").fillna(0).astype(int)
    print(f"급속 샘플: {int((is_fast == 1).sum()):,} / {len(is_fast):,}")

    is_train, train_dates, test_dates = split_by_date(meta, ratio=0.8)
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise SystemExit("날짜 holdout을 만들 수 없습니다.")

    rapid_test = (~is_train) & (is_fast == 1)
    rapid_train = is_train & (is_fast == 1)
    if int(rapid_test.sum()) == 0:
        raise SystemExit("급속 test 표본이 없습니다.")
    if int(rapid_train.sum()) == 0:
        raise SystemExit("급속 train 표본이 없습니다.")

    print(
        f"holdout: train_days={len(train_dates)} test_days={len(test_dates)}  "
        f"rapid_train={int(rapid_train.sum()):,} rapid_test={int(rapid_test.sum()):,}"
    )

    # --- 통합: train 전체 → 급속 test ---
    print("통합 HGB (train 전체) fit...")
    X_tr, _X_te_all, y_tr, _y_te_all = _prepare_fold(X, y, is_train)
    # _prepare_fold 가 만든 전체 test 사본은 안 쓴다(급속 test 만 따로 뽑는다).
    del _X_te_all, _y_te_all
    gc.collect()
    unified = _new_model()
    unified.fit(X_tr, y_tr)
    del X_tr, y_tr
    gc.collect()
    # score only rapid test rows (same feature columns / median as unified fold)
    med_u = X.loc[is_train].median(numeric_only=True)
    X_rapid_te = X.loc[rapid_test].fillna(med_u)
    y_rapid_te = y.loc[rapid_test]
    eta_rapid_te = meta.loc[rapid_test, "eta_minutes"]
    proba_unified = unified.predict_proba(X_rapid_te)[:, 1]
    unified_block = _score_block(y_rapid_te, proba_unified, eta_rapid_te)
    print(
        f"  unified@rapid: AUC={unified_block['overall']['roc_auc']}  "
        f"불가R={unified_block['overall']['unavailable_recall']}  "
        f"ETA20R={unified_block['eta_20_unavailable_recall']}  "
        f"ETA30R={unified_block['eta_30_unavailable_recall']}"
    )

    # --- 급속 전용: rapid train → 동일 급속 test ---
    print("급속 전용 HGB (rapid train) fit...")
    med_r = X.loc[rapid_train].median(numeric_only=True)
    X_r_tr = X.loc[rapid_train].fillna(med_r)
    y_r_tr = y.loc[rapid_train]
    X_r_te = X.loc[rapid_test].fillna(med_r)
    rapid_model = _new_model()
    rapid_model.fit(X_r_tr, y_r_tr)
    proba_rapid = rapid_model.predict_proba(X_r_te)[:, 1]
    rapid_block = _score_block(y_rapid_te, proba_rapid, eta_rapid_te)
    print(
        f"  rapid-only@rapid: AUC={rapid_block['overall']['roc_auc']}  "
        f"불가R={rapid_block['overall']['unavailable_recall']}  "
        f"ETA20R={rapid_block['eta_20_unavailable_recall']}  "
        f"ETA30R={rapid_block['eta_30_unavailable_recall']}"
    )

    u_o = unified_block["overall"]
    r_o = rapid_block["overall"]
    delta = {
        "roc_auc": None
        if u_o["roc_auc"] is None or r_o["roc_auc"] is None
        else float(r_o["roc_auc"] - u_o["roc_auc"]),
        "unavailable_recall": float(
            r_o["unavailable_recall"] - u_o["unavailable_recall"]
        ),
        "pr_auc_unavailable": None
        if u_o["pr_auc_unavailable"] is None or r_o["pr_auc_unavailable"] is None
        else float(r_o["pr_auc_unavailable"] - u_o["pr_auc_unavailable"]),
        "brier_score": float(r_o["brier_score"] - u_o["brier_score"]),
        "eta_20_unavailable_recall": None
        if unified_block["eta_20_unavailable_recall"] is None
        or rapid_block["eta_20_unavailable_recall"] is None
        else float(
            rapid_block["eta_20_unavailable_recall"]
            - unified_block["eta_20_unavailable_recall"]
        ),
        "eta_30_unavailable_recall": None
        if unified_block["eta_30_unavailable_recall"] is None
        or rapid_block["eta_30_unavailable_recall"] is None
        else float(
            rapid_block["eta_30_unavailable_recall"]
            - unified_block["eta_30_unavailable_recall"]
        ),
    }

    # 실험 artifact (서비스 MODEL_PATH와 분리)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    feature_columns = X_r_tr.columns.tolist()
    base_cols = list(FEATURE_COLS)
    schema_hash = compute_feature_schema_hash(base_cols, feature_columns)
    artifact = {
        "model": rapid_model,
        "feature_columns": feature_columns,
        "base_feature_cols": base_cols,
        "feature_schema_hash": schema_hash,
        "medians": med_r.to_dict(),
        "trained_at": now_utc.isoformat(),
        "model_version": now_utc.strftime("%Y%m%dT%H%M%SZ") + "_rapid_exp",
        "model_name": "HistGradientBoosting_rapid_only",
        "horizons": HORIZONS,
        "n_samples": int(len(X_r_tr)),
        "positive_rate": float(y_r_tr.mean()),
        "experiment_only": True,
        "not_service_canonical": True,
        "speed_definition": "is_fast = (output_kw >= 50)",
    }
    joblib.dump(artifact, RAPID_MODEL_PATH)
    print(f"실험 모델 저장: {RAPID_MODEL_PATH}")

    compare = {
        "generated_at": now_utc.isoformat(),
        "match_tolerance_min": MATCH_TOLERANCE_MIN,
        "speed_definition": "is_fast = (output_kw >= 50)",
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
        "n_rapid_train": int(rapid_train.sum()),
        "n_rapid_test": int(rapid_test.sum()),
        "service_promotion": False,
        "promotion_note": (
            "급속 전용 성능이 명확히 우수하고 Rolling에서도 유지될 때만 "
            "rapid_horizon_hgb 서비스 승격 검토. 이번 실행은 실험만."
        ),
        "unified_on_rapid_holdout": unified_block,
        "rapid_only_on_rapid_holdout": rapid_block,
        "delta_rapid_minus_unified": delta,
        "experiment_model_path": str(RAPID_MODEL_PATH),
        "experiment_model_version": artifact["model_version"],
    }
    save_metrics(compare, COMPARE_PATH)
    print(f"비교 저장: {COMPARE_PATH}")
    print(
        "delta(rapid-unified): "
        f"AUC={delta['roc_auc']}  불가R={delta['unavailable_recall']}  "
        f"ETA20R={delta['eta_20_unavailable_recall']}  "
        f"ETA30R={delta['eta_30_unavailable_recall']}"
    )


if __name__ == "__main__":
    main()
