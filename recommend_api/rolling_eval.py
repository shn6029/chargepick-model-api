"""Rolling 날짜 평가만 재실행해 metrics JSON의 rolling 섹션을 갱신.

모델 파일은 변경하지 않음.

사용:
  py -m recommend_api.rolling_eval
  py -m recommend_api.rolling_eval --max-rows 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import HORIZONS, METRICS_PATH, MODEL_PATH
from recommend_api.holdout_eval import evaluate_rolling_folds, save_metrics
from recommend_api.model_store import (
    build_horizon_dataset,
    load_artifact,
    load_joined,
    load_label_series,
    make_xy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh rolling holdout metrics")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    print("JOIN 로드 중...")
    df = load_joined(limit=args.max_rows)
    print(f"JOIN: {len(df):,}행")
    # 라벨 원천은 정본 학습과 맞춘다(delta+snapshot). model_store.load_label_series 참고.
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()
    hz = build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"horizon 샘플: {len(hz):,}")
    X, y, _med, meta = make_xy(hz, fill_median=False)

    print("Rolling fold 평가 중...")
    rolling = evaluate_rolling_folds(X, y, meta, min_train_days=2)

    model_version = None
    if MODEL_PATH.exists():
        try:
            model_version = load_artifact().get("model_version")
        except Exception:
            model_version = None

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = {"model_version": model_version}

    if model_version:
        metrics["model_version"] = model_version
    metrics["rolling"] = rolling
    metrics["rolling_evaluated_at"] = datetime.now(timezone.utc).isoformat()
    save_metrics(metrics, METRICS_PATH)

    if rolling:
        s = rolling["summary"]["roc_auc"]
        print(
            f"완료: folds={rolling['n_folds']}  "
            f"AUC mean={s['mean']} std={s['std']}  → {METRICS_PATH}"
        )
    else:
        print(f"완료: rolling 결과 없음 → {METRICS_PATH}")


if __name__ == "__main__":
    main()
