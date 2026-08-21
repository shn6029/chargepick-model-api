"""현재 joblib/metrics에 스키마 해시·git 등 메타만 스탬프 (재학습 없음).

사용:
  py -m recommend_api.stamp_artifact_meta
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import (
    DURATION_CAP_MIN,
    FEATURE_COLS,
    LONG_STATE_DURATION_MIN,
    METRICS_PATH,
    MODEL_PATH,
    STALE_UPDATE_AGE_MIN,
)
from recommend_api.model_store import (
    compute_feature_schema_hash,
    get_git_commit,
    validate_artifact_features,
)


def stamp() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    artifact = joblib.load(MODEL_PATH)
    feature_columns = list(artifact.get("feature_columns") or [])
    base_cols = list(FEATURE_COLS)
    # 구 artifact: 앞부분이 FEATURE_COLS와 맞는지 확인
    msgs = validate_artifact_features(
        {**artifact, "base_feature_cols": None}, strict=False
    )
    schema_hash = compute_feature_schema_hash(base_cols, feature_columns)
    git_commit = get_git_commit()

    artifact["base_feature_cols"] = base_cols
    artifact["feature_schema_hash"] = schema_hash
    artifact["feature_count"] = len(feature_columns)
    artifact["git_commit"] = git_commit or artifact.get("git_commit")
    artifact["stale_threshold_min"] = STALE_UPDATE_AGE_MIN
    artifact["stale_update_age_min"] = STALE_UPDATE_AGE_MIN
    artifact["duration_cap_min"] = DURATION_CAP_MIN
    artifact["long_state_duration_min"] = LONG_STATE_DURATION_MIN
    if "training_row_count" not in artifact and artifact.get("n_samples"):
        # horizon samples ≠ raw; leave training_row_count for full retrain
        pass

    joblib.dump(artifact, MODEL_PATH)

    metrics: dict = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics["feature_schema_hash"] = schema_hash
    metrics["feature_count"] = len(feature_columns)
    metrics["n_features"] = metrics.get("n_features") or len(feature_columns)
    metrics["base_feature_cols"] = base_cols
    metrics["git_commit"] = git_commit or metrics.get("git_commit")
    metrics["stale_threshold_min"] = STALE_UPDATE_AGE_MIN
    metrics["duration_cap_min"] = DURATION_CAP_MIN
    metrics["long_state_duration_min"] = LONG_STATE_DURATION_MIN
    if artifact.get("model_version"):
        metrics["model_version"] = artifact["model_version"]
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    out = {
        "model_path": str(MODEL_PATH),
        "model_version": artifact.get("model_version"),
        "feature_schema_hash": schema_hash,
        "feature_count": len(feature_columns),
        "git_commit": git_commit,
        "validate_warnings": msgs,
    }
    print(out)
    return out


def main() -> None:
    stamp()


if __name__ == "__main__":
    main()
