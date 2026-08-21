"""[Deprecated wrapper] API 정본 모델 학습으로 위임.

정본:
  py -m recommend_api.train
  → recommend_api/artifacts/horizon_hgb.joblib
  → recommend_api/artifacts/horizon_hgb_metrics.json

이 스크립트는 호환을 위해 동일 학습을 호출합니다.
루트 artifacts/availability_model.joblib 은 더 이상 갱신하지 않습니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.model_store import train_and_save


def main() -> None:
    print(
        "[안내] train_availability.py 는 recommend_api.train 래퍼입니다.\n"
        "  정본 모델: recommend_api/artifacts/horizon_hgb.joblib\n"
        "  권장 명령: py -m recommend_api.train\n"
        "  위치: scripts/etl/train_availability.py"
    )
    parser = argparse.ArgumentParser(
        description="Wrapper → recommend_api.train (canonical horizon_hgb)"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    result = train_and_save(max_rows=args.max_rows, skip_eval=args.skip_eval)
    print("완료:", result)


if __name__ == "__main__":
    main()
