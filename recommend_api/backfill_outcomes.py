"""예측 로그에 실제 미래 상태 백필.

라벨 규칙은 학습과 동일한 LOCF 다(config.LABEL_METHOD / LABEL_MAX_STALENESS_MIN).

사용:
  py -m recommend_api.backfill_outcomes
  py -m recommend_api.backfill_outcomes --limit 2000
  py -m recommend_api.backfill_outcomes --relabel          # 방식 변경 후 1회 전량 재라벨
  py -m recommend_api.backfill_outcomes --method asof_nearest  # 구방식 비교용
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.prediction_log import backfill_outcomes, ensure_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill actual_stat on prediction logs")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--method",
        choices=["locf", "asof_nearest"],
        default=None,
        help="기본값은 config.LABEL_METHOD(학습과 동일). asof_nearest 는 구방식 비교용",
    )
    parser.add_argument(
        "--max-staleness-min",
        type=int,
        default=None,
        help="locf 로 상태를 끌고 올 최대 나이(분). 기본 config.LABEL_MAX_STALENESS_MIN",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="다른 방식으로 라벨된 행까지 전부 재라벨(방식 변경 직후 1회)",
    )
    args = parser.parse_args()
    ensure_table()
    result = backfill_outcomes(
        limit=args.limit,
        method=args.method,
        max_staleness_min=args.max_staleness_min,
        relabel=args.relabel,
    )
    print("백필 완료:", result)


if __name__ == "__main__":
    main()
