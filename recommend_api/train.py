"""모델 학습·평가 후 artifacts/horizon_hgb.joblib 저장 (API 정본)

사용:
  cd f:\\dev\\scheduler
  py -m recommend_api.train
  py -m recommend_api.train --max-rows 30000   # 빠른 테스트 (배선 확인 전용)
  py -m recommend_api.train --skip-eval        # 평가 생략 후 전체 fit만

--max-rows 로 만든 모델을 정본으로 쓰지 말 것. LIMIT 이 stat_id 정렬 뒤에 붙어
무작위 표본이 아니라 '앞쪽 충전기만' 남는다(사업자 통째로 소멸). 상세는
model_store.load_joined docstring 참고.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.model_store import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train canonical horizon HGB model (+ holdout eval)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "JOIN 로드 행 수 제한(빠른 테스트용). 생략 시 전체. "
            "무작위 표본이 아니라 stat_id 앞쪽만 남으므로 정본 학습에 쓰지 말 것 "
            "— 2026-08-03 400000 으로 사업자 61→40 소멸."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="날짜 홀드아웃/rolling/calibration 평가 생략",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help=(
            "저메모리 단계 적재 경로로 학습(recommend_api/staged_training.py). "
            "지평을 하나씩 전개해 parquet 으로 내리고 float32 memmap 으로 적합한다. "
            "같은 데이터면 기본 경로와 **같은 모델**이 나오지만, 평가는 "
            "date_holdout 까지만 한다(rolling/logistic 생략). "
            "RAM 이 부족할 때만 쓸 것 — 가용 메모리에 따라 자동 전환하지 않는다. "
            "그러면 크롬을 켜뒀는지에 따라 산출 모델이 달라져 재현이 안 된다."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        metavar="DIR",
        help="--staged 중간 산출(parts·memmap) 위치. 지정하면 지우지 않아 재개할 수 있다.",
    )
    parser.add_argument(
        "--matrix-dtype",
        choices=["float64", "float32"],
        default="float64",
        help=(
            "--staged 설계행렬 memmap 의 dtype. 기본 float64 — sklearn 1.9.0 HGB 가 "
            "float64 만 제로카피로 읽어서, float32 로 두면 적합 때 RAM 에 전량 "
            "업캐스트 복사한다(2026-08-18 실측 피크 23GB vs 11GB). float32 는 디스크를 "
            "절반으로 줄이므로 **RAM 이 아니라 디스크가 부족할 때만** 쓸 것."
        ),
    )
    args = parser.parse_args()
    if args.staged:
        from recommend_api.staged_training import train_and_save_staged

        result = train_and_save_staged(
            max_rows=args.max_rows,
            skip_eval=args.skip_eval,
            work_dir=args.work_dir,
            matrix_dtype=args.matrix_dtype,
        )
    else:
        result = train_and_save(max_rows=args.max_rows, skip_eval=args.skip_eval)
    print("완료:", result)


if __name__ == "__main__":
    main()
