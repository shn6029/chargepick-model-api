"""로컬 `recommend_api/` 의 코드 지문을 출력한다. 서버 /health 값과 대조용.

    로컬:  py scripts/code_fingerprint.py
    서버:  curl -s localhost:8000/health | tr ',' '\n' | grep code_fingerprint

두 값이 다르면 서버와 로컬 코드가 어긋난 것이다. 서버 `~/scheduler` 는 git 클론이
아니라 파일만 손으로 올라간 상태라, 커밋 해시로는 대조할 수 없어서 내용 해시를 쓴다.
계산 규칙은 recommend_api/code_stamp.py docstring 참고.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.code_stamp import code_file_count, code_fingerprint  # noqa: E402


def main() -> None:
    pkg = ROOT / "recommend_api"
    print(f"code_fingerprint: {code_fingerprint(pkg)}")
    print(f"files:            {code_file_count(pkg)}")
    print(f"dir:              {pkg}")


if __name__ == "__main__":
    main()
