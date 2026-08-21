"""배포된 코드가 무엇인지 밖에서 확인하는 수단.

왜 필요한가
----------
서버 `~/scheduler` 는 **git 클론이 아니라 파일만 손으로 올라간 상태**다. 그래서
"지금 서버가 어느 코드를 돌고 있나"를 물을 방법이 없었고, 2026-08-03~04 에만
세 번 어긋났다(compose 잔존 인자, 디렉터리 통째 누락, 편집이 로컬에만 반영).

`/health` 에 `git_commit` 이 이미 있었지만 그건 **배포 코드가 아니라 모델 학습
시점 커밋**이다(`horizon_hgb.joblib` 의 필드를 그대로 읽는다). 2026-08-12 확인
시점의 값은 `8090f93` = "first commit" 으로, 재빌드해도 안 바뀌었다. 즉 드리프트
탐지에는 아무 쓸모가 없었다.

두 값을 노출한다
--------------
`code_commit`      빌드 시 `--build-arg CODE_COMMIT` 으로 박은 커밋. **주장값**이다.
                   빌드할 때 인자를 안 주면 `unknown` 이고, 잘못 주면 잘못 나온다.

`code_fingerprint` 컨테이너 안 `recommend_api/*.py` 를 실제로 읽어 만든 해시.
                   **거짓말을 못 한다.** 같은 값을 로컬에서도 뽑을 수 있으므로
                   (`py scripts/code_fingerprint.py`) 양쪽을 대조하면 끝난다.

    서버:   curl -s localhost:8000/health | grep code_fingerprint
    로컬:   py scripts/code_fingerprint.py

    두 값이 다르면 서버와 로컬이 어긋난 것이다. `git_commit` 과 달리
    "누가 무엇을 박았는지" 와 무관하게 파일 내용만 본다.

지문 계산 규칙
------------
- 대상은 `recommend_api/` 아래 `*.py` 전부. 서빙에 안 쓰는 `experiment_*` 도
  **포함한다** — 부분 복사가 드리프트의 원인이었으므로 "디렉터리 전체가 같은가"를
  묻는 게 맞다. 서버에도 디렉터리째 올릴 것.
- `__pycache__` 는 뺀다(런타임 산물).
- 개행은 `\r` 을 지우고 계산한다. Windows 체크아웃은 CRLF, 컨테이너는 LF 라
  정규화하지 않으면 내용이 같아도 값이 갈린다.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def code_commit() -> str | None:
    """빌드 시 박힌 커밋. 안 박았으면 None."""
    v = (os.getenv("CODE_COMMIT") or "").strip()
    if not v or v == "unknown":
        return None
    return v


@lru_cache(maxsize=1)
def code_fingerprint(pkg_dir: Path | None = None) -> str:
    """`recommend_api/*.py` 내용으로 만든 12자 해시. 개행 정규화 후 계산."""
    root = Path(pkg_dir) if pkg_dir else _PKG_DIR
    h = hashlib.sha256()
    files = sorted(
        (p for p in root.rglob("*.py") if "__pycache__" not in p.parts),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        body = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(body).digest())
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def code_file_count(pkg_dir: Path | None = None) -> int:
    """지문에 들어간 파일 수. 개수만 달라도 부분 복사를 의심할 수 있다."""
    root = Path(pkg_dir) if pkg_dir else _PKG_DIR
    return sum(1 for p in root.rglob("*.py") if "__pycache__" not in p.parts)
