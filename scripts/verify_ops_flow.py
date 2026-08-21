"""Thin wrapper: py scripts/verify_ops_flow.py → recommend_api.verify_ops."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_module("recommend_api.verify_ops", run_name="__main__")
