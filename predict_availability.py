"""호환 래퍼: scripts/etl/predict_availability.py 실행."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "etl" / "predict_availability.py"
    runpy.run_path(str(target), run_name="__main__")
