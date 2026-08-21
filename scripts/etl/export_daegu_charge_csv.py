# -*- coding: utf-8 -*-
"""대구 충전이력 → 분석용 CSV (charge_history ingest 위임)."""

from pathlib import Path

from charge_history.ingest import ingest_month

OUT_DIR = Path(r"f:\dev\scheduler")
OUT_CSV = OUT_DIR / "daegu_charge_history_202605.csv"
OUT_DL = Path(r"f:\Users\상현\Downloads") / "daegu_charge_history_202605.csv"


def main() -> None:
    out = ingest_month("2026-05")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    out.to_csv(OUT_DL, index=False, encoding="utf-8-sig")
    print(f"rows={len(out):,}")
    print(f"saved={OUT_CSV}")
    print(f"saved={OUT_DL}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
