# -*- coding: utf-8 -*-
"""CLI: py -m charge_history <command> [--scope daegu|nationwide]"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import DEFAULT_DATA_RAW, DEFAULT_DOWNLOADS, scope_prefix
from .compare_eval import run_compare_eval
from .expanding_eval import run_expanding_eval
from .features import load_or_build_features
from .hgb_demand_eval import run_hgb_demand_eval
from .ingest import ingest_all, load_sessions
from .loro_eval import run_loro_eval
from .panel import load_or_build_panel
from .overpredict_monitor import run_overpredict_monitor
from .segment_eval import run_segment_eval
from .transfer_eval import run_transfer_eval
from .weather_asos import backfill_asos
from .weather_extreme_eval import run_weather_extreme_eval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Charge-history ingest & monthly demand eval"
    )
    parser.add_argument(
        "command",
        choices=[
            "ingest",
            "panel",
            "weather",
            "features",
            "expanding_eval",
            "compare_eval",
            "transfer_eval",
            "segment_eval",
            "loro_eval",
            "weather_extreme_eval",
            "hgb_demand_eval",
            "overpredict_monitor",
            "all",
        ],
    )
    parser.add_argument(
        "--scope",
        choices=["daegu", "nationwide"],
        default="daegu",
        help="daegu=지역 필터, nationwide=전국(필터 없음)",
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=None,
        help="xlsx 폴더 (기본: daegu→Downloads, nationwide→data/raw)",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    args = parser.parse_args(argv)

    scope = args.scope
    downloads = args.downloads
    if downloads is None:
        downloads = (
            DEFAULT_DATA_RAW
            if scope_prefix(scope) == "nationwide"
            else DEFAULT_DOWNLOADS
        )

    if args.command in ("ingest", "all"):
        ingest_all(downloads=downloads, force=args.force, scope=scope)
    if args.command in ("panel", "all"):
        load_or_build_panel(
            force=args.force or args.command == "all",
            scope=scope,
            downloads=downloads,
        )
    if args.command in ("weather", "all"):
        backfill_asos(force=args.force, scope=scope_prefix(scope))
    if args.command in ("features", "all"):
        load_or_build_features(
            force=args.force or args.command == "all",
            scope=scope,
        )
    if args.command in ("expanding_eval", "all"):
        if args.command == "expanding_eval" and args.force:
            load_sessions(force_ingest=False, scope=scope, downloads=downloads)
            load_or_build_panel(force=True, scope=scope, downloads=downloads)
        run_expanding_eval(scope=scope)
    if args.command == "compare_eval":
        if args.force:
            backfill_asos(force=False, scope=scope_prefix(scope))
            load_or_build_panel(force=True, scope=scope, downloads=downloads)
            load_or_build_features(force=True, scope=scope)
        run_compare_eval(force_features=False, scope=scope)
    if args.command == "all":
        run_compare_eval(force_features=False, scope=scope)

    if args.command == "transfer_eval":
        run_transfer_eval()
    if args.command == "segment_eval":
        run_segment_eval()
    if args.command == "loro_eval":
        run_loro_eval()
    if args.command == "weather_extreme_eval":
        run_weather_extreme_eval()
    if args.command == "hgb_demand_eval":
        run_hgb_demand_eval()
    if args.command == "overpredict_monitor":
        run_overpredict_monitor(scope=scope, top_n=20)


if __name__ == "__main__":
    main()
