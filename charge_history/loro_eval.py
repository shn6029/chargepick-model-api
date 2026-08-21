# -*- coding: utf-8 -*-
"""Leave-one-region-out: 핵심 5개 시도."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ARTIFACTS_DIR, MONTHS
from .compare_eval import freeze_test_lag_features, month_start_ts, naive_blend_predict
from .features import load_or_build_features
from .transfer_eval import (
    TARGET,
    ensure_district_key,
    label_cold_start_groups,
    _err_stats,
    _pct_improve,
)

LORO_REGIONS = [
    "서울특별시",
    "경기도",
    "강원특별자치도",
    "제주특별자치도",
    "대구광역시",
]


def run_loro_eval(
    *,
    regions: list[str] | None = None,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict[str, Any]:
    regions = regions or list(LORO_REGIONS)
    feat = ensure_district_key(load_or_build_features(scope="nationwide"))
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()

    by_region: dict[str, Any] = {}

    for region in regions:
        print(f"[loro] region={region}")
        held = feat["region"].astype(str) == region
        if not held.any():
            by_region[region] = {"skipped": True, "reason": "region not found"}
            continue

        pool_y: list[np.ndarray] = []
        pool_pred: list[np.ndarray] = []
        pool_y_cold: list[np.ndarray] = []
        pool_pred_cold: list[np.ndarray] = []
        # baseline: train includes region (in-region train) for comparison
        pool_y_in: list[np.ndarray] = []
        pool_pred_in: list[np.ndarray] = []

        fold_rows = []
        for i in range(len(MONTHS) - 1):
            test_month = MONTHS[i + 1]
            train_months = MONTHS[: i + 1]
            m_start = month_start_ts(test_month)
            hist = feat[feat["day"] < m_start]
            train_all = feat[
                feat["year_month"].isin(train_months) & feat["is_observed"]
            ]
            train_loo = train_all[train_all["region"].astype(str) != region]
            test_raw = feat[
                (feat["year_month"] == test_month)
                & feat["is_observed"]
                & (feat["region"].astype(str) == region)
            ].copy()
            if train_loo.empty or test_raw.empty:
                continue

            test = freeze_test_lag_features(hist, test_raw, m_start)
            # cold groups relative to in-region history would be unfair;
            # use held-out train absence: stations never seen in train_loo
            test = label_cold_start_groups(train_loo, test)
            y = test[TARGET].to_numpy(dtype=float)

            pred_loo, _, _ = naive_blend_predict(
                train_loo, test, TARGET, use_speed_weekday=True, return_tiers=True
            )
            pred_in, _, _ = naive_blend_predict(
                train_all, test, TARGET, use_speed_weekday=True, return_tiers=True
            )

            cold = test["cold_group"].isin(["new_station", "sparse"]).to_numpy()
            pool_y.append(y)
            pool_pred.append(pred_loo)
            pool_y_in.append(y)
            pool_pred_in.append(pred_in)
            if cold.any():
                pool_y_cold.append(y[cold])
                pool_pred_cold.append(pred_loo[cold])

            fold_rows.append(
                {
                    "test_month": test_month,
                    "n_test": int(len(test)),
                    "n_cold": int(cold.sum()),
                    "loo_mae": float(np.mean(np.abs(y - pred_loo))),
                    "inregion_mae": float(np.mean(np.abs(y - pred_in))),
                }
            )
            print(
                f"  [{region}] → {test_month} n={len(test)} "
                f"loo={fold_rows[-1]['loo_mae']:.3f} "
                f"in={fold_rows[-1]['inregion_mae']:.3f}"
            )

        if not pool_y:
            by_region[region] = {"skipped": True, "reason": "no folds"}
            continue

        y = np.concatenate(pool_y)
        pred = np.concatenate(pool_pred)
        y_in = np.concatenate(pool_y_in)
        pred_in = np.concatenate(pool_pred_in)
        loo = _err_stats(y, pred)
        inn = _err_stats(y_in, pred_in)
        if pool_y_cold:
            yc = np.concatenate(pool_y_cold)
            pc = np.concatenate(pool_pred_cold)
            cold_stats = _err_stats(yc, pc)
        else:
            cold_stats = _err_stats(np.array([]), np.array([]))

        by_region[region] = {
            "skipped": False,
            "loo": loo,
            "inregion_train": inn,
            "cold_start_loo": cold_stats,
            "loo_vs_inregion_mae_improve": _pct_improve(inn.get("mae"), loo.get("mae")),
            "folds": fold_rows,
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Leave-one-region-out: 해당 시도 제외 학습 → 해당 시도 예측. "
            "inregion_train은 비교용(해당 시도 포함 학습)."
        ),
        "target": TARGET,
        "window": "expanding",
        "regions": regions,
        "by_region": by_region,
    }

    out_json = out_json or (ARTIFACTS_DIR / "charge_history_loro_eval.json")
    out_md = out_md or (ARTIFACTS_DIR / "charge_history_loro_eval.md")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[loro_eval] wrote {out_json}")
    print(f"[loro_eval] wrote {out_md}")
    return result


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    return f"{100.0 * float(v):+.1f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Leave-one-region-out 평가",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        "| region | loo MAE | in-region MAE | cold loo MAE | loo vs in |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in result.get("regions") or []:
        b = (result.get("by_region") or {}).get(region) or {}
        if b.get("skipped"):
            lines.append(f"| {region} | — | — | — | skipped |")
            continue
        lines.append(
            "| {r} | {loo} | {inn} | {cold} | {imp} |".format(
                r=region,
                loo=_fmt((b.get("loo") or {}).get("mae")),
                inn=_fmt((b.get("inregion_train") or {}).get("mae")),
                cold=_fmt((b.get("cold_start_loo") or {}).get("mae")),
                imp=_fmt_pct(b.get("loo_vs_inregion_mae_improve")),
            )
        )
    lines.extend(["", "산출 JSON: `charge_history_loro_eval.json`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run_loro_eval()
