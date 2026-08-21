# -*- coding: utf-8 -*-
"""전국 expanding · 시도별 / fallback tier / 콜드스타트 세그먼트 평가."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ARTIFACTS_DIR, MONTHS
from .compare_eval import (
    TIER_DISTRICT_SPEED_WD,
    TIER_GLOBAL_WD,
    TIER_NAMES,
    TIER_SPEED_WD,
    freeze_test_lag_features,
    month_start_ts,
    naive_blend_predict,
)
from .features import load_or_build_features
from .transfer_eval import (
    TARGET,
    ensure_district_key,
    label_cold_start_groups,
    _err_stats,
)

LOW_N = 30


def _lookup_tables(train: pd.DataFrame, geo: str = "district_key"):
    dsw = train.groupby([geo, "speed_class", "weekday"], sort=False)[TARGET].mean()
    spd = train.groupby(["speed_class", "weekday"], sort=False)[TARGET].mean()
    gwd = train.groupby("weekday", sort=False)[TARGET].mean()
    global_mean = float(train[TARGET].mean()) if len(train) else 0.0
    return dsw, spd, gwd, global_mean


def _predict_from_tables(
    test: pd.DataFrame,
    dsw: pd.Series,
    spd: pd.Series,
    gwd: pd.Series,
    global_mean: float,
    geo: str = "district_key",
) -> dict[str, np.ndarray]:
    """동일 테스트 행에 대해 dsw / speed_wd / global_wd 예측."""
    n = len(test)
    dsw_pred = np.full(n, np.nan)
    spd_pred = np.full(n, np.nan)
    gwd_pred = np.full(n, np.nan)
    for i, row in enumerate(test.itertuples()):
        geo_val = getattr(row, geo, "unknown")
        d_key = (geo_val, row.speed_class, int(row.weekday))
        s_key = (row.speed_class, int(row.weekday))
        wd = int(row.weekday)
        if d_key in dsw.index:
            dsw_pred[i] = float(dsw.loc[d_key])
        if s_key in spd.index:
            spd_pred[i] = float(spd.loc[s_key])
        if wd in gwd.index:
            gwd_pred[i] = float(gwd.loc[wd])
        else:
            gwd_pred[i] = global_mean
    # fill missing with coarser
    dsw_f = np.where(np.isfinite(dsw_pred), dsw_pred, np.where(np.isfinite(spd_pred), spd_pred, gwd_pred))
    spd_f = np.where(np.isfinite(spd_pred), spd_pred, gwd_pred)
    return {
        "district_speed_weekday": np.clip(dsw_f, 0, None),
        "speed_weekday": np.clip(spd_f, 0, None),
        "global_weekday": np.clip(gwd_pred, 0, None),
    }


def run_segment_eval(
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict[str, Any]:
    feat = ensure_district_key(load_or_build_features(scope="nationwide"))
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()

    region_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {"y": [], "pred": [], "y_cold": [], "pred_cold": [], "y_fb": [], "pred_fb": [], "stations": set(), "stations_cold": set()}
    )
    tier_acc: dict[str, dict[str, list]] = defaultdict(lambda: {"y": [], "pred": []})
    ablation_acc: dict[str, dict[str, list]] = {
        k: {"y": [], "pred": []}
        for k in ("district_speed_weekday", "speed_weekday", "global_weekday")
    }
    # ablation only on rows where chosen tier was fallback (>= dsw)
    ablation_fb_rows = {k: {"y": [], "pred": []} for k in ablation_acc}

    for i in range(len(MONTHS) - 1):
        test_month = MONTHS[i + 1]
        train_months = MONTHS[: i + 1]
        m_start = month_start_ts(test_month)
        hist = feat[feat["day"] < m_start]
        train_obs = feat[
            feat["year_month"].isin(train_months) & feat["is_observed"]
        ]
        test_raw = feat[
            (feat["year_month"] == test_month) & feat["is_observed"]
        ].copy()
        if train_obs.empty or test_raw.empty:
            continue

        test = freeze_test_lag_features(hist, test_raw, m_start)
        test = label_cold_start_groups(train_obs, test)
        y = test[TARGET].to_numpy(dtype=float)
        pred, _counts, tiers = naive_blend_predict(
            train_obs, test, TARGET, use_speed_weekday=True, return_tiers=True
        )

        dsw, spd, gwd, gmean = _lookup_tables(train_obs)
        ab = _predict_from_tables(test, dsw, spd, gwd, gmean)

        cold_mask = test["cold_group"].isin(["new_station", "sparse"]).to_numpy()
        fb_mask = tiers >= TIER_DISTRICT_SPEED_WD

        for idx in range(len(test)):
            region = str(test.iloc[idx]["region"])
            st = test.iloc[idx]["station_name"]
            region_acc[region]["y"].append(y[idx])
            region_acc[region]["pred"].append(pred[idx])
            region_acc[region]["stations"].add(st)
            if cold_mask[idx]:
                region_acc[region]["y_cold"].append(y[idx])
                region_acc[region]["pred_cold"].append(pred[idx])
                region_acc[region]["stations_cold"].add(st)
            if fb_mask[idx]:
                region_acc[region]["y_fb"].append(y[idx])
                region_acc[region]["pred_fb"].append(pred[idx])

            tname = TIER_NAMES.get(int(tiers[idx]), "unknown")
            tier_acc[tname]["y"].append(y[idx])
            tier_acc[tname]["pred"].append(pred[idx])

        for k, p in ab.items():
            ablation_acc[k]["y"].append(y)
            ablation_acc[k]["pred"].append(p)
            if fb_mask.any():
                ablation_fb_rows[k]["y"].append(y[fb_mask])
                ablation_fb_rows[k]["pred"].append(p[fb_mask])

        print(
            f"[segment] → {test_month} n={len(test)} "
            f"cold={int(cold_mask.sum())} fb={int(fb_mask.sum())}"
        )

    by_region = []
    for region, acc in sorted(region_acc.items(), key=lambda x: x[0]):
        y = np.asarray(acc["y"], dtype=float)
        p = np.asarray(acc["pred"], dtype=float)
        st = _err_stats(y, p)
        yc = np.asarray(acc["y_cold"], dtype=float)
        pc = np.asarray(acc["pred_cold"], dtype=float)
        yf = np.asarray(acc["y_fb"], dtype=float)
        pf = np.asarray(acc["pred_fb"], dtype=float)
        by_region.append(
            {
                "region": region,
                "n_stations": len(acc["stations"]),
                "n_test_rows": st["n"],
                "naive_blend_mae": st["mae"],
                "cold_start_mae": _err_stats(yc, pc)["mae"],
                "cold_start_n": int(len(yc)),
                "fallback_only_mae": _err_stats(yf, pf)["mae"],
                "fallback_only_n": int(len(yf)),
                "low_n": st["n"] < LOW_N or len(acc["stations"]) < 5,
            }
        )

    by_tier = []
    for tname, acc in tier_acc.items():
        st = _err_stats(
            np.asarray(acc["y"], dtype=float), np.asarray(acc["pred"], dtype=float)
        )
        by_tier.append({"tier": tname, **st})

    def _cat_stats(store: dict) -> dict[str, Any]:
        out = {}
        for k, acc in store.items():
            if not acc["y"]:
                out[k] = _err_stats(np.array([]), np.array([]))
                continue
            y = np.concatenate(acc["y"])
            p = np.concatenate(acc["pred"])
            out[k] = _err_stats(y, p)
        return out

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "전국 expanding naive_blend 세그먼트: 시도별·tier별·콜드스타트. "
            "표본 n이 작은 시도는 low_n=true."
        ),
        "target": TARGET,
        "window": "expanding",
        "by_region": by_region,
        "by_tier": by_tier,
        "fallback_ablation_all_rows": _cat_stats(ablation_acc),
        "fallback_ablation_fallback_rows": _cat_stats(ablation_fb_rows),
    }

    out_json = out_json or (
        ARTIFACTS_DIR / "charge_history_segment_eval_nationwide.json"
    )
    out_md = out_md or (
        ARTIFACTS_DIR / "charge_history_segment_eval_nationwide.md"
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[segment_eval] wrote {out_json}")
    print(f"[segment_eval] wrote {out_md}")
    return result


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 전국 세그먼트 평가 (시도 · tier · 콜드스타트)",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        "## 시도별",
        "",
        "| region | n_stations | n_test_rows | blend MAE | cold MAE | cold n | fb-only MAE | low_n |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in result.get("by_region") or []:
        lines.append(
            "| {region} | {ns} | {nr} | {m} | {cm} | {cn} | {fm} | {ln} |".format(
                region=r["region"],
                ns=r["n_stations"],
                nr=r["n_test_rows"],
                m=_fmt(r.get("naive_blend_mae")),
                cm=_fmt(r.get("cold_start_mae")),
                cn=r.get("cold_start_n"),
                fm=_fmt(r.get("fallback_only_mae")),
                ln="Y" if r.get("low_n") else "",
            )
        )

    lines.extend(
        [
            "",
            "## Fallback tier별 (해당 tier가 선택된 행)",
            "",
            "| tier | n | MAE |",
            "|---|---:|---:|",
        ]
    )
    for t in result.get("by_tier") or []:
        lines.append(f"| {t['tier']} | {t['n']} | {_fmt(t.get('mae'))} |")

    lines.extend(
        [
            "",
            "## Fallback ablation (동일 행에 dsw / speed_wd / global_wd)",
            "",
            "### 전체 행",
            "",
            "| predictor | n | MAE |",
            "|---|---:|---:|",
        ]
    )
    for k, st in (result.get("fallback_ablation_all_rows") or {}).items():
        lines.append(f"| {k} | {st.get('n')} | {_fmt(st.get('mae'))} |")
    lines.extend(
        [
            "",
            "### Fallback 적용 행만 (tier ≥ district_speed_weekday)",
            "",
            "| predictor | n | MAE |",
            "|---|---:|---:|",
        ]
    )
    for k, st in (result.get("fallback_ablation_fallback_rows") or {}).items():
        lines.append(f"| {k} | {st.get('n')} | {_fmt(st.get('mae'))} |")

    lines.extend(["", "산출 JSON: `charge_history_segment_eval_nationwide.json`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run_segment_eval()
