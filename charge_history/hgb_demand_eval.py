# -*- coding: utf-8 -*-
"""수요 전용 HGB challenger (ETA horizon_hgb와 분리).

station_name one-hot 금지. 전체 / new_station / sparse MAE 비교.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from . import ARTIFACTS_DIR, MONTHS
from .compare_eval import (
    freeze_test_lag_features,
    month_start_ts,
    naive_blend_predict,
    ridge_predict,
)
from .features import load_or_build_features
from .transfer_eval import (
    TARGET,
    ensure_district_key,
    label_cold_start_groups,
    _err_stats,
)

CAT_COLS = ["region", "district_key", "speed_class"]
NUM_COLS = [
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "is_long_weekend",
    "power_kw",
    "sessions_rolling_4w_same_weekday",
    "sessions_lag_7d",
    "sessions_lag_14d",
    "sessions_lag_28d",
    "sessions_recent_trend",
    "temperature_mean",
    "precipitation",
    "weather_missing",
    "station_mean",
    "train_session_count",
]


def _add_station_stats(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    st_mean = train.groupby("station_name")[TARGET].mean()
    st_cnt = train.groupby("station_name")[TARGET].sum()
    global_mean = float(train[TARGET].mean()) if len(train) else 0.0
    out["station_mean"] = out["station_name"].map(st_mean).fillna(global_mean)
    out["train_session_count"] = out["station_name"].map(st_cnt).fillna(0.0)
    return out


def _design_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    tr = train.copy()
    te = test.copy()
    for c in CAT_COLS + NUM_COLS:
        if c not in tr.columns:
            tr[c] = np.nan if c in NUM_COLS else "unknown"
        if c not in te.columns:
            te[c] = np.nan if c in NUM_COLS else "unknown"
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_tr = enc.fit_transform(tr[CAT_COLS].astype(str))
    X_cat_te = enc.transform(te[CAT_COLS].astype(str))

    def num_block(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in NUM_COLS:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            if not np.isfinite(med):
                med = 0.0
            v = np.where(np.isfinite(v), v, med)
            cols.append(v)
        return np.column_stack(cols)

    return (
        np.hstack([X_cat_tr, num_block(tr)]),
        np.hstack([X_cat_te, num_block(te)]),
    )


def run_hgb_demand_eval(
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
    max_train: int = 300_000,
) -> dict[str, Any]:
    feat = ensure_district_key(load_or_build_features(scope="nationwide"))
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()
    if "month" not in feat.columns:
        feat["month"] = feat["day"].dt.month

    stores: dict[str, dict[str, list]] = {
        g: {m: {"y": [], "pred": []} for m in ("naive_blend", "ridge_seasonal", "hgb")}
        for g in ("all", "new_station", "sparse")
    }

    for i in range(len(MONTHS) - 1):
        test_month = MONTHS[i + 1]
        train_months = MONTHS[: i + 1]
        m_start = month_start_ts(test_month)
        hist = feat[feat["day"] < m_start]
        train_obs = feat[
            feat["year_month"].isin(train_months) & feat["is_observed"]
        ].copy()
        test_raw = feat[
            (feat["year_month"] == test_month) & feat["is_observed"]
        ].copy()
        if train_obs.empty or test_raw.empty:
            continue

        test = freeze_test_lag_features(hist, test_raw, m_start)
        test = label_cold_start_groups(train_obs, test)
        train_obs = _add_station_stats(train_obs, train_obs)
        test = _add_station_stats(train_obs, test)
        y = test[TARGET].to_numpy(dtype=float)

        blend, _ = naive_blend_predict(
            train_obs, test, TARGET, use_speed_weekday=True
        )
        tr_ridge = train_obs
        if len(tr_ridge) > max_train:
            tr_ridge = tr_ridge.sample(n=max_train, random_state=42)
        ridge = ridge_predict(tr_ridge, test, TARGET, variant="seasonal")

        tr_hgb = train_obs
        if len(tr_hgb) > max_train:
            tr_hgb = tr_hgb.sample(n=max_train, random_state=42)
        X_tr, X_te = _design_matrix(tr_hgb, test)
        model = HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=120,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(X_tr, tr_hgb[TARGET].to_numpy(dtype=float))
        hgb = np.clip(model.predict(X_te), 0, None)

        preds = {"naive_blend": blend, "ridge_seasonal": ridge, "hgb": hgb}
        masks = {
            "all": np.ones(len(test), dtype=bool),
            "new_station": (test["cold_group"] == "new_station").to_numpy(),
            "sparse": (test["cold_group"] == "sparse").to_numpy(),
        }
        for g, mask in masks.items():
            if not mask.any():
                continue
            for mname, p in preds.items():
                stores[g][mname]["y"].append(y[mask])
                stores[g][mname]["pred"].append(p[mask])

        print(
            f"[hgb_demand] → {test_month} n={len(test)} "
            f"blend={float(np.mean(np.abs(y - blend))):.3f} "
            f"ridge={float(np.mean(np.abs(y - ridge))):.3f} "
            f"hgb={float(np.mean(np.abs(y - hgb))):.3f}"
        )

    summary: dict[str, Any] = {}
    for g, models in stores.items():
        summary[g] = {}
        for mname, acc in models.items():
            if not acc["y"]:
                summary[g][mname] = _err_stats(np.array([]), np.array([]))
                continue
            summary[g][mname] = _err_stats(
                np.concatenate(acc["y"]), np.concatenate(acc["pred"])
            )

    # promotion gate
    new_b = (summary.get("new_station") or {}).get("naive_blend", {}).get("mae")
    new_h = (summary.get("new_station") or {}).get("hgb", {}).get("mae")
    promote = bool(
        new_b is not None and new_h is not None and new_h < new_b * 0.98
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "수요 전용 HGB challenger. station_name one-hot 없음. "
            "ETA horizon_hgb와 별개. train 샘플 상한 "
            f"{max_train}."
        ),
        "target": TARGET,
        "window": "expanding",
        "summary": summary,
        "promotion": {
            "eligible": promote,
            "rule": "new_station에서 HGB MAE가 naive_blend보다 ≥2% 개선",
            "verdict": (
                "HGB를 수요 기본 모델로 검토할 수 있음"
                if promote
                else "신규 충전소에서 HGB가 blend를 이기지 못함 → 승격 없음"
            ),
        },
    }

    out_json = out_json or (ARTIFACTS_DIR / "charge_history_hgb_demand_eval.json")
    out_md = out_md or (ARTIFACTS_DIR / "charge_history_hgb_demand_eval.md")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[hgb_demand] wrote {out_json}")
    print(f"[hgb_demand] wrote {out_md}")
    return result


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 수요 HGB Challenger 평가",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        f"**승격:** {(result.get('promotion') or {}).get('verdict')}",
        "",
        "| group | naive_blend | ridge_seasonal | HGB |",
        "|---|---:|---:|---:|",
    ]
    for g in ("all", "new_station", "sparse"):
        block = (result.get("summary") or {}).get(g) or {}
        lines.append(
            "| {g} | {b} | {r} | {h} |".format(
                g=g,
                b=_fmt((block.get("naive_blend") or {}).get("mae")),
                r=_fmt((block.get("ridge_seasonal") or {}).get("mae")),
                h=_fmt((block.get("hgb") or {}).get("mae")),
            )
        )
    lines.extend(["", "산출 JSON: `charge_history_hgb_demand_eval.json`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run_hgb_demand_eval()
