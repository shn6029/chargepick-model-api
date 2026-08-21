# -*- coding: utf-8 -*-
"""극단 기상일 ablation: naive_blend vs ridge_seasonal (weather on/off)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ARTIFACTS_DIR, MONTHS
from .compare_eval import (
    freeze_test_lag_features,
    month_start_ts,
    naive_blend_predict,
    ridge_predict,
)
from .features import load_or_build_features
from .transfer_eval import TARGET, ensure_district_key, _err_stats


def _extreme_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return named boolean masks aligned to df rows."""
    temp = pd.to_numeric(df.get("temperature_mean"), errors="coerce")
    precip = pd.to_numeric(df.get("precipitation"), errors="coerce").fillna(0.0)
    masks: dict[str, np.ndarray] = {}
    masks["precip_any"] = (precip > 0).to_numpy()
    if precip.notna().any():
        q95 = float(precip.quantile(0.95))
        masks["precip_top5pct"] = (precip >= q95).to_numpy()
    if temp.notna().any():
        q05 = float(temp.quantile(0.05))
        q95t = float(temp.quantile(0.95))
        masks["temp_bottom5pct"] = (temp <= q05).to_numpy()
        masks["temp_top5pct"] = (temp >= q95t).to_numpy()
    return masks


def _zero_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in (
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "precipitation",
    ):
        if c in out.columns:
            out[c] = 0.0
    if "weather_missing" in out.columns:
        out["weather_missing"] = 1
    return out


def run_weather_extreme_eval(
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict[str, Any]:
    feat = ensure_district_key(load_or_build_features(scope="nationwide"))
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()

    # pool per mask × model
    stores: dict[str, dict[str, list]] = {}

    def _add(mask_name: str, model: str, y: np.ndarray, p: np.ndarray) -> None:
        stores.setdefault(mask_name, {}).setdefault(model, {"y": [], "pred": []})
        stores[mask_name][model]["y"].append(y)
        stores[mask_name][model]["pred"].append(p)

    for i in range(len(MONTHS) - 1):
        test_month = MONTHS[i + 1]
        train_months = MONTHS[: i + 1]
        m_start = month_start_ts(test_month)
        hist = feat[feat["day"] < m_start]
        train_dense = feat[feat["year_month"].isin(train_months)].copy()
        train_obs = train_dense[train_dense["is_observed"]]
        test_raw = feat[
            (feat["year_month"] == test_month) & feat["is_observed"]
        ].copy()
        if train_obs.empty or test_raw.empty:
            continue

        test = freeze_test_lag_features(hist, test_raw, m_start)
        y = test[TARGET].to_numpy(dtype=float)

        blend, _ = naive_blend_predict(
            train_obs, test, TARGET, use_speed_weekday=True
        )
        # subsample train_dense for ridge speed: observed-only is enough & faster
        tr_ridge = train_obs
        if len(tr_ridge) > 400_000:
            tr_ridge = tr_ridge.sample(n=400_000, random_state=42)

        ridge_on = ridge_predict(tr_ridge, test, TARGET, variant="seasonal")
        ridge_off = ridge_predict(
            _zero_weather(tr_ridge),
            _zero_weather(test),
            TARGET,
            variant="seasonal",
        )

        masks = _extreme_masks(test)
        for name, mask in masks.items():
            if not mask.any():
                continue
            _add(name, "naive_blend", y[mask], blend[mask])
            _add(name, "ridge_seasonal_weather", y[mask], ridge_on[mask])
            _add(name, "ridge_seasonal_no_weather", y[mask], ridge_off[mask])

        print(
            f"[weather_extreme] → {test_month} "
            + " ".join(f"{k}={int(v.sum())}" for k, v in masks.items())
        )

    by_mask: dict[str, Any] = {}
    for mask_name, models in stores.items():
        by_mask[mask_name] = {}
        for model, acc in models.items():
            y = np.concatenate(acc["y"])
            p = np.concatenate(acc["pred"])
            by_mask[mask_name][model] = _err_stats(y, p)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "극단 기상일만 평가. 시도 대표 ASOS 피처; "
            "전체 평균 재비교는 하지 않음. ridge는 observed 샘플(최대 40만) 적합."
        ),
        "target": TARGET,
        "window": "expanding",
        "by_mask": by_mask,
        "interpretation_template": (
            "시도 대표 기상 관측 피처는 전국 평균 및 극단 기상일 세그먼트에서 "
            "수요 예측 성능을 유의미하게 개선하지 못했다."
        ),
    }

    out_json = out_json or (
        ARTIFACTS_DIR / "charge_history_weather_extreme_eval.json"
    )
    out_md = out_md or (ARTIFACTS_DIR / "charge_history_weather_extreme_eval.md")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[weather_extreme] wrote {out_json}")
    print(f"[weather_extreme] wrote {out_md}")
    return result


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 극단 기상일 수요 예측 ablation",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
    ]
    for mask, models in (result.get("by_mask") or {}).items():
        lines.extend(
            [
                f"## {mask}",
                "",
                "| model | n | MAE |",
                "|---|---:|---:|",
            ]
        )
        for model, st in models.items():
            lines.append(
                f"| {model} | {st.get('n')} | {_fmt(st.get('mae'))} |"
            )
        lines.append("")

    # auto verdict
    verdict = result.get("interpretation_template")
    improved = False
    for mask, models in (result.get("by_mask") or {}).items():
        b = (models.get("naive_blend") or {}).get("mae")
        on = (models.get("ridge_seasonal_weather") or {}).get("mae")
        off = (models.get("ridge_seasonal_no_weather") or {}).get("mae")
        if b and on and on < b * 0.98 and off and on < off * 0.98:
            improved = True
            break
    if not improved:
        lines.extend(["## 결론", "", verdict or "", ""])
    else:
        lines.extend(
            [
                "## 결론",
                "",
                "일부 극단 기상 세그먼트에서 weather-on ridge가 blend·weather-off보다 개선됨. "
                "세그먼트표를 개별 확인.",
                "",
            ]
        )
    lines.append("산출 JSON: `charge_history_weather_extreme_eval.json`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_weather_extreme_eval()
