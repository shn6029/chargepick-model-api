# -*- coding: utf-8 -*-
"""B_hybrid 신규·희소 세그먼트 과대예측 모니터.

사용:
  py -m charge_history overpredict_monitor
  py -m charge_history overpredict_monitor --scope nationwide --top 20

산출:
  recommend_api/artifacts/charge_history_overpredict_monitor_MMDD.md

동작:
- nationwide 피처 기준 expanding-window 마지막 fold (최신 월이 test)
- B_hybrid 및 naive_blend 예측 vs actual n_sessions
- new_station / sparse 세그먼트 × 시도 × 속도군 별 bias(pred - actual) 집계
- prior 보정 후보(양(+) 편향 큰 세그먼트)만 목록 출력 — 코드 적용·HGB 승격 없음
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ARTIFACTS_DIR, MONTHS
from .compare_eval import (
    TIER_NAMES,
    freeze_test_lag_features,
    month_start_ts,
    naive_blend_predict,
)
from .features import load_or_build_features
from .transfer_eval import (
    TARGET,
    ensure_district_key,
    label_cold_start_groups,
)

_MMDD = datetime.now().strftime("%m%d")
OUT_MD = ARTIFACTS_DIR / f"charge_history_overpredict_monitor_{_MMDD}.md"
OUT_JSON = ARTIFACTS_DIR / f"charge_history_overpredict_monitor_{_MMDD}.json"

MIN_BIAS_SESSIONS = 0.5  # 편향 후보 최소 abs bias (세션/월)
TOP_N_DEFAULT = 15


def _bhybrid_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """B_hybrid = 대구 district/blend fallback + 전국 speed_wd/global fallback."""
    dsw_daegu = train[train["region"].astype(str).str.contains("대구", na=False)].groupby(
        ["district_key", "speed_class", "weekday"], sort=False
    )[TARGET].mean()
    spd = train.groupby(["speed_class", "weekday"], sort=False)[TARGET].mean()
    gwd = train.groupby("weekday", sort=False)[TARGET].mean()
    global_mean = float(train[TARGET].mean()) if len(train) else 0.0

    preds = []
    for row in ensure_district_key(test).itertuples():
        dk = getattr(row, "district_key", "unknown")
        sc = row.speed_class
        wd = int(row.weekday)
        val = dsw_daegu.get((dk, sc, wd), np.nan)
        if np.isnan(val):
            val = spd.get((sc, wd), np.nan)
        if np.isnan(val):
            val = gwd.get(wd, global_mean)
        preds.append(float(val))
    return np.array(preds)


def _naive_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    return naive_blend_predict(train, test, tiers=None)


def run_overpredict_monitor(
    scope: str = "nationwide",
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    feats = load_or_build_features(force=False, scope=scope)
    feats = ensure_district_key(feats)

    months = sorted(feats["year_month"].dropna().unique())
    if len(months) < 3:
        return {"error": "피처 데이터 부족 (최소 3개월 필요)", "months": list(months)}

    # expanding: 최근 2개월 중 마지막 1개월 test
    train_months = months[:-1]
    test_month = months[-1]
    train = feats[feats["year_month"].isin(train_months)].copy()
    test = feats[feats["year_month"] == test_month].copy()
    train = freeze_test_lag_features(train, test)

    test = label_cold_start_groups(train, test)
    test = ensure_district_key(test)

    bhyb_pred = _bhybrid_predict(train, test)
    naive_pred = _naive_predict(train, test)

    test = test.copy()
    test["pred_bhybrid"] = bhyb_pred
    test["pred_naive"] = naive_pred
    test["bias_bhybrid"] = test["pred_bhybrid"] - test[TARGET]
    test["bias_naive"] = test["pred_naive"] - test[TARGET]

    cold_groups = ["new_station", "sparse"]

    def _agg(df: pd.DataFrame, group_label: str, dim: str) -> list[dict[str, Any]]:
        rows = []
        for key, g in df.groupby(dim):
            n = len(g)
            bias_b = float(g["bias_bhybrid"].mean())
            bias_n = float(g["bias_naive"].mean())
            mae_b = float(g["bias_bhybrid"].abs().mean())
            rows.append(
                {
                    "group": group_label,
                    "dim": dim,
                    "key": str(key),
                    "n": n,
                    "bias_bhybrid": round(bias_b, 3),
                    "bias_naive": round(bias_n, 3),
                    "mae_bhybrid": round(mae_b, 3),
                }
            )
        return rows

    records: list[dict[str, Any]] = []
    for grp in cold_groups:
        sub = test[test["group"] == grp] if "group" in test.columns else pd.DataFrame()
        if sub.empty:
            continue
        for dim in ("district_key", "speed_class"):
            if dim in sub.columns:
                records.extend(_agg(sub, grp, dim))

    # 전체 세그먼트도 포함
    for dim in ("district_key", "speed_class"):
        if dim in test.columns:
            records.extend(_agg(test, "all", dim))

    rec_df = pd.DataFrame(records) if records else pd.DataFrame()

    # 과대예측 후보: new_station|sparse에서 bias_bhybrid > MIN_BIAS_SESSIONS
    if rec_df.empty:
        candidates: list[dict] = []
    else:
        cand = rec_df[
            rec_df["group"].isin(cold_groups)
            & (rec_df["bias_bhybrid"] > MIN_BIAS_SESSIONS)
        ].sort_values("bias_bhybrid", ascending=False)
        candidates = cand.head(top_n).to_dict(orient="records")

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_month": test_month,
        "train_months": list(train_months),
        "n_test": int(len(test)),
        "cold_groups_n": {g: int((test.get("group", pd.Series()) == g).sum()) for g in cold_groups},
        "overall_bias": {
            "bhybrid": round(float(test["bias_bhybrid"].mean()), 3),
            "naive": round(float(test["bias_naive"].mean()), 3),
        },
        "overprediction_candidates": candidates,
        "note": (
            "prior 보정 후보 목록만 출력. "
            "코드 적용·HGB 승격 없음. "
            "positive bias = 과대예측(pred > actual)."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _write_md(result, top_n)
    return result


def _write_md(result: dict[str, Any], top_n: int) -> None:
    lines = [
        f"# B_hybrid 과대예측 모니터 ({_MMDD})",
        "",
        f"생성: {result['generated_at']}  ",
        f"테스트 월: **{result['test_month']}**  ",
        f"학습 월: {result['train_months']}  ",
        f"테스트 행: {result['n_test']}  ",
        f"cold 세그먼트: {result['cold_groups_n']}",
        "",
        "## 전체 편향",
        "",
        "| 모델 | mean bias (pred - actual) |",
        "|------|--------------------------|",
        f"| B_hybrid | {result['overall_bias']['bhybrid']} |",
        f"| naive_blend | {result['overall_bias']['naive']} |",
        "",
        f"## 과대예측 후보 (new_station·sparse, bias > {MIN_BIAS_SESSIONS}, top {top_n})",
        "",
    ]

    cands = result.get("overprediction_candidates", [])
    if not cands:
        lines.append("*후보 없음*")
    else:
        lines += [
            "| group | dim | key | n | bias_bhybrid | bias_naive | mae_bhybrid |",
            "|-------|-----|-----|--:|-------------:|-----------:|------------:|",
        ]
        for c in cands:
            lines.append(
                f"| {c['group']} | {c['dim']} | {c['key']} "
                f"| {c['n']} | {c['bias_bhybrid']} | {c['bias_naive']} | {c['mae_bhybrid']} |"
            )

    lines += [
        "",
        "## 판단 기준",
        "",
        "- `bias_bhybrid > 0`: B_hybrid이 과대예측 (실제보다 높게 예측)",
        "- 보정 후보 = new_station / sparse 세그먼트에서 편향이 큰 시도·속도군",
        "- **prior 보정 목록만 출력. 코드 적용·HGB 승격 없음.**",
        "",
        f"상세: `{OUT_JSON.name}`",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[overpredict_monitor] saved: {OUT_MD}")
    print(f"[overpredict_monitor] saved: {OUT_JSON}")
