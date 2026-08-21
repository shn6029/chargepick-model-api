# -*- coding: utf-8 -*-
"""전국→대구 콜드스타트 전이 평가 (naive_blend A/B/C).

A: 대구-only train + 대구 fallback
B: 대구 station/blend + 전국 fallback (서비스 후보)
C: 전국 train + 전국 fallback
테스트는 전국 피처의 대구 행만.
"""

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
    TIER_NAMES,
    freeze_test_lag_features,
    month_start_ts,
    naive_blend_predict,
)
from .features import load_or_build_features

MIN_COMBO = 14  # district×speed×weekday train 관측 최소
TARGET = "n_sessions"
MODEL_KEYS = ("A_daegu_only", "B_hybrid", "C_nationwide")
GROUP_KEYS = ("existing", "sparse", "new_station", "new_combo", "all")


def is_daegu_region(region: pd.Series) -> pd.Series:
    return region.astype(str).str.contains("대구", na=False)


def ensure_district_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "district_key" not in out.columns:
        out = out.copy()
        region = out["region"] if "region" in out.columns else "unknown"
        district = out["district"] if "district" in out.columns else "unknown"
        out["district_key"] = (
            region.astype(str) + " " + district.astype(str)
        ).str.strip()
    return out


def label_cold_start_groups(
    train_obs: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_combo: int = MIN_COMBO,
) -> pd.DataFrame:
    """테스트 행에 group 라벨 부여 (행 단위; station은 existing/sparse/new)."""
    te = ensure_district_key(test).copy()
    tr = ensure_district_key(train_obs)

    vol = tr.groupby("station_name")[TARGET].sum()
    q20 = float(vol.quantile(0.2)) if len(vol) else 0.0
    train_stations = set(vol.index)

    combo_counts = (
        tr.groupby(["district_key", "speed_class", "weekday"]).size()
        if len(tr)
        else pd.Series(dtype=int)
    )

    groups: list[str] = []
    new_combo_flags: list[bool] = []
    for row in te.itertuples():
        st = row.station_name
        if st not in train_stations:
            g = "new_station"
        elif float(vol.get(st, 0.0)) <= q20:
            g = "sparse"
        else:
            g = "existing"
        groups.append(g)
        key = (getattr(row, "district_key", "unknown"), row.speed_class, int(row.weekday))
        if len(combo_counts) and key in combo_counts.index:
            n_combo = int(combo_counts.loc[key])
        else:
            n_combo = 0
        new_combo_flags.append(n_combo < min_combo)

    te["cold_group"] = groups
    te["new_combo"] = new_combo_flags
    te["train_volume"] = te["station_name"].map(vol).fillna(0.0)
    return te


def _err_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    if len(y) == 0:
        return {
            "n": 0,
            "mae": None,
            "p90_abs_err": None,
            "mean_bias": None,
            "over_frac": None,
            "under_frac": None,
        }
    err = pred - y
    abs_err = np.abs(err)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(abs_err)),
        "p90_abs_err": float(np.quantile(abs_err, 0.9)),
        "mean_bias": float(np.mean(err)),
        "over_frac": float(np.mean(err > 0)),
        "under_frac": float(np.mean(err < 0)),
    }


def _pct_improve(base: float | None, other: float | None) -> float | None:
    if base is None or other is None or base <= 0:
        return None
    return float((base - other) / base)


def run_transfer_eval(
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
    min_combo: int = MIN_COMBO,
) -> dict[str, Any]:
    feat = load_or_build_features(scope="nationwide")
    feat = ensure_district_key(feat)
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"]).dt.normalize()
    feat["is_daegu"] = is_daegu_region(feat["region"])

    # aggregate across folds
    agg_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    tier_counts: dict[str, dict[str, int]] = {k: defaultdict(int) for k in MODEL_KEYS}

    for i in range(len(MONTHS) - 1):
        test_month = MONTHS[i + 1]
        train_months = MONTHS[: i + 1]
        m_start = month_start_ts(test_month)

        hist = feat[feat["day"] < m_start]
        train_dense = feat[feat["year_month"].isin(train_months)].copy()
        train_obs = train_dense[train_dense["is_observed"]]
        train_daegu = train_obs[train_obs["is_daegu"]]
        train_nat = train_obs
        # 전국 prior는 대구를 제외해야 A와 차별화됨 (동일 district_key 누수 방지)
        train_nat_ex_daegu = train_obs[~train_obs["is_daegu"]]

        test_raw = feat[
            (feat["year_month"] == test_month)
            & feat["is_observed"]
            & feat["is_daegu"]
        ].copy()
        if train_daegu.empty or test_raw.empty:
            fold_summaries.append(
                {
                    "test_month": test_month,
                    "skipped": True,
                    "reason": "empty train_daegu or test",
                }
            )
            continue

        test = freeze_test_lag_features(hist, test_raw, m_start)
        test = label_cold_start_groups(train_daegu, test, min_combo=min_combo)
        y = test[TARGET].to_numpy(dtype=float)

        preds: dict[str, np.ndarray] = {}
        tiers: dict[str, np.ndarray] = {}

        # A: daegu only
        p, c, t = naive_blend_predict(
            train_daegu,
            test,
            TARGET,
            use_speed_weekday=True,
            return_tiers=True,
        )
        preds["A_daegu_only"] = p
        tiers["A_daegu_only"] = t
        for k, v in c.items():
            tier_counts["A_daegu_only"][k] += v

        # B: hybrid — station from Daegu, fallback from nationwide EXCLUDING Daegu
        p, c, t = naive_blend_predict(
            train_daegu,
            test,
            TARGET,
            fallback_train=train_nat_ex_daegu if len(train_nat_ex_daegu) else train_nat,
            use_speed_weekday=True,
            return_tiers=True,
        )
        preds["B_hybrid"] = p
        tiers["B_hybrid"] = t
        for k, v in c.items():
            tier_counts["B_hybrid"][k] += v

        # C: train on nationwide excluding Daegu (pure transfer into Daegu)
        p, c, t = naive_blend_predict(
            train_nat_ex_daegu if len(train_nat_ex_daegu) else train_nat,
            test,
            TARGET,
            use_speed_weekday=True,
            return_tiers=True,
        )
        preds["C_nationwide"] = p
        tiers["C_nationwide"] = t
        for k, v in c.items():
            tier_counts["C_nationwide"][k] += v

        fold_block: dict[str, Any] = {
            "test_month": test_month,
            "train_through": train_months[-1],
            "n_train_daegu": int(len(train_daegu)),
            "n_train_nat": int(len(train_nat)),
            "n_train_nat_ex_daegu": int(len(train_nat_ex_daegu)),
            "n_test": int(len(test)),
            "skipped": False,
            "by_group": {},
        }

        masks = {
            "all": np.ones(len(test), dtype=bool),
            "existing": (test["cold_group"] == "existing").to_numpy(),
            "sparse": (test["cold_group"] == "sparse").to_numpy(),
            "new_station": (test["cold_group"] == "new_station").to_numpy(),
            "new_combo": test["new_combo"].to_numpy(),
        }

        for gname, mask in masks.items():
            gstats = {"n_rows": int(mask.sum()), "models": {}}
            for mk in MODEL_KEYS:
                st = _err_stats(y[mask], preds[mk][mask])
                gstats["models"][mk] = st
                agg_rows.append(
                    {
                        "test_month": test_month,
                        "group": gname,
                        "model": mk,
                        **st,
                        "y": y[mask],
                        "pred": preds[mk][mask],
                    }
                )
            # A vs B improvement on this fold/group
            a_mae = gstats["models"]["A_daegu_only"]["mae"]
            b_mae = gstats["models"]["B_hybrid"]["mae"]
            c_mae = gstats["models"]["C_nationwide"]["mae"]
            gstats["B_vs_A_mae_improve"] = _pct_improve(a_mae, b_mae)
            gstats["C_vs_A_mae_improve"] = _pct_improve(a_mae, c_mae)
            fold_block["by_group"][gname] = gstats

        fold_summaries.append(fold_block)
        print(
            f"[transfer] → {test_month} n={len(test)} "
            f"new={int(masks['new_station'].sum())} sparse={int(masks['sparse'].sum())} | "
            f"A={fold_block['by_group']['all']['models']['A_daegu_only']['mae']:.3f} "
            f"B={fold_block['by_group']['all']['models']['B_hybrid']['mae']:.3f} "
            f"C={fold_block['by_group']['all']['models']['C_nationwide']['mae']:.3f}"
        )

    summary = _summarize_pooled(agg_rows)
    success = _success_gate(summary)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "전국→대구 콜드스타트 전이: A 대구-only / B 하이브리드 / C 전국. "
            "전체 MAE보다 신규·희소 그룹을 우선 해석."
        ),
        "target": TARGET,
        "window": "expanding",
        "min_combo": min_combo,
        "models": {
            "A_daegu_only": "대구 train + 대구 fallback",
            "B_hybrid": "대구 blend/station + 전국(대구 제외) geo/speed/global fallback",
            "C_nationwide": "전국(대구 제외) train + fallback → 대구 테스트 (순수 전이)",
        },
        "success_criteria": {
            "cold_start_mae_improve_min": 0.10,
            "existing_mae_degrade_max": 0.01,
            "description": (
                "신규·희소에서 B 또는 C가 A 대비 MAE ≥10% 개선, "
                "기존 충전소 MAE 악화 <1%"
            ),
        },
        "success_gate": success,
        "tier_counts_total": {k: dict(v) for k, v in tier_counts.items()},
        "summary": summary,
        "folds": fold_summaries,
    }

    out_json = out_json or (ARTIFACTS_DIR / "charge_history_transfer_eval.json")
    out_md = out_md or (ARTIFACTS_DIR / "charge_history_transfer_eval.md")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # strip large arrays from serializable copy
    serial = json.loads(
        json.dumps(result, ensure_ascii=False, default=_json_default)
    )
    out_json.write_text(
        json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[transfer_eval] wrote {out_json}")
    print(f"[transfer_eval] wrote {out_md}")
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return None  # drop arrays from JSON
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    return str(o)


def _summarize_pooled(agg_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool all fold rows per group×model (recompute from concatenated errors)."""
    # rebuild by concatenating y/pred stored per fold chunk
    buckets: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(
        list
    )
    for row in agg_rows:
        if row.get("n", 0) == 0:
            continue
        y = row.get("y")
        pred = row.get("pred")
        if y is None or pred is None:
            continue
        buckets[(row["group"], row["model"])].append((y, pred))

    out: dict[str, Any] = {}
    for g in GROUP_KEYS:
        out[g] = {"models": {}}
        for mk in MODEL_KEYS:
            parts = buckets.get((g, mk), [])
            if not parts:
                out[g]["models"][mk] = _err_stats(np.array([]), np.array([]))
                continue
            y = np.concatenate([p[0] for p in parts])
            pred = np.concatenate([p[1] for p in parts])
            out[g]["models"][mk] = _err_stats(y, pred)
        a = out[g]["models"]["A_daegu_only"]["mae"]
        b = out[g]["models"]["B_hybrid"]["mae"]
        c = out[g]["models"]["C_nationwide"]["mae"]
        out[g]["B_vs_A_mae_improve"] = _pct_improve(a, b)
        out[g]["C_vs_A_mae_improve"] = _pct_improve(a, c)
    return out


def _success_gate(summary: dict[str, Any]) -> dict[str, Any]:
    existing = summary.get("existing") or {}
    a_ex = (existing.get("models") or {}).get("A_daegu_only", {}).get("mae")

    def model_ok(model_key: str, improve_key: str) -> dict[str, Any]:
        cold_hit = False
        cold_detail = {}
        for g in ("new_station", "sparse"):
            imp = (summary.get(g) or {}).get(improve_key)
            cold_detail[g] = imp
            if imp is not None and imp >= 0.10:
                cold_hit = True
        mae = (existing.get("models") or {}).get(model_key, {}).get("mae")
        degrade = None
        if a_ex is not None and mae is not None and a_ex > 0:
            degrade = (mae - a_ex) / a_ex
        stable = degrade is None or degrade < 0.01
        return {
            "cold_improved": cold_hit,
            "existing_degrade": degrade,
            "existing_stable": stable,
            "cold_detail": cold_detail,
            "passed": bool(cold_hit and stable),
        }

    b = model_ok("B_hybrid", "B_vs_A_mae_improve")
    c = model_ok("C_nationwide", "C_vs_A_mae_improve")
    passed = bool(b["passed"] or c["passed"])
    winner = (
        "B_hybrid"
        if b["passed"]
        else ("C_nationwide" if c["passed"] else None)
    )
    return {
        "passed": passed,
        "winner": winner,
        "B_hybrid": b,
        "C_nationwide": c,
        "verdict": (
            f"{winner} 기준 통과: 신규·희소 개선 + 기존 충전소 유지"
            if passed
            else "성공 기준 미달 — 그룹별 MAE·개선율을 개별 해석"
        ),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    return f"{100.0 * float(v):+.1f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    s = result.get("summary") or {}
    gate = result.get("success_gate") or {}
    lines = [
        "# 전국→대구 콜드스타트 전이 평가",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        f"**성공 게이트:** {'PASS' if gate.get('passed') else 'FAIL'} — {gate.get('verdict')}",
        "",
        "## 모델",
        "",
        "| key | 설명 |",
        "|---|---|",
    ]
    for k, v in (result.get("models") or {}).items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## 그룹별 pooled MAE (n_sessions)",
            "",
            "| group | n(A) | A MAE | B MAE | C MAE | B vs A | C vs A |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for g in GROUP_KEYS:
        block = s.get(g) or {}
        models = block.get("models") or {}
        n = (models.get("A_daegu_only") or {}).get("n")
        lines.append(
            "| {g} | {n} | {a} | {b} | {c} | {bi} | {ci} |".format(
                g=g,
                n=n if n is not None else "—",
                a=_fmt((models.get("A_daegu_only") or {}).get("mae")),
                b=_fmt((models.get("B_hybrid") or {}).get("mae")),
                c=_fmt((models.get("C_nationwide") or {}).get("mae")),
                bi=_fmt_pct(block.get("B_vs_A_mae_improve")),
                ci=_fmt_pct(block.get("C_vs_A_mae_improve")),
            )
        )

    lines.extend(
        [
            "",
            "## P90 |error| · bias",
            "",
            "| group | model | P90 | bias | over% | under% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for g in ("new_station", "sparse", "existing", "all"):
        models = (s.get(g) or {}).get("models") or {}
        for mk in MODEL_KEYS:
            m = models.get(mk) or {}
            lines.append(
                f"| {g} | {mk} | {_fmt(m.get('p90_abs_err'))} | "
                f"{_fmt(m.get('mean_bias'))} | {_fmt_pct(m.get('over_frac'))} | "
                f"{_fmt_pct(m.get('under_frac'))} |"
            )

    lines.extend(["", "## Fallback tier 사용량 (전 폴드 합)", "", "| model | tier | n |", "|---|---|---:|"])
    for mk, counts in (result.get("tier_counts_total") or {}).items():
        for tier, n in counts.items():
            lines.append(f"| {mk} | {tier} | {n} |")

    lines.extend(
        [
            "",
            "## 해석 가이드",
            "",
            "- **핵심:** `new_station` / `sparse`에서 B 또는 C가 A 대비 MAE 개선인가.",
            "- **B**가 서비스 후보: 이력이 있으면 대구 station blend, 없으면 전국 `district_key×speed×weekday`.",
            "- 전체(`all`) MAE만으로 전국 데이터 가치를 판단하지 말 것.",
            "",
            f"산출 JSON: `{ARTIFACTS_DIR.name}/charge_history_transfer_eval.json`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    run_transfer_eval()
