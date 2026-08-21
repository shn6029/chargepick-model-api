# -*- coding: utf-8 -*-
"""확장 윈도우: 누적 학습 → 다음 달 예측 (naive vs Ridge)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from . import ARTIFACTS_DIR, MONTHS, scope_label, scope_prefix
from .panel import build_hour_distribution, load_or_build_panel


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = y_true > 1e-6
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _tier_mae(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_vol: pd.Series,
) -> dict[str, float | None]:
    """학습 구간 이용량 기준 상위/하위 20% 충전소 MAE."""
    if train_vol.empty:
        return {"top20_mae": None, "bottom20_mae": None}
    q80 = train_vol.quantile(0.8)
    q20 = train_vol.quantile(0.2)
    stations = df["station_name"].to_numpy()
    top = np.isin(stations, train_vol[train_vol >= q80].index)
    bot = np.isin(stations, train_vol[train_vol <= q20].index)
    return {
        "top20_mae": float(np.mean(np.abs(y_true[top] - y_pred[top]))) if top.any() else None,
        "bottom20_mae": float(np.mean(np.abs(y_true[bot] - y_pred[bot]))) if bot.any() else None,
    }


def _naive_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """직전 학습 구간에서 station×weekday 평균 (없으면 station 평균 → 전체 평균)."""
    sw = train.groupby(["station_name", "weekday"])[target].mean()
    s_mean = train.groupby("station_name")[target].mean()
    global_mean = float(train[target].mean()) if len(train) else 0.0
    preds = []
    for row in test.itertuples():
        key = (row.station_name, int(row.weekday))
        if key in sw.index:
            preds.append(float(sw.loc[key]))
        elif row.station_name in s_mean.index:
            preds.append(float(s_mean.loc[row.station_name]))
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=float)


def _encode_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    cat_cols = ["district", "speed_class"]
    # sklearn 버전 호환
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat_tr = enc.fit_transform(train[cat_cols].astype(str))
    X_cat_te = enc.transform(test[cat_cols].astype(str))

    # station 평균 이용(누수 방지: train만)
    st_sess = train.groupby("station_name")["n_sessions"].mean()
    st_kwh = train.groupby("station_name")["kwh"].mean()
    global_s = float(train["n_sessions"].mean())
    global_k = float(train["kwh"].mean())

    def num_block(df: pd.DataFrame) -> np.ndarray:
        wd = df["weekday"].to_numpy().astype(float)
        # cyclic weekday
        ang = 2 * np.pi * wd / 7.0
        pk = df["power_kw"].fillna(0).to_numpy().astype(float)
        ss = df["station_name"].map(st_sess).fillna(global_s).to_numpy()
        sk = df["station_name"].map(st_kwh).fillna(global_k).to_numpy()
        return np.column_stack([wd, np.sin(ang), np.cos(ang), pk, ss, sk])

    X_tr = np.hstack([X_cat_tr, num_block(train)])
    X_te = np.hstack([X_cat_te, num_block(test)])
    return X_tr, X_te


def _ridge_predict(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> np.ndarray:
    if len(train) < 50:
        return _naive_predict(train, test, target)
    X_tr, X_te = _encode_features(train, test)
    y = train[target].to_numpy(dtype=float)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y)
    pred = model.predict(X_te)
    return np.clip(pred, 0, None)


def _hour_cosine(train_months: list[str], test_month: str, hour_df: pd.DataFrame) -> float | None:
    if hour_df.empty:
        return None
    tr = hour_df[hour_df["year_month"].isin(train_months)]
    te = hour_df[hour_df["year_month"] == test_month]
    if tr.empty or te.empty:
        return None
    tr_share = tr.groupby("hour")["n"].sum()
    te_share = te.groupby("hour")["n"].sum()
    hours = list(range(24))
    a = np.array([tr_share.get(h, 0.0) for h in hours], dtype=float)
    b = np.array([te_share.get(h, 0.0) for h in hours], dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return None
    a = a / a.sum()
    b = b / b.sum()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def eval_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    train_vol: pd.Series,
) -> dict[str, Any]:
    y = test[target].to_numpy(dtype=float)
    naive = _naive_predict(train, test, target)
    ridge = _ridge_predict(train, test, target)
    out: dict[str, Any] = {
        "n": int(len(test)),
        "naive": {
            "mae": _mae(y, naive),
            "mape": _mape(y, naive),
            **_tier_mae(test, y, naive, train_vol),
        },
        "ridge": {
            "mae": _mae(y, ridge),
            "mape": _mape(y, ridge),
            **_tier_mae(test, y, ridge, train_vol),
        },
    }
    n_mae = out["naive"]["mae"]
    r_mae = out["ridge"]["mae"]
    out["ridge_vs_naive_mae_improvement"] = (
        float((n_mae - r_mae) / n_mae) if n_mae and n_mae > 0 else None
    )
    return out


def run_expanding_eval(
    panel: pd.DataFrame | None = None,
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
    scope: str = "daegu",
) -> dict[str, Any]:
    if panel is None:
        panel = load_or_build_panel(scope=scope)
    panel = panel.copy()
    panel["day"] = pd.to_datetime(panel["day"])

    from .ingest import load_sessions

    hour_df = build_hour_distribution(load_sessions(scope=scope))

    folds: list[dict[str, Any]] = []
    # train through MONTHS[i], test MONTHS[i+1]
    for i in range(len(MONTHS) - 1):
        train_months = MONTHS[: i + 1]
        test_month = MONTHS[i + 1]
        train = panel[panel["year_month"].isin(train_months)]
        test = panel[panel["year_month"] == test_month]
        if train.empty or test.empty:
            folds.append(
                {
                    "train_months": train_months,
                    "test_month": test_month,
                    "skipped": True,
                    "reason": "empty train or test",
                }
            )
            continue

        train_vol = train.groupby("station_name")["n_sessions"].sum()
        fold: dict[str, Any] = {
            "train_months": train_months,
            "train_through": train_months[-1],
            "test_month": test_month,
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_train_stations": int(train["station_name"].nunique()),
            "n_test_stations": int(test["station_name"].nunique()),
            "n_sessions": eval_target(train, test, "n_sessions", train_vol),
            "kwh": eval_target(train, test, "kwh", train_vol),
            "hour_profile_cosine": _hour_cosine(train_months, test_month, hour_df),
            "skipped": False,
        }
        folds.append(fold)
        print(
            f"[fold] train≤{train_months[-1]} → test={test_month} | "
            f"sess MAE naive={fold['n_sessions']['naive']['mae']:.3f} "
            f"ridge={fold['n_sessions']['ridge']['mae']:.3f}"
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "세션 이력 기반 수요(일별 세션수·kWh) 확장 검증. "
            "ETA 가용확률(status) 모델과 동일 지표가 아님."
        ),
        "scope": scope_label(scope),
        "scope_prefix": scope_prefix(scope),
        "months": MONTHS,
        "n_folds": len(folds),
        "folds": folds,
        "summary": _summarize(folds),
    }

    prefix = scope_prefix(scope)
    suffix = "" if prefix == "daegu" else f"_{prefix}"
    date_tag = datetime.now().strftime("%m%d")
    out_json = out_json or (
        ARTIFACTS_DIR / f"charge_history_expanding_eval{suffix}.json"
    )
    out_md = out_md or (
        ARTIFACTS_DIR / f"charge_history_expanding_eval{suffix}_{date_tag}.md"
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_to_markdown(result), encoding="utf-8")
    print(f"[eval] wrote {out_json}")
    print(f"[eval] wrote {out_md}")
    return result


def _summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    active = [f for f in folds if not f.get("skipped")]
    if not active:
        return {}

    def series(target: str, model: str, metric: str) -> list[float]:
        vals = []
        for f in active:
            v = f[target][model].get(metric)
            if v is not None:
                vals.append(float(v))
        return vals

    def avg(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    return {
        "n_active_folds": len(active),
        "n_sessions_naive_mae_mean": avg(series("n_sessions", "naive", "mae")),
        "n_sessions_ridge_mae_mean": avg(series("n_sessions", "ridge", "mae")),
        "kwh_naive_mae_mean": avg(series("kwh", "naive", "mae")),
        "kwh_ridge_mae_mean": avg(series("kwh", "ridge", "mae")),
        "ridge_beat_naive_session_folds": sum(
            1
            for f in active
            if (f["n_sessions"].get("ridge_vs_naive_mae_improvement") or 0) > 0
        ),
    }


def _to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 충전이력 월별 확장 검증",
        "",
        f"**생성:** `{result['generated_at']}`",
        "",
        f"> {result['note']}",
        "",
        f"범위: **{result['scope']}** · folds: {result['n_folds']}",
        "",
        "## 요약",
        "",
    ]
    s = result.get("summary") or {}
    lines.append("| 지표 | 값 |")
    lines.append("|---|---|")
    for k, v in s.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend(["", "## 폴드별 (n_sessions MAE)", "", "| train≤ | test | naive MAE | ridge MAE | improve |", "|---|---|---|---|---|"])
    for f in result["folds"]:
        if f.get("skipped"):
            lines.append(
                f"| {f.get('train_through', f['train_months'][-1])} | {f['test_month']} | — | — | skipped |"
            )
            continue
        ns = f["n_sessions"]
        imp = ns.get("ridge_vs_naive_mae_improvement")
        imp_s = f"{imp*100:.1f}%" if imp is not None else "—"
        lines.append(
            f"| {f['train_through']} | {f['test_month']} | "
            f"{ns['naive']['mae']:.3f} | {ns['ridge']['mae']:.3f} | {imp_s} |"
        )
    lines.extend(
        [
            "",
            "## 모델",
            "",
            "- **naive:** station×weekday 평균 (없으면 station / global)",
            "- **ridge:** district·speed_class one-hot + weekday 주기 + station 평균 이용 + power_kw",
            "",
            "산출 JSON: `charge_history_expanding_eval.json`",
            "",
        ]
    )
    return "\n".join(lines)
