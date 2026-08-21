"""API 정본 모델(horizon_hgb)용 홀드아웃 평가.

- 날짜 80/20 홀드아웃 + ETA별 지표
- Expanding rolling 날짜 fold + 평균/표준편차
- Horizon별 calibration PNG
- 가용확률 threshold 스윕 → rank/warn 권고
- 동일 holdout LogisticRegression baseline 비교
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recommend_api.eval_metrics import (
    always_available_baseline,
    calibration_table,
    evaluate_classifier,
    iter_expanding_date_folds,
    plot_calibration_curves,
    split_by_date,
)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RANK_THRESHOLD = 0.5


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )


def _new_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _prepare_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_mask is None:
        test_mask = ~train_mask
    X_tr_raw, X_te_raw = X.loc[train_mask], X.loc[test_mask]
    y_tr, y_te = y.loc[train_mask], y.loc[test_mask]
    med = X_tr_raw.median(numeric_only=True)
    return X_tr_raw.fillna(med), X_te_raw.fillna(med), y_tr, y_te


def _per_eta_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    eta: pd.Series,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    tmp = pd.DataFrame({"y": y_true.values, "proba": proba, "eta_minutes": eta.values})
    rows = []
    for h, g in tmp.groupby("eta_minutes"):
        m = evaluate_classifier(g["y"], g["proba"], threshold=threshold)
        rows.append(
            {
                "eta_minutes": int(h),
                "n": m["n"],
                "roc_auc": m["roc_auc"],
                "unavailable_recall": m["unavailable_recall"],
                "unavailable_precision": m["unavailable_precision"],
                "unavailable_f1": m["unavailable_f1"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "brier_score": m["brier_score"],
                "pr_auc_unavailable": m["pr_auc_unavailable"],
            }
        )
    return sorted(rows, key=lambda r: r["eta_minutes"])


def _fit_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    X_tr, X_te, y_tr, y_te = _prepare_fold(X, y, train_mask, test_mask)
    model = _new_model()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    return proba, y_te, X_te.index.to_series(index=X_te.index)


def _fit_predict_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    X_tr, X_te, y_tr, y_te = _prepare_fold(X, y, train_mask, test_mask)
    model = _new_logistic()
    model.fit(X_tr.to_numpy(dtype=float), y_tr)
    proba = model.predict_proba(X_te.to_numpy(dtype=float))[:, 1]
    return proba, y_te, X_te.index.to_series(index=X_te.index)


def _segment_holdout_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    eta: pd.Series,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    """holdout test 예측을 마스크로 잘라 세그먼트 지표 계산."""
    mask_arr = np.asarray(mask, dtype=bool)
    if not mask_arr.any():
        return None
    y_s = pd.Series(np.asarray(y_true)[mask_arr])
    p_s = np.asarray(proba, dtype=float)[mask_arr]
    eta_s = pd.Series(np.asarray(eta)[mask_arr])
    overall = evaluate_classifier(y_s, p_s)
    return {
        "n": int(len(y_s)),
        "unavailable_rate": float(1.0 - overall["positive_rate"]),
        "overall": overall,
        "per_eta": _per_eta_metrics(y_s, p_s, eta_s),
        "calibration_bins": calibration_table(y_s, p_s).to_dict(orient="records"),
        "_y": y_s,
        "_proba": p_s,
        "_eta": eta_s,
    }


def _public_segment(seg: dict[str, Any] | None) -> dict[str, Any] | None:
    if seg is None:
        return None
    return {k: v for k, v in seg.items() if not str(k).startswith("_")}


def evaluate_date_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    ratio: float = 0.8,
) -> dict[str, Any] | None:
    is_train, train_dates, test_dates = split_by_date(meta, ratio=ratio)
    if len(test_dates) == 0 or len(train_dates) == 0:
        return None

    proba, y_te, te_index = _fit_predict(X, y, is_train)
    eta_te = meta.loc[te_index, "eta_minutes"]
    overall = evaluate_classifier(y_te, proba)
    baseline = always_available_baseline(y_te)
    per_eta = _per_eta_metrics(y_te, proba, eta_te)

    rapid_only = None
    slow_only = None
    if "is_fast" in meta.columns:
        is_fast_te = (
            pd.to_numeric(meta.loc[te_index, "is_fast"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        rapid_only = _segment_holdout_metrics(y_te, proba, eta_te, is_fast_te == 1)
        slow_only = _segment_holdout_metrics(y_te, proba, eta_te, is_fast_te == 0)

    # 승격 기준 모집단: 실제로 추천 후보가 되는 행만 (급속 ∧ access 하드제외 아님).
    # 전체 기준 지표는 서빙되지 않는 아파트 충전기(예측이 쉬움)가 섞여 낙관적이다.
    servable = None
    if "is_servable" in meta.columns:
        sv = meta.loc[te_index, "is_servable"]
        if sv.notna().any():
            servable = _segment_holdout_metrics(
                y_te, proba, eta_te, sv.fillna(False).astype(bool).to_numpy()
            )

    return {
        "train_days": int(len(train_dates)),
        "test_days": int(len(test_dates)),
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "overall": overall,
        "always_available_baseline": baseline,
        "per_eta": per_eta,
        "rapid_only": rapid_only,
        "slow_only": slow_only,
        "servable": servable,
        "_proba": proba,
        "_y_test": y_te,
        "_eta_test": eta_te,
        "_is_train": is_train,
        "_te_index": te_index,
    }


def evaluate_logistic_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    is_train: pd.Series,
    train_dates: list[Any] | None = None,
    test_dates: list[Any] | None = None,
) -> dict[str, Any]:
    """동일 날짜 holdout 분할로 LogisticRegression 기준선 평가."""
    proba, y_te, te_index = _fit_predict_logistic(X, y, is_train)
    eta_te = meta.loc[te_index, "eta_minutes"]
    overall = evaluate_classifier(y_te, proba)
    return {
        "model_name": "LogisticRegression",
        "train_dates": [str(d) for d in (train_dates or [])],
        "test_dates": [str(d) for d in (test_dates or [])],
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "overall": {
            "roc_auc": overall["roc_auc"],
            "pr_auc_unavailable": overall["pr_auc_unavailable"],
            "brier_score": overall["brier_score"],
            "accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
            "unavailable_recall": overall["unavailable_recall"],
            "unavailable_precision": overall["unavailable_precision"],
            "unavailable_f1": overall["unavailable_f1"],
            "n": overall["n"],
        },
        "per_eta": _per_eta_metrics(y_te, proba, eta_te),
    }


def evaluate_rolling_folds(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    min_train_days: int = 2,
) -> dict[str, Any] | None:
    folds: list[dict[str, Any]] = []
    for is_train, train_dates, test_date in iter_expanding_date_folds(
        meta, min_train_days=min_train_days
    ):
        is_test = pd.to_datetime(meta["created_at"]).dt.date == test_date
        proba, y_te, te_index = _fit_predict(X, y, is_train, is_test)
        eta_te = meta.loc[te_index, "eta_minutes"]
        overall = evaluate_classifier(y_te, proba)
        folds.append(
            {
                "train_dates": [str(d) for d in train_dates],
                "test_date": str(test_date),
                "n_train": int(is_train.sum()),
                "n_test": int(is_test.sum()),
                "overall": {
                    "roc_auc": overall["roc_auc"],
                    "accuracy": overall["accuracy"],
                    "balanced_accuracy": overall["balanced_accuracy"],
                    "unavailable_recall": overall["unavailable_recall"],
                    "unavailable_precision": overall["unavailable_precision"],
                    "unavailable_f1": overall["unavailable_f1"],
                    "pr_auc_unavailable": overall["pr_auc_unavailable"],
                },
                "per_eta": _per_eta_metrics(y_te, proba, eta_te),
            }
        )

    if not folds:
        return None

    def _agg(key: str) -> dict[str, float | None]:
        vals = [f["overall"][key] for f in folds if f["overall"].get(key) is not None]
        if not vals:
            return {"mean": None, "std": None}
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    # ETA별 fold 평균
    eta_keys = sorted({row["eta_minutes"] for f in folds for row in f["per_eta"]})
    per_eta_summary = []
    for h in eta_keys:
        aucs, recalls = [], []
        for f in folds:
            match = next((r for r in f["per_eta"] if r["eta_minutes"] == h), None)
            if not match:
                continue
            if match["roc_auc"] is not None:
                aucs.append(match["roc_auc"])
            recalls.append(match["unavailable_recall"])
        per_eta_summary.append(
            {
                "eta_minutes": h,
                "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
                "roc_auc_std": float(np.std(aucs, ddof=0)) if aucs else None,
                "unavailable_recall_mean": float(np.mean(recalls)) if recalls else None,
                "unavailable_recall_std": float(np.std(recalls, ddof=0)) if recalls else None,
                "n_folds": len(aucs) if aucs else len(recalls),
            }
        )

    return {
        "n_folds": len(folds),
        "folds": folds,
        "summary": {
            "roc_auc": _agg("roc_auc"),
            "accuracy": _agg("accuracy"),
            "balanced_accuracy": _agg("balanced_accuracy"),
            "unavailable_recall": _agg("unavailable_recall"),
            "unavailable_precision": _agg("unavailable_precision"),
            "unavailable_f1": _agg("unavailable_f1"),
            "pr_auc_unavailable": _agg("pr_auc_unavailable"),
        },
        "per_eta_summary": per_eta_summary,
    }


def evaluate_threshold_sweep(
    y_true: pd.Series,
    proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or THRESHOLDS
    rows = []
    for t in thresholds:
        m = evaluate_classifier(y_true, proba, threshold=t)
        rows.append(
            {
                "threshold": t,
                "unavailable_precision": m["unavailable_precision"],
                "unavailable_recall": m["unavailable_recall"],
                "unavailable_f1": m["unavailable_f1"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
            }
        )

    # warn: 가용 판정 임계값을 올려 사용불가 Recall을 높임. unavailable F1 최대
    best = max(rows, key=lambda r: (r["unavailable_f1"], r["unavailable_recall"]))
    base = next((r for r in rows if abs(r["threshold"] - RANK_THRESHOLD) < 1e-9), rows[0])

    return {
        "sweep": rows,
        "recommended": {
            "rank_threshold": RANK_THRESHOLD,
            "warn_threshold": best["threshold"],
            "rank_unavailable_recall": base["unavailable_recall"],
            "warn_unavailable_recall": best["unavailable_recall"],
            "warn_unavailable_precision": best["unavailable_precision"],
            "warn_unavailable_f1": best["unavailable_f1"],
            "note": (
                "rank_threshold: 추천 순위용(기본 0.5). "
                "warn_threshold: 사용불가 경고용(가용 확률 상향 → 불가 Recall↑)."
            ),
        },
    }


def write_horizon_calibrations(
    y_true: pd.Series,
    proba: np.ndarray,
    eta: pd.Series,
    out_dir: Path,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    tmp = pd.DataFrame({"y": y_true.values, "proba": proba, "eta_minutes": eta.values})
    for h, g in sorted(tmp.groupby("eta_minutes"), key=lambda x: x[0]):
        path = out_dir / f"calibration_h{int(h)}.png"
        plot_calibration_curves(
            {f"H={int(h)}min": (g["y"].to_numpy(), g["proba"].to_numpy())},
            path,
        )
        table = calibration_table(g["y"], g["proba"])
        summaries.append(
            {
                "eta_minutes": int(h),
                "n": int(len(g)),
                "path": str(path),
                "bins": table.to_dict(orient="records"),
            }
        )
    return summaries


def _print_speed_segment_summary(date_holdout: dict[str, Any]) -> None:
    for name in ("servable", "rapid_only", "slow_only"):
        seg = date_holdout.get(name)
        if not seg:
            print(f"  {name}: (없음)")
            continue
        o = seg["overall"]
        print(
            f"  {name}: n={seg['n']:,}  unavail={seg['unavailable_rate']:.1%}  "
            f"AUC={o.get('roc_auc')}  불가R={o.get('unavailable_recall')}  "
            f"Brier={o.get('brier_score')}"
        )
        if name == "rapid_only":
            for h in (15, 20, 30):
                row = next(
                    (r for r in seg.get("per_eta") or [] if r["eta_minutes"] == h),
                    None,
                )
                if row:
                    print(
                        f"    ETA {h}m: 불가R={row.get('unavailable_recall')}  "
                        f"AUC={row.get('roc_auc')}  n={row.get('n')}"
                    )


def run_full_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    artifacts_dir: Path,
    *,
    include_rolling: bool = True,
    include_logistic: bool = True,
) -> dict[str, Any]:
    """전체 홀드아웃 평가 실행. 내부 `_` 키는 JSON 저장 전 제거."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("날짜 홀드아웃 평가 중...")
    date_holdout = evaluate_date_holdout(X, y, meta)
    if date_holdout is None:
        print("  날짜가 부족해 date holdout을 건너뜁니다.")
    else:
        _print_speed_segment_summary(date_holdout)

    baseline_logistic: dict[str, Any] | None = None
    if include_logistic and date_holdout is not None:
        print("LogisticRegression 기준선(동일 holdout) 평가 중...")
        try:
            baseline_logistic = evaluate_logistic_baseline(
                X,
                y,
                meta,
                date_holdout["_is_train"],
                train_dates=date_holdout.get("train_dates"),
                test_dates=date_holdout.get("test_dates"),
            )
            lo = baseline_logistic["overall"]
            ho = date_holdout["overall"]
            print(
                f"  LR  AUC={lo['roc_auc']}  Brier={lo['brier_score']}  "
                f"불가R={lo['unavailable_recall']}"
            )
            print(
                f"  HGB AUC={ho['roc_auc']}  Brier={ho.get('brier_score')}  "
                f"불가R={ho['unavailable_recall']}"
            )
        except Exception as exc:  # noqa: BLE001 — 기준선 실패해도 HGB 평가는 유지
            print(f"  Logistic baseline 실패(무시): {exc}")
            baseline_logistic = {"error": str(exc)}

    rolling = None
    if include_rolling:
        print("Rolling 날짜 fold 평가 중...")
        rolling = evaluate_rolling_folds(X, y, meta, min_train_days=2)
        if rolling is None:
            print("  fold 생성 불가 — rolling 평가를 건너뜁니다.")
        else:
            s = rolling["summary"]["roc_auc"]
            print(
                f"  folds={rolling['n_folds']}  "
                f"AUC mean={s['mean']}  std={s['std']}"
            )

    thresholds = None
    calibrations = []
    if date_holdout is not None:
        print("임계값 스윕 / horizon calibration...")
        thresholds = evaluate_threshold_sweep(
            date_holdout["_y_test"], date_holdout["_proba"]
        )
        calibrations = write_horizon_calibrations(
            date_holdout["_y_test"],
            date_holdout["_proba"],
            date_holdout["_eta_test"],
            artifacts_dir,
        )
        # overall calibration
        plot_calibration_curves(
            {
                "HistGradientBoosting": (
                    date_holdout["_y_test"].to_numpy(),
                    date_holdout["_proba"],
                ),
                "항상 사용 가능(기준모델)": (
                    date_holdout["_y_test"].to_numpy(),
                    np.full(len(date_holdout["_y_test"]), 0.999),
                ),
            },
            artifacts_dir / "calibration_overall.png",
        )
        rapid = date_holdout.get("rapid_only")
        if rapid and rapid.get("_y") is not None and len(rapid["_y"]) > 0:
            plot_calibration_curves(
                {
                    "rapid_only": (
                        np.asarray(rapid["_y"]),
                        np.asarray(rapid["_proba"]),
                    ),
                },
                artifacts_dir / "calibration_rapid_only.png",
            )
            print(f"  rapid calibration: {artifacts_dir / 'calibration_rapid_only.png'}")

    # strip private keys for JSON
    date_public = None
    if date_holdout is not None:
        _segs = ("rapid_only", "slow_only", "servable")
        date_public = {
            k: v
            for k, v in date_holdout.items()
            if not str(k).startswith("_") and k not in _segs
        }
        for _s in _segs:
            date_public[_s] = _public_segment(date_holdout.get(_s))

    return {
        "date_holdout": date_public,
        "baseline_logistic": baseline_logistic,
        "rolling": rolling,
        "thresholds": thresholds,
        "horizon_calibration": calibrations,
    }


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
