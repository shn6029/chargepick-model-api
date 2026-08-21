"""station_horizon_training_v1.parquet 다중 모델 벤치마크.

DA① 풀데이터 팩(2026-08-04)의 충전소 단위 horizon 학습셋으로
여러 분류기를 같은 시간 분할·같은 피처셋에서 비교한다.

핵심 원칙
---------
1. 분할은 파일에 이미 들어 있는 `split`(train/valid/test) 을 그대로 쓴다.
   train 2026-07-22~07-29 / valid 07-30~08-01 / test 08-02~08-04 (시간 순).
   랜덤 분할 금지 — 같은 충전소의 인접 tick 이 train/test 에 갈라지면 누수다.
2. 라벨 파생 컬럼(label_*)과 도착 시점 관측 컬럼(target_*)은 피처에서 제외한다.
   `label_quality` 는 target_available 과 1:1 이라 넣으면 AUC 1.0 이 나온다.
3. 양성률 92.5% 이므로 accuracy 는 무의미하다. 판단은
   PR-AUC(사용불가 기준) · 사용불가 recall · Brier · ECE 로 한다.
4. 확률이 점수(available_prob)로 바로 나가므로 캘리브레이션을 같이 잰다.
   valid 를 prefit calibration 셋으로 쓴 변형도 함께 돌린다.

사용법
------
    py scripts/analysis/benchmark_station_models.py --sample 200000   # 빠른 확인
    py scripts/analysis/benchmark_station_models.py                   # 전량
    py scripts/analysis/benchmark_station_models.py --models hgb,xgb,lgbm
    py scripts/analysis/benchmark_station_models.py --featureset full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts" / "etl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# joblib 의 resource_tracker 는 임시폴더 경로를 ascii 로 인코딩해서 보낸다.
# 윈도우 기본 temp 가 C:\Users\<한글이름>\... 이면 UnicodeEncodeError 로 죽는다
# (EBM·RandomForest 등 멀티프로세싱을 쓰는 모델에서 발생). ascii 경로로 돌린다.
_JOBLIB_TMP = _ROOT / ".joblib_tmp"
_JOBLIB_TMP.mkdir(exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_JOBLIB_TMP))

from build_station_timeseries import (  # noqa: E402
    TIMESERIES_FEATURES,
    TRAINING_V1,
    TRAINING_V2,
)
from recommend_api.eval_metrics import evaluate_classifier  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_DATA = TRAINING_V1
OUT_DIR = _ROOT / "data" / "reports"

# 피처에서 반드시 빼야 하는 컬럼.
# label_*  : 라벨을 만들 때 나온 부산물. label_quality 는 y 와 1:1 (완전 누수).
# target_* : 도착 시점(t+h)에 관측된 값. 예측 시점에는 알 수 없다.
LEAKY_COLS = [
    "target_available",
    "label_reason",
    "label_quality",
    "label_source",
    "label_observed_at",
    "label_match_delta_minutes",
    "target_known_chargers",
    "target_total_chargers",
    "target_observation_coverage",
]
ID_COLS = [
    "station_id",
    "station_name",
    "feature_as_of",
    "target_time",
    "split",
    "source_snapshot_id",
    "feature_date",
]

# DA① 핸드오프 MVP_v1 제안 (usage 3종·요일/주말 제외)
FEATURESETS: dict[str, list[str]] = {
    "mvp": [
        "available_count",
        "total_chargers",
        "known_charger_count",
        "observation_coverage",
        "observation_age_minutes",
        "horizon_minutes",
    ],
    "mvp_plus": [
        "available_count",
        "total_chargers",
        "known_charger_count",
        "observation_coverage",
        "observation_age_minutes",
        "horizon_minutes",
        "minutes_since_last_change",
    ],
    # usage 2종(결측 92.7%)만 빼고 관측 계열 전부. 과적합 상한 확인용.
    "full": [
        "available_count",
        "usable_count",
        "known_charger_count",
        "direct_observed_count",
        "total_chargers",
        "observation_coverage",
        "direct_observation_coverage",
        "observation_age_minutes",
        "available_count_delta_1tick",
        "minutes_since_last_change",
        "unobserved_rate",
        "horizon_minutes",
        "hour",
        "weekday",
        "is_weekend",
    ],
}

# 정식 편입 피처셋. full + tick panel 시계열 8종.
# 이 피처셋은 v1 에 없는 컬럼을 쓰므로 학습셋 v2 가 필요하다
# (scripts/etl/build_station_timeseries.py 로 생성).
FEATURESETS["v2"] = FEATURESETS["full"] + TIMESERIES_FEATURES

# featureset 이름 → 필요한 학습셋 기본 경로
DATA_FOR_FEATURESET = {"v2": TRAINING_V2}


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """|평균예측 - 실제양성비율| 을 분위 구간별 표본수로 가중평균."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    df = pd.DataFrame({"y": y, "p": p})
    if np.ptp(p) == 0:  # 상수 예측(항상가용 기준선) — 구간이 하나뿐
        return float(abs(p[0] - y.mean()))
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=n_bins)
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mp=("p", "mean"), fp=("y", "mean")
    )
    return float((g["n"] * (g["mp"] - g["fp"]).abs()).sum() / g["n"].sum())


# ---------------------------------------------------------------- 모델 정의

def _hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = dict(max_depth=8, learning_rate=0.08, max_iter=250, random_state=42)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def _rf(**kw):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    params = dict(
        n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42
    )
    params.update(kw)
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**params)),
    ])


def _et():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42
        )),
    ])


def _mlp():
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64), early_stopping=True,
            max_iter=60, random_state=42
        )),
    ])


def _xgb(**kw):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=400, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    params.update(kw)
    return XGBClassifier(**params)


def _lgbm(**kw):
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=400, num_leaves=63, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=42, verbose=-1,
    )
    params.update(kw)
    return LGBMClassifier(**params)


def _catboost(**kw):
    from catboost import CatBoostClassifier

    params = dict(
        iterations=500, depth=8, learning_rate=0.08,
        verbose=0, random_seed=42, allow_writing_files=False,
    )
    params.update(kw)
    return CatBoostClassifier(**params)


def _ebm():
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(random_state=42)


# name -> (설명, 생성자)
MODELS: dict[str, tuple[str, Callable[[], Any]]] = {
    "hgb":          ("현행 정본 (HistGradientBoosting)", _hgb),
    "hgb_deep":     ("HGB max_depth=None·iter=400", lambda: _hgb(max_depth=None, max_iter=400)),
    "hgb_balanced": ("HGB class_weight=balanced", lambda: _hgb(class_weight="balanced")),
    "logreg":       ("로지스틱 회귀 (선형 기준선)", _logreg),
    "rf":           ("RandomForest", _rf),
    "rf_balanced":  ("RandomForest class_weight=balanced_subsample",
                     lambda: _rf(class_weight="balanced_subsample")),
    "et":           ("ExtraTrees", _et),
    "mlp":          ("MLP (128,64)", _mlp),
    "xgb":          ("XGBoost hist", _xgb),
    "xgb_spw":      ("XGBoost scale_pos_weight=0.08 (사용불가 가중)",
                     lambda: _xgb(scale_pos_weight=0.081)),
    "lgbm":         ("LightGBM", _lgbm),
    "lgbm_balanced":("LightGBM class_weight=balanced",
                     lambda: _lgbm(class_weight="balanced")),
    "catboost":     ("CatBoost (기본 캘리브레이션 우수)", _catboost),
    "ebm":          ("ExplainableBoostingMachine (설명 가능 GAM)", _ebm),
}

# 규칙 기준선 — 학습 없이 확률을 만드는 비교군
RULE_BASELINES = {
    "base_always_available": "항상 사용가능 (다수 클래스)",
    "base_current_state": "현재 available_count>0 이면 가용 (지속성 규칙)",
    "base_avail_ratio": "available_count / known_charger_count",
}


def rule_proba(name: str, X: pd.DataFrame, prior: float) -> np.ndarray:
    if name == "base_always_available":
        return np.full(len(X), min(prior, 0.999))
    if name == "base_current_state":
        has = (X["available_count"].fillna(0) > 0).to_numpy()
        return np.where(has, 0.95, 0.60)
    if name == "base_avail_ratio":
        num = X["available_count"].fillna(0)
        den = X["known_charger_count"].fillna(0).replace(0, np.nan)
        r = (num / den).fillna(prior).clip(0.02, 0.98)
        # 관측 없는 소는 사전확률로 되돌린다
        return r.to_numpy()
    raise ValueError(name)


# ---------------------------------------------------------------- 실행

def load_data(path: Path, sample: int | None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample and sample < len(df):
        # 시간 구조를 깨지 않도록 split 별 비율 유지 층화 추출.
        # groupby.apply 는 pandas 2.2 에서 그룹 키 컬럼을 떨궈서 인덱스로 뽑는다.
        frac = sample / len(df)
        idx = np.concatenate([
            g.sample(max(int(len(g) * frac), 1), random_state=seed).index.to_numpy()
            for _, g in df.groupby("split", sort=False)
        ])
        df = df.loc[idx].reset_index(drop=True)
    return df


def score_block(y, proba, horizons, threshold: float = 0.5) -> dict:
    m = evaluate_classifier(y, proba, threshold=threshold)
    out = {
        "n": m["n"],
        "roc_auc": m["roc_auc"],
        "pr_auc_unavailable": m["pr_auc_unavailable"],
        "brier": m["brier_score"],
        "log_loss": m["log_loss"],
        "ece": expected_calibration_error(y, proba),
        "unavailable_recall": m["unavailable_recall"],
        "unavailable_precision": m["unavailable_precision"],
        "unavailable_f1": m["unavailable_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "accuracy": m["accuracy"],
    }
    per_h = {}
    tmp = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba), "h": np.asarray(horizons)})
    for h, g in tmp.groupby("h"):
        if g["y"].nunique() < 2:
            continue
        mh = evaluate_classifier(g["y"], g["p"], threshold=threshold)
        per_h[int(h)] = {
            "n": mh["n"],
            "roc_auc": mh["roc_auc"],
            "pr_auc_unavailable": mh["pr_auc_unavailable"],
            "brier": mh["brier_score"],
            "unavailable_recall": mh["unavailable_recall"],
        }
    out["per_horizon"] = per_h
    return out


def run(args) -> dict:
    print(f"데이터 로드: {args.data}")
    df = load_data(Path(args.data), args.sample)
    feats = FEATURESETS[args.featureset]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"피처 없음: {missing}")

    leak_present = [c for c in LEAKY_COLS if c in feats]
    if leak_present:
        raise SystemExit(f"누수 컬럼이 피처셋에 있음: {leak_present}")

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "valid"]
    te = df[df["split"] == "test"]
    X_tr, y_tr = tr[feats], tr["target_available"].astype(int)
    X_va, y_va = va[feats], va["target_available"].astype(int)
    X_te, y_te = te[feats], te["target_available"].astype(int)
    prior = float(y_tr.mean())

    print(
        f"피처셋={args.featureset} ({len(feats)}개) | "
        f"train {len(tr):,} / valid {len(va):,} / test {len(te):,} | "
        f"train 양성률 {prior:.3%}"
    )

    wanted = (
        [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    )
    results: dict[str, Any] = {}

    # 규칙 기준선
    for name, desc in RULE_BASELINES.items():
        results[name] = {
            "desc": desc,
            "kind": "rule",
            "fit_sec": 0.0,
            "valid": score_block(y_va, rule_proba(name, X_va, prior), va["horizon_minutes"]),
            "test": score_block(y_te, rule_proba(name, X_te, prior), te["horizon_minutes"]),
        }
        print(f"  [rule] {name:22s} test AUC={results[name]['test']['roc_auc']:.4f}")

    for name in wanted:
        if name not in MODELS:
            print(f"  ! 알 수 없는 모델: {name} (건너뜀)")
            continue
        desc, ctor = MODELS[name]
        try:
            model = ctor()
        except ImportError as e:
            print(f"  - {name:22s} 라이브러리 없음 ({e.name}) → 건너뜀")
            results[name] = {"desc": desc, "skipped": f"missing:{e.name}"}
            continue

        t0 = time.perf_counter()
        try:
            model.fit(X_tr, y_tr)
        except Exception as e:  # 개별 모델 실패가 전체를 죽이지 않게
            print(f"  ! {name:22s} 학습 실패: {type(e).__name__}: {e}")
            results[name] = {"desc": desc, "error": f"{type(e).__name__}: {e}"}
            continue
        fit_sec = time.perf_counter() - t0

        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        entry = {
            "desc": desc,
            "kind": "model",
            "fit_sec": round(fit_sec, 1),
            "valid": score_block(y_va, p_va, va["horizon_minutes"]),
            "test": score_block(y_te, p_te, te["horizon_minutes"]),
        }

        # valid 로 prefit 캘리브레이션 → test 재평가 (확률이 점수로 나가므로 중요)
        if args.calibrate:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.frozen import FrozenEstimator

                # sklearn 1.6에서 cv="prefit" 제거됨 → FrozenEstimator 로 감싼다.
                cal = CalibratedClassifierCV(
                    FrozenEstimator(model), method="isotonic"
                )
                cal.fit(X_va, y_va)
                p_te_cal = cal.predict_proba(X_te)[:, 1]
                entry["test_calibrated"] = score_block(
                    y_te, p_te_cal, te["horizon_minutes"]
                )
            except Exception as e:
                entry["calibration_error"] = f"{type(e).__name__}: {e}"

        results[name] = entry
        t = entry["test"]
        print(
            f"  [{name:14s}] AUC={t['roc_auc']:.4f} "
            f"PR-AUC(불가)={t['pr_auc_unavailable']:.4f} "
            f"Brier={t['brier']:.4f} ECE={t['ece']:.4f} "
            f"불가recall={t['unavailable_recall']:.3f} ({fit_sec:.0f}s)"
        )

    return {
        "data_path": str(args.data),
        "featureset": args.featureset,
        "features": feats,
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
        "n_test": int(len(te)),
        "train_positive_rate": prior,
        "sample": args.sample,
        "calibrated": bool(args.calibrate),
        "results": results,
    }


def to_markdown(payload: dict) -> str:
    rows = []
    for name, r in payload["results"].items():
        if "test" not in r:
            rows.append((name, r.get("desc", ""), None))
            continue
        t = r["test"]
        rows.append((name, r["desc"], t))
    rows.sort(key=lambda x: -(x[2]["pr_auc_unavailable"] or 0) if x[2] else 1)

    lines = [
        f"# 충전소 horizon 모델 벤치마크 ({payload['featureset']})",
        "",
        f"- 데이터: `{Path(payload['data_path']).name}`",
        f"- 피처 {len(payload['features'])}개: {', '.join(payload['features'])}",
        f"- train {payload['n_train']:,} / valid {payload['n_valid']:,} / test {payload['n_test']:,}"
        f" · train 양성률 {payload['train_positive_rate']:.2%}",
        "",
        "정렬 기준은 **PR-AUC(사용불가)** 다. 양성률이 92.5%라 accuracy·AUC 는 잘 안 벌어진다.",
        "",
        "| 모델 | 설명 | ROC-AUC | PR-AUC(불가) | Brier | ECE | 불가 recall | 학습(초) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, desc, t in rows:
        if t is None:
            lines.append(f"| `{name}` | {desc} | — | — | — | — | — | 건너뜀 |")
            continue
        fit = payload["results"][name].get("fit_sec", 0)
        lines.append(
            f"| `{name}` | {desc} | {t['roc_auc']:.4f} | {t['pr_auc_unavailable']:.4f} "
            f"| {t['brier']:.4f} | {t['ece']:.4f} | {t['unavailable_recall']:.3f} | {fit:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="기본: featureset 에 맞는 학습셋 (v2 는 v2 parquet)")
    ap.add_argument("--featureset", default="mvp_plus", choices=list(FEATURESETS))
    ap.add_argument("--models", default=None, help="쉼표 구분. 기본=전체")
    ap.add_argument("--sample", type=int, default=None, help="층화 추출 행 수")
    ap.add_argument("--calibrate", action="store_true", help="valid prefit isotonic 추가")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    args = ap.parse_args()

    if args.data is None:
        args.data = str(DATA_FOR_FEATURESET.get(args.featureset, DEFAULT_DATA))
    if not Path(args.data).exists():
        raise SystemExit(
            f"학습셋이 없습니다: {args.data}\n"
            "먼저 `py scripts/etl/build_station_timeseries.py` 를 실행하세요."
        )

    payload = run(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"station_model_bench_{args.featureset}"
    out_json = Path(args.out) if args.out else OUT_DIR / f"{stem}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(payload), encoding="utf-8")
    print(f"\n저장: {out_json}\n      {out_md}")


if __name__ == "__main__":
    main()
