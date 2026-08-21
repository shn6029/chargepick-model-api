"""최종 후보 모델 rolling fold 대결.

단판(train/valid/test 고정 분할) 결과는 PR-AUC 4째 자리에서 갈렸다.
그 정도 차이는 fold 하나 바꾸면 뒤집히므로, 날짜 expanding fold 로
**차이가 재현되는지**를 본다. 순위 자체보다 폴드 간 표준편차가 중요하다.

프로토콜
--------
- fold: train D0..Dk-1 → test Dk (recommend_api.eval_metrics.iter_expanding_date_folds)
- 날짜 = feature_as_of 기준. 충전소 단위 tick 이라 같은 날 안에서는 섞이지만
  fold 경계는 항상 날짜이므로 미래 정보가 train 으로 새지 않는다.
- 각 fold 마다 train 중앙값으로 결측 대치 (fold 밖 통계 사용 금지).

판정
----
기준선(HGB) 대비 폴드별 PR-AUC 차이를 모아 mean ± std 로 본다.
|mean| < std 면 "차이 없음"이 결론이다. 승패 카운트도 같이 낸다.

사용법
------
    py scripts/analysis/rolling_model_race.py
    py scripts/analysis/rolling_model_race.py --min-train-days 7 --sample 400000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_station_models import (  # noqa: E402
    DATA_FOR_FEATURESET,
    DEFAULT_DATA,
    FEATURESETS,
    MODELS,
    OUT_DIR,
    expected_calibration_error,
    load_data,
)
from recommend_api.eval_metrics import (  # noqa: E402
    evaluate_classifier,
    iter_expanding_date_folds,
)

# 사용자 확정 최종 경쟁군 + 참고 기준점
FINALISTS = ["hgb", "lgbm", "xgb", "catboost", "rf"]
BASELINE = "hgb"


def fold_metrics(y_true, proba) -> dict[str, float]:
    m = evaluate_classifier(y_true, proba, threshold=0.5)
    return {
        "n": m["n"],
        "roc_auc": m["roc_auc"],
        "pr_auc_unavailable": m["pr_auc_unavailable"],
        "brier": m["brier_score"],
        "ece": expected_calibration_error(y_true, proba),
        "unavailable_recall": m["unavailable_recall"],
        "unavailable_precision": m["unavailable_precision"],
    }


def run(args) -> dict[str, Any]:
    df = load_data(Path(args.data), args.sample)
    feats = FEATURESETS[args.featureset]
    X_all = df[feats]
    y_all = df["target_available"].astype(int)
    meta = df[["feature_as_of", "horizon_minutes"]]

    names = [m.strip() for m in args.models.split(",")] if args.models else FINALISTS
    ready: dict[str, Any] = {}
    for name in names:
        if name not in MODELS:
            print(f"  ! 알 수 없는 모델: {name} (건너뜀)")
            continue
        try:
            MODELS[name][1]()  # 임포트 가능 여부만 확인
            ready[name] = MODELS[name]
        except ImportError as e:
            print(f"  - {name}: 라이브러리 없음 ({e.name}) → 건너뜀")

    folds = list(
        iter_expanding_date_folds(
            meta, date_col="feature_as_of", min_train_days=args.min_train_days
        )
    )
    print(
        f"피처셋={args.featureset}({len(feats)}) · 행 {len(df):,} · "
        f"fold {len(folds)}개 · 모델 {list(ready)}"
    )

    per_fold: dict[str, list[dict]] = {n: [] for n in ready}
    for fi, (is_train, train_dates, test_date) in enumerate(folds, 1):
        is_test = pd.to_datetime(meta["feature_as_of"]).dt.date == test_date
        X_tr_raw, y_tr = X_all[is_train], y_all[is_train]
        X_te_raw, y_te = X_all[is_test], y_all[is_test]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            print(f"  fold{fi} {test_date}: 단일 클래스 → 건너뜀")
            continue
        med = X_tr_raw.median(numeric_only=True)
        X_tr, X_te = X_tr_raw.fillna(med), X_te_raw.fillna(med)

        print(
            f"  fold{fi} train {len(train_dates)}일/{len(X_tr):,}행 "
            f"→ test {test_date}/{len(X_te):,}행 (양성률 {y_te.mean():.3%})"
        )
        for name, (_desc, ctor) in ready.items():
            t0 = time.perf_counter()
            model = ctor()
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            m = fold_metrics(y_te, proba)
            m["fold"] = fi
            m["test_date"] = str(test_date)
            m["fit_sec"] = round(time.perf_counter() - t0, 1)
            per_fold[name].append(m)
            print(
                f"      {name:10s} PR-AUC={m['pr_auc_unavailable']:.4f} "
                f"Brier={m['brier']:.4f} recall={m['unavailable_recall']:.3f}"
            )

    # 요약: mean ± std, 그리고 기준선 대비 폴드별 차이
    summary: dict[str, Any] = {}
    base_by_fold = {
        r["fold"]: r["pr_auc_unavailable"] for r in per_fold.get(BASELINE, [])
    }
    for name, rows in per_fold.items():
        if not rows:
            continue
        agg = {}
        for k in (
            "roc_auc", "pr_auc_unavailable", "brier", "ece",
            "unavailable_recall", "unavailable_precision",
        ):
            v = np.array([r[k] for r in rows], dtype=float)
            agg[f"{k}_mean"] = float(v.mean())
            agg[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg["n_folds"] = len(rows)
        agg["fit_sec_mean"] = float(np.mean([r["fit_sec"] for r in rows]))

        if name != BASELINE and base_by_fold:
            d = np.array(
                [
                    r["pr_auc_unavailable"] - base_by_fold[r["fold"]]
                    for r in rows
                    if r["fold"] in base_by_fold
                ],
                dtype=float,
            )
            if len(d):
                agg["delta_vs_baseline_mean"] = float(d.mean())
                agg["delta_vs_baseline_std"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                agg["wins_vs_baseline"] = int((d > 0).sum())
                agg["losses_vs_baseline"] = int((d < 0).sum())
                # 폴드 간 흔들림이 평균 차이보다 크면 우열을 주장할 수 없다
                agg["decisive"] = bool(
                    len(d) > 1 and abs(d.mean()) > d.std(ddof=1)
                )
        summary[name] = agg

    return {
        "data_path": str(args.data),
        "featureset": args.featureset,
        "features": feats,
        "baseline": BASELINE,
        "min_train_days": args.min_train_days,
        "n_rows": int(len(df)),
        "n_folds": len(folds),
        "sample": args.sample,
        "per_fold": per_fold,
        "summary": summary,
    }


def to_markdown(p: dict) -> str:
    base = p["baseline"]
    rows = sorted(
        p["summary"].items(),
        key=lambda kv: -kv[1]["pr_auc_unavailable_mean"],
    )
    lines = [
        "# 최종 후보 rolling fold 대결",
        "",
        f"- 데이터: `{Path(p['data_path']).name}` · {p['n_rows']:,}행",
        f"- 피처셋 `{p['featureset']}` ({len(p['features'])}개)",
        f"- expanding 날짜 fold {p['n_folds']}개 (최소 train {p['min_train_days']}일)",
        f"- 기준선: `{base}`",
        "",
        "`Δ` = 폴드별 PR-AUC(사용불가) 차이의 평균 ± 표준편차. ",
        "**|Δ평균| < Δ표준편차 이면 차이를 주장할 수 없다** (판정 열 참고).",
        "",
        "| 모델 | PR-AUC(불가) | Brier | ECE | 불가 recall | Δ vs " + base + " | 승/패 | 판정 | 학습(초) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, s in rows:
        pr = f"{s['pr_auc_unavailable_mean']:.4f} ± {s['pr_auc_unavailable_std']:.4f}"
        br = f"{s['brier_mean']:.4f}"
        ec = f"{s['ece_mean']:.4f}"
        rc = f"{s['unavailable_recall_mean']:.3f}"
        if name == base:
            dlt, wl, verdict = "—", "—", "기준선"
        elif "delta_vs_baseline_mean" in s:
            dlt = f"{s['delta_vs_baseline_mean']:+.4f} ± {s['delta_vs_baseline_std']:.4f}"
            wl = f"{s['wins_vs_baseline']}/{s['losses_vs_baseline']}"
            verdict = "**우세**" if s.get("decisive") and s["delta_vs_baseline_mean"] > 0 else (
                "**열세**" if s.get("decisive") else "차이 없음"
            )
        else:
            dlt, wl, verdict = "—", "—", "—"
        lines.append(
            f"| `{name}` | {pr} | {br} | {ec} | {rc} | {dlt} | {wl} | {verdict} "
            f"| {s['fit_sec_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="기본: featureset 에 맞는 학습셋 (v2 는 v2 parquet)")
    ap.add_argument("--featureset", default="mvp_plus", choices=list(FEATURESETS))
    ap.add_argument("--models", default=None, help=f"기본={','.join(FINALISTS)}")
    ap.add_argument("--min-train-days", type=int, default=5)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", default=None)
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
    stem = f"station_model_rolling_race_{args.featureset}"
    out_json = Path(args.out) if args.out else OUT_DIR / f"{stem}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_json.with_suffix(".md").write_text(to_markdown(payload), encoding="utf-8")
    print(f"\n저장: {out_json}\n      {out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
