"""확장창 워크포워드 평가 — **staged 학습 작업 디렉터리를 재사용**하는 저메모리 경로.

왜 필요한가 — 창 하나짜리 지표는 창의 기저율에 인질로 잡힌다
------------------------------------------------------------
`date_holdout` 은 마지막 20% 하루뭉치 하나로만 잰다. 그 창의 기저 가용률이
흔들리면 모델 성능과 무관하게 지표가 통째로 움직인다. 2026-08-18 재학습에서
실제로 겪었다:

    구 모델 date_holdout (test 08-07~08-10, 기저 0.7581)  AUC 0.9707  Brier 0.0437
    신 모델 date_holdout (test 08-13~08-18, 기저 0.7736)  AUC 0.9746  Brier 0.0389
    -> "AUC +0.0040 개선" 처럼 보인다

    같은 08-13~18 행에 두 모델을 태우면                    AUC +0.0002  Brier -0.0002
    -> 개선의 실체는 0. 전부 창 차이였다.

폴드가 여러 개면 **폴드 평균과 표준편차**가 나오므로 이 착시를 구조적으로 막는다.
"차이가 폴드 표준편차보다 작으면 신호가 아니다" 를 바로 읽을 수 있다.

`holdout_eval.evaluate_rolling_folds` 와 무엇이 다른가
-----------------------------------------------------
정책(확장창·폴드마다 재학습·같은 하이퍼파라미터)은 같고 출력 구조도 맞췄다
(`summary` / `per_eta_summary` / `folds`). 다른 것은 두 가지다.

1. **입력.** 그쪽은 pandas DataFrame 을 받는다. 현재 데이터량(14,498,475 x 105)
   에서는 설계행렬만 float64 로 12.18GB 라 RAM 에 못 올린다. 여기서는
   `train --staged --work-dir DIR` 이 남긴 memmap 을 그대로 연다.
2. **폴드 간격.** 그쪽은 테스트일마다(=하루 간격) 재학습이라 28일이면 26폴드다.
   누적 학습 행이 약 1억 9천만이라 이 머신에서 7시간대다. 여기서는 간격을
   `--test-days` 로 열어 뒀고 기본 3일이다(8폴드 · 누적 약 6,400만 행 · 약 2.5시간).
   폴드끼리 학습 데이터가 대부분 겹치므로 하루 간격이 3배의 정보를 주지는 않는다.

초기 폴드 주의
--------------
첫 폴드는 학습이 `--min-train-days`(기본 5일)뿐이다. 지표가 낮은 건 "모델이 나쁘다"
가 아니라 "그때는 데이터가 없었다" 이다. 폴드 간 **추이**와 뒤쪽 폴드를 읽을 것.

메모리
------
폴드마다 train 부분집합을 memmap 으로 떠서 적합한다(`_date_holdout` 과 같은 방식).
sklearn 1.9.0 HGB 는 `check_array(dtype=[float64])` 라 float64 memmap 만 제로카피로
읽는다 — float32 로 만든 work-dir 을 물리면 업캐스트 복사가 생겨 피크가 두 배가 된다.
상세는 `recommend_api/staged_training.py` 독스트링.

사용:
  py scripts/analysis/walkforward_staged.py --work-dir F:/tmp/scheduler_train_20260818
  py scripts/analysis/walkforward_staged.py --work-dir DIR --test-days 2 --write-metrics

산출:
  artifacts/walkforward.json           폴드별 + 요약
  --write-metrics 를 주면 horizon_hgb_metrics.json 의 `rolling` 키에 요약을 병합한다.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from recommend_api.config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH
from recommend_api.eval_metrics import evaluate_classifier
from recommend_api.holdout_eval import _new_model
from recommend_api.staged_training import MATRIX_MEMMAP_NAME

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from rapid_thresholds_staged import load_meta  # noqa: E402

CHUNK = 400_000
SUMMARY_KEYS = [
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "unavailable_recall",
    "unavailable_precision",
    "unavailable_f1",
    "pr_auc_unavailable",
    "brier_score",
]


def open_matrix(work: Path, n_rows: int, n_cols: int) -> np.memmap:
    """work-dir 의 설계행렬 memmap 을 연다. 크기까지 대조해 짝이 안 맞으면 즉시 죽는다."""
    for name in ("float64", "float32"):
        p = work / MATRIX_MEMMAP_NAME[name]
        if p.exists() and p.stat().st_size == n_rows * n_cols * np.dtype(name).itemsize:
            if name == "float32":
                print(
                    "  경고: float32 memmap 이다. sklearn 이 적합마다 float64 로 전량 "
                    "복사하므로 RAM 피크가 두 배가 된다.",
                    flush=True,
                )
            print(f"설계행렬 {p.name} (dtype={name})", flush=True)
            return np.memmap(p, dtype=np.dtype(name), mode="r", shape=(n_rows, n_cols))
    have = sorted(x.name for x in work.glob("X_*.memmap"))
    raise SystemExit(
        f"쓸 수 있는 memmap 이 없다 (기대 {n_rows:,}행 x {n_cols}열). 있는 것: {have or '없음'}"
    )


def per_eta_metrics(y: np.ndarray, proba: np.ndarray, eta: np.ndarray) -> list[dict]:
    """holdout_eval._per_eta_metrics 와 같은 필드 집합."""
    rows = []
    for h in sorted(set(eta.tolist())):
        m = eta == h
        mm = evaluate_classifier(pd.Series(y[m]), proba[m])
        rows.append(
            {
                "eta_minutes": int(h),
                "n": mm["n"],
                "roc_auc": mm["roc_auc"],
                "unavailable_recall": mm["unavailable_recall"],
                "unavailable_precision": mm["unavailable_precision"],
                "unavailable_f1": mm["unavailable_f1"],
                "accuracy": mm["accuracy"],
                "balanced_accuracy": mm["balanced_accuracy"],
                "brier_score": mm["brier_score"],
                "pr_auc_unavailable": mm["pr_auc_unavailable"],
            }
        )
    return rows


def run_fold(
    X: np.memmap, y: np.ndarray, tr: np.ndarray, te: np.ndarray, work: Path
) -> np.ndarray:
    """train 부분집합을 memmap 으로 떠서 적합하고 test 예측을 돌려준다."""
    sub = work / "X_wf_train.memmap"
    Xt = np.memmap(sub, dtype=X.dtype, mode="w+", shape=(len(tr), X.shape[1]))
    for s in range(0, len(tr), CHUNK):
        e = min(s + CHUNK, len(tr))
        Xt[s:e] = X[tr[s:e]]
    Xt.flush()
    model = _new_model()
    model.fit(Xt, y[tr])
    del Xt
    gc.collect()
    try:
        sub.unlink()
    except OSError:
        pass

    proba = np.empty(len(te), dtype=np.float64)
    for s in range(0, len(te), CHUNK):
        e = min(s + CHUNK, len(te))
        proba[s:e] = model.predict_proba(X[te[s:e]])[:, 1]
    del model
    gc.collect()
    return proba


def main() -> None:
    ap = argparse.ArgumentParser(description="staged work-dir 재사용 확장창 워크포워드")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--test-days", type=int, default=3, help="폴드당 평가 구간(일)")
    ap.add_argument("--min-train-days", type=int, default=5, help="첫 폴드 최소 학습일")
    ap.add_argument("--model", default=str(MODEL_PATH), help="열 수 확인용 아티팩트")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    ap.add_argument(
        "--write-metrics",
        action="store_true",
        help="horizon_hgb_metrics.json 의 `rolling` 키에 요약을 병합한다",
    )
    args = ap.parse_args()

    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    art = joblib.load(args.model)
    n_cols = len(art["feature_columns"])
    model_version = art.get("model_version")
    print(f"아티팩트 {model_version} · 피처 {n_cols}개", flush=True)

    parts_meta = json.loads((work / "parts_meta.json").read_text(encoding="utf-8"))
    n_rows = int(parts_meta["n_rows"])
    meta = load_meta(work, n_rows)
    print(f"메타 {len(meta):,}행", flush=True)
    X = open_matrix(work, n_rows, n_cols)

    dates = meta["created_at"].dt.date
    date_arr = dates.to_numpy()
    uniq = np.sort(dates.unique())
    y = meta["y"].to_numpy()
    eta_all = meta["eta_minutes"].to_numpy()
    print(f"전체 {len(uniq)}일: {uniq[0]} ~ {uniq[-1]}", flush=True)

    folds: list[dict[str, Any]] = []
    i = args.min_train_days
    t0 = time.time()
    while i < len(uniq):
        train_dates = uniq[:i]
        test_dates = uniq[i : i + args.test_days]
        if len(test_dates) == 0:
            break
        tr = np.flatnonzero(np.isin(date_arr, train_dates))
        te = np.flatnonzero(np.isin(date_arr, test_dates))
        if len(te) == 0 or len(tr) == 0:
            i += args.test_days
            continue
        print(
            f"\n=== 폴드 {len(folds)+1}: 학습 {len(train_dates)}일 {len(tr):,}행 "
            f"-> 평가 {test_dates[0]}~{test_dates[-1]} {len(te):,}행 ===",
            flush=True,
        )
        fs = time.time()
        proba = run_fold(X, y, tr, te, work)
        overall = evaluate_classifier(pd.Series(y[te]), proba)
        rec = {
            "fold": len(folds) + 1,
            "train_dates": [str(d) for d in train_dates],
            "test_dates": [str(d) for d in test_dates],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "positive_rate_test": float(y[te].mean()),
            "overall": {k: overall.get(k) for k in SUMMARY_KEYS},
            "per_eta": per_eta_metrics(y[te], proba, eta_all[te]),
        }
        fast = meta["is_fast"].to_numpy()[te] == 1
        if fast.sum() > 200:
            rm = evaluate_classifier(pd.Series(y[te][fast]), proba[fast])
            rec["rapid_only"] = {k: rm.get(k) for k in SUMMARY_KEYS + ["n"]}
        folds.append(rec)
        print(
            f"  AUC {overall['roc_auc']:.4f} · Brier {overall['brier_score']:.4f} · "
            f"불가R {overall['unavailable_recall']:.4f} · 기저 {y[te].mean():.4f} "
            f"({time.time()-fs:.0f}초, 누적 {(time.time()-t0)/60:.0f}분)",
            flush=True,
        )
        del proba
        gc.collect()
        i += args.test_days

    if not folds:
        raise SystemExit("폴드가 만들어지지 않았다 — --min-train-days 를 확인할 것")

    def agg(key: str) -> dict[str, float | None]:
        vals = [f["overall"][key] for f in folds if f["overall"].get(key) is not None]
        if not vals:
            return {"mean": None, "std": None}
        a = np.asarray(vals, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}

    eta_keys = sorted({r["eta_minutes"] for f in folds for r in f["per_eta"]})
    per_eta_summary = []
    for h in eta_keys:
        aucs, recalls, briers = [], [], []
        for f in folds:
            m = next((r for r in f["per_eta"] if r["eta_minutes"] == h), None)
            if not m:
                continue
            if m["roc_auc"] is not None:
                aucs.append(m["roc_auc"])
            if m["unavailable_recall"] is not None:
                recalls.append(m["unavailable_recall"])
            if m["brier_score"] is not None:
                briers.append(m["brier_score"])
        per_eta_summary.append(
            {
                "eta_minutes": h,
                "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
                "roc_auc_std": float(np.std(aucs, ddof=0)) if aucs else None,
                "unavailable_recall_mean": float(np.mean(recalls)) if recalls else None,
                "unavailable_recall_std": float(np.std(recalls, ddof=0)) if recalls else None,
                "brier_score_mean": float(np.mean(briers)) if briers else None,
                "brier_score_std": float(np.std(briers, ddof=0)) if briers else None,
                "n_folds": len(aucs),
            }
        )

    result = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "scripts/analysis/walkforward_staged.py",
        "model_version": model_version,
        "method": "expanding_window",
        "test_days": args.test_days,
        "min_train_days": args.min_train_days,
        "n_folds": len(folds),
        "oos_dates": [folds[0]["test_dates"][0], folds[-1]["test_dates"][-1]],
        "summary": {k: agg(k) for k in SUMMARY_KEYS},
        "per_eta_summary": per_eta_summary,
        "folds": folds,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }

    print("\n=== 폴드 요약 (평균 ± 표준편차) ===")
    for k in SUMMARY_KEYS:
        s = result["summary"][k]
        if s["mean"] is not None:
            print(f"  {k:24} {s['mean']:.4f} ± {s['std']:.4f}")
    print("\n=== ETA별 (폴드 평균) ===")
    print(f"  {'ETA':>4} | {'AUC':>16} | {'Brier':>16} | {'불가R':>16}")
    for r in per_eta_summary:
        print(
            f"  {r['eta_minutes']:>4} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
            f"{r['brier_score_mean']:.4f} ± {r['brier_score_std']:.4f} | "
            f"{r['unavailable_recall_mean']:.4f} ± {r['unavailable_recall_std']:.4f}"
        )

    out = out_dir / "walkforward.json"
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n저장: {out}")

    if args.write_metrics:
        mp = Path(METRICS_PATH)
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["rolling"] = {
            k: result[k]
            for k in (
                "method",
                "test_days",
                "min_train_days",
                "n_folds",
                "oos_dates",
                "summary",
                "per_eta_summary",
                "generated_at",
                "source",
            )
        }
        mp.write_text(
            json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"metrics 병합: {mp} (rolling)")


if __name__ == "__main__":
    main()
