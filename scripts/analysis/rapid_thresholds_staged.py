"""급속 shadow warn 임계값 재산출 — **staged 학습 작업 디렉터리를 재사용**하는 저메모리 경로.

왜 새로 만들었나
----------------
`recommend_api/analyze_rapid_thresholds.py` 와 `scripts/analysis/eta_thresholds.py` 는
둘 다 `build_horizon_dataset(df, HORIZONS)` 로 7개 지평을 **한 번에** 전개한다.
`recommend_api/staged_training.py` 가 피하려고 만들어진 바로 그 피크다. 2026-08-10
시점에 이미 약 21.5GB 였고, 2026-08-18 재학습에서 base 행이 1,509,849 -> 2,107,920
(+39.6%) 으로 늘어 총 RAM 15.6GB 머신에서는 더 이상 돌지 않는다. 그런데 `config.py`
주석은 **재학습할 때마다 이 곡선을 다시 뽑으라**고 지시한다. 지시를 지킬 수 있는
경로가 없어진 셈이라, staged 가 이미 디스크에 만들어 둔 산출물을 그대로 쓴다.

무엇을 재사용하나 — `train --staged --work-dir DIR` 이 남긴 것
------------------------------------------------------------
    h{0005..0060}.parquet   지평별 전개 결과 (라벨·메타 포함)
    X_f32.memmap            정본과 같은 행 순서의 float32 설계행렬
    parts_meta.json         n_rows / 데이터 구간

행렬을 다시 만들지 않는다. 열 순서·중앙값 대체는 학습이 저장한 아티팩트의
`feature_columns` 로 확정되므로, memmap 을 (n_rows, len(feature_columns)) 로 열기만
하면 학습이 본 것과 **같은 행렬**이다. 라벨·메타(y·created_at·eta_minutes·is_fast)만
parquet 에서 4개 컬럼으로 다시 읽는다 — 컬럼 저장이라 수십 MB 다.

분할과 적합은 기존 두 스크립트와 같은 눈금
-----------------------------------------
- 날짜 8:2 분할. `eval_metrics.split_by_date(ratio=0.8)` 와
  `staged_training._date_holdout` 은 같은 규칙(`int(n_days*0.8)` 일까지 train)이라
  여기서도 그대로 쓴다. 즉 새 곡선은 metrics 의 `date_holdout` 과 같은 test 구간이다.
- 홀드아웃 모델은 `holdout_eval._new_model()` == `staged_training._new_hgb()`
  (max_depth=8 · lr=0.08 · max_iter=250 · seed 42). 정본 joblib 를 읽어 예측하지
  않는다 — 정본은 test 구간까지 학습했으므로 임계값을 그것으로 뽑으면 낙관된다.
- 적합은 한 번만 한다. 구 스크립트 두 개는 같은 분할에 같은 모델을 각자 적합했다.

사용:
  py scripts/analysis/rapid_thresholds_staged.py --work-dir F:/tmp/scheduler_train_20260818

산출(기본 `recommend_api/artifacts/`):
  rapid_threshold_analysis.json   목표 Recall(0.60/0.70/0.80) 별 최소 thr — 공통값 근거
  eta_thresholds.json             ETA별 곡선 + 정책 비교 — _BY_ETA 근거
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recommend_api.config import (
    ARTIFACTS_DIR,
    HORIZONS,
    MODEL_PATH,
    RAPID_SHADOW_WARN_THRESHOLD_BY_ETA,
    RAPID_SHADOW_WARN_THRESHOLD_COMMON,
)
from recommend_api.holdout_eval import _new_model
from recommend_api.staged_training import MATRIX_MEMMAP_NAME

# 구 스크립트와 같은 격자·정책 상수
GRID = np.round(np.arange(0.05, 0.996, 0.01), 2)
RECALL_TARGETS = [0.60, 0.70, 0.80]
TARGET_R = 0.70
WARN_CAP = 0.12
FIT_CHUNK = 400_000  # memmap -> memmap 복사 청크. _date_holdout 과 동일


def load_meta(work: Path, n_rows: int) -> pd.DataFrame:
    """parquet parts 에서 라벨·메타 4개 컬럼만 HORIZONS 순서로 이어 붙인다.

    build_matrix 가 X_f32.memmap 을 채운 순서와 **같아야** 행이 대응된다.
    staged_training.part_files 가 HORIZONS 순서를 강제하므로 여기서도 그대로 쓴다.

    `pd.concat` 으로 7개 조각을 합치지 않는다. 합치는 순간 조각 전부와 결과가
    동시에 살아 있어 피크가 두 배가 되는데, 이 스크립트가 도는 상황이 애초에
    "RAM 이 없어서 staged 로 학습한" 직후다(2026-08-18 실행 시 가용 540MB).
    build_matrix 와 같은 방식으로 미리 잡아 두고 조각별로 채운다 — 피크가
    한 조각(약 207만 행 x 4열)으로 묶인다.
    """
    y = np.empty(n_rows, dtype=np.int8)
    created = np.empty(n_rows, dtype="datetime64[ns]")
    eta = np.empty(n_rows, dtype=np.int16)
    is_fast = np.empty(n_rows, dtype=np.int8)

    off = 0
    for h in HORIZONS:
        f = work / f"h{h:04d}.parquet"
        if not f.exists():
            raise SystemExit(f"지평 parquet 누락: {f} — --work-dir 이 맞는지 확인할 것")
        t = pq.read_table(f, columns=["y", "created_at", "eta_minutes", "is_fast"])
        m = t.num_rows
        if off + m > n_rows:
            raise SystemExit(f"parts 행수가 parts_meta 의 n_rows({n_rows:,}) 를 넘는다")
        y[off : off + m] = t.column("y").to_numpy(zero_copy_only=False).astype(np.int8)
        created[off : off + m] = (
            t.column("created_at").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
        )
        eta[off : off + m] = (
            t.column("eta_minutes").to_numpy(zero_copy_only=False).astype(np.int16)
        )
        fast = t.column("is_fast").to_numpy(zero_copy_only=False).astype("float64")
        is_fast[off : off + m] = np.nan_to_num(fast, nan=0.0).astype(np.int8)
        off += m
        del t, fast
        gc.collect()
    if off != n_rows:
        raise SystemExit(f"적재 행수 불일치 {off:,} != {n_rows:,}")
    return pd.DataFrame(
        {"y": y, "created_at": created, "eta_minutes": eta, "is_fast": is_fast}
    )


def curve(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """thr 별 사용불가 recall/precision/warn 발화율. 사용불가 예측 = p < thr.

    격자가 91점이라 구 스크립트처럼 매번 전체 마스크를 만들면 91회 전량 스캔이 된다
    (급속 test 가 100만 행대라 무시할 수 없다). 한 번 정렬해 누적합으로 바꾸면
    thr 마다 `searchsorted` 한 번이면 되고 결과는 동일하다.
    """
    unavail = y == 0
    n_unavail = int(unavail.sum())
    n = len(y)
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    unavail_cum = np.cumsum(unavail[order])
    rows = []
    for thr in GRID:
        # p < thr 인 행 수 = 정렬된 p 에서 thr 의 왼쪽 삽입 위치
        k = int(np.searchsorted(p_sorted, thr, side="left"))
        tp = int(unavail_cum[k - 1]) if k > 0 else 0
        rows.append(
            {
                "threshold": float(thr),
                "unavailable_recall": tp / n_unavail if n_unavail else None,
                "unavailable_precision": tp / k if k else None,
                "warning_rate": k / n,
                "n": n,
            }
        )
    return rows


def pick_target(rows: list[dict], target: float) -> dict | None:
    """목표 Recall 을 처음 넘기는 가장 낮은 thr (thr↑ = 불가 판정 확대)."""
    return next((r for r in rows if (r["unavailable_recall"] or 0) >= target), None)


def pick_capped(rows: list[dict], cap: float) -> dict:
    ok = [r for r in rows if r["warning_rate"] <= cap]
    return max(ok, key=lambda r: r["unavailable_recall"] or 0)


def at(rows: list[dict], thr: float) -> dict:
    return min(rows, key=lambda r: abs(r["threshold"] - thr))


def apply_policy(y, p, eta, thr_by_eta: dict[int, float]) -> dict:
    pred = np.zeros(len(y), dtype=bool)
    for h, thr in thr_by_eta.items():
        m = eta == h
        pred[m] = p[m] < thr
    unavail = y == 0
    tp = int((pred & unavail).sum())
    return {
        "unavailable_recall": tp / int(unavail.sum()),
        "unavailable_precision": tp / int(pred.sum()) if pred.sum() else None,
        "warning_rate": float(pred.mean()),
        "n_warn": int(pred.sum()),
    }


def holdout_predict(work: Path, n_cols: int, meta: pd.DataFrame):
    """설계행렬 memmap 을 열어 날짜 8:2 분할 후 train 적합 -> test 예측.

    dtype 은 파일명으로 고른다(`MATRIX_MEMMAP_NAME`). 크기까지 대조해서, 학습과
    다른 work-dir 이나 다른 아티팩트를 물리면 조용히 어긋나는 대신 즉시 죽는다.
    """
    n_rows = len(meta)
    found = []
    for name, path in ((n, work / f) for n, f in MATRIX_MEMMAP_NAME.items()):
        if path.exists() and path.stat().st_size == n_rows * n_cols * np.dtype(name).itemsize:
            found.append((name, path))
    if not found:
        have = sorted(p.name for p in work.glob("X_*.memmap"))
        raise SystemExit(
            f"쓸 수 있는 memmap 이 없다 (기대 {n_rows:,}행 x {n_cols}열). "
            f"작업 디렉터리에 있는 것: {have or '없음'} — --work-dir/--model 이 "
            "학습과 같은 짝인지 확인할 것."
        )
    dtype_name, mm_path = found[0]
    print(f"설계행렬 {mm_path.name} (dtype={dtype_name})", flush=True)
    X = np.memmap(mm_path, dtype=np.dtype(dtype_name), mode="r", shape=(n_rows, n_cols))

    dates = meta["created_at"].dt.date
    unique_dates = np.sort(dates.unique())
    split = max(1, int(len(unique_dates) * 0.8))
    train_dates, test_dates = unique_dates[:split], unique_dates[split:]
    is_train = dates.isin(set(train_dates)).to_numpy()
    tr = np.flatnonzero(is_train)
    te = np.flatnonzero(~is_train)
    print(
        f"날짜 홀드아웃: train {len(train_dates)}일 {len(tr):,}행 / "
        f"test {len(test_dates)}일 {len(te):,}행 -> {[str(d) for d in test_dates]}",
        flush=True,
    )

    # train 부분집합도 memmap 으로 떠서 RAM 피크를 만들지 않는다(_date_holdout 과 동일)
    sub = work / "X_thr_train.memmap"
    Xt = np.memmap(sub, dtype=X.dtype, mode="w+", shape=(len(tr), n_cols))
    for s in range(0, len(tr), FIT_CHUNK):
        e = min(s + FIT_CHUNK, len(tr))
        Xt[s:e] = X[tr[s:e]]
    Xt.flush()
    y = meta["y"].to_numpy()
    print("홀드아웃 HGB fit...", flush=True)
    model = _new_model()
    model.fit(Xt, y[tr])
    del Xt
    gc.collect()
    try:
        sub.unlink()
    except OSError:
        pass

    proba = np.empty(len(te), dtype=np.float64)
    for s in range(0, len(te), FIT_CHUNK):
        e = min(s + FIT_CHUNK, len(te))
        proba[s:e] = model.predict_proba(X[te[s:e]])[:, 1]
    del model, X
    gc.collect()
    return proba, te, [str(d) for d in train_dates], [str(d) for d in test_dates]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="staged work-dir 재사용 급속 shadow warn 임계값 재산출"
    )
    ap.add_argument(
        "--work-dir", required=True, help="train --staged --work-dir 이 남긴 디렉터리"
    )
    ap.add_argument("--model", default=str(MODEL_PATH), help="열 수 확인용 아티팩트")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    art = joblib.load(args.model)
    feature_columns = art["feature_columns"]
    model_version = art.get("model_version")
    print(f"아티팩트 {model_version} · 피처 {len(feature_columns)}개", flush=True)

    parts_meta = json.loads((work / "parts_meta.json").read_text(encoding="utf-8"))
    n_rows = int(parts_meta["n_rows"])
    meta = load_meta(work, n_rows)
    print(
        f"메타 {len(meta):,}행 (학습 구간 {parts_meta.get('data_start')} ~ "
        f"{parts_meta.get('data_end')})",
        flush=True,
    )
    proba, te, train_dates, test_dates = holdout_predict(
        work, len(feature_columns), meta
    )

    m_te = meta.iloc[te]
    fast = m_te["is_fast"].to_numpy() == 1
    y_arr = m_te["y"].to_numpy()[fast]
    p_arr = proba[fast]
    eta_arr = m_te["eta_minutes"].to_numpy()[fast].astype(int)
    unavail_rate = float(1 - y_arr.mean())
    print(f"급속 test n={len(y_arr):,} 사용불가율={unavail_rate:.4f}", flush=True)

    # --- (1) 급속 전체 곡선: 목표 Recall 별 최소 thr = 공통값 근거 ---
    overall_rows = curve(y_arr, p_arr)
    targets = []
    print("\n== 급속 전체: 목표 Recall 별 최소 thr ==")
    for t in RECALL_TARGETS:
        r = pick_target(overall_rows, t)
        targets.append({"target_recall": t, **(r or {})})
        if r:
            print(
                f" R>={t:.2f} -> thr {r['threshold']:.2f}  "
                f"R {r['unavailable_recall']:.4f}  P {r['unavailable_precision']:.4f}  "
                f"warn {r['warning_rate']*100:.2f}%"
            )
        else:
            print(f" R>={t:.2f} -> 달성불가")
    cur_common = at(overall_rows, RAPID_SHADOW_WARN_THRESHOLD_COMMON)
    print(
        f" (현행 공통 {RAPID_SHADOW_WARN_THRESHOLD_COMMON:.2f} -> "
        f"R {cur_common['unavailable_recall']:.4f} P {cur_common['unavailable_precision']:.4f} "
        f"warn {cur_common['warning_rate']*100:.2f}%)"
    )

    # --- (2) ETA별 곡선 + 정책 비교 ---
    per_eta: dict[int, Any] = {}
    thr_B: dict[int, float] = {}
    thr_C: dict[int, float] = {}
    print(
        "\n ETA |        n | 불가율 | 현행별 R/P/warn | R>=0.70 thr(P,warn) | warn<=12% R(thr,P)"
    )
    for h in sorted(set(eta_arr.tolist())):
        m = eta_arr == h
        rows = curve(y_arr[m], p_arr[m])
        cur = at(
            rows,
            RAPID_SHADOW_WARN_THRESHOLD_BY_ETA.get(
                int(h), RAPID_SHADOW_WARN_THRESHOLD_COMMON
            ),
        )
        tgt = pick_target(rows, TARGET_R)
        cap = pick_capped(rows, WARN_CAP)
        thr_B[int(h)] = float(tgt["threshold"]) if tgt else float(GRID[-1])
        thr_C[int(h)] = float(cap["threshold"])
        per_eta[int(h)] = {
            "n": int(m.sum()),
            "unavailable_rate": float(1 - y_arr[m].mean()),
            "at_current": cur,
            "target_r70": tgt,
            "warn_capped": cap,
        }
        tgt_s = (
            f"{tgt['threshold']:.2f} (P{tgt['unavailable_precision']:.3f}, {tgt['warning_rate']*100:4.1f}%)"
            if tgt
            else "달성불가"
        )
        print(
            f" h{h:<3}|{int(m.sum()):>9,}| {1-y_arr[m].mean():.3f} | "
            f"{cur['unavailable_recall']:.3f}/{cur['unavailable_precision']:.3f}/{cur['warning_rate']*100:4.1f}% | "
            f"{tgt_s} | {cap['unavailable_recall']:.3f} (thr{cap['threshold']:.2f}, P{cap['unavailable_precision']:.3f})",
            flush=True,
        )

    policies = {
        "A_common_current": apply_policy(
            y_arr,
            p_arr,
            eta_arr,
            {h: RAPID_SHADOW_WARN_THRESHOLD_COMMON for h in per_eta},
        ),
        "A2_per_eta_current": apply_policy(
            y_arr,
            p_arr,
            eta_arr,
            {
                h: RAPID_SHADOW_WARN_THRESHOLD_BY_ETA.get(
                    h, RAPID_SHADOW_WARN_THRESHOLD_COMMON
                )
                for h in per_eta
            },
        ),
        "B_per_eta_r70": apply_policy(y_arr, p_arr, eta_arr, thr_B),
        "C_per_eta_warn_cap12": apply_policy(y_arr, p_arr, eta_arr, thr_C),
    }
    print("\n== 급속 전체 정책 비교 ==")
    for k, v in policies.items():
        print(
            f" {k:<22} R={v['unavailable_recall']:.4f} P={v['unavailable_precision']:.4f} "
            f"warn={v['warning_rate']*100:.2f}% (n_warn={v['n_warn']:,})"
        )

    common = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "scripts/analysis/rapid_thresholds_staged.py",
        "model_version": model_version,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "n_rapid_test": int(len(y_arr)),
        "rapid_unavailable_rate": unavail_rate,
    }
    (out_dir / "rapid_threshold_analysis.json").write_text(
        json.dumps(
            {
                **common,
                "recall_targets": targets,
                "at_current_common": cur_common,
                "current_common_threshold": RAPID_SHADOW_WARN_THRESHOLD_COMMON,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (out_dir / "eta_thresholds.json").write_text(
        json.dumps(
            {
                **common,
                "current_common_threshold": RAPID_SHADOW_WARN_THRESHOLD_COMMON,
                "current_by_eta": RAPID_SHADOW_WARN_THRESHOLD_BY_ETA,
                "policy_target_recall": TARGET_R,
                "warn_cap": WARN_CAP,
                "per_eta": per_eta,
                "thresholds_B_r70": thr_B,
                "thresholds_C_warn_cap": thr_C,
                "policy_comparison": policies,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {out_dir/'rapid_threshold_analysis.json'}")
    print(f"저장: {out_dir/'eta_thresholds.json'}")


if __name__ == "__main__":
    main()
