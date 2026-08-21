"""충전소 단위 분해 — 풀링 지표가 가리는 것을 본다.

왜 필요한가
-----------
`date_holdout` 도 워크포워드도 전 충전기를 한 통에 넣고 잰다. 그래서 "h30 이상
불가 Recall 0.74~0.84" 가 **전 충전소에 고르게 퍼진 한계**인지, 아니면 **일부
충전소가 끌어내리는 것**인지 구별하지 못한다. 둘은 다음 할 일이 완전히 다르다.

  고르게 퍼져 있다  -> 모델·피처의 한계. 장거리 예측 자체를 손봐야 한다.
  일부가 끌어내린다 -> 그 충전소들의 공통점을 찾는 게 먼저다(사업자·출력·접근성 등).

모델은 충전소를 식별하지 않는다는 점을 유의할 것
------------------------------------------------
`FEATURE_COLS` 에 `stat_id`·`chger_id` 는 없다. 충전기 정체성은 두 번 기각됐다
(2026-08-03 요일 prior, 2026-08-12 세션 해저드 계층). 여기서 `stat_id` 는 **평가를
쪼개는 열쇠일 뿐 학습 입력이 아니다.** 설계행렬에도 들어가지 않는다.

무엇을 하나
-----------
1. DB 에서 새로 전개한다(`write_parts`). 기존 work-dir 의 parts 에는 `stat_id` 가
   없어서 재사용할 수 없다 — `META_COLS` 에 2026-08-18 에야 추가했다.
2. `date_holdout` 과 **같은 날짜 8:2 분할**로 홀드아웃 모델을 적합한다. 정본 joblib
   을 쓰지 않는 이유는 늘 같다 — 정본은 test 구간까지 학습했다.
3. test 예측을 충전소별로 쪼갠다. 충전소 하나의 표본은 작아서 개별 AUC 는 잡음이다.
   그래서 **분포와 집중도**를 본다:
     - 최소 표본을 넘긴 충전소들의 불가 Recall 십분위
     - 놓친 불가(FN)가 상위 몇 % 충전소에 몰려 있는가 (집중도 곡선)
     - 하위 충전소의 사업자·출력 구성이 전체와 다른가

사용:
  py scripts/analysis/station_breakdown_staged.py --work-dir F:/tmp/station_eval_20260818
  py scripts/analysis/station_breakdown_staged.py --work-dir DIR --eta 30 --min-unavail 50
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recommend_api.config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS
from recommend_api.eval_metrics import evaluate_classifier
from recommend_api.holdout_eval import _new_model
from recommend_api.staged_training import (
    MATRIX_DTYPE,
    MATRIX_MEMMAP_NAME,
    build_matrix,
    dummy_levels,
    exact_medians,
    feature_columns_of,
    part_files,
    write_parts,
)

CHUNK = 400_000
SEG_COLS = [
    "stat_id", "chger_id", "created_at", "eta_minutes", "is_fast", "y",
    "busi_id", "output_kw",
]
# 불가율 구간 — "Recall 이 낮은 충전소" 와 "불가가 드문 충전소" 를 가르기 위한 눈금.
# 2026-08-18 1차 실행에서 최저 Recall 15곳의 불가율이 1.4~8.0% 였다(전체 22.4%).
# 희소 사건이라 못 잡는 것인지 정말 못 맞히는 것인지 구별할 칸이 출력에 없었다.
UNAVAIL_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 1.01]
UNAVAIL_LABELS = ["<5%", "5~10%", "10~20%", "20~30%", ">=30%"]


def load_seg_meta(work: Path, n_rows: int) -> pd.DataFrame:
    """parts 에서 평가·분해에 쓸 컬럼만 HORIZONS 순서로 읽는다(build_matrix 와 같은 순서)."""
    frames = []
    for h in HORIZONS:
        t = pq.read_table(work / f"h{h:04d}.parquet", columns=SEG_COLS)
        frames.append(t.to_pandas())
        del t
        gc.collect()
    m = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    if len(m) != n_rows:
        raise SystemExit(f"메타 행수 불일치 {len(m):,} != {n_rows:,}")
    m["created_at"] = pd.to_datetime(m["created_at"])
    m["y"] = m["y"].astype(np.int8)
    m["eta_minutes"] = m["eta_minutes"].astype(np.int16)
    m["stat_id"] = m["stat_id"].astype("category")
    return m


def concentration(fn_per_station: pd.Series, total_fn: int) -> list[dict]:
    """놓친 불가(FN)가 상위 몇 % 충전소에 몰려 있는지 — 집중도 곡선."""
    s = fn_per_station.sort_values(ascending=False)
    n = len(s)
    out = []
    for pct in (5, 10, 20, 30, 50):
        k = max(1, int(n * pct / 100))
        out.append(
            {
                "top_pct_stations": pct,
                "n_stations": k,
                "fn_share": float(s.iloc[:k].sum() / total_fn) if total_fn else None,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="충전소 단위 홀드아웃 분해")
    ap.add_argument("--work-dir", required=True, help="새로 전개할 작업 디렉터리")
    ap.add_argument("--eta", type=int, default=30, help="집중 분석할 ETA(분)")
    ap.add_argument(
        "--min-unavail",
        type=int,
        default=50,
        help="충전소별 Recall 을 신뢰할 최소 불가 표본 수",
    )
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    meta_p = work / "parts_meta.json"
    if meta_p.exists() and all((work / f"h{h:04d}.parquet").exists() for h in HORIZONS):
        print("기존 parts 재사용")
        base_meta = json.loads(meta_p.read_text(encoding="utf-8"))
    else:
        base_meta = write_parts(work)
    n_rows = int(base_meta["n_rows"])

    files = part_files(work)
    levels = dummy_levels(files)
    feature_columns = feature_columns_of(levels)
    print(f"피처 {len(feature_columns)}개", flush=True)
    med = exact_medians(files, levels, n_rows)
    mm_path = work / MATRIX_MEMMAP_NAME[np.dtype(MATRIX_DTYPE).name]
    expect = n_rows * len(feature_columns) * np.dtype(MATRIX_DTYPE).itemsize
    if mm_path.exists() and mm_path.stat().st_size == expect:
        # 같은 parts + 같은 FEATURE_COLS 면 build_matrix 는 결정적이라 결과가 같다.
        # 열 수가 바뀌면 크기가 어긋나 자동으로 다시 만든다(약 7분 절약).
        print(f"기존 행렬 재사용 {mm_path.name}", flush=True)
        X = np.memmap(mm_path, dtype=MATRIX_DTYPE, mode="r", shape=(n_rows, len(feature_columns)))
        y_arr = None
    else:
        X, y_arr, _ = build_matrix(
            files, feature_columns, levels, med, n_rows, mm_path, dtype=MATRIX_DTYPE,
        )
    print(f"행렬 {n_rows:,} x {len(feature_columns)} ({time.time()-t0:.0f}초)", flush=True)

    seg = load_seg_meta(work, n_rows)
    if y_arr is None:  # 행렬을 재사용했으면 라벨은 parts 에서 온 것을 쓴다
        y_arr = seg["y"].to_numpy()
    dates = seg["created_at"].dt.date
    uniq = np.sort(dates.unique())
    split = max(1, int(len(uniq) * 0.8))
    tr = np.flatnonzero(dates.isin(set(uniq[:split])).to_numpy())
    te = np.flatnonzero(dates.isin(set(uniq[split:])).to_numpy())
    print(
        f"날짜 홀드아웃: train {split}일 {len(tr):,}행 / test {len(uniq)-split}일 {len(te):,}행",
        flush=True,
    )

    sub = work / "X_sb_train.memmap"
    Xt = np.memmap(sub, dtype=X.dtype, mode="w+", shape=(len(tr), X.shape[1]))
    for s in range(0, len(tr), CHUNK):
        e = min(s + CHUNK, len(tr))
        Xt[s:e] = X[tr[s:e]]
    Xt.flush()
    print("홀드아웃 HGB fit...", flush=True)
    model = _new_model()
    model.fit(Xt, y_arr[tr])
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
    del model, X
    gc.collect()
    print(f"예측 완료 ({time.time()-t0:.0f}초)", flush=True)

    t = seg.iloc[te].copy()
    t["proba"] = proba
    t["pred_unavail"] = proba < 0.5
    t["is_unavail"] = t["y"] == 0

    # --- test 예측 덤프 ---------------------------------------------------
    # 이게 없으면 세그먼트를 새로 자를 때마다 홀드아웃을 재적합해야 한다.
    # 2026-08-18 에 그 대가를 두 번 치렀다(임계값 25분 · 충전소 32분). 약 40MB 면
    # 사업자별·출력별·시간대별 어떤 분해든 재적합 없이 몇 초에 끝난다.
    pred_path = out_dir / "station_breakdown_predictions.parquet"
    t[[
        "stat_id", "chger_id", "created_at", "eta_minutes", "is_fast",
        "busi_id", "output_kw", "y", "proba",
    ]].to_parquet(pred_path, index=False, compression="zstd")
    print(f"예측 덤프: {pred_path} ({pred_path.stat().st_size/1e6:.1f}MB)", flush=True)

    result: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "scripts/analysis/station_breakdown_staged.py",
        "test_dates": [str(d) for d in uniq[split:]],
        "n_test": int(len(te)),
        "n_stations_test": int(t["stat_id"].nunique()),
        "eta_focus": args.eta,
        "min_unavail": args.min_unavail,
        "overall": evaluate_classifier(pd.Series(t["y"].to_numpy()), proba),
    }
    print(f"\n충전소 {result['n_stations_test']:,}곳 · test {len(te):,}행")

    f = t[t["eta_minutes"] == args.eta]
    g = f.groupby("stat_id", observed=True)
    st = pd.DataFrame({
        "n": g.size(),
        "n_unavail": g["is_unavail"].sum(),
        "tp": g.apply(lambda d: int((d["pred_unavail"] & d["is_unavail"]).sum())),
        "n_warn": g["pred_unavail"].sum(),
    })
    st["fn"] = st["n_unavail"] - st["tp"]
    st["recall"] = np.where(st["n_unavail"] > 0, st["tp"] / st["n_unavail"], np.nan)
    st["precision"] = np.where(st["n_warn"] > 0, st["tp"] / st["n_warn"], np.nan)
    st["unavail_rate"] = st["n_unavail"] / st["n"]
    st["busi_id"] = f.groupby("stat_id", observed=True)["busi_id"].first()
    total_fn = int(st["fn"].sum())
    total_unavail = int(st["n_unavail"].sum())

    # 충전소별 전체 표 저장 — 십분위·최저 15곳만 뽑고 버리던 것을 그대로 남긴다
    st_path = out_dir / "station_breakdown_stations.csv"
    st.reset_index().to_csv(st_path, index=False, encoding="utf-8-sig")
    print(f"충전소 표: {st_path} ({len(st):,}행)")

    print(f"\n=== ETA {args.eta}분 · 충전소 분해 ===")
    print(f"  전체 불가 {total_unavail:,} · 놓친 불가(FN) {total_fn:,} · "
          f"풀링 Recall {1 - total_fn/total_unavail:.4f}")

    ok = st[st["n_unavail"] >= args.min_unavail]
    print(f"  불가 표본 {args.min_unavail}건 이상 충전소: {len(ok):,}곳 "
          f"(전체 {len(st):,}곳 중, 불가의 {ok['n_unavail'].sum()/total_unavail*100:.1f}% 보유)")
    if len(ok):
        q = ok["recall"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print("  충전소별 Recall 십분위: " + " · ".join(
            f"p{int(k*100)} {v:.3f}" for k, v in q.items()))
        result["station_recall_quantiles"] = {f"p{int(k*100)}": float(v) for k, v in q.items()}
        result["n_stations_scored"] = int(len(ok))

    # --- 불가율 구간별 Recall: "희소해서 못 잡나, 정말 못 맞히나" ------------
    #
    # 2026-08-18 1차 실행에서 Recall 최저 15곳이 전부 불가율 1.4~8.0% 였다(전체 22.4%).
    # 잘 보정된 모델이 희소 사건에서 Recall 이 낮은 건 정상 동작이므로, 이걸 구별하지
    # 못하면 "이 충전소들이 고장났다" 는 틀린 결론으로 간다. 구간별로 갈라 본다.
    #   Recall 이 불가율과 함께 오른다  -> 희소 사건 아티팩트. 손댈 대상이 아니다.
    #   구간과 무관하게 평평/제각각     -> 정말 못 맞히는 충전소가 따로 있다.
    scored = st[st["n_unavail"] >= args.min_unavail].copy()
    bin_rows = []
    if len(scored):
        scored["bin"] = pd.cut(
            scored["unavail_rate"], bins=UNAVAIL_BINS, labels=UNAVAIL_LABELS, right=False
        )
        print(f"\n  불가율 구간별 (불가 {args.min_unavail}건 이상 {len(scored):,}곳):")
        print(f"    {'구간':<8} {'충전소':>7} {'평균Recall':>11} {'중앙Recall':>11} {'FN비중':>8}")
        for lab, grp in scored.groupby("bin", observed=True):
            if not len(grp):
                continue
            row = {
                "bin": str(lab),
                "n_stations": int(len(grp)),
                "recall_mean": float(grp["recall"].mean()),
                "recall_median": float(grp["recall"].median()),
                "fn_share": float(grp["fn"].sum() / total_fn) if total_fn else None,
                "unavail_rate_mean": float(grp["unavail_rate"].mean()),
            }
            bin_rows.append(row)
            share = "  n/a" if row["fn_share"] is None else f"{row['fn_share']*100:>6.1f}%"
            print(f"    {row['bin']:<8} {row['n_stations']:>7,} {row['recall_mean']:>11.3f} "
                  f"{row['recall_median']:>11.3f} {share:>8}")
        v = scored[["unavail_rate", "recall"]].dropna()
        # Recall 이 전 충전소에서 같으면(예: 전부 1.0) 상관이 정의되지 않아 nan 이 온다.
        if len(v) > 3 and v["recall"].nunique() > 1 and v["unavail_rate"].nunique() > 1:
            from scipy.stats import spearmanr

            rho, p = spearmanr(v["unavail_rate"], v["recall"])
            print(f"    불가율 vs Recall  스피어만 rho={rho:+.3f} (p={p:.2e}, n={len(v):,})")
            result["unavail_rate_vs_recall"] = {
                "spearman_rho": float(rho), "p_value": float(p), "n": int(len(v)),
            }
    result["unavail_rate_bins"] = bin_rows

    conc = concentration(st["fn"], total_fn)
    print("\n  놓친 불가의 집중도:")
    for c in conc:
        share = "n/a (FN 0건)" if c["fn_share"] is None else f"{c['fn_share']*100:.1f}%"
        print(f"    상위 {c['top_pct_stations']:>2}% 충전소({c['n_stations']:,}곳)가 "
              f"FN 의 {share}")
    result["fn_concentration"] = conc
    result["totals"] = {"n_unavail": total_unavail, "n_fn": total_fn,
                        "pooled_recall": float(1 - total_fn / total_unavail) if total_unavail else None}

    if len(ok):
        worst = ok.nsmallest(15, "recall")
        print(f"\n  Recall 최저 15곳 (불가 {args.min_unavail}건 이상):")
        print(f"    {'stat_id':<12} {'n':>7} {'불가':>7} {'Recall':>8}")
        rows = []
        for sid, r in worst.iterrows():
            print(f"    {str(sid):<12} {int(r['n']):>7,} {int(r['n_unavail']):>7,} {r['recall']:>8.3f}")
            rows.append({"stat_id": str(sid), "n": int(r["n"]),
                         "n_unavail": int(r["n_unavail"]), "recall": float(r["recall"])})
        result["worst_stations"] = rows

        # 하위 충전소가 특정 사업자·출력에 몰려 있는가
        lo = set(ok.nsmallest(max(1, len(ok) // 5), "recall").index)
        base = f.groupby("busi_id", observed=True).size()
        lo_mix = f[f["stat_id"].isin(lo)].groupby("busi_id", observed=True).size()
        mix = pd.DataFrame({"전체": base / base.sum(), "하위20%": lo_mix / lo_mix.sum()}).fillna(0)
        mix["차이"] = mix["하위20%"] - mix["전체"]
        top = mix.reindex(mix["차이"].abs().sort_values(ascending=False).index).head(8)
        print("\n  하위 20% 충전소의 사업자 구성 (전체 대비):")
        print(f"    {'busi_id':<12} {'전체':>8} {'하위20%':>8} {'차이':>8}")
        for b, r in top.iterrows():
            print(f"    {str(b):<12} {r['전체']*100:>7.1f}% {r['하위20%']*100:>7.1f}% {r['차이']*100:>+7.1f}%p")
        result["worst_quintile_busi_mix"] = {
            str(b): {"overall": float(r["전체"]), "worst20": float(r["하위20%"]),
                     "diff": float(r["차이"])} for b, r in top.iterrows()
        }

    out = out_dir / "station_breakdown.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {out}  (총 {(time.time()-t0)/60:.0f}분)")


if __name__ == "__main__":
    main()
