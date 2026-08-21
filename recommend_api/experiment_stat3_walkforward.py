"""실험: stat=3 블렌드 비중을 **워크포워드**로 재검증한다.

experiment_stat3_blend.py 는 창이 2.1일뿐이었다. `horizon_hgb.joblib` 이
2026-08-10 까지 학습해서 그 이전은 전부 in-sample 이었기 때문이다. 여기서는
**폴드마다 HGB 를 새로 학습**해 수집 시작 직후부터 전 구간을 out-of-sample 로 만든다.

설계
----
확장창(expanding). 폴드 k 는 cut 이전 전부로 학습하고 cut ~ cut+3일 을 평가한다.
평가 대상은 **급속 stat=3 행**이고, 비교 대상은

    HGB 단독
    운영 블렌드    0.25*HGB + 0.75*free_by_eta_score(잔여시간 ML)
    해저드 블렌드  0.25*HGB + 0.75*(충전소 해저드 x 종료->가용 전환)
    + 비중 스윕

해저드 테이블과 전환율도 폴드 cut 이전 데이터로만 만든다. 잔여시간 ML 은 외부
아티팩트라 재학습할 수 없다 — 2026-07-24 학습본이고 status 수집(2026-07-22~)과
기간이 사실상 겹치지 않으므로 누수로 보지 않는다. 다만 폴드마다 고정이라는 점은
감안할 것(뒤쪽 폴드일수록 그 모델에는 불리하지도 유리하지도 않다).

정본과 다른 점 (의도적)
---------------------
지평을 [10,20,30] 으로 좁혔다. 정본은 7개(5~60)이고 전체 1,038만 샘플인데
로컬 가용 RAM 이 6GB 라 폴드마다 그걸 재학습할 수 없다. 그래서

  - 이 실험의 **절대 지표를 운영 모델 성능으로 읽지 말 것.**
  - 폴드 간·모델 간 **비교**는 같은 조건에서 하므로 유효하다. 결론(비중)은
    그 비교에서만 낸다.

eta_minutes 가 피처라 10/20/30 만 학습하면 그 사이 보간은 못 배운다. 평가도 같은
세 값에서만 하므로 이 실험 안에서는 문제되지 않는다.

초기 폴드 주의
------------
첫 폴드는 학습이 5일뿐이다. 지표가 낮게 나오는 건 "모델이 나쁘다"가 아니라
"그때는 데이터가 없었다"이다. 폴드 간 **모델 순위**만 읽을 것.

사용:
  py -m recommend_api.experiment_stat3_walkforward
  py -m recommend_api.experiment_stat3_walkforward --test-days 3 --min-train-days 5
  py -m recommend_api.experiment_stat3_walkforward --json data/reports/x.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import (
    FEATURE_COLS,
    HGB_BLEND_WEIGHT,
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    REMAINING_BLEND_WEIGHT,
)
from .experiment_session_hazard import (
    ELAPSED_BINS,
    SHRINK_BUCKET,
    SHRINK_GLOBAL,
    SHRINK_STATION,
    _auc,
    _brier,
    _eb,
    _expand_risk_set,
    _logloss,
    attach_output,
    bucket_of,
    build_runs,
    charging_sessions,
    ml_remaining_minutes,
)
from .model_store import build_horizon_dataset, load_joined, load_label_series
from .remaining_time import free_by_eta_score, model_available

ETAS = [10, 20, 30]
WEIGHTS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]


def _new_model() -> HistGradientBoostingClassifier:
    """holdout_eval._new_model 과 동일 하이퍼파라미터."""
    return HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
    )


def _hazard_for(sess: pd.DataFrame, rows: pd.DataFrame, eta: int) -> np.ndarray:
    """cut 이전 세션으로 (충전소, 경과구간) 해저드를 만들어 rows 에 붙인다."""
    TR = _expand_risk_set(sess, eta)
    g_prior = float(TR["y"].mean())
    t_g = TR.groupby("ebin")["y"].agg(gk="sum", gn="size").reset_index()
    t_b = TR.groupby(["bucket", "ebin"])["y"].agg(bk="sum", bn="size").reset_index()
    t_s = TR.groupby(["stat_id", "ebin"])["y"].agg(sk="sum", sn="size").reset_index()
    R = (
        rows.merge(t_g, on="ebin", how="left")
        .merge(t_b, on=["bucket", "ebin"], how="left")
        .merge(t_s, on=["stat_id", "ebin"], how="left")
    )
    for c in ("gk", "gn", "bk", "bn", "sk", "sn"):
        R[c] = R[c].fillna(0.0)
    pg = np.where(R["gn"] > 0, _eb(R["gk"], R["gn"], g_prior, SHRINK_GLOBAL), g_prior)
    pb = _eb(R["bk"], R["bn"], pg, SHRINK_BUCKET)
    return np.asarray(_eb(R["sk"], R["sn"], pb, SHRINK_STATION))


def _conversion(sess: pd.DataFrame, runs: pd.DataFrame, cut: pd.Timestamp, eta: int) -> float:
    """P(도착 시점 stat==2 | ETA 안에 세션 종료), cut 이전 세션만으로."""
    steps = runs[["key", "start", "stat"]].sort_values(["start", "key"]).reset_index(drop=True)
    rows = []
    for e in (5, 15, 30, 45, 60, 90):
        s = sess[sess["dur_min"] > e]
        if s.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "key": s["key"].values,
                    "t_arr": s["start"].values + pd.to_timedelta(e + eta, unit="m"),
                    "ended": (s["dur_min"] <= e + eta).astype(int).values,
                }
            )
        )
    if not rows:
        return 1.0
    P = pd.concat(rows, ignore_index=True)
    P = P[(P["ended"] == 1) & (P["t_arr"] < cut)]
    if len(P) < 200:
        return 1.0
    P = P.sort_values("t_arr")
    P = pd.merge_asof(
        P,
        steps.rename(columns={"start": "t_arr", "stat": "arr_stat"}),
        on="t_arr",
        by="key",
        direction="backward",
    ).dropna(subset=["arr_stat"])
    return float((P["arr_stat"].astype(int) == 2).mean())


def _build_matrix(hz: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """make_xy 와 같은 설계행렬을 float32 ndarray 로. 폴드마다 재생성하면 RAM 이 못 버틴다."""
    type_d = pd.get_dummies(hz["chger_type"].fillna("UNK"), prefix="ctype")
    kind_d = pd.get_dummies(hz["kind"].fillna("UNK"), prefix="kind")
    busi_d = pd.get_dummies(hz["busi_id"].fillna("UNK"), prefix="busi")
    X = pd.concat([hz[FEATURE_COLS], type_d, kind_d, busi_d], axis=1)
    del type_d, kind_d, busi_d
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    cols = X.columns.tolist()
    Xv = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    del X
    gc.collect()
    return Xv, cols


def _build_meta(hz: pd.DataFrame) -> pd.DataFrame:
    elapsed = pd.to_numeric(hz["time_since_charge_started"], errors="coerce")
    elapsed = elapsed.fillna(pd.to_numeric(hz["capped_state_duration"], errors="coerce"))
    meta = pd.DataFrame(
        {
            "created_at": hz["created_at"].to_numpy(),
            "eta_minutes": hz["eta_minutes"].astype(int).to_numpy(),
            "stat": pd.to_numeric(hz["stat"], errors="coerce").fillna(0).astype(int).to_numpy(),
            "stat_id": hz["stat_id"].astype(str).to_numpy(),
            "output_kw": pd.to_numeric(hz["output_kw"], errors="coerce").to_numpy(),
            "chger_type": hz["chger_type"].astype(str).to_numpy(),
            "addr": hz["addr"].astype(str).to_numpy(),
            "elapsed": elapsed.fillna(0.0).clip(lower=0.0).to_numpy(),
        }
    )
    meta["bucket"] = meta["output_kw"].map(bucket_of)
    meta["ebin"] = pd.cut(
        meta["elapsed"], bins=ELAPSED_BINS, right=False, labels=ELAPSED_BINS[:-1]
    ).astype(float)
    meta["date"] = pd.to_datetime(meta["created_at"]).dt.normalize()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-days", type=int, default=3, help="폴드당 평가 구간(일)")
    ap.add_argument("--min-train-days", type=int, default=5, help="첫 폴드 최소 학습일")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    print("JOIN 로드 중...")
    df = load_joined()
    print(f"  {len(df):,}행  {df['created_at'].min()} ~ {df['created_at'].max()}")
    label_df = load_label_series()

    print(f"horizon 데이터셋 생성 (지평 {ETAS})...")
    hz = build_horizon_dataset(
        df, ETAS, label_df=label_df, method=LABEL_METHOD,
        max_staleness_min=LABEL_MAX_STALENESS_MIN,
    )
    del df, label_df
    gc.collect()
    print(f"  {len(hz):,} 샘플")

    Xv, cols = _build_matrix(hz)
    print(f"  설계행렬 {Xv.shape} float32 = {Xv.nbytes / 2**30:.2f} GB, 피처 {len(cols)}")
    y = hz["y"].astype(np.int8).to_numpy()
    meta = _build_meta(hz)
    del hz
    gc.collect()

    runs, _ = build_runs()
    runs = attach_output(runs)
    all_sess = charging_sessions(runs, fast_only=True)

    dates = np.sort(meta["date"].unique())
    print(f"  날짜 {len(dates)}일: {pd.Timestamp(dates[0]):%m-%d} ~ {pd.Timestamp(dates[-1]):%m-%d}")

    date_arr = meta["date"].to_numpy()
    is_target = (meta["stat"].to_numpy() == 3) & (meta["output_kw"].to_numpy() >= 50)
    out: dict[str, Any] = {
        "etas": ETAS,
        "n_features": len(cols),
        "n_samples": int(len(y)),
        "note": "지평 3개(정본 7개)로 축소 학습 — 절대 지표는 운영 모델과 다르다",
        "folds": [],
    }

    i = args.min_train_days
    while i < len(dates):
        cut = pd.Timestamp(dates[i])
        test_end = cut + pd.Timedelta(days=args.test_days)
        tr_idx = np.flatnonzero(date_arr < np.datetime64(cut))
        te_idx = np.flatnonzero(
            (date_arr >= np.datetime64(cut)) & (date_arr < np.datetime64(test_end)) & is_target
        )
        if len(tr_idx) < 10000 or len(te_idx) < 500:
            i += args.test_days
            continue

        ts = time.time()
        model = _new_model()
        model.fit(Xv[tr_idx], y[tr_idx])
        p_all = model.predict_proba(Xv[te_idx])[:, 1]
        fit_s = time.time() - ts

        sess = all_sess[all_sess["start"] < cut]
        M = meta.iloc[te_idx].reset_index(drop=True)
        yy_all = y[te_idx]

        fold: dict[str, Any] = {
            "cut": f"{cut:%Y-%m-%d}",
            "test_end": f"{test_end:%Y-%m-%d}",
            "train_days": int(i),
            "n_train": int(len(tr_idx)),
            "n_test_stat3_fast": int(len(te_idx)),
            "fit_sec": round(fit_s, 1),
            "sessions_for_hazard": int(len(sess)),
            "etas": {},
        }
        print(
            f"\n=== 폴드 cut {cut:%m-%d} (학습 {i}일 {len(tr_idx):,}샘플, "
            f"평가 {cut:%m-%d}~{test_end - pd.Timedelta(days=1):%m-%d} "
            f"stat3급속 {len(te_idx):,}) fit {fit_s:.0f}s ==="
        )

        for eta in ETAS:
            sel = M["eta_minutes"].to_numpy() == eta
            if sel.sum() < 300:
                continue
            R = M.loc[sel].reset_index(drop=True)
            yy = yy_all[sel]
            if yy.min() == yy.max():
                continue
            hgb = p_all[sel]

            conv = _conversion(sess, runs, cut, eta)
            haz_c = _hazard_for(sess, R, eta) * conv

            if model_available():
                probes = pd.DataFrame(
                    {
                        "t": R["created_at"].to_numpy(),
                        "time_since_charge_started": R["elapsed"].to_numpy(),
                        "output_kw": R["output_kw"].to_numpy(),
                        "chger_type": R["chger_type"].to_numpy(),
                        "addr": R["addr"].to_numpy(),
                    }
                )
                rem = ml_remaining_minutes(probes)
                comp_ml = np.array([free_by_eta_score(float(r), eta) for r in rem])
            else:
                comp_ml = np.full(len(yy), np.nan)

            prod = HGB_BLEND_WEIGHT * hgb + REMAINING_BLEND_WEIGHT * comp_ml
            prop = HGB_BLEND_WEIGHT * hgb + REMAINING_BLEND_WEIGHT * haz_c

            rec: dict[str, Any] = {
                "n": int(len(yy)),
                "actual": round(float(yy.mean()), 4),
                "conv": round(conv, 4),
                "models": {},
                "sweep": {},
            }
            print(f"  ETA={eta:2d}  n={len(yy):,}  실제 {yy.mean():.3f}  전환 {conv:.3f}")
            for name, p in (("HGB단독", hgb), ("운영블렌드", prod), ("해저드블렌드", prop)):
                if np.isnan(p).all():
                    continue
                b, a = _brier(yy, p), _auc(yy, p)
                rec["models"][name] = {
                    "brier": round(b, 4),
                    "auc": round(a, 4),
                    "logloss": round(_logloss(yy, p), 4),
                    "mean_pred": round(float(np.nanmean(p)), 4),
                }
                print(f"    {name:12s} brier {b:.4f}  auc {a:.4f}  평균 {np.nanmean(p):.3f}")
            for w in WEIGHTS:
                bm = (
                    _brier(yy, (1 - w) * hgb + w * comp_ml)
                    if not np.isnan(comp_ml).all()
                    else float("nan")
                )
                rec["sweep"][str(w)] = {
                    "ml": round(bm, 4),
                    "hazard": round(_brier(yy, (1 - w) * hgb + w * haz_c), 4),
                }
            fold["etas"][f"eta{eta}"] = rec

        out["folds"].append(fold)
        del model
        gc.collect()
        i += args.test_days

    _summarize(out)
    print(f"\n총 소요 {(time.time() - t0) / 60:.1f}분")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {p}")


def _summarize(out: dict[str, Any]) -> None:
    print("\n" + "=" * 84)
    print("폴드 종합 — Brier (낮을수록 좋음)")
    for eta in ETAS:
        rows = [(f["cut"], f["etas"].get(f"eta{eta}")) for f in out["folds"]]
        rows = [(c, r) for c, r in rows if r]
        if not rows:
            continue
        print(f"\nETA={eta}분")
        print(
            f"  {'cut':<8}{'n':>7}{'실제':>7}{'HGB단독':>10}{'운영0.75':>10}"
            f"{'해저드0.75':>11}{'최적w_ML':>10}{'최적w_haz':>11}"
        )
        agg: dict[str, list[float]] = {"hgb": [], "prod": [], "haz": []}
        for cut, r in rows:
            m, sw = r["models"], r["sweep"]
            best_ml = min(WEIGHTS, key=lambda w: sw[str(w)]["ml"])
            best_hz = min(WEIGHTS, key=lambda w: sw[str(w)]["hazard"])
            agg["hgb"].append(m["HGB단독"]["brier"])
            if "운영블렌드" in m:
                agg["prod"].append(m["운영블렌드"]["brier"])
            agg["haz"].append(m["해저드블렌드"]["brier"])
            prod_b = m.get("운영블렌드", {}).get("brier", float("nan"))
            print(
                f"  {cut:<8}{r['n']:>7,}{r['actual']:>7.3f}"
                f"{m['HGB단독']['brier']:>10.4f}{prod_b:>10.4f}"
                f"{m['해저드블렌드']['brier']:>11.4f}{best_ml:>10}{best_hz:>11}"
            )
        print(
            f"  {'평균':<8}{'':>7}{'':>7}{np.mean(agg['hgb']):>10.4f}"
            f"{(np.mean(agg['prod']) if agg['prod'] else float('nan')):>10.4f}"
            f"{np.mean(agg['haz']):>11.4f}   (폴드 {len(agg['hgb'])}개)"
        )


if __name__ == "__main__":
    main()
