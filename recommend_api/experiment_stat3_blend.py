"""실험: 충전중(stat=3) **전체 서빙 점수**를 재현해 블렌드 비중을 정한다.

experiment_session_hazard.py 는 잔여시간 **성분 단독**을 비교했다. 운영 점수는

    available_prob = HGB_BLEND_WEIGHT * HGB + REMAINING_BLEND_WEIGHT * 성분
                   = 0.25 * HGB        + 0.75 * free_by_eta_score(...)

라 HGB 가 성분의 오차를 일부 상쇄할 수 있다. 여기서 그 상쇄분을 직접 잰다.

Out-of-time 창을 다시 잡는 이유
------------------------------
`horizon_hgb.joblib` 의 `training_data_end` 는 2026-08-10 15:14 다. 세션 해저드
실험에서 쓴 홀드아웃 분할점(08-06 14:14)은 **HGB 학습 구간 안**이라, 그 창에서
HGB 를 재면 in-sample 이라 과대평가된다. 그래서 이 실험은 평가창을
`training_data_end` **이후**로 잡고, 해저드 테이블도 그 이전 세션으로만 만든다.
두 성분 모두 평가창을 못 본 상태가 된다.

대가는 창이 짧다는 것이다(약 2일). 라벨 품질이 날마다 달라 일 단위 지표가
±0.02 쯤 흔들리므로(스냅샷 앵커 주기 영향), **소수점 셋째 자리를 읽지 말 것.**
결론은 그보다 큰 차이에서만 낸다.

사용:
  py -m recommend_api.experiment_stat3_blend
  py -m recommend_api.experiment_stat3_blend --horizon-model <경로>
  py -m recommend_api.experiment_stat3_blend --json data/reports/x.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    HGB_BLEND_WEIGHT,
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    MODEL_PATH,
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
from .model_store import (
    align_features,
    build_horizon_dataset,
    load_joined,
    load_label_series,
    make_xy,
)
from .remaining_time import free_by_eta_score, model_available

ETAS = [10, 20, 30]


def _load_horizon_artifact(path: str | None) -> dict[str, Any]:
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"horizon 모델이 없습니다: {p}\n"
            "--horizon-model 로 경로를 주거나 artifacts/ 에 두세요 "
            "(워크트리에는 *.joblib 이 git 으로 안 따라옵니다)."
        )
    return joblib.load(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon-model", default=None, help="horizon_hgb.joblib 경로")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    art = _load_horizon_artifact(args.horizon_model)
    train_end = pd.Timestamp(art["training_data_end"])
    print(f"HGB {art['model_version']}  학습 {art['training_data_start']} ~ {train_end}")
    print(f"잔여시간 ML 모델: {'사용' if model_available() else '없음'}")

    # ---------------------------------------------------------- 평가 표본
    print("\nJOIN 로드 중...")
    df = load_joined()
    kw = pd.to_numeric(df.get("output_kw"), errors="coerce")
    oot = df[(kw >= 50) & (df["created_at"] > train_end)].copy()
    print(f"  전체 {len(df):,}행 → 급속·OOT {len(oot):,}행 "
          f"({oot['created_at'].min()} ~ {oot['created_at'].max()})")
    del df

    label_df = load_label_series()
    hz = build_horizon_dataset(
        oot, ETAS, label_df=label_df, method=LABEL_METHOD,
        max_staleness_min=LABEL_MAX_STALENESS_MIN,
    )
    hz = hz[hz["stat"].astype(int) == 3].reset_index(drop=True)
    print(f"  stat=3 평가 샘플 {len(hz):,} (충전기 {hz.groupby(['stat_id','chger_id']).ngroups:,}대)")

    # ---------------------------------------------------------- HGB 예측
    X, y_s, _, _ = make_xy(hz, fill_median=True)
    X = align_features(X, art["feature_columns"], art["medians"])
    p_hgb = art["model"].predict_proba(X.astype(art.get("input_dtype", "float32")))[:, 1]
    y = y_s.to_numpy()
    del X

    # ---------------------------------------------------------- 해저드 테이블
    # 평가창을 못 보게 train_end 이전 세션만 쓴다.
    runs, _ = build_runs()
    runs = attach_output(runs)
    sess = charging_sessions(runs, fast_only=True)
    sess = sess[sess["start"] < train_end]
    print(f"  해저드 학습 세션 {len(sess):,}건 (~{train_end:%m-%d %H:%M})")

    hz["bucket"] = pd.to_numeric(hz["output_kw"], errors="coerce").map(bucket_of)
    elapsed = pd.to_numeric(hz.get("time_since_charge_started"), errors="coerce")
    elapsed = elapsed.fillna(pd.to_numeric(hz.get("capped_state_duration"), errors="coerce"))
    hz["elapsed"] = elapsed.fillna(0.0).clip(lower=0.0)
    hz["ebin"] = pd.cut(hz["elapsed"], bins=ELAPSED_BINS, right=False,
                        labels=ELAPSED_BINS[:-1]).astype(float)
    print(f"  평가 표본 경과시간 중앙 {hz['elapsed'].median():.0f}분 "
          f"/ 45분 초과 {float((hz['elapsed'] > 45).mean())*100:.1f}%")

    out: dict[str, Any] = {
        "model_version": art["model_version"],
        "train_end": str(train_end),
        "n_samples": int(len(hz)),
        "etas": {},
    }

    for eta in ETAS:
        m = hz["eta_minutes"].astype(int) == eta
        if m.sum() < 500:
            continue
        H = hz.loc[m]
        yy = y[m.to_numpy()]
        hgb = p_hgb[m.to_numpy()]

        # 해저드: (충전소, 경과구간) → 부모로 shrink
        TR = _expand_risk_set(sess, eta)
        g_prior = float(TR["y"].mean())
        t_g = TR.groupby("ebin")["y"].agg(gk="sum", gn="size").reset_index()
        t_b = TR.groupby(["bucket", "ebin"])["y"].agg(bk="sum", bn="size").reset_index()
        t_s = TR.groupby(["stat_id", "ebin"])["y"].agg(sk="sum", sn="size").reset_index()
        R = (
            H.merge(t_g, on="ebin", how="left")
            .merge(t_b, on=["bucket", "ebin"], how="left")
            .merge(t_s, on=["stat_id", "ebin"], how="left")
        )
        for c in ("gk", "gn", "bk", "bn", "sk", "sn"):
            R[c] = R[c].fillna(0.0)
        pg = np.where(R["gn"] > 0, _eb(R["gk"], R["gn"], g_prior, SHRINK_GLOBAL), g_prior)
        pb = _eb(R["bk"], R["bn"], pg, SHRINK_BUCKET)
        haz = np.asarray(_eb(R["sk"], R["sn"], pb, SHRINK_STATION))

        # 종료→가용 전환율도 train 기간 세션에서 뽑는다(평가창 누수 방지)
        conv = _conversion_rate(sess, runs, train_end, eta)
        haz_c = haz * conv

        # 잔여시간 ML 성분.
        # build_horizon_dataset 결과에는 중복 컬럼명이 섞여 있어(merge_asof 산물)
        # 그대로 넘기면 remaining_time._elapsed_min 의 row.get() 이 Series 를 받아
        # 터진다. 모델이 실제로 읽는 컬럼만 뽑아 최소 프레임으로 만든다.
        probes = pd.DataFrame(
            {
                "t": H["created_at"].to_numpy(),
                "time_since_charge_started": H["elapsed"].to_numpy(),
                "output_kw": pd.to_numeric(H["output_kw"], errors="coerce").to_numpy(),
                "chger_type": _first_col(H, "chger_type"),
                "addr": _first_col(H, "addr"),
            }
        )
        if model_available():
            rem = ml_remaining_minutes(probes)
            comp_ml = np.array([free_by_eta_score(float(r), eta) for r in rem])
        else:
            comp_ml = np.full(len(yy), np.nan)

        prod = HGB_BLEND_WEIGHT * hgb + REMAINING_BLEND_WEIGHT * comp_ml
        prop = HGB_BLEND_WEIGHT * hgb + REMAINING_BLEND_WEIGHT * haz_c

        print(f"\n=== ETA={eta}분  n={len(yy):,}  실제 가용률 {yy.mean():.3f} "
              f"| 종료→가용 전환 {conv:.3f} ===")
        rows = {}
        for name, p in (
            ("HGB 단독", hgb),
            ("성분 ML 단독", comp_ml),
            ("성분 해저드 단독", haz_c),
            ("운영 블렌드 0.25/0.75(ML)", prod),
            ("제안 블렌드 0.25/0.75(해저드)", prop),
        ):
            if np.isnan(p).all():
                continue
            ll, br, au = _logloss(yy, p), _brier(yy, p), _auc(yy, p)
            print(f"  {name:28s} logloss {ll:.4f}  brier {br:.4f}  auc {au:.4f}  평균 {p.mean():.3f}")
            rows[name] = {"logloss": round(ll, 4), "brier": round(br, 4),
                          "auc": round(au, 4), "mean_pred": round(float(p.mean()), 4)}

        # 비중 스윕 — 지금 0.75 가 맞는 값인가
        print("  성분 비중 스윕 (Brier):")
        sweep = {}
        for w in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
            b_ml = _brier(yy, (1 - w) * hgb + w * comp_ml) if not np.isnan(comp_ml).all() else float("nan")
            b_hz = _brier(yy, (1 - w) * hgb + w * haz_c)
            sweep[str(w)] = {"ml": round(b_ml, 4), "hazard": round(b_hz, 4)}
            print(f"    w={w:<5} ML {b_ml:.4f}   해저드 {b_hz:.4f}")

        # 해저드 증분이 잡음인지 — 충전소 클러스터 부트스트랩.
        # 같은 충전소 행끼리 상관이 있어 행 단위 부트스트랩은 CI 를 과소평가한다.
        boot = _cluster_bootstrap(yy, hgb, 0.75 * hgb + 0.25 * haz_c,
                                  _first_col(R, "stat_id"), n=400)
        print(f"  해저드 w=0.25 증분 (HGB 단독 대비, 충전소 부트스트랩 400회):")
        print(f"    ΔBrier {boot['d_brier']:+.4f}  [{boot['d_brier_lo']:+.4f}, {boot['d_brier_hi']:+.4f}]"
              f"   (음수 = 개선)")
        print(f"    ΔAUC   {boot['d_auc']:+.4f}  [{boot['d_auc_lo']:+.4f}, {boot['d_auc_hi']:+.4f}]")

        # 창이 2일뿐이라 지표가 날마다 ±0.02 흔들린다(스냅샷 앵커 주기).
        # 해저드 증분(~0.002)은 그 눈금보다 훨씬 작으므로 날짜별로 재현되는지 본다.
        day = pd.to_datetime(_first_col(R, "created_at")).astype("datetime64[ns]")
        day = pd.Series(day).dt.date.to_numpy()
        blend25 = 0.75 * hgb + 0.25 * haz_c
        by_day = {}
        for d in sorted(set(day)):
            dm = day == d
            if dm.sum() < 500 or yy[dm].min() == yy[dm].max():
                continue
            db = _brier(yy[dm], blend25[dm]) - _brier(yy[dm], hgb[dm])
            da = _auc(yy[dm], blend25[dm]) - _auc(yy[dm], hgb[dm])
            by_day[str(d)] = {"n": int(dm.sum()), "d_brier": round(db, 4), "d_auc": round(da, 4)}
            print(f"      {d} n={int(dm.sum()):,}  ΔBrier {db:+.4f}  ΔAUC {da:+.4f}")

        out["etas"][f"eta{eta}"] = {
            "n": int(len(yy)), "actual": round(float(yy.mean()), 4),
            "conv": round(conv, 4), "models": rows, "weight_sweep": sweep,
            "hazard_increment_w025": boot, "by_day": by_day,
        }

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {p}")


def _cluster_bootstrap(
    y: np.ndarray, p_base: np.ndarray, p_alt: np.ndarray,
    cluster: np.ndarray, n: int = 400, seed: int = 0,
) -> dict[str, float]:
    """충전소 단위 클러스터 부트스트랩으로 (alt − base) 지표 차의 95% CI.

    같은 충전소 행은 서로 독립이 아니다(같은 이용자·같은 시간대). 행 단위로
    재표집하면 CI 가 실제보다 좁게 나와 잡음을 유의한 이득으로 착각한다.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cluster, return_inverse=True)
    idx_by_cluster = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    d_b, d_a = [], []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        d_b.append(_brier(yb, p_alt[idx]) - _brier(yb, p_base[idx]))
        d_a.append(_auc(yb, p_alt[idx]) - _auc(yb, p_base[idx]))

    def q(a, lo):
        return float(np.percentile(a, lo)) if a else float("nan")

    return {
        "d_brier": round(_brier(y, p_alt) - _brier(y, p_base), 4),
        "d_brier_lo": round(q(d_b, 2.5), 4),
        "d_brier_hi": round(q(d_b, 97.5), 4),
        "d_auc": round(_auc(y, p_alt) - _auc(y, p_base), 4),
        "d_auc_lo": round(q(d_a, 2.5), 4),
        "d_auc_hi": round(q(d_a, 97.5), 4),
    }


def _first_col(df: pd.DataFrame, name: str) -> np.ndarray:
    """중복 컬럼명이 있어도 첫 번째 것만 1차원으로 꺼낸다."""
    if name not in df.columns:
        return np.full(len(df), "unknown", dtype=object)
    v = df[name]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    return v.to_numpy()


def _conversion_rate(
    sess: pd.DataFrame, runs: pd.DataFrame, train_end: pd.Timestamp, eta: int
) -> float:
    """P(도착 시점 stat==2 | 세션이 ETA 안에 종료), **train 기간 세션만으로** 추정.

    experiment_session_hazard 가 잰 것과 같은 양이지만, 거기서는 평가창(08-06~)에서
    쟀다. 이 실험의 평가창(08-10~)과 겹치므로 그 값을 그대로 가져오면 누수다.
    여기서는 train_end 이전 세션에 경과 격자를 찍어 다시 잰다.
    """
    steps = runs[["key", "start", "stat"]].sort_values(["start", "key"]).reset_index(drop=True)
    rows = []
    for e in (5, 15, 30, 45, 60, 90):
        s = sess[sess["dur_min"] > e]
        if s.empty:
            continue
        rows.append(pd.DataFrame({
            "key": s["key"].values,
            "t_arr": s["start"].values + pd.to_timedelta(e + eta, unit="m"),
            "ended": (s["dur_min"] <= e + eta).astype(int).values,
        }))
    if not rows:
        return 1.0
    P = pd.concat(rows, ignore_index=True)
    P = P[(P["ended"] == 1) & (P["t_arr"] < train_end)]
    if len(P) < 200:
        return 1.0
    P = P.sort_values("t_arr")
    P = pd.merge_asof(
        P,
        steps.rename(columns={"start": "t_arr", "stat": "arr_stat"}),
        on="t_arr", by="key", direction="backward",
    ).dropna(subset=["arr_stat"])
    return float((P["arr_stat"].astype(int) == 2).mean())


if __name__ == "__main__":
    main()
