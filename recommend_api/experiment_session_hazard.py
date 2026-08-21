"""실험: 충전중(stat=3) 세션 조건부 종료확률(해저드)이 현재 잔여시간 성분을 대체하는가.

운영 아티팩트는 건드리지 않는다. "이 방식이 값을 하는가"만 본다.

무엇을 검증하나
--------------
제안된 구조는 이랬다. stat=3 은 HGB 대신 **충전기별 세션 종료확률**로 판단하고,
데이터가 부족하면 충전소 → 출력·유형 → 전체 순으로 폴백한다.

    P(전체 충전시간 <= 현재 경과 + ETA | 전체 충전시간 > 현재 경과)

방향은 맞다. 다만 세 가지가 실측과 어긋나서 그대로 만들면 안 된다.
아래 네 실험이 각각 그 지점을 잰다(2026-08-12 실행, status 21.0일 · 273만행).

  (1) 세션 분포      — 평균이 아니라 중앙값을 쓰라는 지적은 옳다. 다만 급속의
                       평균은 291분이 아니라 38~108분이다. 291분급 꼬리는 완속
                       (7kW 중앙 520분)이 섞였을 때 나온다. 급속만 보면
                       중앙 25~45분 / 평균 38~108분이다.
  (2) 사다리 홀드아웃 — **충전기 계층은 충전소 위에서 이득이 0이다.** 사다리의
                       첫 칸을 충전소로 시작해야 한다. 위약 대조 포함.
  (3) 해저드 모양     — 해저드는 단조가 아니다. 경과 45~60분에서 정점을 찍고
                       그 뒤 무너진다(ETA10 기준 0.428 -> 0.096). "중앙값 - 경과"
                       같은 선형 잔여시간은 이 구간을 정확히 거꾸로 맞힌다.
  (4) 실라벨 비교     — 목표는 '세션이 끝나는가'가 아니라 '도착 시점에 비어 있는가'
                       다. 종료->가용 전환율은 ETA10 0.943 / ETA30 0.867 로,
                       "종료 예상 = 확정 가용" 으로 쓰면 안 된다는 지적도 옳다.

왜 status 인가 (charge_history 가 아니라)
----------------------------------------
build_session_priors.py 의 세션 이력은 기후에너지환경부 급속기에만 있어 대구
24,659대 중 285대(1.2%)뿐이고, 이건 구조적 결손이라 시간이 지나도 안 는다.
반면 status 의 stat=3 런에서 세션을 복원하면 **19,664대(급속 1,598대)**가 잡힌다.
같은 '세션 길이'라도 커버리지가 두 자릿수 배 차이라 이쪽이 유일한 실용 경로다.

대신 status 쪽은 5분 delta 해상도라 세션 경계에 최대 ±5분 양자화가 있고,
5분 안에 끝난 세션은 아예 안 잡힌다. 급속 중앙값이 25~45분이라 감수 가능하지만
완속(중앙 520분)은 6시간 스냅샷에만 걸리는 구간이 섞여 있어 그대로 믿지 말 것.

절단 처리
--------
충전기별 **첫 런은 좌측 절단**(언제 시작했는지 모른다)이라 버린다. 21일 창에서
급속 14,376건이 여기서 빠진다. **마지막 런은 우측 절단**이라 세션 집합에서 빼고
(2,426건), 라벨 프로브도 t+ETA 가 그 충전기의 마지막 관측을 넘으면 버린다.
이걸 안 하면 "관측이 끊긴 것"을 "상태가 안 변한 것"으로 세어 해저드가 눌린다.

사용:
  py -m recommend_api.experiment_session_hazard
  py -m recommend_api.experiment_session_hazard --days 14   # 창 좁혀 재현성 확인
  py -m recommend_api.experiment_session_hazard --json data/reports/x.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_store import get_connection, load_label_series
from .remaining_time import (
    _load_remaining_artifact,
    _lookup_remaining,
    build_remaining_feature_frame,
    free_by_eta_score,
    model_available,
)

# 경과시간 구간. 급속 중앙값이 25~45분이라 그 근처를 촘촘히 둔다.
ELAPSED_BINS = [0, 10, 20, 30, 45, 60, 90, 120, 180, 1e9]
ELAPSED_GRID = [5, 15, 30, 45, 60, 90, 120, 180]
ETAS = [10, 20, 30]
FAST_KW = 50.0

# 경험적 베이즈 shrinkage 강도(부모 사전확률에 실을 가상 표본수).
# 충전소당 세션이 21일에 중앙 14건이라 20 이면 절반쯤 부모로 당겨진다.
SHRINK_BUCKET = 50.0
SHRINK_STATION = 20.0
SHRINK_CHARGER = 20.0
SHRINK_GLOBAL = 20.0


# ------------------------------------------------------------------ 세션 복원

def build_runs(days: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """status(delta+snapshot) → 상태 런 테이블 + 충전기별 마지막 관측 시각.

    delta 는 '변경분'이라 행이 없다 = 상태가 그대로였다는 뜻이다(CLAUDE.md).
    그래서 연속 관측을 stat 이 바뀌는 지점에서만 끊어 런으로 묶는다. snapshot 행은
    같은 stat 을 반복 확인해 주는 앵커라 런을 나누지 않는다.
    """
    st = load_label_series(days=days).dropna(subset=["stat"])
    st["stat"] = st["stat"].astype(int)
    st = st.sort_values(["stat_id", "chger_id", "created_at"], kind="mergesort")
    st["key"] = st["stat_id"].astype(str) + "|" + st["chger_id"].astype(str)

    changed = (st["stat"] != st["stat"].shift()) | (st["key"] != st["key"].shift())
    st["run_id"] = changed.cumsum()

    runs = (
        st.groupby("run_id")
        .agg(
            key=("key", "first"),
            stat_id=("stat_id", "first"),
            stat=("stat", "first"),
            start=("created_at", "first"),
            last_obs=("created_at", "last"),
        )
        .reset_index(drop=True)
    )
    runs["next_key"] = runs["key"].shift(-1)
    runs["next_start"] = pd.to_datetime(runs["start"].shift(-1))
    runs["ended"] = runs["next_key"] == runs["key"]
    runs["end"] = pd.to_datetime(np.where(runs["ended"], runs["next_start"], runs["last_obs"]))
    runs["dur_min"] = (runs["end"] - runs["start"]).dt.total_seconds() / 60.0
    runs["is_first"] = runs["key"] != runs["key"].shift()

    horizon_end = st.groupby("key")["created_at"].max()
    return runs, horizon_end


def attach_output(runs: pd.DataFrame) -> pd.DataFrame:
    """output_kw 외에 chger_type·addr 도 붙인다 — 잔여시간 ML 모델의 범주형 입력."""
    conn = get_connection()
    try:
        info = pd.read_sql(
            "SELECT stat_id, chger_id, output, chger_type, addr FROM ev_charger_info "
            "WHERE (del_yn IS NULL OR del_yn <> 'Y')",
            conn,
        )
    finally:
        conn.close()
    info["key"] = info["stat_id"].astype(str) + "|" + info["chger_id"].astype(str)
    info["output_kw"] = pd.to_numeric(info["output"], errors="coerce")
    cols = ["key", "output_kw", "chger_type", "addr"]
    return runs.merge(info[cols].drop_duplicates("key"), on="key", how="left")


def bucket_of(kw: float) -> str:
    """remaining_time.snap_power_kw 와 같은 버킷 경계(±25kW)."""
    if not np.isfinite(kw):
        return "unknown"
    for b in (50, 100, 200, 350, 400):
        if abs(kw - b) <= 25:
            return f"f{b}"
    return f"f_other_{int(kw)}"


def charging_sessions(runs: pd.DataFrame, fast_only: bool = True) -> pd.DataFrame:
    ch = runs[(runs["stat"] == 3) & runs["ended"] & ~runs["is_first"]].copy()
    if fast_only:
        ch = ch[ch["output_kw"] >= FAST_KW]
    ch["bucket"] = ch["output_kw"].map(bucket_of)
    return ch.sort_values(["key", "start"])


# ------------------------------------------------------------------ 유틸

def _eb(k, n, prior, m):
    """경험적 베이즈: 자식 관측(k/n)을 부모 사전확률로 당긴다."""
    return (k + m * prior) / (n + m)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, p))
    except ValueError:  # 한쪽 클래스만 있는 경우
        return float("nan")


def ml_remaining_minutes(probes: pd.DataFrame) -> np.ndarray:
    """잔여시간 ML 모델(daegu_remaining_time_hgb)로 과거 프로브를 재생한다.

    서빙(predict_remaining_minutes)은 hour/dow 를 `pd.Timestamp.now()` 로 채운다.
    요청 시점이 곧 '지금'이라 서빙에서는 맞지만, 과거 프로브를 그 함수로 그냥
    돌리면 21일치 전부에 오늘 시각이 박혀 시간 피처가 통째로 틀어진다.
    그래서 build_remaining_feature_frame(now=...) 을 **프로브 시각의 시(hour)
    단위로 묶어** 직접 호출한다. 그 외 경로(버킷 스냅·범주형·기상 기본값)는
    서빙과 동일하다.
    """
    artifact = _load_remaining_artifact()
    if artifact is None:
        return np.full(len(probes), np.nan)
    pipe = artifact["pipeline"]
    cols = artifact["numeric_features"] + artifact["categorical_features"]

    out = np.full(len(probes), np.nan)
    src = probes.rename(columns={"elapsed": "time_since_charge_started"})
    keys = pd.to_datetime(probes["t"]).dt.floor("h")
    for when, idx in keys.groupby(keys).groups.items():
        chunk = src.loc[idx]
        feat = build_remaining_feature_frame(chunk, now=pd.Timestamp(when))
        for c in artifact["categorical_features"]:
            feat[c] = feat[c].astype(str).fillna("unknown")
        for c in artifact["numeric_features"]:
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
        out[probes.index.get_indexer(idx)] = np.clip(pipe.predict(feat[cols]), 0, None)
    return out


def _expand_risk_set(sess: pd.DataFrame, eta: int) -> pd.DataFrame:
    """세션 → (경과구간, 종료여부) 프로브. 각 구간의 위험집합만 남긴다."""
    rows = []
    for lo in ELAPSED_BINS[:-1]:
        s = sess[sess["dur_min"] > lo]
        if s.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "key": s["key"].values,
                    "stat_id": s["stat_id"].values,
                    "bucket": s["bucket"].values,
                    "ebin": float(lo),
                    "y": (s["dur_min"] <= lo + eta).astype(int).values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ (1) 분포

def report_distribution(ch_all: pd.DataFrame) -> dict[str, Any]:
    print("\n=== (1) stat=3 완결 세션 지속시간 분포(분) ===")
    out: dict[str, Any] = {}
    g = (
        ch_all.groupby("bucket")["dur_min"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
        )
        .sort_values("n", ascending=False)
    )
    print(g.round(1).head(12).to_string())
    out["by_bucket"] = g.round(2).to_dict(orient="index")
    return out


# ------------------------------------------------------------------ (2) 신뢰도

def report_reliability(fast: pd.DataFrame) -> dict[str, Any]:
    """분할반분: 충전기별 중앙 세션길이가 홀/짝 세션에서 재현되는가.

    원값 상관은 '완속 vs 급속'만으로도 높게 나오므로, 버킷 중앙값 대비 로그비
    (버킷 효과를 뺀 잔차)로도 함께 잰다. 재현되면 충전기 계층이 잡음은 아니다.
    다만 '잡음이 아니다'와 '충전소 위에서 이득이 있다'는 다른 질문이다 — (3) 참고.
    """
    print("\n=== (2) 충전기별 세션길이 분할반분 신뢰도 (급속) ===")
    fast = fast.copy()
    fast["idx"] = fast.groupby("key").cumcount()
    cnt = fast.groupby("key").size()
    bmed = fast.groupby("bucket")["dur_min"].transform("median")
    fast["resid"] = np.log(fast["dur_min"].clip(lower=0.5) / bmed.clip(lower=0.5))

    out: dict[str, Any] = {}
    for min_n in (6, 10, 20):
        sub = fast[fast["key"].map(cnt) >= min_n]
        for col, name in (("dur_min", "원값"), ("resid", "버킷잔차")):
            a = sub[sub["idx"] % 2 == 0].groupby("key")[col].median()
            b = sub[sub["idx"] % 2 == 1].groupby("key")[col].median()
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(j) < 20:
                print(f"  min_n={min_n} {name}: 표본 부족 ({len(j)}대)")
                continue
            r = float(j["a"].corr(j["b"], method="spearman"))
            sb = 2 * r / (1 + r)  # Spearman-Brown (반분 → 전체 신뢰도)
            print(f"  min_n={min_n:2d} {name:5s}  n={len(j):,}대  spearman={r:.3f}  SB={sb:.3f}")
            out[f"n{min_n}_{name}"] = {"n": int(len(j)), "spearman": round(r, 4), "sb": round(sb, 4)}
    return out


# ------------------------------------------------------------------ (3) 사다리

def report_ladder(fast: pd.DataFrame, train_frac: float = 0.72) -> dict[str, Any]:
    """시간 홀드아웃에서 사다리 각 단계가 log-loss 를 줄이는가 + 위약 대조.

    위약은 필수다. 충전기·충전소 고정 계층은 대조군 없이 판단하면 안 된다는 게
    experiment_detour_eta.py(2026-08-10)·experiment_session_prior.py 의 교훈이다.
    라벨을 셔플하면 커버리지·표본수는 그대로 두고 '누구의 이력인가'만 파괴된다.
    """
    print("\n=== (3) 사다리 홀드아웃 (급속, 목표=세션 종료) ===")
    cut = fast["start"].quantile(train_frac)
    tr, te = fast[fast["start"] < cut], fast[fast["start"] >= cut]
    print(f"  train {len(tr):,}건 (~{cut:%m-%d %H:%M})   test {len(te):,}건")

    out: dict[str, Any] = {"cut": str(cut)}
    rng = np.random.default_rng(0)

    for eta in ETAS:
        TR, TE = _expand_risk_set(tr, eta), _expand_risk_set(te, eta)
        g_prior = float(TR["y"].mean())

        t_glob = TR.groupby("ebin")["y"].agg(gk="sum", gn="size").reset_index()
        t_buck = TR.groupby(["bucket", "ebin"])["y"].agg(bk="sum", bn="size").reset_index()
        t_stn = TR.groupby(["stat_id", "ebin"])["y"].agg(sk="sum", sn="size").reset_index()
        t_chg = TR.groupby(["key", "ebin"])["y"].agg(ck="sum", cn="size").reset_index()

        # 위약: 충전소/충전기 라벨만 섞는다
        stns = t_stn["stat_id"].unique()
        t_stn_p = t_stn.assign(stat_id=t_stn["stat_id"].map(dict(zip(stns, rng.permutation(stns)))))
        keys = t_chg["key"].unique()
        t_chg_p = t_chg.assign(key=t_chg["key"].map(dict(zip(keys, rng.permutation(keys)))))

        def stack(stn_tbl, chg_tbl):
            E = (
                TE.merge(t_glob, on="ebin", how="left")
                .merge(t_buck, on=["bucket", "ebin"], how="left")
                .merge(stn_tbl, on=["stat_id", "ebin"], how="left")
                .merge(chg_tbl, on=["key", "ebin"], how="left")
            )
            for c in ("gk", "gn", "bk", "bn", "sk", "sn", "ck", "cn"):
                E[c] = E[c].fillna(0.0)
            pg = np.where(E["gn"] > 0, _eb(E["gk"], E["gn"], g_prior, SHRINK_GLOBAL), g_prior)
            pb = _eb(E["bk"], E["bn"], pg, SHRINK_BUCKET)
            ps = _eb(E["sk"], E["sn"], pb, SHRINK_STATION)
            pc = _eb(E["ck"], E["cn"], ps, SHRINK_CHARGER)
            return E["y"].values, pg, pb, ps, pc

        y, p_g, p_b, p_s, p_c = stack(t_stn, t_chg)
        _, _, _, p_s_plac, _ = stack(t_stn_p, t_chg)
        _, _, _, _, p_c_plac = stack(t_stn, t_chg_p)
        base = np.full(len(y), g_prior)

        print(f"\n  ETA={eta}분  프로브 {len(y):,}  실제 종료율 {y.mean():.3f}")
        rows = {}
        for name, p in (
            ("상수", base),
            ("전역x경과", p_g),
            ("+출력버킷", p_b),
            ("+충전소", p_s),
            ("  위약(충전소셔플)", p_s_plac),
            ("+충전기", p_c),
            ("  위약(충전기셔플)", p_c_plac),
        ):
            ll, br = _logloss(y, p), _brier(y, p)
            print(f"    {name:18s} logloss {ll:.4f}   brier {br:.4f}")
            rows[name.strip()] = {"logloss": round(ll, 4), "brier": round(br, 4)}
        out[f"eta{eta}"] = rows
    return out


# ------------------------------------------------------------------ (4) 실라벨

def report_against_label(
    fast: pd.DataFrame,
    runs: pd.DataFrame,
    horizon_end: pd.Series,
    train_frac: float = 0.72,
) -> dict[str, Any]:
    """목표를 '도착 시점 stat==2' 로 바꿔 현재 점수 성분과 직접 비교한다.

    P0 persistence   충전중이면 계속 충전중(p=0) — 관성 베이스라인
    P1 현재 폴백      free_by_eta_score(_lookup_remaining(kw, e), eta)
    P1m 현재 ML       free_by_eta_score(daegu_remaining_time_hgb 예측, eta) — 운영 경로
    P2 종료해저드     충전소 사다리
    P3 해저드x전환    P2 x P(도착시 stat==2 | 세션 종료) — '다음 차' 보정
    """
    print("\n=== (4) 실제 가용성 라벨 대비 (급속 stat=3) ===")
    cut = fast["start"].quantile(train_frac)
    tr, te = fast[fast["start"] < cut], fast[fast["start"] >= cut]
    steps = runs[["key", "start", "stat"]].sort_values(["start", "key"]).reset_index(drop=True)

    probes = []
    for e in ELAPSED_GRID:
        s = te[te["dur_min"] > e].copy()
        if s.empty:
            continue
        s["elapsed"] = float(e)
        s["t"] = s["start"] + pd.to_timedelta(e, unit="m")
        probes.append(
            s[["key", "stat_id", "bucket", "output_kw", "chger_type", "addr",
               "dur_min", "elapsed", "t"]]
        )
    P = pd.concat(probes, ignore_index=True)
    P["ebin"] = pd.cut(P["elapsed"], bins=ELAPSED_BINS, right=False,
                       labels=ELAPSED_BINS[:-1]).astype(float)

    out: dict[str, Any] = {}
    for eta in ETAS:
        Q = P.copy()
        Q["t_arr"] = Q["t"] + pd.to_timedelta(eta, unit="m")
        Q["obs_end"] = Q["key"].map(horizon_end)
        Q = Q[Q["t_arr"] <= Q["obs_end"]]          # 우측 절단 밖은 라벨 불가
        if Q.empty:
            continue
        # 도착 시점 상태 = t_arr 직전 런의 stat (LOCF). 충전기별 루프 대신 merge_asof.
        Q = Q.sort_values("t_arr")
        Q = pd.merge_asof(
            Q,
            steps.rename(columns={"start": "t_arr", "stat": "arr_stat"}),
            on="t_arr",
            by="key",
            direction="backward",
        ).dropna(subset=["arr_stat"])
        y = (Q["arr_stat"].astype(int) == 2).astype(int).values
        ended = (Q["dur_min"] <= Q["elapsed"] + eta).astype(int).values

        TR = _expand_risk_set(tr, eta)
        g_prior = float(TR["y"].mean())
        t_glob = TR.groupby("ebin")["y"].agg(gk="sum", gn="size").reset_index()
        t_buck = TR.groupby(["bucket", "ebin"])["y"].agg(bk="sum", bn="size").reset_index()
        t_stn = TR.groupby(["stat_id", "ebin"])["y"].agg(sk="sum", sn="size").reset_index()
        R = (
            Q.merge(t_glob, on="ebin", how="left")
            .merge(t_buck, on=["bucket", "ebin"], how="left")
            .merge(t_stn, on=["stat_id", "ebin"], how="left")
        )
        for c in ("gk", "gn", "bk", "bn", "sk", "sn"):
            R[c] = R[c].fillna(0.0)
        pg = np.where(R["gn"] > 0, _eb(R["gk"], R["gn"], g_prior, SHRINK_GLOBAL), g_prior)
        pb = _eb(R["bk"], R["bn"], pg, SHRINK_BUCKET)
        p_haz = np.asarray(_eb(R["sk"], R["sn"], pb, SHRINK_STATION))

        conv = float(y[ended == 1].mean()) if ended.any() else float("nan")
        P0 = np.zeros(len(y))
        P1 = np.array(
            [
                free_by_eta_score(_lookup_remaining(float(kw), float(e)), eta)
                for kw, e in zip(R["output_kw"].values, R["elapsed"].values)
            ]
        )
        P2 = p_haz
        P3 = p_haz * conv

        cands: list[tuple[str, np.ndarray]] = [
            ("P0 persistence", P0),
            ("P1 현재 폴백", P1),
        ]
        if model_available():
            R = R.reset_index(drop=True)
            rem_ml = ml_remaining_minutes(R)
            P1m = np.array([free_by_eta_score(float(r), eta) for r in rem_ml])
            cands.append(("P1m 현재 ML", P1m))
        else:
            P1m = None
        cands += [("P2 종료해저드", P2), ("P3 해저드x전환", P3)]

        print(
            f"\n  ETA={eta}분  프로브 {len(y):,}  실제 가용률 {y.mean():.3f}"
            f"  | 세션종료율 {ended.mean():.3f}  | 종료→가용 전환 {conv:.3f}"
        )
        rows = {}
        for name, p in cands:
            ll, br, au = _logloss(y, p), _brier(y, p), _auc(y, p)
            print(f"    {name:16s} logloss {ll:.4f}  brier {br:.4f}  auc {au:.4f}  평균예측 {p.mean():.3f}")
            rows[name] = {
                "logloss": round(ll, 4),
                "brier": round(br, 4),
                "auc": round(au, 4),
                "mean_pred": round(float(p.mean()), 4),
            }

        # 학습범위 안팎 분해. 아티팩트 elapsed_grid 는 0~45분까지라
        # 경과 60분 이상은 외삽 구간이다(트리라 사실상 상수로 눌린다).
        if P1m is not None:
            e_arr = R["elapsed"].values
            for label, mask in (("경과<=45", e_arr <= 45), ("경과>45", e_arr > 45)):
                if mask.sum() < 200:
                    continue
                print(
                    f"      [{label}] n={int(mask.sum()):,} 실제 {y[mask].mean():.3f}"
                    f" | ML brier {_brier(y[mask], P1m[mask]):.4f} (평균 {P1m[mask].mean():.3f})"
                    f" | 해저드 brier {_brier(y[mask], P3[mask]):.4f} (평균 {P3[mask].mean():.3f})"
                )
                rows.setdefault("_by_elapsed", {})[label] = {
                    "n": int(mask.sum()),
                    "actual": round(float(y[mask].mean()), 4),
                    "ml_brier": round(_brier(y[mask], P1m[mask]), 4),
                    "hazard_brier": round(_brier(y[mask], P3[mask]), 4),
                }

        out[f"eta{eta}"] = {
            "n": int(len(y)),
            "actual_avail": round(float(y.mean()), 4),
            "end_to_avail": round(conv, 4),
            "models": rows,
        }
    return out


# ------------------------------------------------------------------ 해저드 곡선

def report_hazard_curve(fast: pd.DataFrame) -> dict[str, Any]:
    """해저드가 단조인지 본다. 단조가 아니면 '중앙값 - 경과' 는 못 쓴다."""
    print("\n=== (3b) 경험적 조건부 종료확률 (급속 전체) ===")
    dur = fast["dur_min"].values
    out: dict[str, Any] = {}
    for eta in ETAS:
        cells, row = {}, []
        for e in ELAPSED_GRID:
            at_risk = dur > e
            if at_risk.sum() < 50:
                row.append(f"e={e:3d}:  n/a")
                continue
            p = float(((dur > e) & (dur <= e + eta)).sum() / at_risk.sum())
            cells[str(e)] = round(p, 4)
            row.append(f"e={e:3d}: {p:.3f}")
        print(f"  ETA={eta:2d}분  " + "  ".join(row))
        out[f"eta{eta}"] = cells
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=None, help="status 조회 창(일). 기본 전체")
    ap.add_argument("--train-frac", type=float, default=0.72, help="시간 홀드아웃 분할점")
    ap.add_argument("--json", type=str, default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    runs, horizon_end = build_runs(days=args.days)
    runs = attach_output(runs)

    span_h = (runs["last_obs"].max() - runs["start"].min()).total_seconds() / 3600
    print(f"status {runs['start'].min()} ~ {runs['last_obs'].max()}  ({span_h/24:.1f}일)")
    from .config import REMAINING_MODEL_PATH

    print(
        f"잔여시간 ML 모델: {'사용' if model_available() else '없음 → 폴백만 평가'}"
        f"  ({REMAINING_MODEL_PATH})"
    )

    ch_all = charging_sessions(runs, fast_only=False)
    fast = charging_sessions(runs, fast_only=True)
    n_left = int(((runs["stat"] == 3) & runs["is_first"]).sum())
    n_right = int(((runs["stat"] == 3) & ~runs["ended"]).sum())
    print(
        f"stat=3 완결 세션 {len(ch_all):,}건 / 충전기 {ch_all['key'].nunique():,}대"
        f"  (좌측절단 제외 {n_left:,} · 우측절단 제외 {n_right:,})"
    )
    print(f"  급속 {len(fast):,}건 / 충전기 {fast['key'].nunique():,}대 / 충전소 {fast['stat_id'].nunique():,}곳")

    result: dict[str, Any] = {
        "span_days": round(span_h / 24, 2),
        "sessions_all": int(len(ch_all)),
        "sessions_fast": int(len(fast)),
        "distribution": report_distribution(ch_all),
        "reliability": report_reliability(fast),
        "hazard_curve": report_hazard_curve(fast),
        "ladder": report_ladder(fast, args.train_frac),
        "against_label": report_against_label(fast, runs, horizon_end, args.train_frac),
    }

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
