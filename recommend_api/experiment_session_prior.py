"""실험: 세션이력 요일 prior 피처 효과 측정.

운영 아티팩트(horizon_hgb.joblib)는 건드리지 않는다. "이 피처가 값을 하는가"만 본다.

설계: 같은 데이터·같은 날짜 홀드아웃에서 **피처 집합만** 바꿔 비교한다.
  A(baseline)  = config.FEATURE_COLS (25) + chger_type/kind/busi 더미
  B(+prior)    = A + occupancy_prior, kwh_per_day_prior           (계층 shrinkage 값)
  C(+charger)  = B + charger_occupancy, charger_n, has_charger_history (개별 이력 원값)

왜 이 피처인가
--------------
ev_charger_status 는 13일치(2026-07-22~)라 충전기 x 요일 셀당 관측이 1~2회다.
그래서 FEATURE_COLS 의 day_of_week/arrival_weekday 는 **전역** 요일 효과만 배운다.
그런데 세션 이력(2025-01~2026-05, 515일)으로 재보면 충전기별 주말/평일 점유율 비가
p10 0.31 ~ p90 1.42 로 벌어진다. 전역 평균 0.90 하나로는 양쪽 꼬리를 다 틀린다.
scripts/etl/build_session_priors.py 가 만든 prior 를 도착 요일 기준으로 붙여 본다.

기간이 겹치지 않아(세션 ~2026-05, status 2026-07~) 누수는 없다.

커버리지
--------
prior 는 기후에너지환경부 급속기에만 개별 이력이 있다(대구 24,660대 중 229대).
지역x속도군 그룹 prior 까지 포함해도 1,777대(7.2%)뿐인데, 나머지는 완속이라
애초에 급속 랭킹 대상이 아니다. 그래서 전체 지표보다 **급속 부분집합**과
**개별 이력 보유 부분집합**을 따로 본다. 전체 지표는 희석돼서 효과가 안 보인다.

재실험 설계 (2026-08-10 추가)
---------------------------
2026-08-03 실행은 **기각**됐다. AUC +0.0018 인데 PR-AUC −0.0164 · unavail-recall
−0.0055 · Brier +0.0009 로, 랭킹만 미세하게 좋아지고 "이 충전기는 찼을 것"을 더 못
잡는 교환이었다. 기각 사유는 신호 부재가 아니라 **status 가 13일뿐**이라는 것이다.
prior 는 (충전기, 요일) 고정값이라, 학습 10일/평가 3일에서는 요일 패턴이 아니라
충전기별 기저율을 외운다(Brier 악화가 그 증거). ETA별로는 h5 +0.0000 -> h60 +0.0020
으로 단조 증가해 신호 자체는 실재했다.

그래서 두 가지를 더한다.

1. **4주 게이트** — status 기간이 --min-days(기본 28) 미만이면 중단한다.
   "홀드아웃이 충전기 ID 와 요일 패턴을 분리할 수 있을 것"이 재실험 조건이었다.

2. **위약 대조군 두 개** — 충전기 고정값 피처는 대조군 없이 판단하면 안 된다.
   2026-08-10 우회계수 실험에서 위약이 실제 피처보다 나은 결과가 나왔다
   (recommend_api/experiment_detour_eta.py).

   P1 (충전기 상수)  충전기당 난수 하나. 요일과 무관하고 커버리지만 같다.
                     -> 여기서 오르는 만큼이 순수 "충전기 ID 암기" 능력이다.
   P2 (요일 셔플)    실제 prior 값을 쓰되 충전기 안에서 요일 라벨만 섞는다.
                     -> 충전기별 값 분포는 완전히 보존되고 **요일 패턴만 파괴**된다.

   판정의 핵심은 B(진짜) vs P2(요일 셔플)다. 둘이 같으면 이득의 정체가 요일 패턴이
   아니라 충전기 기저율이므로 기각한다. P1 은 그 기저율이 얼마나 강한지를 잰다.

사용:
  py -m recommend_api.experiment_session_prior                # 4주 미만이면 중단
  py -m recommend_api.experiment_session_prior --min-days 0   # 게이트 무시(진단용)
  py -m recommend_api.experiment_session_prior --max-rows 400000
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS
from .eval_metrics import evaluate_classifier, split_by_date
from .holdout_eval import _new_model, _per_eta_metrics, _prepare_fold
from .model_store import (
    build_horizon_dataset,
    get_connection,
    load_joined,
    load_label_series,
    table_exists,
)

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

OUT_DIR = ARTIFACTS_DIR / "experiments"

PRIOR_COLS = ["occupancy_prior", "kwh_per_day_prior"]
CHARGER_COLS = ["charger_occupancy", "charger_n", "has_charger_history"]
PLACEBO_CONST_COLS = ["placebo_charger_const"]
PLACEBO_SHUFFLE_COLS = ["placebo_dow_shuffled"]
SEED = 20260810


def add_placebos(df: pd.DataFrame) -> pd.DataFrame:
    """위약 두 개를 붙인다. 커버리지는 occupancy_prior 와 정확히 같게 맞춘다.

    P1 placebo_charger_const
        (충전기)당 난수 상수. 요일과 무관하므로 요일 패턴은 전혀 없고, 모델이
        충전기를 구분해 얻을 수 있는 이득만 남는다.

    P2 placebo_dow_shuffled
        occupancy_prior 의 값을 **충전기 안에서 요일에만 재배치**한다. 충전기별
        값 집합(따라서 평균·분산·기저율)은 완전히 보존되고 요일 대응만 깨진다.
        B 가 P2 를 못 이기면 이득은 요일 패턴이 아니다.
    """
    rng = np.random.default_rng(SEED)
    covered = df["occupancy_prior"].notna()

    key = df["stat_id"].astype(str) + "|" + df["chger_id"].astype(str)
    uniq = key[covered].unique()
    src = df.loc[covered, "occupancy_prior"].to_numpy()
    const_map = pd.Series(
        rng.choice(src, size=len(uniq), replace=True), index=uniq
    )
    df["placebo_charger_const"] = key.map(const_map).where(covered)

    # 충전기 안에서 요일 라벨만 셔플 -> 값 집합은 그대로, 요일 대응만 파괴
    def _shuffle(group: pd.Series) -> pd.Series:
        vals = group.to_numpy(copy=True)
        rng.shuffle(vals)
        return pd.Series(vals, index=group.index)

    shuffled = (
        df.loc[covered]
        .groupby(key[covered], sort=False)["occupancy_prior"]
        .transform(_shuffle)
    )
    df["placebo_dow_shuffled"] = shuffled.reindex(df.index)

    for c in PLACEBO_CONST_COLS + PLACEBO_SHUFFLE_COLS:
        got = df[c].notna()
        if not got.equals(covered):
            raise RuntimeError(
                f"{c} 커버리지가 occupancy_prior 와 다르다 "
                f"({int(got.sum())} vs {int(covered.sum())}) — 대조가 성립하지 않는다"
            )
    return df


def load_prior() -> pd.DataFrame:
    """ev_charger_session_prior 전량(172,620행). 작아서 통째로 읽어 merge 한다."""
    conn = get_connection()
    try:
        if not table_exists(conn, "ev_charger_session_prior"):
            raise SystemExit(
                "ev_charger_session_prior 없음 — "
                "py scripts/etl/build_session_priors.py 를 먼저 실행할 것"
            )
        df = pd.read_sql(
            """
            SELECT stat_id, chger_id, day_of_week,
                   occupancy_prior, kwh_per_day_prior,
                   charger_occupancy, charger_n, has_charger_history
            FROM ev_charger_session_prior
            """,
            conn,
        )
    finally:
        conn.close()
    for c in PRIOR_COLS + CHARGER_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def attach_prior(hz: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """도착 요일 기준으로 붙인다.

    기준 시점(day_of_week)이 아니라 **도착 시점**(arrival_weekday)이다.
    예측 대상이 도착 시점의 가용 여부이므로 요일도 그쪽을 따라가야 하고,
    ETA 60분이 자정을 넘으면 두 값이 실제로 갈린다.
    """
    out = hz.merge(
        prior.rename(columns={"day_of_week": "arrival_weekday"}),
        on=["stat_id", "chger_id", "arrival_weekday"],
        how="left",
    )
    if len(out) != len(hz):
        raise RuntimeError(
            f"merge 후 행수 변동 {len(hz):,} → {len(out):,} — prior PK 중복 의심"
        )
    # 이력이 아예 없는 충전기: 플래그는 0, 나머지는 NaN 유지(HGB 가 결측을 직접 처리)
    out["has_charger_history"] = out["has_charger_history"].fillna(0).astype(int)
    out["charger_n"] = out["charger_n"].fillna(0).astype(int)
    return out


def build_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """make_xy 와 동일한 더미 구성. 피처 목록만 바꿔 끼운다."""
    dummies = [
        pd.get_dummies(df["chger_type"].fillna("UNK"), prefix="ctype"),
        pd.get_dummies(df["kind"].fillna("UNK"), prefix="kind"),
        pd.get_dummies(df["busi_id"].fillna("UNK"), prefix="busi"),
    ]
    X = pd.concat([df[cols], *dummies], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def _subset(y_te, proba, mask, min_n: int = 200) -> dict[str, Any] | None:
    m = np.asarray(mask, dtype=bool)
    if m.sum() < min_n or len(np.unique(np.asarray(y_te)[m])) < 2:
        return None
    return evaluate_classifier(np.asarray(y_te)[m], np.asarray(proba)[m])


def run_arm(
    df: pd.DataFrame, cols: list[str], train_mask: pd.Series, label: str
) -> dict[str, Any]:
    X = build_matrix(df, cols)
    y = df["y"]
    X_tr, X_te, y_tr, y_te = _prepare_fold(X, y, train_mask)

    model = _new_model()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    te = df.loc[X_te.index]
    overall = evaluate_classifier(y_te, proba)
    fast = _subset(y_te, proba, te["is_fast"] == 1)
    covered = _subset(y_te, proba, te["has_charger_history"] == 1)
    per_eta = _per_eta_metrics(y_te, proba, te["eta_minutes"])

    print(f"\n[{label}]  피처 {X.shape[1]}개")
    for name, m in (("전체", overall), ("급속", fast), ("개별이력", covered)):
        if m is None:
            print(f"  {name:<6} 표본 부족")
            continue
        print(
            f"  {name:<6} AUC {m['roc_auc']:.4f} | "
            f"unavail-recall {m['unavailable_recall']:.3f} | "
            f"PR-AUC {m['pr_auc_unavailable']:.4f} | "
            f"Brier {m['brier_score']:.4f} | n={m['n']:,}"
        )

    return {
        "label": label,
        "n_features": int(X.shape[1]),
        "overall": overall,
        "fast": fast,
        "charger_history": covered,
        "per_eta": per_eta,
    }


def _diff_block(base: dict, arm: dict, keys=("overall", "fast", "charger_history")) -> None:
    for scope in keys:
        a, b = base.get(scope), arm.get(scope)
        if not a or not b:
            continue
        print(f"  [{scope}]", end="")
        for k, lab in (
            ("roc_auc", "AUC"),
            ("unavailable_recall", "unavail-R"),
            ("pr_auc_unavailable", "PR-AUC"),
            ("brier_score", "Brier↓"),
        ):
            print(f"  {lab} {b[k] - a[k]:+.4f}", end="")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="세션이력 요일 prior 피처 효과 실험")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "load_joined 상한. 기본은 전량. "
            "load_joined 이 stat_id 순으로 정렬하므로 상한을 걸면 CV* 충전소만 남고 "
            "prior 가 있는 ME* 가 잘려나간다 — 이 실험에서는 권장하지 않는다."
        ),
    )
    p.add_argument(
        "--min-days",
        type=int,
        default=28,
        metavar="N",
        help=(
            "status 최소 축적일. 기본 28(4주). 2026-08-03 기각 사유가 "
            "'13일뿐이라 요일 패턴 대신 충전기 기저율을 외운다'였으므로, "
            "홀드아웃이 둘을 분리할 수 있을 만큼 쌓이기 전에는 재실험이 무의미하다. "
            "0 을 주면 게이트를 무시한다(진단용)."
        ),
    )
    args = p.parse_args()

    if args.max_rows:
        print(
            f"주의: --max-rows {args.max_rows:,} 는 stat_id 정렬 상한이라 "
            "prior 보유 충전기(ME*)가 잘릴 수 있다."
        )

    print("데이터 로드...")
    base = load_joined(limit=args.max_rows)
    print(f"  status JOIN {len(base):,}행")

    span = pd.to_datetime(base["created_at"])
    n_days = int((span.max().normalize() - span.min().normalize()).days) + 1
    print(f"  status 축적 {n_days}일 ({span.min():%Y-%m-%d} ~ {span.max():%Y-%m-%d})")
    if n_days < args.min_days:
        raise SystemExit(
            f"중단: status 가 {n_days}일뿐이다(필요 {args.min_days}일).\n"
            "  2026-08-03 기각은 신호가 없어서가 아니라 기간이 짧아 prior 가 충전기 ID 로\n"
            "  작동했기 때문이다. 지금 돌리면 같은 결론이 같은 이유로 반복된다.\n"
            f"  약 {args.min_days - n_days}일 뒤 다시 실행할 것 "
            "(게이트를 무시하려면 --min-days 0)."
        )

    print("라벨 시계열 로드 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성...")
    hz = build_horizon_dataset(base, HORIZONS, label_df=label_df).reset_index(drop=True)
    print(f"  {len(hz):,}샘플 / positive_rate {hz['y'].mean():.4f}")
    if hz.empty:
        raise SystemExit("horizon 샘플 없음")

    print("prior 로드 및 결합...")
    prior = load_prior()
    df = attach_prior(hz, prior)
    df = add_placebos(df)

    n = len(df)
    cov_prior = float(df["occupancy_prior"].notna().mean())
    cov_charger = float((df["has_charger_history"] == 1).mean())
    fast_mask = df["is_fast"] == 1
    cov_prior_fast = float(df.loc[fast_mask, "occupancy_prior"].notna().mean())
    cov_charger_fast = float((df.loc[fast_mask, "has_charger_history"] == 1).mean())
    print(
        f"  prior 결합률   전체 {100*cov_prior:.1f}%  급속 {100*cov_prior_fast:.1f}%\n"
        f"  개별이력 보유  전체 {100*cov_charger:.1f}%  급속 {100*cov_charger_fast:.1f}%\n"
        f"  급속 샘플 {int(fast_mask.sum()):,} / {n:,}"
    )
    if cov_prior_fast < 0.01:
        raise SystemExit(
            "급속 표본에 prior 가 거의 안 붙었다. "
            "build_session_priors.py 재실행 또는 매핑 확인 필요."
        )

    train_mask, train_dates, test_dates = split_by_date(df)
    print(
        f"\n날짜 홀드아웃: train {len(train_dates)}일 / test {len(test_dates)}일"
        f"  ({int(train_mask.sum()):,} vs {int((~train_mask).sum()):,}샘플)"
    )

    a = run_arm(df, FEATURE_COLS, train_mask, "A baseline")
    b = run_arm(df, FEATURE_COLS + PRIOR_COLS, train_mask, "B +prior")
    c = run_arm(df, FEATURE_COLS + PRIOR_COLS + CHARGER_COLS, train_mask, "C +charger")
    p1 = run_arm(df, FEATURE_COLS + PLACEBO_CONST_COLS, train_mask, "P1 위약(충전기 상수)")
    p2 = run_arm(df, FEATURE_COLS + PLACEBO_SHUFFLE_COLS, train_mask, "P2 위약(요일 셔플)")

    print("\n" + "=" * 74)
    print("차이 (B - A) : 계층 shrinkage prior 추가")
    print("=" * 74)
    _diff_block(a, b)
    print("\n차이 (C - A) : + 개별 충전기 원값")
    print("-" * 74)
    _diff_block(a, c)
    print("\n차이 (P1 - A) : 위약 · 충전기당 난수 상수 — 순수 충전기 ID 암기분")
    print("-" * 74)
    _diff_block(a, p1)
    print("\n차이 (P2 - A) : 위약 · 요일 셔플 — 충전기 분포는 보존, 요일 패턴만 파괴")
    print("-" * 74)
    _diff_block(a, p2)

    # ── 판정 ──────────────────────────────────────────────────────────
    # 관심 모집단은 prior 가 실제로 붙은 부분집합이다. 전체는 커버리지 7%라 희석된다.
    scope = "charger_history" if a.get("charger_history") else "fast"
    ra, rb, rp2 = a.get(scope), b.get(scope), p2.get(scope)
    reasons: list[str] = []
    if not (ra and rb and rp2):
        verdict = "INCONCLUSIVE"
        reasons.append(f"{scope} 부분집합 표본 부족")
    else:
        d_pr = rb["pr_auc_unavailable"] - ra["pr_auc_unavailable"]
        d_rec = rb["unavailable_recall"] - ra["unavailable_recall"]
        d_br = rb["brier_score"] - ra["brier_score"]
        net = d_pr - (rp2["pr_auc_unavailable"] - ra["pr_auc_unavailable"])
        if d_pr <= 0 or d_rec <= 0:
            reasons.append(f"PR-AUC {d_pr:+.4f} / recall {d_rec:+.4f} — 사용불가 탐지 개선 없음")
        if d_br > 0.0005:
            reasons.append(f"Brier {d_br:+.4f} 악화 — 충전기 기저율 암기 징후")
        if net <= 0.002:
            reasons.append(
                f"요일 셔플 위약 대비 순이득 {net:+.4f} — 이득이 요일 패턴이 아니라 "
                "충전기 기저율로 설명됨"
            )
        verdict = "REJECT" if reasons else "ADOPT_CANDIDATE"

    print("\n" + "=" * 74)
    print(f"판정({scope} 부분집합): {verdict}")
    for r in reasons:
        print(f"  - {r}")
    if verdict == "ADOPT_CANDIDATE":
        print("  요일 패턴이 충전기 기저율을 넘어서는 신호를 준다. config.FEATURE_COLS 편입 검토.")
    print("=" * 74)

    print(f"\n{'ETA':>5}{'A AUC':>10}{'B AUC':>10}{'C AUC':>10}{'C-A':>9}")
    print("-" * 46)
    bm = {r["eta_minutes"]: r for r in b["per_eta"]}
    cm = {r["eta_minutes"]: r for r in c["per_eta"]}
    for ra in a["per_eta"]:
        h = ra["eta_minutes"]
        print(
            f"{h:>5}{ra['roc_auc']:>10.4f}{bm[h]['roc_auc']:>10.4f}"
            f"{cm[h]['roc_auc']:>10.4f}{cm[h]['roc_auc'] - ra['roc_auc']:>+9.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "session_prior_weekday",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "운영 아티팩트 미변경. 승격 판단 아님.",
        "n_samples": int(n),
        "positive_rate": float(df["y"].mean()),
        "split": {
            "n_train_dates": len(train_dates),
            "n_test_dates": len(test_dates),
            "train_dates": [str(d) for d in train_dates],
            "test_dates": [str(d) for d in test_dates],
        },
        "coverage": {
            "prior_pct": 100 * cov_prior,
            "prior_pct_fast": 100 * cov_prior_fast,
            "charger_history_pct": 100 * cov_charger,
            "charger_history_pct_fast": 100 * cov_charger_fast,
            "n_fast": int(fast_mask.sum()),
        },
        "status_days": n_days,
        "min_days_gate": args.min_days,
        "verdict": verdict,
        "verdict_scope": scope,
        "verdict_reasons": reasons,
        "arms": {
            "A_baseline": a,
            "B_plus_prior": b,
            "C_plus_charger": c,
            "P1_placebo_charger_const": p1,
            "P2_placebo_dow_shuffled": p2,
        },
    }
    path = OUT_DIR / "session_prior_experiment.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
