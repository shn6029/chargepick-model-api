"""실험: TMAP 실측 도로기하 피처(우회계수) 효과 측정.

운영 아티팩트(horizon_hgb.joblib)는 건드리지 않는다. "이 피처가 값을 하는가"만 본다.

설계: 같은 데이터·같은 날짜 홀드아웃에서 **피처 집합만** 바꿔 비교한다.
  A(baseline)  = config.FEATURE_COLS (25) + chger_type/kind/busi 더미
  B(+detour)   = A + detour_ratio, eta_per_km              (소별 상수 원값)
  C(+placebo)  = A + placebo_station_const                 (소별 난수 — 대조군)
  D(+adjusted) = A + eta_road_adj                          (eta_minutes 와의 상호작용)

왜 이 피처인가
--------------
FEATURE_COLS 의 eta_minutes 는 HORIZONS = [5,10,15,20,30,45,60] **합성 격자**다.
실제 도로 주행시간이 아니라 "몇 분 뒤를 볼 것인가"라는 지평 표식일 뿐이라,
같은 30분이어도 직선 3km 도심 충전소와 직선 12km 외곽 충전소가 구분되지 않는다.

DA① 팩의 TMAP 실측(동대구 고정 origin, 1,848소)에서 도로/직선 비율을 뽑으면
소별로 p25 1.320 ~ p75 1.529 로 갈린다(중앙 1.405, km당 2.59분). 이 비율은
"이 충전소에 실제로 닿는 데 얼마나 돌아가야 하는가"라는 도로망 특성이고,
DA① 팩으로 학습한 별도 모델의 leave-one-out 에서 30분 지평 ROC 기여가
-0.054 로 available_count 다음으로 컸다.

**C(placebo) 를 반드시 같이 볼 것**
-----------------------------------
detour_ratio 는 (충전소) 고정값이다. config.FEATURE_COLS 주석에 적힌 대로
session_prior 가 기각된 이유가 정확히 이 구조였다 — 소/충전기 고정값은
status 가 짧을 때 요일·기하 패턴이 아니라 **충전소별 기저율**을 외운다.

그래서 같은 커버리지·같은 분포를 갖되 **정보가 0인** 난수 상수를 C 로 넣는다.
  B - C ≈ 0  이면 이득의 정체는 도로 기하가 아니라 충전소 ID 다 → 기각.
  B - C > 0  이면 기하 정보가 실재한다 → 채택 검토.

D 는 같은 정보를 지평과 곱해 소별 상수성을 깬 변형이다. 소 암기가 문제라면
D 가 B 보다 나아야 한다.

누수 없음: TMAP ETA 는 2026-08-06 1회 측정한 **도로망 기하**이고 충전기 상태와
무관하다. 라벨(future_stat)과 시간적·인과적 접점이 없다.

커버리지 — 결합 부분집합이 주 판독면인 이유
------------------------------------------
TMAP 실측은 공용·coord_ok 1,848소뿐이다(대구 전체 4,224 statId 대비 43.7%).
게다가 holdout_eval._prepare_fold 는 결측을 **train 중앙값으로 대치**한다
(HGB 네이티브 결측 처리가 아니다). 즉 미결합 소는 detour_ratio 가 중앙값으로
채워져 A 와 사실상 같은 입력이 된다.

그래서 전체 지표는 두 배로 희석된다. 판정은 **결합 부분집합(covered)** 으로 하고
전체·급속은 참고로만 본다.

판정 기준 (config.FEATURE_COLS 주석의 session_prior 기각 선례를 따른다)
  채택: 결합 부분집합에서 PR-AUC(불가)·unavail-recall 이 함께 오르고
        Brier 가 나빠지지 않으며, B - C 가 그 이득의 대부분을 설명할 것
  기각: AUC 만 오르고 PR-AUC/recall 이 내려가거나, B ≈ C

사용:
  py -m recommend_api.experiment_detour_eta
  py -m recommend_api.experiment_detour_eta --detour data/da1_refined/station_eta_detour.parquet
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, FEATURE_COLS, HORIZONS, SCHEDULER_ROOT
from .eval_metrics import evaluate_classifier, split_by_date
from .holdout_eval import _new_model, _per_eta_metrics, _prepare_fold
from .model_store import build_horizon_dataset, load_joined, load_label_series

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

OUT_DIR = ARTIFACTS_DIR / "experiments"
DEFAULT_DETOUR = SCHEDULER_ROOT / "data" / "da1_refined" / "station_eta_detour.parquet"

DETOUR_COLS = ["detour_ratio", "eta_per_km"]
PLACEBO_COLS = ["placebo_station_const"]
ADJUSTED_COLS = ["eta_road_adj"]

SEED = 20260810


def load_detour(path: Path) -> pd.DataFrame:
    """DA① 팩 정제본. 1,848소 × 소별 1행이라 통째로 읽어 merge 한다."""
    if not path.exists():
        raise SystemExit(
            f"우회계수 파일 없음: {path}\n"
            "DA① 팩 정제 추출(refine_da1.py)의 station_eta_detour.parquet 를 놓을 것."
        )
    df = pd.read_parquet(path)
    need = {"stat_id", "detour_ratio", "eta_per_km", "haversine_km", "tmap_road_km"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"필수 컬럼 누락: {sorted(missing)}")

    df = df[["stat_id", "detour_ratio", "eta_per_km", "haversine_km", "tmap_road_km"]].copy()
    df["stat_id"] = df["stat_id"].astype(str)
    for c in ("detour_ratio", "eta_per_km"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # 0 나눗셈·API 이상치 방어. 실측 분포는 p1~p99 가 1.1~2.6 안에 들어온다.
        df.loc[~np.isfinite(df[c]), c] = np.nan
    df = df.drop_duplicates("stat_id")

    lo, hi = df["detour_ratio"].quantile([0.001, 0.999])
    n_out = int(((df.detour_ratio < lo) | (df.detour_ratio > hi)).sum())
    if n_out:
        print(f"  detour_ratio 극단값 {n_out}건 → NaN")
        df.loc[(df.detour_ratio < lo) | (df.detour_ratio > hi), "detour_ratio"] = np.nan
    return df


def attach_detour(hz: pd.DataFrame, detour: pd.DataFrame) -> pd.DataFrame:
    """충전소 단위로 붙인다. 같은 소의 모든 충전기가 같은 값을 받는다."""
    out = hz.merge(detour, on="stat_id", how="left")
    if len(out) != len(hz):
        raise RuntimeError(
            f"merge 후 행수 변동 {len(hz):,} → {len(out):,} — stat_id 중복 의심"
        )

    # C: 같은 커버리지·같은 주변분포를 갖되 정보가 0인 소별 난수.
    #    detour_ratio 가 붙은 소에만 값을 주어야 커버리지 효과가 상쇄된다.
    rng = np.random.default_rng(SEED)
    covered = detour.loc[detour.detour_ratio.notna(), "stat_id"].unique()
    src = detour.detour_ratio.dropna().to_numpy()
    placebo = pd.DataFrame({
        "stat_id": covered,
        "placebo_station_const": rng.permutation(
            rng.choice(src, size=len(covered), replace=True)
        ),
    })
    out = out.merge(placebo, on="stat_id", how="left")

    # D: 지평과 곱해 소별 상수성을 깬 변형. 중앙값으로 나눠 eta_minutes 스케일 유지.
    med = float(np.nanmedian(out["detour_ratio"])) or 1.0
    out["eta_road_adj"] = out["eta_minutes"] * (out["detour_ratio"] / med)
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
    covered = _subset(y_te, proba, te["detour_ratio"].notna())
    covered_fast = _subset(y_te, proba, te["detour_ratio"].notna() & (te["is_fast"] == 1))
    per_eta = _per_eta_metrics(y_te, proba, te["eta_minutes"])
    # 결합 부분집합 안에서의 ETA별 — 장거리 개선 여부가 이 실험의 핵심.
    # _per_eta_metrics 는 .values 를 쓰므로 ndarray 가 아니라 Series 로 넘긴다.
    cov_mask = te["detour_ratio"].notna().to_numpy()
    per_eta_covered = _per_eta_metrics(
        pd.Series(np.asarray(y_te)[cov_mask]),
        np.asarray(proba)[cov_mask],
        pd.Series(te.loc[te["detour_ratio"].notna(), "eta_minutes"].to_numpy()),
    )

    print(f"\n[{label}]  피처 {X.shape[1]}개")
    for name, m in (("전체", overall), ("급속", fast),
                    ("결합", covered), ("결합·급속", covered_fast)):
        if m is None:
            print(f"  {name:<9} 표본 부족")
            continue
        print(
            f"  {name:<9} AUC {m['roc_auc']:.4f} | "
            f"unavail-recall {m['unavailable_recall']:.3f} | "
            f"PR-AUC {m['pr_auc_unavailable']:.4f} | "
            f"Brier {m['brier_score']:.4f} | n={m['n']:,}"
        )

    return {
        "label": label,
        "n_features": int(X.shape[1]),
        "overall": overall,
        "fast": fast,
        "covered": covered,
        "covered_fast": covered_fast,
        "per_eta": per_eta,
        "per_eta_covered": per_eta_covered,
    }


def _diff_block(base: dict, arm: dict, keys=("overall", "fast", "covered", "covered_fast")) -> None:
    for scope in keys:
        a, b = base.get(scope), arm.get(scope)
        if not a or not b:
            continue
        print(f"  [{scope:<12}]", end="")
        for k, lab in (
            ("roc_auc", "AUC"),
            ("unavailable_recall", "unavail-R"),
            ("pr_auc_unavailable", "PR-AUC"),
            ("brier_score", "Brier↓"),
        ):
            print(f"  {lab} {b[k] - a[k]:+.4f}", end="")
        print()


def _verdict(a: dict, b: dict, c: dict) -> tuple[str, list[str]]:
    """session_prior 기각 선례와 같은 기준으로 자동 판정한다."""
    reasons: list[str] = []
    cov_a, cov_b, cov_c = a.get("covered"), b.get("covered"), c.get("covered")
    if not (cov_a and cov_b and cov_c):
        return "INCONCLUSIVE", ["결합 부분집합 표본 부족"]

    d_pr = cov_b["pr_auc_unavailable"] - cov_a["pr_auc_unavailable"]
    d_rec = cov_b["unavailable_recall"] - cov_a["unavailable_recall"]
    d_brier = cov_b["brier_score"] - cov_a["brier_score"]
    placebo_pr = cov_c["pr_auc_unavailable"] - cov_a["pr_auc_unavailable"]
    net = d_pr - placebo_pr

    if d_pr <= 0 or d_rec <= 0:
        reasons.append(f"PR-AUC {d_pr:+.4f} / recall {d_rec:+.4f} — 사용불가 탐지가 개선되지 않음")
    if d_brier > 0.0005:
        reasons.append(f"Brier {d_brier:+.4f} 악화 — 충전소 기저율 암기 징후")
    if net <= 0.002:
        reasons.append(
            f"위약 대비 순이득 {net:+.4f} — 이득이 도로 기하가 아니라 충전소 ID 로 설명됨"
        )
    return ("REJECT" if reasons else "ADOPT_CANDIDATE"), reasons


def main() -> None:
    p = argparse.ArgumentParser(description="TMAP 우회계수 피처 효과 실험")
    p.add_argument("--detour", type=Path, default=DEFAULT_DETOUR,
                   help=f"우회계수 parquet 경로 (기본 {DEFAULT_DETOUR})")
    p.add_argument("--max-rows", type=int, default=None, metavar="N",
                   help=("load_joined 상한. 기본은 전량. load_joined 이 stat_id 순으로 "
                         "정렬하므로 상한을 걸면 모집단이 잘린다 — 권장하지 않는다."))
    args = p.parse_args()

    if args.max_rows:
        print(f"주의: --max-rows {args.max_rows:,} 는 stat_id 정렬 상한이라 "
              "TMAP 결합 충전소가 치우쳐 잘릴 수 있다.")

    print("우회계수 로드...")
    detour = load_detour(args.detour)
    print(f"  {len(detour):,}소 · detour_ratio 중앙 {detour.detour_ratio.median():.3f} "
          f"(p25 {detour.detour_ratio.quantile(.25):.3f} / p75 {detour.detour_ratio.quantile(.75):.3f})")

    print("데이터 로드...")
    base = load_joined(limit=args.max_rows)
    print(f"  status JOIN {len(base):,}행")

    print("라벨 시계열 로드 (delta+snapshot)...")
    label_df = load_label_series()

    print("horizon 데이터셋 생성...")
    hz = build_horizon_dataset(base, HORIZONS, label_df=label_df).reset_index(drop=True)
    print(f"  {len(hz):,}샘플 / positive_rate {hz['y'].mean():.4f}")
    if hz.empty:
        raise SystemExit("horizon 샘플 없음")

    print("우회계수 결합...")
    df = attach_detour(hz, detour)

    n = len(df)
    cov = float(df["detour_ratio"].notna().mean())
    fast_mask = df["is_fast"] == 1
    cov_fast = float(df.loc[fast_mask, "detour_ratio"].notna().mean())
    print(f"  결합률  전체 {100*cov:.1f}%  급속 {100*cov_fast:.1f}%")
    print(f"  결합 충전소 {df.loc[df.detour_ratio.notna(),'stat_id'].nunique():,} / "
          f"{df.stat_id.nunique():,}  ·  급속 샘플 {int(fast_mask.sum()):,} / {n:,}")
    if cov_fast < 0.01:
        raise SystemExit("급속 표본에 우회계수가 거의 안 붙었다. stat_id 형식 불일치 확인 필요.")

    train_mask, train_dates, test_dates = split_by_date(df)
    print(f"\n날짜 홀드아웃: train {len(train_dates)}일 / test {len(test_dates)}일"
          f"  ({int(train_mask.sum()):,} vs {int((~train_mask).sum()):,}샘플)")

    a = run_arm(df, FEATURE_COLS, train_mask, "A baseline")
    b = run_arm(df, FEATURE_COLS + DETOUR_COLS, train_mask, "B +detour")
    c = run_arm(df, FEATURE_COLS + PLACEBO_COLS, train_mask, "C +placebo(대조)")
    d = run_arm(df, FEATURE_COLS + ADJUSTED_COLS, train_mask, "D +eta_road_adj")

    print("\n" + "=" * 78)
    print("차이 (B - A) : 우회계수 원값 추가")
    print("=" * 78)
    _diff_block(a, b)
    print("\n차이 (C - A) : 위약 소별 상수 — 여기 오르는 만큼은 충전소 ID 효과")
    print("-" * 78)
    _diff_block(a, c)
    print("\n차이 (D - A) : 지평 상호작용 변형")
    print("-" * 78)
    _diff_block(a, d)

    print(f"\n{'ETA':>5}{'A AUC':>10}{'B AUC':>10}{'B-A':>9}{'B PR':>10}{'B-A PR':>9}   (결합 부분집합)")
    print("-" * 62)
    bm = {r["eta_minutes"]: r for r in b["per_eta_covered"]}
    for ra in a["per_eta_covered"]:
        h = ra["eta_minutes"]
        rb = bm.get(h)
        if not rb:
            continue
        print(f"{h:>5}{ra['roc_auc']:>10.4f}{rb['roc_auc']:>10.4f}"
              f"{rb['roc_auc']-ra['roc_auc']:>+9.4f}"
              f"{rb['pr_auc_unavailable']:>10.4f}"
              f"{rb['pr_auc_unavailable']-ra['pr_auc_unavailable']:>+9.4f}")

    verdict, reasons = _verdict(a, b, c)
    print("\n" + "=" * 78)
    print(f"판정: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    if verdict == "ADOPT_CANDIDATE":
        print("  결합 부분집합에서 사용불가 탐지가 개선되고 위약으로 설명되지 않음.")
        print("  → config.FEATURE_COLS 에 추가 검토. 단, 서빙은 사용자 위치 기반 ETA 라")
        print("    detour_ratio 를 origin 무관 소 속성으로 쓸 수 있는지 먼저 확인할 것.")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"detour_eta_experiment_{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detour_source": str(args.detour),
        "n_samples": int(n),
        "positive_rate": float(df["y"].mean()),
        "coverage_all": cov,
        "coverage_fast": cov_fast,
        "train_dates": [str(x) for x in train_dates],
        "test_dates": [str(x) for x in test_dates],
        "verdict": verdict,
        "verdict_reasons": reasons,
        "arms": {"A_baseline": a, "B_detour": b, "C_placebo": c, "D_adjusted": d},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n저장: {out}")
    print("운영 아티팩트(horizon_hgb.joblib)는 변경하지 않았다.")


if __name__ == "__main__":
    main()
