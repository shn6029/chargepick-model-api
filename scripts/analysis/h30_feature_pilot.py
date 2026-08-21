"""h30 개선 파일럿 — tick panel 기반 시계열 피처.

문제
----
현행 충전소 학습셋(station_horizon_training_v1)의 피처는 **전부 현재 시점 스냅샷**이다.
available_count / total_chargers / observation_coverage / observation_age_minutes ...
"지금 몇 대 비었나"는 있는데 "얼마나 자주 비는 곳인가", "지금 비는 중인가 차는 중인가"가 없다.

h5·h10 은 현재 상태가 거의 그대로 유지되므로(라벨 일치율 99.7%) 스냅샷만으로 충분하다.
h30 은 라벨이 h5 와 8.3% 어긋나는데, 그 8.3% 를 맞히려면 **변화의 방향과 속도**가 필요하다.
스냅샷 피처에는 그 정보가 없다.

이 스크립트는 station_tick_panel.parquet(698만 행, 5분 grid)에서
회전율·추세·이력 피처를 만들어 h30 이 실제로 개선되는지 측정한다.

주의
----
- 롤링 창은 **과거만** 본다(현재 tick 포함). 미래 tick 이 들어가면 누수다.
- 충전소 단위 사전확률은 폐기했다(REJECTED_FEATURES 주석 참고).
- 평가는 파일의 split 을 그대로 쓰고, horizon 별로 나눠 본다.
- horizon 마다 양성률이 다르므로 PR-AUC 절대값은 비교 불가. lift(무작위 대비 배수)로 본다.

사용법
------
    py scripts/analysis/h30_feature_pilot.py
    py scripts/analysis/h30_feature_pilot.py --sample 400000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_station_models import (  # noqa: E402
    DEFAULT_DATA,
    FEATURESETS,
    OUT_DIR,
    expected_calibration_error,
)
from recommend_api.eval_metrics import evaluate_classifier  # noqa: E402

PANEL_PATH = DEFAULT_DATA.parent / "station_tick_panel.parquet"

# 채택 피처 — 셋 다 h30 을 개선한다 (ablation 근거는 아래 REJECTED 주석 참고)
NEW_FEATURES = [
    # 회전율: "얼마나 자주 비는 곳인가"       h30 +0.0920
    "avail_ratio_30m",
    "avail_ratio_60m",
    "avail_ratio_180m",
    # 추세: "지금 비는 중인가 차는 중인가"     h30 +0.0618
    "avail_count_mean_60m",
    "avail_count_trend_60m",
    "changes_60m",
    # 경과시간: "이 상태가 얼마나 됐나"        h30 +0.0358
    "min_since_avail",
    "min_since_full",
]

# 폐기 — 충전소별 사전확률 2종. 넣으면 **성능이 반토막 난다.**
#   baseline h5 0.7989 / h30 0.4627
#   +priors  h5 0.6336 / h30 0.2197
# 충전소 단위 집계라 사실상 station_id 를 외우는 고차원 인코딩으로 작동한다.
# train 기간의 충전소별 가용률을 암기하고 test 기간에 그대로 적용하는데,
# 충전소별 패턴은 주 단위로 바뀌므로 전부 틀린 사전확률이 된다.
# 시간대별 사전확률이 필요하면 충전소 단위가 아니라 **군집 단위**로 묶을 것.
REJECTED_FEATURES = ["station_hour_prior", "station_base_rate"]


def build_panel_features(panel: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """tick panel → 충전소×시각 시계열 피처.

    available_recon 을 쓴다(관측 결측을 앞 상태로 채운 복원값). available_observed 는
    관측 안 된 tick 에서 0 이 되어 '비어 있음'과 '모름'이 구분되지 않는다.
    """
    p = panel[["station_id", "panel_time", "available_recon", "known_recon"]].copy()
    p = p.sort_values(["station_id", "panel_time"])
    p["is_avail"] = (p["available_recon"] > 0).astype(np.float32)
    p["available_recon"] = p["available_recon"].astype(np.float32)

    out = [p[["station_id", "panel_time"]].reset_index(drop=True)]
    g = p.set_index("panel_time").groupby("station_id", sort=False)

    # 롤링 가용 비율 — "얼마나 자주 비는 곳인가"
    for win, name in (("30min", "avail_ratio_30m"),
                      ("60min", "avail_ratio_60m"),
                      ("180min", "avail_ratio_180m")):
        s = g["is_avail"].rolling(win, closed="both").mean()
        out.append(s.reset_index(drop=True).rename(name))

    m60 = g["available_recon"].rolling("60min", closed="both").mean()
    out.append(m60.reset_index(drop=True).rename("avail_count_mean_60m"))

    # 회전율 — 최근 1시간 상태 전환 횟수
    ch = g["is_avail"].rolling("60min", closed="both").apply(
        lambda a: float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0, raw=True
    )
    out.append(ch.reset_index(drop=True).rename("changes_60m"))

    feat = pd.concat(out, axis=1)
    feat["available_recon"] = p["available_recon"].to_numpy()
    feat["is_avail"] = p["is_avail"].to_numpy()

    # 추세 — 현재값이 최근 평균보다 높은가(비는 중) 낮은가(차는 중)
    feat["avail_count_trend_60m"] = (
        feat["available_recon"] - feat["avail_count_mean_60m"]
    )

    # 마지막으로 비었던/꽉 찼던 뒤 경과 분
    t = feat["panel_time"]
    for flag, name in ((feat["is_avail"] == 1, "min_since_avail"),
                       (feat["is_avail"] == 0, "min_since_full")):
        marked = t.where(flag)
        last = marked.groupby(feat["station_id"]).ffill()
        feat[name] = (t - last).dt.total_seconds() / 60.0

    keep = ["station_id", "panel_time"] + NEW_FEATURES
    return feat[keep]


def evaluate(df: pd.DataFrame, feats: list[str], label: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    med = tr[feats].median()
    model = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
    )
    model.fit(tr[feats].fillna(med), tr["target_available"])
    proba = model.predict_proba(te[feats].fillna(med))[:, 1]

    res = {"label": label, "n_features": len(feats), "features": feats, "per_horizon": {}}
    overall = evaluate_classifier(te["target_available"], proba)
    res["overall"] = {
        "pr_auc_unavailable": overall["pr_auc_unavailable"],
        "roc_auc": overall["roc_auc"],
        "brier": overall["brier_score"],
        "ece": expected_calibration_error(te["target_available"], proba),
        "unavailable_recall": overall["unavailable_recall"],
    }
    tmp = pd.DataFrame({
        "y": te["target_available"].to_numpy(),
        "p": proba,
        "h": te["horizon_minutes"].to_numpy(),
    })
    for h, g in tmp.groupby("h"):
        if g["y"].nunique() < 2:
            continue
        m = evaluate_classifier(g["y"], g["p"])
        neg = 1.0 - float(g["y"].mean())
        res["per_horizon"][int(h)] = {
            "n": m["n"],
            "negative_rate": neg,
            "pr_auc_unavailable": m["pr_auc_unavailable"],
            # 양성률이 horizon 마다 달라 PR-AUC 절대값은 비교 불가.
            # 무작위 기준선(=음성 비율) 대비 배수로 정규화한다.
            "pr_auc_lift": m["pr_auc_unavailable"] / neg if neg > 0 else None,
            "unavailable_recall": m["unavailable_recall"],
            "brier": m["brier_score"],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"학습셋 로드: {Path(args.data).name}")
    df = pd.read_parquet(args.data)
    train_end = df.loc[df["split"] == "train", "feature_as_of"].max()
    print(f"train 종료: {train_end}")

    print(f"tick panel 로드: {Path(args.panel).name}")
    panel = pd.read_parquet(args.panel)
    print(f"  {len(panel):,}행 / 충전소 {panel.station_id.nunique():,}")

    print("시계열 피처 생성 중...")
    feat = build_panel_features(panel, train_end)
    print(f"  피처 {len(feat):,}행")

    merged = df.merge(
        feat, left_on=["station_id", "feature_as_of"],
        right_on=["station_id", "panel_time"], how="left",
    )
    hit = merged[NEW_FEATURES].notna().any(axis=1).mean()
    print(f"  조인 성공률 {hit:.2%}")
    if args.sample and args.sample < len(merged):
        idx = np.concatenate([
            g.sample(max(int(len(g) * args.sample / len(merged)), 1), random_state=42).index.to_numpy()
            for _, g in merged.groupby("split", sort=False)
        ])
        merged = merged.loc[idx].reset_index(drop=True)

    base_feats = FEATURESETS["mvp_plus"]
    full_feats = FEATURESETS["full"]
    runs = [
        ("baseline (mvp_plus)", base_feats),
        ("full", full_feats),
        ("mvp_plus + 시계열", base_feats + NEW_FEATURES),
        ("full + 시계열", full_feats + NEW_FEATURES),
    ]
    results = []
    for label, feats in runs:
        print(f"\n--- {label} ({len(feats)}개) ---")
        r = evaluate(merged, feats, label)
        results.append(r)
        for h, m in sorted(r["per_horizon"].items()):
            print(
                f"  h{h:<3d} PR-AUC={m['pr_auc_unavailable']:.4f} "
                f"(lift {m['pr_auc_lift']:.1f}x) recall={m['unavailable_recall']:.3f}"
            )

    print("\n=== h30 요약 ===")
    for r in results:
        m = r["per_horizon"].get(30)
        if m:
            print(
                f"  {r['label']:22s} PR-AUC={m['pr_auc_unavailable']:.4f} "
                f"lift={m['pr_auc_lift']:.1f}x recall={m['unavailable_recall']:.3f}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "h30_feature_pilot.json"
    out.write_text(
        json.dumps({"runs": results, "new_features": NEW_FEATURES},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
