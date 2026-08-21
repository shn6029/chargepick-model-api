"""날짜별 성능이 오르는 게 모델 때문인지 '라벨·표본 품질' 때문인지 가른다.

08-04 AUC 0.9578 -> 08-06 0.9755 로 고정 모델의 지표가 올라갔다. 후보 원인:
  (a) 라벨 나이(LOCF staleness)가 짧아졌다 = 수집이 촘촘해졌다
  (b) delta 누락(=delta 라벨과 snapshot 라벨의 불일치)이 줄었다
  (c) 그냥 시간대 구성 차이(08-07 은 오전만 있다)
per-day 로 (a)(b) 를 재고, (c) 는 같은 시간대(00~09시)만 잘라 통제한다.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\dev\scheduler")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from recommend_api import model_store as ms
from recommend_api.config import HORIZONS
from recommend_api.eval_metrics import evaluate_classifier

sys.path.insert(0, str(Path(__file__).parent))
from oot_eval import CUTOFF, load_joined_since  # noqa: E402
from oot_label_ab import label_series  # noqa: E402

KEY = ["stat_id", "chger_id", "created_at", "eta_minutes"]

df = load_joined_since(CUTOFF)
print(f"표본 원본 {len(df):,}행")

art = ms.load_artifact()
hz_all = ms.build_horizon_dataset(df, HORIZONS, label_df=label_series("all"))
X, y, _m, meta = ms.make_xy(hz_all, fill_median=False)
proba = art["model"].predict_proba(ms.align_features(X, art["feature_columns"], art["medians"]))[:, 1]

base = hz_all[KEY].copy()
base["y"] = y.values
base["proba"] = proba
base["age"] = hz_all["label_staleness_min"].values
base["is_fast"] = meta["is_fast"].values
base["date"] = base["created_at"].dt.date.astype(str)
base["hour"] = base["created_at"].dt.hour

hz_delta = ms.build_horizon_dataset(df, HORIZONS, label_df=label_series("delta"))
d = hz_delta[KEY].copy()
d["y_delta"] = hz_delta["y"].values
merged = base.merge(d, on=KEY, how="inner")

print("\n== 날짜별 품질 ==")
print(" 날짜        표본     pos   라벨나이(중앙/p90)  delta불일치  AUC     PR-AUC  불가R")
for dt, g in merged.groupby("date"):
    m = evaluate_classifier(g["y"], g["proba"].to_numpy(), threshold=0.5)
    dis = (g["y"] != g["y_delta"]).mean()
    print(
        f" {dt}  {len(g):>9,} {g['y'].mean():.3f}   {g['age'].median():5.1f} /{g['age'].quantile(0.9):6.1f}"
        f"     {dis*100:5.2f}%   {m['roc_auc']:.4f} {m['pr_auc_unavailable']:.4f} {m['unavailable_recall']:.4f}"
    )

print("\n== 같은 시간대(00~09시)만 ==")
sub = merged[merged["hour"] < 9]
print(" 날짜        표본     pos   라벨나이중앙  delta불일치  AUC     PR-AUC  불가R")
for dt, g in sub.groupby("date"):
    if len(g) < 5000:
        continue
    m = evaluate_classifier(g["y"], g["proba"].to_numpy(), threshold=0.5)
    dis = (g["y"] != g["y_delta"]).mean()
    print(
        f" {dt}  {len(g):>9,} {g['y'].mean():.3f}      {g['age'].median():5.1f}      {dis*100:5.2f}%   "
        f"{m['roc_auc']:.4f} {m['pr_auc_unavailable']:.4f} {m['unavailable_recall']:.4f}"
    )

print("\n== 급속만, 같은 시간대(00~09시) ==")
for dt, g in sub[sub["is_fast"] == 1].groupby("date"):
    if len(g) < 3000:
        continue
    m = evaluate_classifier(g["y"], g["proba"].to_numpy(), threshold=0.5)
    print(
        f" {dt}  n={len(g):>8,} 불가율={1-g['y'].mean():.3f} AUC={m['roc_auc']:.4f} "
        f"PR-AUC={m['pr_auc_unavailable']:.4f} 불가R={m['unavailable_recall']:.4f}"
    )
