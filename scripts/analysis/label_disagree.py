"""delta-only LOCF 라벨 vs delta+snapshot LOCF 라벨의 불일치율.

같은 (stat_id, chger_id, created_at, eta) 표본에서 y 가 갈리는 비율 = delta 스트림이
놓친 상태 변화의 크기. CLAUDE.md 의 '07-31 스냅샷 대비 LOCF 정확도 97.7%' 를
신규 2일(스냅샷 6시간 주기 정상화 이후)에서 다시 잰다.
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

import pandas as pd

from recommend_api import model_store as ms
from recommend_api.config import HORIZONS

sys.path.insert(0, str(Path(__file__).parent))
from oot_eval import CUTOFF, load_joined_since  # noqa: E402
from oot_label_ab import label_series  # noqa: E402

KEY = ["stat_id", "chger_id", "created_at", "eta_minutes"]

df = load_joined_since(CUTOFF)
frames = {}
for src in ("delta", "all"):
    hz = ms.build_horizon_dataset(df, HORIZONS, label_df=label_series(src))
    frames[src] = hz[KEY + ["y", "future_stat", "label_staleness_min"]].rename(
        columns={"y": f"y_{src}", "future_stat": f"stat_{src}", "label_staleness_min": f"age_{src}"}
    )

m = frames["delta"].merge(frames["all"], on=KEY, how="inner")
print(f"공통 표본: {len(m):,}")
dis = m["y_delta"] != m["y_all"]
print(f"y 불일치: {dis.sum():,} ({dis.mean()*100:.2f}%)")
sd = m["stat_delta"] != m["stat_all"]
print(f"stat 불일치: {sd.sum():,} ({sd.mean()*100:.2f}%)")
print("\n불일치 방향 (delta라벨 -> snapshot라벨):")
print(m.loc[dis].groupby(["y_delta", "y_all"]).size().to_string())
print("\n라벨 나이대별 y 불일치율 (delta 라벨 기준):")
bins = [0, 5, 15, 30, 60, 120, 240, 481]
m["age_bin"] = pd.cut(m["age_delta"], bins=bins, right=False)
print((m.groupby("age_bin", observed=True)
        .agg(n=("y_delta", "size"), 불일치율=(("y_delta"), lambda s: 0))
        .drop(columns="불일치율")
        .join(m.assign(d=dis).groupby("age_bin", observed=True)["d"].mean().rename("불일치율"))
      ).to_string())
print("\nETA별 y 불일치율:")
print(m.assign(d=dis).groupby("eta_minutes")["d"].agg(["size", "mean"]).to_string())
