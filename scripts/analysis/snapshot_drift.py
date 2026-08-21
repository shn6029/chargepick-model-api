"""스냅샷 배치마다 'delta LOCF 로 복원한 상태'가 실제와 얼마나 어긋나는지 잰다.

모델과 무관한 순수 수집 품질 지표다. 배치별로 추세를 보면 08-04 의 높은 불일치가
(a) 스냅샷 공백기(08-01~03)에 쌓인 drift 가 한꺼번에 드러난 것인지
(b) 그냥 그날 수집이 나빴던 것인지 구분할 수 있다.
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

from recommend_api.config import LABEL_MAX_STALENESS_MIN
from recommend_api.model_store import get_connection

conn = get_connection()
try:
    st = pd.read_sql(
        """
        SELECT stat_id, chger_id, created_at, stat, source, snapshot_batch_id
        FROM ev_charger_status
        WHERE created_at >= '2026-08-02'
        ORDER BY created_at
        """,
        conn,
    )
finally:
    conn.close()
st["created_at"] = pd.to_datetime(st["created_at"])
delta = st[st["source"] != "snapshot"].copy()
snap = st[st["source"] == "snapshot"].copy()
print(f"delta {len(delta):,} · snapshot {len(snap):,}")

KEY = ["stat_id", "chger_id"]
delta_sorted = (
    delta[KEY + ["created_at", "stat"]]
    .rename(columns={"created_at": "delta_at", "stat": "delta_stat"})
    .sort_values("delta_at", kind="mergesort")
    .reset_index(drop=True)
)

rows = []
for batch, g in snap.groupby("snapshot_batch_id"):
    at = g["created_at"].min()
    left = (
        g[KEY + ["created_at", "stat"]]
        .rename(columns={"stat": "snap_stat"})
        .sort_values("created_at", kind="mergesort")
    )
    m = pd.merge_asof(
        left,
        delta_sorted,
        left_on="created_at",
        right_on="delta_at",
        by=KEY,
        direction="backward",
        tolerance=pd.Timedelta(minutes=LABEL_MAX_STALENESS_MIN),
    ).dropna(subset=["delta_stat"])
    if m.empty:
        continue
    m["age_min"] = (m["created_at"] - m["delta_at"]).dt.total_seconds() / 60
    mismatch = m["delta_stat"].astype(int) != m["snap_stat"].astype(int)
    # 가용/불가 이분법으로 봤을 때(라벨 기준)
    y_mismatch = (m["delta_stat"].astype(int) == 2) != (m["snap_stat"].astype(int) == 2)
    rows.append(
        {
            "snapshot_at": at,
            "n_compared": len(m),
            "stat_불일치%": round(mismatch.mean() * 100, 2),
            "y_불일치%": round(y_mismatch.mean() * 100, 2),
            "delta나이_중앙": round(m["age_min"].median(), 1),
            "delta나이_p90": round(m["age_min"].quantile(0.9), 1),
        }
    )

out = pd.DataFrame(rows).sort_values("snapshot_at")
print("\n== 스냅샷 배치별 delta LOCF 복원 오차 ==")
print(out.to_string(index=False))
