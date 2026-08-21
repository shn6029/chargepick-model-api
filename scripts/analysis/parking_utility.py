# -*- coding: utf-8 -*-
"""주차 정보를 UI 로 노출할 때 실제로 얼마나 쓸모가 있는지 평가.

모델 피처로는 기각됐지만(잔차 상관 R^2 0.54%), '표시용 정보'로서의 가치는
별개다. 그건 정확도가 아니라 아래 넷으로 판단한다.
  1) 요청 관점 커버리지 — 추천 결과 top-K 중 몇 개에 주차 정보가 붙나
  2) 데이터가 살아 있나 — 값이 고정된 죽은 주차장은 없나
  3) 정보량 — 점유율이 실제로 변하나(항상 여유면 알려줄 게 없다)
  4) 신선도 — 요청 시점에 몇 분 전 데이터인가
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(r"F:\dev\scheduler\.env")
c = pymysql.connect(
    host=os.getenv("DB_HOST"), port=3306, user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_NAME"),
    charset="utf8mb4", connect_timeout=60, read_timeout=900, autocommit=True)

st = pd.read_pickle(r"F:\dev\scheduler\data\reports\_parking_match.pkl")

print("=" * 68)
print("[1] 요청 관점 커버리지 — 반경 2km 안에서 몇 %가 주차 정보를 갖나")
# 대구 주요 지점에서 반경 2km 급속 충전소 중 주차정보 보유 비율
spots = {
    "동성로(중구)": (35.8693, 128.5947),
    "수성구청":     (35.8583, 128.6311),
    "성서산업단지": (35.8207, 128.4880),
    "칠곡(북구)":   (35.9436, 128.5432),
    "동대구역":     (35.8797, 128.6285),
}
fast = st[st.output >= 50].copy()
rows = []
for name, (la, lo) in spots.items():
    d = 2 * 6371000 * np.arcsin(np.sqrt(
        np.sin(np.radians(fast.lat.astype(float) - la) / 2) ** 2 +
        np.cos(np.radians(la)) * np.cos(np.radians(fast.lat.astype(float))) *
        np.sin(np.radians(fast.lng.astype(float) - lo) / 2) ** 2))
    near = fast[d <= 2000]
    for r_m, lab in ((100, "100m"), (300, "300m")):
        pass
    rows.append({
        "지점": name, "반경2km_급속": len(near),
        "주차정보(100m)": int((near.nearest_lot_m <= 100).sum()),
        "주차정보(300m)": int((near.nearest_lot_m <= 300).sum()),
    })
cov = pd.DataFrame(rows)
cov["비율100m"] = (cov["주차정보(100m)"] / cov["반경2km_급속"] * 100).round(1)
cov["비율300m"] = (cov["주차정보(300m)"] / cov["반경2km_급속"] * 100).round(1)
print(cov.to_string(index=False))

print("\n" + "=" * 68)
print("[2] 데이터가 살아 있나 — 주차장별 점유율 변동")
pk = pd.read_sql("""
    SELECT pklt_id, collected_at, total_spaces, remaining_spaces, occupancy_rate
    FROM parking_realtime_status
    WHERE collected_at >= NOW() - INTERVAL 7 DAY""", c)
pk["occupancy_rate"] = pd.to_numeric(pk["occupancy_rate"], errors="coerce")
g = pk.groupby("pklt_id")["occupancy_rate"].agg(["count", "mean", "std", "min", "max"])
g["range"] = g["max"] - g["min"]
print(f"  실시간 주차장 {len(g)}곳 (최근 7일)")
dead = g[(g["std"].fillna(0) < 0.01)]
print(f"  값이 고정된 곳(std<0.01): {len(dead)}곳 ({len(dead)/len(g)*100:.0f}%)")
print(f"  점유율 변동폭(max-min) 중앙값 {g['range'].median():.1f}%p")
print("\n  변동폭 분포:")
for lo, hi in [(0,5),(5,20),(20,40),(40,70),(70,101)]:
    k = ((g["range"] >= lo) & (g["range"] < hi)).sum()
    print(f"    {lo:>3}~{hi:>3}%p  {k:>3}곳")

print("\n" + "=" * 68)
print("[3] 정보량 — 실제로 만차가 되나")
print(f"  전체 관측 점유율: 평균 {pk.occupancy_rate.mean():.1f}%  "
      f"중앙값 {pk.occupancy_rate.median():.1f}%")
for th in (70, 80, 90, 95):
    print(f"    {th}% 이상인 관측 비율: {(pk.occupancy_rate>=th).mean()*100:5.2f}%")
print("\n  주차장별 '한 번이라도 90% 넘은 곳':",
      f"{(g['max']>=90).sum()}곳 / {len(g)}곳")

print("\n" + "=" * 68)
print("[4] 신선도 — 지금 시점 데이터 나이")
fresh = pd.read_sql("""
    SELECT TIMESTAMPDIFF(MINUTE, MAX(collected_at), NOW()) age_min
    FROM parking_realtime_status""", c)
print(f"  최신 수집 이후 경과: {int(fresh.age_min[0])}분")
print("  수집 주기 10분 → 요청 시점 데이터 나이는 0~10분")

print("\n" + "=" * 68)
print("[5] 100m 매칭 급속 충전소가 어떤 곳인가")
top = st[(st.output >= 50) & (st.nearest_lot_m <= 100)].nsmallest(10, "nearest_lot_m")
print(top[["stat_nm", "nearest_lot_m"]].to_string(index=False))
c.close()
