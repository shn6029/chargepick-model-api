# -*- coding: utf-8 -*-
"""parking 결합 가능성 평가.

관문 1 커버리지 / 2 시간해상도 / 3 신호. 하나라도 무너지면 나머지는 볼 필요 없다.
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
q = lambda s: pd.read_sql(s, c)

print("=" * 70)
print("[0] 현재 서빙 상태 점검")
t = q("""SELECT COUNT(*) n FROM information_schema.tables
         WHERE table_schema='team_5' AND table_name='ev_charger_parking_map'""").n[0]
print(f"  ev_charger_parking_map 존재: {'예' if t else '아니오'}")
print("  -> parking.py 가 이 테이블을 조인하므로, 없으면 주차 점유율이")
print("     항상 비어서 parking_occupancy_coefficient 가 기본값 1.0 이 된다.")

print("\n" + "=" * 70)
print("[1] 관문 2: 시간 해상도 / 신선도")
print(q("""SELECT MIN(collected_at) oldest, MAX(collected_at) newest,
    COUNT(*) n, COUNT(DISTINCT pklt_id) lots FROM parking_realtime_status
""").to_string(index=False))
print("\n  최근 3시간 10분 단위 수집량:")
print(q("""SELECT FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(collected_at)/600)*600) b,
    COUNT(*) n, COUNT(DISTINCT pklt_id) lots FROM parking_realtime_status
    WHERE collected_at >= NOW() - INTERVAL 3 HOUR
    GROUP BY b ORDER BY b""").to_string(index=False))

print("\n" + "=" * 70)
print("[2] 관문 1: 지리 커버리지 (좌표로 매칭)")
lots = q("""SELECT pklt_id, pklt_nm, lat, lot AS lng, total_spaces
            FROM parking_lot_info
            WHERE lat IS NOT NULL AND lot IS NOT NULL AND lat > 0""")
lots = lots.drop_duplicates("pklt_id")
print(f"  좌표 있는 주차장 {len(lots):,}곳")

st = q("""SELECT DISTINCT stat_id, stat_nm, lat, lng, MAX(output) AS output
          FROM ev_charger_info
          WHERE (del_yn IS NULL OR del_yn<>'Y') AND lat IS NOT NULL
          GROUP BY stat_id, stat_nm, lat, lng""")
st["output"] = pd.to_numeric(st["output"], errors="coerce").fillna(0)
for col in ("lat", "lng"):
    st[col] = pd.to_numeric(st[col], errors="coerce")
st = st.dropna(subset=["lat", "lng"])
print(f"  좌표 있는 충전소 {len(st):,}곳 (급속 {(st.output>=50).sum():,}곳)")

# 실시간 상태가 실제로 들어오는 주차장만
live = set(q("""SELECT DISTINCT pklt_id FROM parking_realtime_status
                WHERE collected_at >= NOW() - INTERVAL 1 DAY""").pklt_id)
lots["live"] = lots.pklt_id.isin(live)
print(f"  그중 최근 24h 실시간 데이터 있는 곳 {lots.live.sum():,}곳")

L = lots[lots.live].copy()
if len(L):
    la1 = np.radians(st.lat.values.astype(float))[:, None]
    lo1 = np.radians(st.lng.values.astype(float))[:, None]
    la2 = np.radians(L.lat.values.astype(float))[None, :]
    lo2 = np.radians(L.lng.values.astype(float))[None, :]
    d = 2 * 6371000 * np.arcsin(np.sqrt(
        np.sin((la2 - la1) / 2) ** 2 +
        np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    nearest = d.min(axis=1)
    st["nearest_lot_m"] = nearest
    st["nearest_lot"] = L.pklt_id.values[d.argmin(axis=1)]
    print("\n  충전소 -> 가장 가까운 (실시간) 주차장 거리 분포:")
    for r in (50, 100, 200, 300, 500, 1000):
        k = (nearest <= r).sum()
        print(f"    {r:>5}m 이내  {k:>6,}곳  ({k/len(st)*100:5.1f}%)")
    fast = st[st.output >= 50]
    print(f"\n  급속 충전소 {len(fast):,}곳 기준 100m 이내: "
          f"{(fast.nearest_lot_m<=100).sum():,}곳 "
          f"({(fast.nearest_lot_m<=100).mean()*100:.1f}%)")
    st.to_pickle(r"F:\dev\scheduler\data\reports\_parking_match.pkl")
c.close()
