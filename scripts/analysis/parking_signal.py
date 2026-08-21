# -*- coding: utf-8 -*-
"""관문 3: 주차 점유율이 충전기 가용성에 '시간대 이상의' 정보를 주는가.

주의 — 원시 상관은 거의 확실히 유의하게 나온다. 주차 점유율과 충전기 점유율은
둘 다 하루 리듬을 따르기 때문이다(우리 모델은 이미 hour/minute_slot 을 쓴다).
그래서 반드시 **시간대 평균을 제거한 잔차끼리** 봐야 한다. 그게 곧
"기존 피처에 얹었을 때 남는 증분"이다.
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
pairs = st[(st.nearest_lot_m <= 100)][["stat_id", "nearest_lot", "nearest_lot_m", "output"]]
print(f"100m 이내 매칭 충전소 {len(pairs):,}곳 (급속 {(pairs.output>=50).sum():,})")

ids = "','".join(pairs.stat_id.tolist())
lots = "','".join(pairs.nearest_lot.unique().tolist())

# 충전기 상태 -> LOCF 점유 패널 (30분)
raw = pd.read_sql(f"""
    SELECT stat_id, chger_id, stat, created_at FROM ev_charger_status
    WHERE stat_id IN ('{ids}') AND stat IN (1,2,3,4,5,9)
    ORDER BY stat_id, chger_id, created_at""", c)
raw["created_at"] = pd.to_datetime(raw["created_at"])
raw["cid"] = raw.stat_id + "|" + raw.chger_id
print(f"충전기 관측 {len(raw):,}행 / 충전기 {raw.cid.nunique():,}대")

ev = (raw.set_index("created_at").groupby("cid")["stat"]
        .resample("30min").last().reset_index())
wide = ev.pivot(index="created_at", columns="cid", values="stat")
wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="30min")).ffill()
cid2stat = dict(zip(raw.cid, raw.stat_id))

# 충전소별 가용률 패널
recs = []
for sid, grp in pd.Series(cid2stat).groupby(lambda k: cid2stat[k]):
    cols = [c_ for c_ in grp.index if c_ in wide.columns]
    if not cols:
        continue
    sub = wide[cols]
    live = sub.isin([2, 3, 4, 5]).sum(axis=1)
    avail = sub.eq(2).sum(axis=1)
    d = pd.DataFrame({"ts": sub.index, "stat_id": sid,
                      "ev_avail": (avail / live.replace(0, np.nan)).values})
    recs.append(d)
evp = pd.concat(recs).dropna()

# 주차 점유율 (30분 격자)
pk = pd.read_sql(f"""
    SELECT pklt_id, collected_at, occupancy_rate FROM parking_realtime_status
    WHERE pklt_id IN ('{lots}') AND occupancy_rate IS NOT NULL""", c)
pk["collected_at"] = pd.to_datetime(pk["collected_at"])
pk["occupancy_rate"] = pd.to_numeric(pk["occupancy_rate"], errors="coerce")
pk = (pk.set_index("collected_at").groupby("pklt_id")["occupancy_rate"]
        .resample("30min").mean().reset_index()
        .rename(columns={"collected_at": "ts", "occupancy_rate": "pk_occ"}))

m = evp.merge(pairs[["stat_id", "nearest_lot"]], on="stat_id") \
       .merge(pk, left_on=["nearest_lot", "ts"], right_on=["pklt_id", "ts"]) \
       .dropna(subset=["ev_avail", "pk_occ"])
m["pk_occ"] = m["pk_occ"] / 100.0 if m["pk_occ"].max() > 1.5 else m["pk_occ"]
m["hour"] = m.ts.dt.hour
print(f"\n결합 표본 {len(m):,}행 / 충전소 {m.stat_id.nunique()}곳 / "
      f"기간 {m.ts.min()} ~ {m.ts.max()}")

print("\n" + "=" * 66)
print("[A] 원시 상관 (시간대 효과 포함 — 부풀려진 값)")
r_raw = m.ev_avail.corr(m.pk_occ)
print(f"  Pearson r = {r_raw:+.4f}")

print("\n[B] 시간대 평균 제거 후 잔차 상관 ← 이게 진짜 증분")
m["ev_res"] = m.ev_avail - m.groupby("hour").ev_avail.transform("mean")
m["pk_res"] = m.pk_occ - m.groupby("hour").pk_occ.transform("mean")
r_res = m.ev_res.corr(m.pk_res)
print(f"  Pearson r = {r_res:+.4f}")
print(f"  설명되는 분산 R^2 = {r_res**2*100:.2f}%")

print("\n[C] 충전소별 잔차 상관 분포")
per = m.groupby("stat_id").apply(
    lambda d: d.ev_res.corr(d.pk_res) if len(d) > 50 and d.pk_res.std() > 0 else np.nan
).dropna()
print(f"  대상 {len(per)}곳  중앙값 r={per.median():+.3f}  "
      f"|r|>0.2 인 곳 {(per.abs()>0.2).sum()}곳 ({(per.abs()>0.2).mean()*100:.0f}%)")

print("\n[D] 참고: 기존 피처(avail_ratio_60m)와 비교")
ref = m.groupby("stat_id").ev_avail.shift(2)   # 1시간 전 자기 가용률
ok = ref.notna()
print(f"  1시간 전 자기 가용률과의 상관 r = {m.ev_avail[ok].corr(ref[ok]):+.4f}")
print("  (모델이 이미 쓰는 avail_ratio_60m 계열 신호의 대리 지표)")
c.close()
