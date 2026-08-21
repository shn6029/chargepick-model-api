"""데이터 품질 검사: 수집 상태 + 좌표가 대구인지."""

from __future__ import annotations
import os
from dotenv import load_dotenv

import pymysql
import pandas as pd

load_dotenv()

# DB 접속정보는 환경변수에서 읽는다(저장소 루트 `.env`).
# 폴백값을 두지 않는다 — 예전에는 접속정보가 여기 박혀 있어서 `.env` 없이 돌리면
# 의도치 않게 운영 DB 에 조용히 붙었다. 지금은 비어 있으면 연결에서 실패한다.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "charset": "utf8mb4",
}

# 대구광역시 대략 경계 (달성군 포함, 여유)
DAEGU_LAT = (35.60, 36.02)
DAEGU_LNG = (128.35, 128.78)


def q(conn, sql, params=None):
    return pd.read_sql(sql, conn, params=params)


def main():
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        print("=" * 64)
        print("1) 테이블 건수 / 기간")
        print("=" * 64)
        for t in [
            "ev_charger_status",
            "ev_charger_info",
            "ev_charger_features",
            "weather",
        ]:
            try:
                n = q(conn, f"SELECT COUNT(*) n FROM {t}").iloc[0, 0]
                print(f"  {t}: {n:,}")
            except Exception as e:
                print(f"  {t}: ERROR {e}")

        st = q(
            conn,
            """
            SELECT COUNT(*) n,
                   MIN(created_at) mn, MAX(created_at) mx,
                   COUNT(DISTINCT created_at) snaps,
                   COUNT(DISTINCT CONCAT(stat_id,'_',chger_id)) chargers
            FROM ev_charger_status
            """,
        )
        print(
            f"  status 기간: {st.iloc[0]['mn']} ~ {st.iloc[0]['mx']} | "
            f"스냅샷 {st.iloc[0]['snaps']} | 관측충전기 {st.iloc[0]['chargers']}"
        )

        print("\n" + "=" * 64)
        print("2) info 좌표/필수값 결측")
        print("=" * 64)
        miss = q(
            conn,
            """
            SELECT
              COUNT(*) total,
              SUM(lat IS NULL OR lng IS NULL) no_coord,
              SUM(stat_nm IS NULL OR stat_nm='') no_name,
              SUM(addr IS NULL OR addr='') no_addr,
              SUM(zcode IS NULL OR zcode='') no_zcode,
              SUM(del_yn='Y') deleted
            FROM ev_charger_info
            """,
        )
        row = miss.iloc[0]
        total = int(row["total"])
        for k in ["no_coord", "no_name", "no_addr", "no_zcode", "deleted"]:
            v = int(row[k])
            print(f"  {k}: {v:,} ({v/total:.1%})")

        print("\n" + "=" * 64)
        print("3) zcode 분포 (27=대구)")
        print("=" * 64)
        z = q(
            conn,
            """
            SELECT COALESCE(zcode,'(null)') zcode, COUNT(*) chargers,
                   COUNT(DISTINCT stat_id) stations
            FROM ev_charger_info
            GROUP BY zcode
            ORDER BY chargers DESC
            """,
        )
        print(z.to_string(index=False))

        print("\n" + "=" * 64)
        print("4) 좌표 범위 / 대구 박스 밖 여부")
        print("=" * 64)
        print(
            f"  대구 판정 박스: lat {DAEGU_LAT[0]}~{DAEGU_LAT[1]}, "
            f"lng {DAEGU_LNG[0]}~{DAEGU_LNG[1]}"
        )
        info = q(
            conn,
            """
            SELECT stat_id, chger_id, stat_nm, addr, lat, lng, zcode
            FROM ev_charger_info
            WHERE lat IS NOT NULL AND lng IS NOT NULL
            """,
        )
        info["lat"] = pd.to_numeric(info["lat"], errors="coerce")
        info["lng"] = pd.to_numeric(info["lng"], errors="coerce")
        info = info.dropna(subset=["lat", "lng"])

        print(
            f"  lat min/max: {info['lat'].min():.5f} ~ {info['lat'].max():.5f}"
        )
        print(
            f"  lng min/max: {info['lng'].min():.5f} ~ {info['lng'].max():.5f}"
        )

        in_box = (
            info["lat"].between(*DAEGU_LAT) & info["lng"].between(*DAEGU_LNG)
        )
        out = info.loc[~in_box].copy()
        print(f"  좌표 있는 충전기: {len(info):,}")
        print(f"  대구 박스 안: {int(in_box.sum()):,} ({in_box.mean():.1%})")
        print(f"  대구 박스 밖: {len(out):,} ({(~in_box).mean():.1%})")

        # 충전소 단위
        stn = (
            info.groupby("stat_id", as_index=False)
            .agg(
                lat=("lat", "first"),
                lng=("lng", "first"),
                stat_nm=("stat_nm", "first"),
                addr=("addr", "first"),
                zcode=("zcode", "first"),
                n=("chger_id", "count"),
            )
        )
        stn_out = stn[
            ~(
                stn["lat"].between(*DAEGU_LAT)
                & stn["lng"].between(*DAEGU_LNG)
            )
        ]
        print(f"  충전소 수: {len(stn):,} | 박스 밖 충전소: {len(stn_out):,}")

        if len(stn_out):
            print("\n  [박스 밖 충전소 샘플 최대 20]")
            sample = stn_out.sort_values("n", ascending=False).head(20)
            print(
                sample[
                    ["stat_id", "stat_nm", "addr", "lat", "lng", "zcode", "n"]
                ].to_string(index=False)
            )

            # 대략 지역 추정
            def region(lat, lng):
                if 37.4 <= lat <= 37.8 and 126.7 <= lng <= 127.2:
                    return "서울권 추정"
                if 35.0 <= lat <= 35.4 and 128.9 <= lng <= 129.3:
                    return "부산권 추정"
                if 35.5 <= lat <= 35.7 and 129.2 <= lng <= 129.5:
                    return "울산권 추정"
                if 35.1 <= lat <= 35.3 and 126.7 <= lng <= 127.0:
                    return "광주권 추정"
                if 36.2 <= lat <= 36.5 and 127.3 <= lng <= 127.5:
                    return "대전권 추정"
                if 35.55 <= lat < 35.70 and 128.35 <= lng <= 128.55:
                    return "대구 달성군(남부) 가능"
                if 36.05 < lat <= 36.40 and 128.4 <= lng <= 128.7:
                    return "경북(구미/의성 방면) 가능"
                if lat < 33.6:
                    return "제주/해외 추정"
                return "기타/확인필요"

            stn_out = stn_out.copy()
            stn_out["region_guess"] = [
                region(a, b) for a, b in zip(stn_out["lat"], stn_out["lng"])
            ]
            print("\n  박스 밖 지역 추정 분포:")
            print(stn_out["region_guess"].value_counts().to_string())

        print("\n" + "=" * 64)
        print("5) 주소는 대구인데 좌표가 박스 밖인 경우")
        print("=" * 64)
        mismatch = stn_out[
            stn_out["addr"].fillna("").str.contains("대구", na=False)
        ]
        print(f"  건수: {len(mismatch)}")
        if len(mismatch):
            print(
                mismatch.head(15)[
                    ["stat_id", "stat_nm", "addr", "lat", "lng"]
                ].to_string(index=False)
            )

        print("\n" + "=" * 64)
        print("6) 좌표는 대구 박스인데 주소에 대구가 없는 경우")
        print("=" * 64)
        stn_in = stn[
            stn["lat"].between(*DAEGU_LAT) & stn["lng"].between(*DAEGU_LNG)
        ]
        addr_odd = stn_in[
            ~stn_in["addr"].fillna("").str.contains("대구", na=False)
        ]
        print(f"  건수: {len(addr_odd)}")
        if len(addr_odd):
            print(
                addr_odd.head(15)[
                    ["stat_id", "stat_nm", "addr", "lat", "lng", "zcode"]
                ].to_string(index=False)
            )

        print("\n" + "=" * 64)
        print("7) status ↔ info 정합성")
        print("=" * 64)
        link = q(
            conn,
            """
            SELECT
              (SELECT COUNT(DISTINCT CONCAT(stat_id,'_',chger_id))
                 FROM ev_charger_status) status_chargers,
              (SELECT COUNT(*) FROM (
                  SELECT DISTINCT s.stat_id, s.chger_id
                  FROM ev_charger_status s
                  LEFT JOIN ev_charger_info i
                    ON s.stat_id=i.stat_id AND s.chger_id=i.chger_id
                  WHERE i.stat_id IS NULL
               ) t) status_without_info
            """,
        )
        print(link.to_string(index=False))

        print("\n" + "=" * 64)
        print("8) status 상태코드 / NULL 비율")
        print("=" * 64)
        sc = q(
            conn,
            """
            SELECT stat, COUNT(*) cnt
            FROM ev_charger_status
            GROUP BY stat
            ORDER BY stat
            """,
        )
        print(sc.to_string(index=False))
        nulls = q(
            conn,
            """
            SELECT
              SUM(stat IS NULL) null_stat,
              SUM(stat_upd_dt IS NULL) null_upd,
              COUNT(*) total
            FROM ev_charger_status
            """,
        )
        print(nulls.to_string(index=False))

        print("\n" + "=" * 64)
        print("요약")
        print("=" * 64)
        out_pct = (~in_box).mean() * 100
        if out_pct < 1:
            verdict = "좌표 품질 양호 (대구 밖 거의 없음)"
        elif out_pct < 5:
            verdict = "대체로 양호, 소량 이상좌표 확인 권장"
        else:
            verdict = "대구 밖 좌표 비중 있음 - 필터/정제 필요"
        print(f"  판정: {verdict}")
        print(f"  대구 밖 충전기 비율: {out_pct:.2f}%")
        print(f"  zcode=27 비중은 위 표 참고")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
