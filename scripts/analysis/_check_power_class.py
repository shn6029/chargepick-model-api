# -*- coding: utf-8 -*-
"""요금표 용량구분 vs DB output 매칭 가능 여부."""

from pathlib import Path

import pandas as pd
import pymysql
from run_info import DB_CONFIG

PRICE = Path(r"f:\Users\상현\Downloads") / "전기차 충전 요금(2026-07-24).xls"
OUT = Path(r"f:\dev\scheduler") / "_power_class_report.txt"


def load_price() -> pd.DataFrame:
    raw = pd.read_excel(PRICE, engine="xlrd", header=None)
    df = raw.iloc[3:].copy()
    df.columns = [
        "seq",
        "operator",
        "fare_type",
        "power_class",
        "member",
        "member_note",
        "nonmember",
        "updated",
    ]
    df = df.dropna(subset=["operator"])
    df["operator"] = df["operator"].astype(str).str.strip()
    df = df[df["operator"].ne("") & df["operator"].ne("사업자명")]
    df["power_class"] = df["power_class"].astype(str).str.strip()
    return df


def classify_output(kw: float) -> str:
    """일반적인 국내 요금/속도 구간 가정."""
    if pd.isna(kw):
        return "unknown"
    if kw < 50:
        return "완속(~50kW미만)"
    if kw < 100:
        return "급속(50~100미만)"
    if kw < 200:
        return "초급속(100~200미만)"
    return "초고속(200+)"


def main() -> None:
    price = load_price()
    lines: list[str] = []

    lines.append("=== price power_class ===")
    vc = price["power_class"].value_counts(dropna=False)
    for k, v in vc.items():
        lines.append(f"  {k!r}: {v}")

    lines.append("")
    lines.append(
        f"operators with >=2 power classes: "
        f"{int(price.groupby('operator')['power_class'].nunique().ge(2).sum())}"
    )
    lines.append("")
    lines.append("=== sample (operator, power_class, member, nonmember) ===")
    for _, r in price.head(25).iterrows():
        lines.append(
            f"  {r['operator']}|{r['power_class']}|{r['member']}|{r['nonmember']}"
        )

    conn = pymysql.connect(**DB_CONFIG, connect_timeout=20)
    try:
        info = pd.read_sql(
            """
            SELECT
              CAST(output AS DECIMAL(10,2)) AS kw,
              chger_type,
              COUNT(*) AS c
            FROM ev_charger_info
            WHERE output REGEXP '^[0-9]'
            GROUP BY CAST(output AS DECIMAL(10,2)), chger_type
            ORDER BY c DESC
            """,
            conn,
        )
        null_cnt = pd.read_sql(
            """
            SELECT
              SUM(CASE WHEN output IS NULL OR output='' OR output NOT REGEXP '^[0-9]' THEN 1 ELSE 0 END) bad,
              COUNT(*) total
            FROM ev_charger_info
            """,
            conn,
        )
    finally:
        conn.close()

    info["bucket"] = info["kw"].map(classify_output)
    lines.append("")
    lines.append("=== DB output buckets (assumed mapping) ===")
    buck = info.groupby("bucket")["c"].sum().sort_values(ascending=False)
    for k, v in buck.items():
        lines.append(f"  {k}: {int(v)}")

    lines.append("")
    lines.append(
        f"DB output parseable: {int(null_cnt['total'].iloc[0] - null_cnt['bad'].iloc[0])}"
        f" / {int(null_cnt['total'].iloc[0])} "
        f"(bad={int(null_cnt['bad'].iloc[0])})"
    )

    lines.append("")
    lines.append("=== DB kw distribution (top) ===")
    by_kw = info.groupby("kw")["c"].sum().sort_values(ascending=False).head(20)
    for k, v in by_kw.items():
        lines.append(f"  {k} kW -> {classify_output(float(k))}: {int(v)}")

    # chger_type hint
    lines.append("")
    lines.append("=== chger_type vs bucket ===")
    ct = (
        info.groupby(["chger_type", "bucket"])["c"]
        .sum()
        .reset_index()
        .sort_values("c", ascending=False)
        .head(30)
    )
    for _, r in ct.iterrows():
        lines.append(f"  type={r['chger_type']}|{r['bucket']}: {int(r['c'])}")

    # How many price rows use text labels vs kW ranges?
    lines.append("")
    lines.append("=== power_class patterns ===")
    for pc in sorted(price["power_class"].dropna().unique()):
        n_ops = price.loc[price["power_class"].eq(pc), "operator"].nunique()
        lines.append(f"  {pc!r}: rows={int((price['power_class']==pc).sum())}, ops={n_ops}")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
