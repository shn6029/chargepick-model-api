# -*- coding: utf-8 -*-
"""요금표 사업자명 vs DB busi 매칭 점검."""

from pathlib import Path

import pandas as pd
import pymysql
from run_info import DB_CONFIG

PRICE_XLS = Path(r"f:\Users\상현\Downloads") / "전기차 충전 요금(2026-07-24).xls"
OUT = Path(r"f:\dev\scheduler") / "_busi_match_report.txt"


def load_price_operators() -> pd.DataFrame:
    raw = pd.read_excel(PRICE_XLS, engine="xlrd", header=None)
    # row1=헤더후보, row2=서브헤더, row3부터 데이터 (0-index: 1,2,3...)
    # 미리보기: row1 = 순번,사업자명,요금유형,용량구분,회원가,...
    header_row = 1
    df = raw.iloc[header_row + 2 :].copy()
    df.columns = [
        "seq",
        "operator_name",
        "fare_type",
        "power_class",
        "member_price",
        "member_note",
        "nonmember_price",
        "updated_at",
    ]
    df = df.dropna(subset=["operator_name"])
    df["operator_name"] = df["operator_name"].astype(str).str.strip()
    df = df[df["operator_name"].ne("") & df["operator_name"].ne("사업자명")]
    return df


def load_db_busi() -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG, connect_timeout=20)
    try:
        q = """
            SELECT busi_id, bnm, busi_nm, COUNT(*) AS chargers,
                   COUNT(DISTINCT stat_id) AS stations
            FROM ev_charger_info
            GROUP BY busi_id, bnm, busi_nm
        """
        return pd.read_sql(q, conn)
    finally:
        conn.close()


def norm(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().lower()
    for ch in [" ", "\u00a0", "(", ")", "-", "_", ".", "·"]:
        t = t.replace(ch, "")
    # 흔한 표기 통일
    repl = {
        "한국전기차충전서비스": "한국전기차충전서비스",
        "한국전력": "한국전력",
        "환경부": "기후에너지환경부",
        "기후에너지환경부": "기후에너지환경부",
        "한국환경공단": "한국환경공단",
    }
    return repl.get(t, t)


def main() -> None:
    price = load_price_operators()
    busi = load_db_busi()

    price_ops = sorted(price["operator_name"].dropna().unique())
    lines = []
    lines.append(f"price_operators={len(price_ops)}")
    lines.append(f"db_busi_rows={len(busi)}")
    lines.append(f"db_unique_busi_id={busi['busi_id'].nunique()}")
    lines.append(f"db_unique_bnm={busi['bnm'].nunique()}")
    lines.append(f"db_unique_busi_nm={busi['busi_nm'].nunique()}")

    # exact match on bnm / busi_nm
    bnm_set = {norm(x): x for x in busi["bnm"].dropna().unique()}
    businm_set = {norm(x): x for x in busi["busi_nm"].dropna().unique()}
    busiid_set = {str(x).strip().upper(): x for x in busi["busi_id"].dropna().unique()}

    exact_bnm = []
    exact_businm = []
    exact_id = []
    fuzzy = []
    unmatched = []

    for op in price_ops:
        n = norm(op)
        if n in bnm_set:
            exact_bnm.append((op, bnm_set[n], "bnm"))
        elif n in businm_set:
            exact_businm.append((op, businm_set[n], "busi_nm"))
        elif op.strip().upper() in busiid_set:
            exact_id.append((op, busiid_set[op.strip().upper()], "busi_id"))
        else:
            # contains either way
            hit = None
            for nb, raw in {**bnm_set, **businm_set}.items():
                if not nb or not n:
                    continue
                if n in nb or nb in n:
                    hit = (op, raw, "contains")
                    break
            if hit:
                fuzzy.append(hit)
            else:
                unmatched.append(op)

    matched = len(exact_bnm) + len(exact_businm) + len(exact_id) + len(fuzzy)
    lines.append(f"\nmatched_ops={matched}/{len(price_ops)} ({matched/max(len(price_ops),1):.1%})")
    lines.append(f"  exact_bnm={len(exact_bnm)}")
    lines.append(f"  exact_busi_nm={len(exact_businm)}")
    lines.append(f"  exact_busi_id={len(exact_id)}")
    lines.append(f"  fuzzy_contains={len(fuzzy)}")
    lines.append(f"  unmatched={len(unmatched)}")

    # coverage by chargers: which DB chargers can get a price via matched operator names
    matched_names = set()
    for lst in (exact_bnm, exact_businm, fuzzy):
        for op, raw, how in lst:
            matched_names.add(norm(raw))
    for op, raw, how in exact_id:
        # id match - mark rows with that id
        pass

    busi["bnm_n"] = busi["bnm"].map(norm)
    busi["busi_nm_n"] = busi["busi_nm"].map(norm)
    covered = busi[
        busi["bnm_n"].isin(matched_names) | busi["busi_nm_n"].isin(matched_names)
    ]
    # also id exact
    id_matched = {op.strip().upper() for op, _, _ in exact_id}
    covered_id = busi[busi["busi_id"].astype(str).str.upper().isin(id_matched)]
    covered_all = pd.concat([covered, covered_id]).drop_duplicates(
        subset=["busi_id", "bnm", "busi_nm"]
    )

    lines.append(
        f"\nDB charger coverage by matched operators: "
        f"{int(covered_all['chargers'].sum()):,}/"
        f"{int(busi['chargers'].sum()):,} "
        f"({covered_all['chargers'].sum()/busi['chargers'].sum():.1%})"
    )
    lines.append(
        f"DB station coverage: "
        f"{int(covered_all['stations'].sum()):,}/"
        f"{int(busi['stations'].sum()):,}"
    )

    lines.append("\n=== exact_bnm samples ===")
    for row in exact_bnm[:30]:
        lines.append(f"  {row}")

    lines.append("\n=== exact_busi_nm samples ===")
    for row in exact_businm[:30]:
        lines.append(f"  {row}")

    lines.append("\n=== fuzzy samples ===")
    for row in fuzzy[:40]:
        lines.append(f"  {row}")

    lines.append("\n=== unmatched operators ===")
    for op in unmatched:
        lines.append(f"  {op}")

    # top DB operators not in price list
    price_n = {norm(x) for x in price_ops}
    busi_top = busi.sort_values("chargers", ascending=False).head(40)
    lines.append("\n=== top DB operators (by chargers) and price hit? ===")
    for _, r in busi_top.iterrows():
        hit = norm(r["bnm"]) in price_n or norm(r["busi_nm"]) in price_n
        lines.append(
            f"  {r['busi_id']}|bnm={r['bnm']}|busi_nm={r['busi_nm']}|chargers={r['chargers']}|hit={hit}"
        )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8")[:2500])
    print("\n... full report:", OUT)


if __name__ == "__main__":
    main()
