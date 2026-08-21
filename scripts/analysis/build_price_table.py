"""
전기차 충전 요금 Excel → DB 테이블 적재

1) ev_operator_fare  : 사업자×용량구분 요금 (원본 정규화 + bnm 매칭)
2) ev_charger_fare   : 충전기별 적용 요금 (bnm + output(kW) → 용량구분)

사용:
  py build_price_table.py
  py build_price_table.py --xls "경로\\전기차 충전 요금.xls"
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import pymysql

from run_info import DB_CONFIG

DEFAULT_XLS = Path(r"f:\Users\상현\Downloads") / "전기차 충전 요금(2026-07-24).xls"

CIRCLED_SUFFIX = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]+$")
NON_DIGIT_START = re.compile(r"^[0-9]")

CREATE_OPERATOR_FARE_SQL = """
CREATE TABLE IF NOT EXISTS ev_operator_fare (
    id                BIGINT       NOT NULL AUTO_INCREMENT,
    operator_name     VARCHAR(100) NOT NULL COMMENT '요금표 사업자명',
    bnm               VARCHAR(50)  NULL COMMENT '매칭된 DB 기관명(bnm)',
    busi_id           VARCHAR(10)  NULL COMMENT '대표 busi_id(동일 bnm 중 1개)',
    power_class_raw   VARCHAR(50)  NOT NULL COMMENT '요금표 용량구분 원문',
    power_class       VARCHAR(30)  NOT NULL COMMENT '정규화 용량구분',
    fare_type         VARCHAR(50)  NULL COMMENT '요금유형',
    member_price      DECIMAL(10,2) NULL COMMENT '회원 단가(원/kWh)',
    nonmember_price   DECIMAL(10,2) NULL COMMENT '비회원 단가(원/kWh)',
    source_updated_at VARCHAR(50)  NULL COMMENT '요금표 갱신일 원문',
    matched           TINYINT      NOT NULL DEFAULT 0 COMMENT 'bnm 매칭 여부',
    loaded_at         DATETIME     NOT NULL COMMENT '적재시각',
    PRIMARY KEY (id),
    UNIQUE KEY uk_operator_raw (operator_name, power_class_raw),
    KEY idx_bnm_class (bnm, power_class),
    KEY idx_matched (matched)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='사업자×용량구분 충전 요금(외부 요금표)'
"""

CREATE_CHARGER_FARE_SQL = """
CREATE TABLE IF NOT EXISTS ev_charger_fare (
    stat_id           VARCHAR(20)  NOT NULL COMMENT '충전소ID',
    chger_id          VARCHAR(10)  NOT NULL COMMENT '충전기ID',
    busi_id           VARCHAR(10)  NULL,
    bnm               VARCHAR(50)  NULL,
    busi_nm           VARCHAR(100) NULL,
    output_kw         DECIMAL(10,2) NULL COMMENT '충전용량(kW)',
    power_class       VARCHAR(30)  NULL COMMENT '적용 용량구분',
    operator_name     VARCHAR(100) NULL COMMENT '매칭된 요금표 사업자명',
    member_price      DECIMAL(10,2) NULL COMMENT '회원 단가(원/kWh)',
    nonmember_price   DECIMAL(10,2) NULL COMMENT '비회원 단가(원/kWh)',
    matched           TINYINT      NOT NULL DEFAULT 0 COMMENT '요금 매칭 여부',
    loaded_at         DATETIME     NOT NULL COMMENT '적재시각',
    PRIMARY KEY (stat_id, chger_id),
    KEY idx_bnm (bnm),
    KEY idx_power_class (power_class),
    KEY idx_matched (matched)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='충전기별 적용 충전 요금(bnm+output)'
"""

INSERT_OPERATOR_SQL = """
INSERT INTO ev_operator_fare (
    operator_name, bnm, busi_id, power_class_raw, power_class,
    fare_type, member_price, nonmember_price, source_updated_at,
    matched, loaded_at
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s
)
"""

INSERT_CHARGER_SQL = """
INSERT INTO ev_charger_fare (
    stat_id, chger_id, busi_id, bnm, busi_nm,
    output_kw, power_class, operator_name,
    member_price, nonmember_price, matched, loaded_at
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s
)
"""


def norm_name(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().lower()
    for ch in [" ", "\u00a0", "(", ")", "-", "_", ".", "·", "㈜", "(주)", "주식회사"]:
        t = t.replace(ch, "")
    return t


def normalize_power_class(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "기타"
    s = str(raw).strip()
    s = CIRCLED_SUFFIX.sub("", s).strip()
    low = s.lower().replace(" ", "")
    if "완속" in s:
        return "완속"
    if "100kw미만" in low or "100㎾미만" in s.replace(" ", ""):
        return "급속(100kW미만)"
    if "100kw이상" in low or "100㎾이상" in s.replace(" ", ""):
        return "급속(100kW이상)"
    if "초급속" in s or "초고속" in s:
        return "급속(100kW이상)"
    if "중속" in s:
        return "중속"
    if "급속" in s:
        return "급속(100kW이상)"
    return s or "기타"


def power_class_from_kw(kw: float | None) -> str | None:
    if kw is None or pd.isna(kw):
        return None
    if kw < 50:
        return "완속"
    if kw < 100:
        return "급속(100kW미만)"
    return "급속(100kW이상)"


def parse_price(v: object) -> Decimal | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace(",", "")
    if not s or s in {"-", "–", "—", "nan", "None"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def parse_kw(v: object) -> Decimal | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or not NON_DIGIT_START.match(s):
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def load_price_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, engine="xlrd", header=None)
    df = raw.iloc[3:].copy()
    df.columns = [
        "seq",
        "operator_name",
        "fare_type",
        "power_class_raw",
        "member_price",
        "member_note",
        "nonmember_price",
        "source_updated_at",
    ]
    df = df.dropna(subset=["operator_name"])
    df["operator_name"] = df["operator_name"].astype(str).str.strip()
    df = df[df["operator_name"].ne("") & df["operator_name"].ne("사업자명")]
    df["power_class_raw"] = df["power_class_raw"].astype(str).str.strip()
    df["power_class"] = df["power_class_raw"].map(normalize_power_class)
    df["member_price"] = df["member_price"].map(parse_price)
    df["nonmember_price"] = df["nonmember_price"].map(parse_price)
    df["fare_type"] = df["fare_type"].map(
        lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
    )
    df["source_updated_at"] = df["source_updated_at"].map(
        lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
    )
    return df


def load_db_busi(conn) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT busi_id, bnm, COUNT(*) AS chargers
        FROM ev_charger_info
        WHERE bnm IS NOT NULL AND bnm <> ''
        GROUP BY busi_id, bnm
        """,
        conn,
    )


def load_db_chargers(conn) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT stat_id, chger_id, busi_id, bnm, busi_nm, output
        FROM ev_charger_info
        """,
        conn,
    )


def build_bnm_lookup(busi: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """norm(bnm) -> (bnm, busi_id) 충전기 수 많은 busi_id 우선."""
    rows = busi.sort_values("chargers", ascending=False)
    out: dict[str, tuple[str, str]] = {}
    for _, r in rows.iterrows():
        key = norm_name(r["bnm"])
        if key and key not in out:
            out[key] = (str(r["bnm"]), str(r["busi_id"]) if r["busi_id"] is not None else None)
    return out


def pick_operator_rows(price: pd.DataFrame) -> pd.DataFrame:
    """
    동일 사업자·정규화 용량구분에 여러 원문이 있으면
    원문에 원문자 없는 행을 우선, 그다음 회원가 있는 행.
    """
    df = price.copy()
    df["_plain"] = ~df["power_class_raw"].astype(str).str.contains(r"[①②③④⑤⑥⑦⑧⑨⑩]")
    df["_has_member"] = df["member_price"].notna()
    df = df.sort_values(
        ["operator_name", "power_class", "_plain", "_has_member"],
        ascending=[True, True, False, False],
    )
    return df.drop_duplicates(subset=["operator_name", "power_class_raw"], keep="first")


def build_fare_lookup(operator_rows: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """
    (norm(operator_name), power_class) -> fare dict
    정규화 클래스당 1행(원문 plain 우선).
    """
    df = operator_rows.copy()
    df["_plain"] = ~df["power_class_raw"].astype(str).str.contains(r"[①②③④⑤⑥⑦⑧⑨⑩]")
    df["_has_member"] = df["member_price"].notna()
    df = df.sort_values(
        ["operator_name", "power_class", "_plain", "_has_member"],
        ascending=[True, True, False, False],
    )
    df = df.drop_duplicates(subset=["operator_name", "power_class"], keep="first")

    lookup: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        key = (norm_name(r["operator_name"]), r["power_class"])
        lookup[key] = {
            "operator_name": r["operator_name"],
            "member_price": r["member_price"],
            "nonmember_price": r["nonmember_price"],
        }
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="충전 요금 테이블 생성/적재")
    parser.add_argument("--xls", type=Path, default=DEFAULT_XLS, help="요금표 xls 경로")
    args = parser.parse_args()

    if not args.xls.exists():
        raise SystemExit(f"요금표 파일 없음: {args.xls}")

    loaded_at = datetime.now().replace(microsecond=0)
    price = load_price_excel(args.xls)
    operator_rows = pick_operator_rows(price)

    conn = pymysql.connect(**DB_CONFIG, connect_timeout=30)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_OPERATOR_FARE_SQL)
            cur.execute(CREATE_CHARGER_FARE_SQL)
            cur.execute("TRUNCATE TABLE ev_operator_fare")
            cur.execute("TRUNCATE TABLE ev_charger_fare")
        conn.commit()

        busi = load_db_busi(conn)
        bnm_lookup = build_bnm_lookup(busi)
        # also index by exact displayed bnm for charger join via matched operator
        fare_by_op_class = build_fare_lookup(operator_rows)

        # operator fare rows
        op_values = []
        matched_ops = set()
        for _, r in operator_rows.iterrows():
            key = norm_name(r["operator_name"])
            hit = bnm_lookup.get(key)
            bnm = hit[0] if hit else None
            busi_id = hit[1] if hit else None
            matched = 1 if hit else 0
            if matched:
                matched_ops.add(r["operator_name"])
            op_values.append(
                (
                    r["operator_name"],
                    bnm,
                    busi_id,
                    r["power_class_raw"],
                    r["power_class"],
                    r["fare_type"],
                    r["member_price"],
                    r["nonmember_price"],
                    r["source_updated_at"],
                    matched,
                    loaded_at,
                )
            )

        # charger fare: match via norm(bnm) == norm(operator)
        chargers = load_db_chargers(conn)
        ch_values = []
        for _, r in chargers.iterrows():
            bnm = r["bnm"]
            kw = parse_kw(r["output"])
            pclass = power_class_from_kw(float(kw) if kw is not None else None)
            op_norm = norm_name(bnm)
            fare = fare_by_op_class.get((op_norm, pclass)) if pclass else None
            matched = 1 if fare else 0
            ch_values.append(
                (
                    r["stat_id"],
                    r["chger_id"],
                    r["busi_id"],
                    bnm,
                    r["busi_nm"],
                    kw,
                    pclass,
                    fare["operator_name"] if fare else None,
                    fare["member_price"] if fare else None,
                    fare["nonmember_price"] if fare else None,
                    matched,
                    loaded_at,
                )
            )

        with conn.cursor() as cur:
            cur.executemany(INSERT_OPERATOR_SQL, op_values)
            cur.executemany(INSERT_CHARGER_SQL, ch_values)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ev_operator_fare")
            n_op = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ev_operator_fare WHERE matched=1")
            n_op_m = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ev_charger_fare")
            n_ch = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ev_charger_fare WHERE matched=1")
            n_ch_m = cur.fetchone()[0]
            cur.execute(
                """
                SELECT power_class, COUNT(*) c, SUM(matched) m
                FROM ev_charger_fare
                GROUP BY power_class
                ORDER BY c DESC
                """
            )
            by_class = cur.fetchall()
            cur.execute(
                """
                SELECT bnm, member_price, nonmember_price, power_class, COUNT(*) c
                FROM ev_charger_fare
                WHERE matched=1
                GROUP BY bnm, member_price, nonmember_price, power_class
                ORDER BY c DESC
                LIMIT 12
                """
            )
            samples = cur.fetchall()
    finally:
        conn.close()

    print(f"xls: {args.xls}")
    print(f"loaded_at: {loaded_at}")
    print(f"ev_operator_fare: {n_op:,} rows (matched bnm={n_op_m:,})")
    print(
        f"ev_charger_fare: {n_ch:,} rows "
        f"(matched price={n_ch_m:,} / {n_ch:,} = {100*n_ch_m/max(n_ch,1):.1f}%)"
    )
    print("by power_class (total / matched):")
    for pc, c, m in by_class:
        print(f"  {pc}: {c:,} / matched {int(m or 0):,}")
    print("top matched samples:")
    for bnm, mem, non, pc, c in samples:
        print(f"  {bnm}|{pc}|member={mem}|nonmember={non}|chargers={c}")


if __name__ == "__main__":
    main()
