# -*- coding: utf-8 -*-
"""ev_charger_status → 요일×24시 사용가능률 히트맵.

Usage:
  py scripts/analysis/plot_status_heatmap.py
  py scripts/analysis/plot_status_heatmap.py --days 14 --out data/reports/status_heatmap.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_status_hour_dow(days: int = 14) -> pd.DataFrame:
    import pymysql

    load_dotenv(ROOT / ".env")
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        connect_timeout=40,
    )
    try:
        q = f"""
            SELECT DAYOFWEEK(created_at) AS dow,
                   HOUR(created_at) AS hr,
                   COUNT(*) AS n,
                   SUM(CASE WHEN stat = 2 THEN 1 ELSE 0 END) AS n_avail,
                   SUM(CASE WHEN stat IN (3, 4, 5) THEN 1 ELSE 0 END) AS n_busy
            FROM ev_charger_status
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL {int(days)} DAY)
            GROUP BY DAYOFWEEK(created_at), HOUR(created_at)
        """
        return pd.read_sql(q, conn)
    finally:
        conn.close()


def plot_heatmap(df: pd.DataFrame, out: Path, days: int) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from recommend_api.eval_metrics import configure_matplotlib_kr

        configure_matplotlib_kr()
    except Exception:
        plt.rcParams["axes.unicode_minus"] = False

    # MySQL DOW 1=Sun … 7=Sat → Mon-first
    order = [2, 3, 4, 5, 6, 7, 1]
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    mat = np.full((7, 24), np.nan)
    for _, r in df.iterrows():
        if int(r["dow"]) not in order:
            continue
        i = order.index(int(r["dow"]))
        n = float(r["n"])
        mat[i, int(r["hr"])] = float(r["n_avail"]) / n if n else np.nan

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=0.9, origin="upper")
    ax.set_yticks(list(range(7)), labels=labels)
    ax.set_xticks(list(range(24)))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title(f"EV charger availability rate (stat=2) · last {days}d · ev_charger_status")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Availability rate")

    # overall 24h profile
    hour = (
        df.groupby("hr", as_index=False)
        .agg(n=("n", "sum"), n_avail=("n_avail", "sum"), n_busy=("n_busy", "sum"))
        .sort_values("hr")
    )
    hour["avail"] = hour["n_avail"] / hour["n"]
    hour["busy"] = hour["n_busy"] / hour["n"]
    ax2 = axes[1]
    ax2.plot(hour["hr"], hour["avail"], label="avail (stat=2)", color="#2ca02c")
    ax2.plot(hour["hr"], hour["busy"], label="busy (stat=3/4/5)", color="#d62728")
    ax2.set_xlim(0, 23)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Share")
    ax2.set_title("24h profile (all weekdays pooled)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Status DB 24h heatmap")
    p.add_argument("--days", type=int, default=14)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reports" / "status_heatmap.png",
    )
    args = p.parse_args()
    df = load_status_hour_dow(args.days)
    path = plot_heatmap(df, args.out, args.days)
    print(f"[heatmap] rows={len(df)} n={int(df['n'].sum()):,} → {path}")


if __name__ == "__main__":
    main()
