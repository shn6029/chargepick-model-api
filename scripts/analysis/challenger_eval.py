"""도전자 vs 정본: 08-04~08-05 데이터를 더 넣으면 나아지는가.

정본 20260803T131029Z 는 08-03 21:10 까지로 학습됐다. 같은 하이퍼파라미터로
**08-05 24:00 까지** 학습한 도전자를 만들고, 두 모델을 모두 08-06~08-07 구간
(양쪽 다 처음 보는 데이터)에서 평가한다.

메모리: horizon 표본이 800만 행대라 문자열 컬럼(stat_nm/addr 등)을 미리 버리고
X 는 float32 로 내린다. HGB 는 내부에서 uint8 로 비닝하므로 정밀도 손실이 없다.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\dev\scheduler")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from recommend_api import model_store as ms
from recommend_api.config import HORIZONS
from recommend_api.eval_metrics import evaluate_classifier

SPLIT = "2026-08-06 00:00:00"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("challenger.json")

DROP_COLS = ["stat_nm", "addr", "lat", "lng", "output", "parking_free", "traffic_yn"]


def load_range(start: str | None, end: str | None) -> pd.DataFrame:
    conn = ms.get_connection()
    try:
        holiday_sel = ms._holiday_feature_select(conn)
        context_sel, context_join = ms._context_select_and_join(conn)
        src = ""
        if ms.column_exists(conn, "ev_charger_status", "source"):
            src = "AND s.source <> 'snapshot'"
        rng = ""
        if start:
            rng += f" AND s.created_at >= '{start}'"
        if end:
            rng += f" AND s.created_at < '{end}'"
        query = f"""
            SELECT
                s.log_id, s.stat_id, s.chger_id, s.busi_id, s.stat, s.created_at,
                s.stat_upd_dt,
                f.hour, f.minute_slot, f.day_of_week, f.is_weekend, f.is_holiday,
                {holiday_sel},
                f.current_state_duration, f.changes_30m,
                f.avail_ratio_15m, f.avail_ratio_30m, f.avail_ratio_60m,
                f.time_since_available, f.time_since_charge_started, f.time_since_charge_ended,
                i.chger_type, i.output, i.kind, i.kind_detail,
                i.parking_free, i.limit_yn, i.limit_detail, i.traffic_yn,
                i.use_time,
                i.stat_nm, i.addr, i.lat, i.lng,
                {context_sel}
            FROM ev_charger_status s
            INNER JOIN ev_charger_features f ON s.log_id = f.log_id
            INNER JOIN ev_charger_info i
                ON s.stat_id = i.stat_id AND s.chger_id = i.chger_id
            {context_join}
            WHERE (i.del_yn IS NULL OR i.del_yn <> 'Y') {rng} {src}
            ORDER BY s.stat_id, s.chger_id, s.created_at, s.log_id
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    df = ms.enrich_frame(df)
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def build(df: pd.DataFrame, label_df: pd.DataFrame):
    hz = ms.build_horizon_dataset(df, HORIZONS, label_df=label_df)
    X, y, _med, meta = ms.make_xy(hz, fill_median=False)
    del hz
    gc.collect()
    return X, y, meta


def seg_metrics(y, proba, meta) -> dict:
    out = {"overall": evaluate_classifier(y, proba, threshold=0.5)}
    for name, m in (
        ("rapid_only", (meta["is_fast"] == 1).to_numpy()),
        ("servable", meta["is_servable"].to_numpy().astype(bool)),
    ):
        out[name] = evaluate_classifier(y[m], proba[m], threshold=0.5)
    rows = []
    for h in sorted(meta["eta_minutes"].unique()):
        m = (meta["eta_minutes"] == h).to_numpy()
        fast = m & (meta["is_fast"] == 1).to_numpy()
        rows.append(
            {
                "eta_minutes": int(h),
                "n": int(m.sum()),
                "pr_auc_unavailable": evaluate_classifier(y[m], proba[m])["pr_auc_unavailable"],
                "unavailable_recall": evaluate_classifier(y[m], proba[m])["unavailable_recall"],
                "rapid_pr_auc": evaluate_classifier(y[fast], proba[fast])["pr_auc_unavailable"],
                "rapid_unavailable_recall": evaluate_classifier(y[fast], proba[fast])[
                    "unavailable_recall"
                ],
            }
        )
    out["per_eta"] = rows
    return out


def main() -> None:
    print("라벨 시계열 로드 (delta+snapshot 전체)...", flush=True)
    label_df = ms.load_label_series()
    print(f"  {len(label_df):,}행", flush=True)

    print(f"[TEST] {SPLIT} ~ 현재 로드...", flush=True)
    df_te = load_range(SPLIT, None)
    print(f"  JOIN {len(df_te):,}행", flush=True)
    X_te, y_te, meta_te = build(df_te, label_df)
    del df_te
    gc.collect()
    print(f"  테스트 표본 {len(y_te):,} (pos {y_te.mean():.3f})", flush=True)

    print(f"[TRAIN] ~ {SPLIT} 로드...", flush=True)
    df_tr = load_range(None, SPLIT)
    print(f"  JOIN {len(df_tr):,}행 ({df_tr['created_at'].min()} ~ {df_tr['created_at'].max()})", flush=True)
    X_tr, y_tr, meta_tr = build(df_tr, label_df)
    n_raw_tr = len(df_tr)
    del df_tr, label_df
    gc.collect()
    print(f"  학습 표본 {len(y_tr):,} (pos {y_tr.mean():.3f})", flush=True)

    med = X_tr.median(numeric_only=True)
    cols = X_tr.columns.tolist()
    X_tr = X_tr.fillna(med).astype(np.float32)
    gc.collect()

    print("도전자 학습 (HGB depth8 lr0.08 iter250)...", flush=True)
    challenger = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
    )
    challenger.fit(X_tr, y_tr)
    del X_tr, y_tr
    gc.collect()

    print("평가...", flush=True)
    X_te_ch = ms.align_features(X_te.copy(), cols, med.to_dict()).astype(np.float32)
    p_ch = challenger.predict_proba(X_te_ch)[:, 1]
    del X_te_ch
    gc.collect()

    art = ms.load_artifact()
    X_te_cur = ms.align_features(X_te, art["feature_columns"], art["medians"]).astype(np.float32)
    p_cur = art["model"].predict_proba(X_te_cur)[:, 1]
    del X_te_cur, X_te
    gc.collect()

    res = {
        "test_window": {"start": SPLIT, "n_samples": int(len(y_te)), "positive_rate": float(y_te.mean())},
        "challenger": {
            "train_end": SPLIT,
            "n_rows_raw": int(n_raw_tr),
            "n_features": len(cols),
            **seg_metrics(y_te, p_ch, meta_te),
        },
        "current": {
            "model_version": art.get("model_version"),
            "n_features": len(art["feature_columns"]),
            **seg_metrics(y_te, p_cur, meta_te),
        },
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n저장: {OUT}")
    for k in ("current", "challenger"):
        o = res[k]["overall"]
        r = res[k]["rapid_only"]
        print(
            f"{k:<10} AUC={o['roc_auc']:.4f} PR-AUC={o['pr_auc_unavailable']:.4f} "
            f"불가R={o['unavailable_recall']:.4f} Brier={o['brier_score']:.4f} | "
            f"급속 PR-AUC={r['pr_auc_unavailable']:.4f} 불가R={r['unavailable_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
