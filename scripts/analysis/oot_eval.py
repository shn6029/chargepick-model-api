"""정본 모델(20260803T131029Z)을 학습 종료 이후 새로 쌓인 데이터에서 평가.

학습 데이터 끝: 2026-08-03 21:10:55 → 그 이후 표본만 out-of-time 테스트로 쓴다.
모델·아티팩트는 건드리지 않는다(읽기 전용). 결과는 JSON 으로만 남긴다.

load_joined 은 전체 기간을 다 끌어오므로, 여기서는 같은 SQL 에 created_at 하한만
붙여 서버 부하를 줄인다(2GB Lightsail — CLAUDE.md §DB 제약).
"""

from __future__ import annotations

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

from recommend_api import model_store as ms
from recommend_api.config import HORIZONS, METRICS_PATH
from recommend_api.eval_metrics import always_available_baseline, evaluate_classifier

CUTOFF = os.environ.get("OOT_CUTOFF", "2026-08-03 21:10:55")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("oot_eval.json")


def load_joined_since(cutoff: str) -> pd.DataFrame:
    conn = ms.get_connection()
    try:
        holiday_sel = ms._holiday_feature_select(conn)
        context_sel, context_join = ms._context_select_and_join(conn)
        source_filter = ""
        if ms.column_exists(conn, "ev_charger_status", "source"):
            source_filter = "AND s.source <> 'snapshot'"
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
            WHERE (i.del_yn IS NULL OR i.del_yn <> 'Y')
              AND s.created_at > '{cutoff}'
              {source_filter}
            ORDER BY s.stat_id, s.chger_id, s.created_at, s.log_id
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return ms.enrich_frame(df)


def per_eta(y, proba, eta, thr=0.5):
    tmp = pd.DataFrame({"y": np.asarray(y), "proba": proba, "eta": np.asarray(eta)})
    rows = []
    for h, g in tmp.groupby("eta"):
        m = evaluate_classifier(g["y"], g["proba"], threshold=thr)
        rows.append(
            {
                "eta_minutes": int(h),
                "n": int(len(g)),
                "positive_rate": float(g["y"].mean()),
                "roc_auc": m.get("roc_auc"),
                "pr_auc_unavailable": m.get("pr_auc_unavailable"),
                "unavailable_recall": m.get("unavailable_recall"),
                "unavailable_precision": m.get("unavailable_precision"),
                "brier_score": m.get("brier_score"),
                "accuracy": m.get("accuracy"),
            }
        )
    return rows


def main() -> None:
    print(f"[1/5] 새 데이터 로드 (created_at > {CUTOFF}) ...")
    df = load_joined_since(CUTOFF)
    print(f"      JOIN: {len(df):,}행  ({df['created_at'].min()} ~ {df['created_at'].max()})")
    if df.empty:
        raise SystemExit("새 표본 없음")

    print("[2/5] 라벨 시계열 로드 (delta+snapshot, 최근 4일) ...")
    label_df = ms.load_label_series(days=4)
    print(f"      라벨 원천: {len(label_df):,}행")

    print("[3/5] horizon 샘플 생성 ...")
    hz = ms.build_horizon_dataset(df, HORIZONS, label_df=label_df)
    print(f"      horizon 샘플: {len(hz):,}")
    X, y, _med, meta = ms.make_xy(hz, fill_median=False)

    print("[4/5] 정본 모델 로드 + 예측 ...")
    art = ms.load_artifact()
    Xa = ms.align_features(X, art["feature_columns"], art["medians"])
    proba = art["model"].predict_proba(Xa)[:, 1]

    print("[5/5] 지표 계산 ...")
    res = {
        "model_version": art.get("model_version"),
        "cutoff": CUTOFF,
        "test_start": str(meta["created_at"].min()),
        "test_end": str(meta["created_at"].max()),
        "n_rows_raw": int(len(df)),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "label_staleness_median_min": hz.attrs.get("label_staleness_median_min"),
        "label_staleness_p90_min": hz.attrs.get("label_staleness_p90_min"),
        "n_dropped_no_match": hz.attrs.get("n_dropped_no_match"),
        "n_excluded_unlabelable_stat": hz.attrs.get("n_excluded_unlabelable_stat"),
    }
    res["overall"] = evaluate_classifier(y, proba, threshold=0.5)
    res["always_available_baseline"] = always_available_baseline(y)
    res["per_eta"] = per_eta(y, proba, meta["eta_minutes"])

    for name, mask in (
        ("rapid_only", meta["is_fast"] == 1),
        ("slow_only", meta["is_fast"] == 0),
        ("servable", meta["is_servable"] == True),  # noqa: E712
    ):
        m = np.asarray(mask)
        if m.sum() == 0:
            continue
        res[name] = {
            "n": int(m.sum()),
            "unavailable_rate": float(1 - y[m].mean()),
            "overall": evaluate_classifier(y[m], proba[m], threshold=0.5),
            "per_eta": per_eta(y[m], proba[m], meta.loc[m, "eta_minutes"]),
        }

    # 날짜별(운영 관점): 새 데이터 하루 단위 성능
    day = meta["created_at"].dt.date.astype(str)
    daily = []
    for d, idx in pd.Series(range(len(y))).groupby(day.values):
        i = idx.values
        daily.append(
            {
                "date": d,
                "n": int(len(i)),
                "positive_rate": float(y.iloc[i].mean()),
                **{
                    k: evaluate_classifier(y.iloc[i], proba[i], threshold=0.5).get(k)
                    for k in ("roc_auc", "pr_auc_unavailable", "unavailable_recall", "brier_score")
                },
            }
        )
    res["daily"] = daily

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {OUT}")
    o = res["overall"]
    print(
        f"OOT n={res['n_samples']:,} pos={res['positive_rate']:.3f} "
        f"AUC={o['roc_auc']:.4f} PR-AUC(불가)={o['pr_auc_unavailable']:.4f} "
        f"불가recall={o['unavailable_recall']:.4f} Brier={o['brier_score']:.4f}"
    )


if __name__ == "__main__":
    main()
