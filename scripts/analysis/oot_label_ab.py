"""OOT 성능 하락이 '모델 드리프트'인지 '라벨 원천 변화'인지 가른다.

전량 스냅샷은 2026-08-04 02:30 부터 6시간 주기로 정상화됐다(그 전 08-01~03 은 거의 없음).
스냅샷은 delta 가 놓친 상태 변화를 드러내므로 LOCF 라벨을 더 '어렵게' 만든다.
같은 표본·같은 모델에 라벨 원천만 delta-only / delta+snapshot 으로 바꿔 비교한다.
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

import pandas as pd

from recommend_api import model_store as ms
from recommend_api.config import HORIZONS
from recommend_api.eval_metrics import evaluate_classifier

sys.path.insert(0, str(Path(__file__).parent))
from oot_eval import CUTOFF, load_joined_since, per_eta  # noqa: E402


def label_series(source: str) -> pd.DataFrame:
    conn = ms.get_connection()
    try:
        flt = "AND source <> 'snapshot'" if source == "delta" else ""
        q = f"""
            SELECT stat_id, chger_id, created_at, stat
            FROM ev_charger_status
            WHERE created_at >= NOW() - INTERVAL 4 DAY {flt}
            ORDER BY stat_id, chger_id, created_at
        """
        out = pd.read_sql(q, conn)
    finally:
        conn.close()
    out["created_at"] = pd.to_datetime(out["created_at"])
    return out


def main() -> None:
    df = load_joined_since(CUTOFF)
    print(f"표본 원본: {len(df):,}행")
    art = ms.load_artifact()

    res = {}
    for src in ("all", "delta"):
        lbl = label_series(src)
        hz = ms.build_horizon_dataset(df, HORIZONS, label_df=lbl)
        X, y, _m, meta = ms.make_xy(hz, fill_median=False)
        Xa = ms.align_features(X, art["feature_columns"], art["medians"])
        proba = art["model"].predict_proba(Xa)[:, 1]
        fast = meta["is_fast"] == 1
        res[src] = {
            "n_label_rows": int(len(lbl)),
            "n_samples": int(len(y)),
            "positive_rate": float(y.mean()),
            "overall": evaluate_classifier(y, proba, threshold=0.5),
            "rapid_only": evaluate_classifier(y[fast.values], proba[fast.values], threshold=0.5),
            "per_eta": per_eta(y, proba, meta["eta_minutes"]),
        }
        o = res[src]["overall"]
        print(
            f"[{src:5}] 라벨원천 {len(lbl):,}행 표본 {len(y):,} pos={y.mean():.3f} "
            f"AUC={o['roc_auc']:.4f} PR-AUC={o['pr_auc_unavailable']:.4f} "
            f"불가R={o['unavailable_recall']:.4f} Brier={o['brier_score']:.4f}"
        )

    Path(sys.argv[1] if len(sys.argv) > 1 else "oot_label_ab.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
