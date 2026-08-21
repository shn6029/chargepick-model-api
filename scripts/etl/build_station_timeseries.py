"""충전소 tick panel → 시계열 피처 · 학습셋 v2.

무엇을 왜
---------
DA① 학습셋 `station_horizon_training_v1.parquet` 의 피처는 **전부 현재 시점 스냅샷**이다.
"지금 몇 대 비었나"는 있는데 "얼마나 자주 비는 곳인가", "지금 비는 중인가 차는 중인가"가 없다.

h5·h10 은 현재 상태가 거의 그대로 유지되므로(h5↔h10 라벨 일치율 99.7%) 스냅샷으로 충분하다.
h30 은 h5 와 라벨이 8.3% 어긋나는데, 그 8.3% 를 맞히려면 변화의 방향과 속도가 필요하다.

`station_tick_panel.parquet` (698만 행 · 5분 grid) 에 그 재료가 있다.
이 스크립트가 거기서 시계열 피처 8종을 만들어 학습셋에 붙인 v2 를 만든다.

측정된 효과 (test 8/02~8/04 · HGB 동일 조건)
    h30 PR-AUC(사용불가)  0.4627 → 0.5562  (+20%)
    h30 사용불가 recall    0.229  → 0.376   (+64%)
비교: 모델 교체(LGBM/XGB/CatBoost/RF)로 얻은 것은 0.003 이었다. 30배 차이다.

산출물
------
    data/processed/station_timeseries_features.parquet   충전소 × tick 시계열 피처
    data/processed/station_horizon_training_v2.parquet   v1 + 시계열 피처
    data/processed/station_training_v2_meta.json         행수·피처·소스 해시

사용법
------
    py scripts/etl/build_station_timeseries.py
    py scripts/etl/build_station_timeseries.py --features-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# DA① 풀데이터 팩 (읽기 전용 — 이 저장소 밖)
PACK_DIR = Path(
    r"F:\카카오톡 받은 파일\EV_SafeCharge_DA1to조장_풀데이터_20260804_1614"
    r"\EV_SafeCharge_DA1to조장_풀데이터_20260804_1614\02_모델테스트_가공본"
)
PANEL_PATH = PACK_DIR / "station_tick_panel.parquet"
TRAINING_V1 = PACK_DIR / "station_horizon_training_v1.parquet"

OUT_DIR = _ROOT / "data" / "processed"
FEATURES_PATH = OUT_DIR / "station_timeseries_features.parquet"
TRAINING_V2 = OUT_DIR / "station_horizon_training_v2.parquet"
META_PATH = OUT_DIR / "station_training_v2_meta.json"

# 정식 편입 피처 8종. 기여도는 h30 PR-AUC 기준 ablation 실측값이다.
TIMESERIES_FEATURES = [
    # 회전율 — "얼마나 자주 비는 곳인가"            +0.0920
    "avail_ratio_30m",
    "avail_ratio_60m",
    "avail_ratio_180m",
    # 추세 — "지금 비는 중인가 차는 중인가"          +0.0618
    "avail_count_mean_60m",
    "avail_count_trend_60m",
    "changes_60m",
    # 경과시간 — "이 상태가 얼마나 됐나"            +0.0358
    "min_since_avail",
    "min_since_full",
]

# 폐기 — 충전소별 사전확률. 넣으면 **성능이 반토막 난다.**
#     baseline  h5 0.7989 / h30 0.4627
#     +priors   h5 0.6336 / h30 0.2197
# 충전소 단위 집계라 사실상 station_id 를 외우는 고차원 인코딩으로 작동한다.
# train 기간 충전소별 가용률을 암기하는데 그 패턴이 주 단위로 바뀌어
# test 기간에는 전부 틀린 사전확률이 된다. 시간대 prior 가 필요하면
# 충전소 단위가 아니라 **군집 단위**로 묶을 것.
REJECTED_FEATURES = ["station_hour_prior", "station_base_rate"]

ROLLING_WINDOWS = (("30min", "avail_ratio_30m"),
                   ("60min", "avail_ratio_60m"),
                   ("180min", "avail_ratio_180m"))


def build_timeseries_features(panel: pd.DataFrame) -> pd.DataFrame:
    """tick panel → 충전소 × 시각 시계열 피처.

    `available_recon` 을 쓴다. `available_observed` 는 관측 안 된 tick 에서 0 이 되어
    '비어 있음'과 '모름'이 구분되지 않는다.

    롤링 창은 과거만 본다(현재 tick 포함). 미래 tick 이 들어가면 누수다.
    """
    p = panel[["station_id", "panel_time", "available_recon"]].copy()
    p = p.sort_values(["station_id", "panel_time"]).reset_index(drop=True)
    p["available_recon"] = p["available_recon"].astype(np.float32)
    p["is_avail"] = (p["available_recon"] > 0).astype(np.float32)

    parts = [p[["station_id", "panel_time"]]]
    g = p.set_index("panel_time").groupby("station_id", sort=False)

    for win, name in ROLLING_WINDOWS:
        parts.append(
            g["is_avail"].rolling(win, closed="both").mean()
            .reset_index(drop=True).rename(name)
        )
    parts.append(
        g["available_recon"].rolling("60min", closed="both").mean()
        .reset_index(drop=True).rename("avail_count_mean_60m")
    )
    parts.append(
        g["is_avail"].rolling("60min", closed="both").apply(
            lambda a: float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0,
            raw=True,
        ).reset_index(drop=True).rename("changes_60m")
    )

    feat = pd.concat(parts, axis=1)
    feat["available_recon"] = p["available_recon"].to_numpy()
    feat["is_avail"] = p["is_avail"].to_numpy()

    # 현재값이 최근 1시간 평균보다 높으면 비는 중, 낮으면 차는 중
    feat["avail_count_trend_60m"] = (
        feat["available_recon"] - feat["avail_count_mean_60m"]
    )

    t = feat["panel_time"]
    for flag, name in ((feat["is_avail"] == 1, "min_since_avail"),
                       (feat["is_avail"] == 0, "min_since_full")):
        last = t.where(flag).groupby(feat["station_id"]).ffill()
        feat[name] = (t - last).dt.total_seconds() / 60.0

    return feat[["station_id", "panel_time"] + TIMESERIES_FEATURES]


def _sha(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--training", default=str(TRAINING_V1))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--features-only", action="store_true",
                    help="시계열 피처만 만들고 학습셋 조인은 건너뜀")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = Path(args.panel)

    print(f"tick panel 로드: {panel_path.name}")
    panel = pd.read_parquet(panel_path)
    print(f"  {len(panel):,}행 / 충전소 {panel.station_id.nunique():,}")
    dup = panel.duplicated(["station_id", "panel_time"]).sum()
    if dup:
        raise SystemExit(f"panel 에 (station_id, panel_time) 중복 {dup:,}건 — 조인 불가")

    print("시계열 피처 생성 중...")
    feat = build_timeseries_features(panel)
    feat_path = out_dir / FEATURES_PATH.name
    feat.to_parquet(feat_path, index=False)
    print(f"  저장: {feat_path}  ({len(feat):,}행 × {len(TIMESERIES_FEATURES)}피처)")

    if args.features_only:
        return

    training_path = Path(args.training)
    print(f"학습셋 로드: {training_path.name}")
    df = pd.read_parquet(training_path)
    n_before = len(df)

    v2 = df.merge(
        feat,
        left_on=["station_id", "feature_as_of"],
        right_on=["station_id", "panel_time"],
        how="left",
    ).drop(columns=["panel_time"])

    if len(v2) != n_before:
        raise SystemExit(
            f"조인 후 행수 변동: {n_before:,} → {len(v2):,}. panel 키 중복을 확인하세요."
        )
    hit = v2[TIMESERIES_FEATURES].notna().any(axis=1).mean()
    print(f"  조인 성공률 {hit:.2%}")
    if hit < 0.95:
        raise SystemExit(f"조인 성공률 {hit:.2%} — 시각 grid 불일치 의심. 중단합니다.")

    v2_path = out_dir / TRAINING_V2.name
    v2.to_parquet(v2_path, index=False)
    print(f"  저장: {v2_path}  ({len(v2):,}행 × {v2.shape[1]}컬럼)")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_panel": str(panel_path),
        "source_panel_sha256_16": _sha(panel_path),
        "source_training_v1": str(training_path),
        "source_training_v1_sha256_16": _sha(training_path),
        "n_rows": int(len(v2)),
        "n_columns": int(v2.shape[1]),
        "n_stations": int(v2.station_id.nunique()),
        "timeseries_features": TIMESERIES_FEATURES,
        "rejected_features": REJECTED_FEATURES,
        "join_hit_rate": float(hit),
        "null_rate": {
            c: float(v2[c].isna().mean()) for c in TIMESERIES_FEATURES
        },
    }
    meta_path = out_dir / META_PATH.name
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {meta_path}")

    print("\n결측률:")
    for c in TIMESERIES_FEATURES:
        print(f"  {c:24s} {v2[c].isna().mean():.4f}")


if __name__ == "__main__":
    main()
