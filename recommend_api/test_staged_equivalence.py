"""단계 적재 경로가 정본 make_xy 경로와 **같은 모델**을 내는지 검증한다.

DB 없이 돈다 — 합성 horizon 프레임을 만들어 두 경로에 똑같이 태운다.

무엇을 보나
-----------
1. feature_columns 가 완전히 같은가 (이름·개수·**순서**)
2. medians 가 같은가 (더미 포함 104개 전부)
3. 결측을 메운 뒤 행렬 값이 같은가
4. 같은 하이퍼파라미터로 적합했을 때 **예측이 같은가**

4번이 본질이다. HGB 의 _BinMapper 가 20만 행을 표본추출해 구간 경계를 정하므로
행 순서가 어긋나면 3번까지 통과해도 4번이 깨진다.

사용:
  py -m recommend_api.test_staged_equivalence
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommend_api.config import FEATURE_COLS, HORIZONS  # noqa: E402
from recommend_api.model_store import make_xy  # noqa: E402
from recommend_api.staged_training import (  # noqa: E402
    _new_hgb,
    build_matrix,
    dummy_levels,
    exact_medians,
    feature_columns_of,
    part_files,
)

SEED = 20260810


def synth_horizon(n_per_h: int = 4000) -> dict[int, pd.DataFrame]:
    """지평별 합성 프레임. 결측·null 범주·무한대를 일부러 섞는다."""
    rng = np.random.default_rng(SEED)
    ctypes = ["02", "04", "07", None]
    kinds = ["A0", "B0", "C0", None]
    busis = [f"B{i:02d}" for i in range(12)] + [None]
    out = {}
    for h in HORIZONS:
        n = n_per_h
        d = {}
        # make_xy 는 meta 를 만들 때 is_stale_status 같은 플래그를 int 로 캐스팅하므로
        # 무한대는 연속형 열에만 넣는다(실데이터에서도 플래그에 inf 는 나오지 않는다).
        inf_ok = {"avail_ratio_15m", "avail_ratio_30m", "avail_ratio_60m",
                  "time_since_available", "capped_state_duration"}
        for c in FEATURE_COLS:
            v = rng.normal(size=n)
            v[rng.random(n) < 0.05] = np.nan          # 결측
            if c in inf_ok:
                v[rng.random(n) < 0.01] = np.inf      # 무한대 → make_xy 가 NaN 으로
            d[c] = v
        d["eta_minutes"] = np.full(n, h, dtype=float)
        d["y"] = (rng.random(n) < 0.75).astype(int)
        d["created_at"] = pd.Timestamp("2026-07-22") + pd.to_timedelta(
            rng.integers(0, 19 * 24 * 60, n), unit="m"
        )
        d["is_fast"] = rng.integers(0, 2, n)
        d["chger_type"] = rng.choice(ctypes, n)
        d["kind"] = rng.choice(kinds, n)
        d["busi_id"] = rng.choice(busis, n)
        d["stat_id"] = rng.choice([f"S{i:04d}" for i in range(200)], n)
        d["chger_id"] = rng.choice(["01", "02"], n)
        d["output_kw"] = rng.choice([7.0, 50.0, 100.0], n)
        out[h] = pd.DataFrame(d)
    return out


def main() -> None:
    parts = synth_horizon()
    # 정본 경로: build_horizon_dataset 은 HORIZONS 순서로 블록을 이어 붙인다
    hz = pd.concat([parts[h] for h in HORIZONS], ignore_index=True)
    n_rows = len(hz)
    print(f"합성 horizon {n_rows:,}행 · 지평 {len(HORIZONS)}개")

    # ── A: 정본 ──
    X_a, y_a, med_a, _meta_a = make_xy(hz, fill_median=False)
    med_a_full = X_a.median(numeric_only=True)
    X_a_filled = X_a.fillna(med_a_full)
    cols_a = X_a_filled.columns.tolist()

    # ── B: 단계 적재 ──
    work = Path(tempfile.mkdtemp(prefix="staged_eq_"))
    try:
        for h in HORIZONS:
            parts[h].to_parquet(work / f"h{h:04d}.parquet", index=False)
        files = part_files(work)
        levels = dummy_levels(files)
        cols_b = feature_columns_of(levels)
        med_b = exact_medians(files, levels, n_rows)
        X_b, y_b, _meta_b = build_matrix(
            files, cols_b, levels, med_b, n_rows, work / "X.memmap"
        )

        ok = True

        # 1) 열 이름·순서
        same_cols = cols_a == cols_b
        print(f"\n1) feature_columns 동일(순서 포함): {same_cols}  ({len(cols_a)} vs {len(cols_b)})")
        if not same_cols:
            ok = False
            print("   A에만:", [c for c in cols_a if c not in cols_b][:8])
            print("   B에만:", [c for c in cols_b if c not in cols_a][:8])

        # 2) 중앙값
        if same_cols:
            da = np.array([med_a_full[c] for c in cols_a], dtype=float)
            db = np.array([med_b[c] for c in cols_a], dtype=float)
            dmax = float(np.nanmax(np.abs(da - db)))
            print(f"2) medians 최대 |차이|: {dmax:.3e}")
            if dmax > 1e-9:
                ok = False
                bad = [(c, med_a_full[c], med_b[c]) for c in cols_a
                       if abs(float(med_a_full[c]) - float(med_b[c])) > 1e-9][:5]
                print("   불일치 예:", bad)

            # 3) 행렬 값
            A = X_a_filled[cols_a].to_numpy(dtype=np.float32)
            B = np.asarray(X_b)
            xmax = float(np.nanmax(np.abs(A - B)))
            print(f"3) 행렬 최대 |차이|: {xmax:.3e}  · shape {A.shape} vs {B.shape}")
            if xmax > 1e-6:
                ok = False
                r, c = np.unravel_index(int(np.nanargmax(np.abs(A - B))), A.shape)
                print(f"   최대 지점 row={r} col={cols_a[c]}  A={A[r,c]} B={B[r,c]}")

            # 4) 예측
            assert (np.asarray(y_a) == y_b).all(), "y 불일치"
            ma, mb = _new_hgb(), _new_hgb()
            ma.fit(A, np.asarray(y_a))
            mb.fit(B, y_b)
            pa, pb = ma.predict_proba(A[:2000])[:, 1], mb.predict_proba(B[:2000])[:, 1]
            pmax = float(np.abs(pa - pb).max())
            print(f"4) 예측 최대 |차이|: {pmax:.3e}  · 완전 동일: {np.array_equal(pa, pb)}")
            if pmax > 1e-9:
                ok = False

        # 대조군: 행 순서를 섞으면 4번이 깨져야 한다 (테스트가 살아있다는 증거)
        rng = np.random.default_rng(1)
        perm = rng.permutation(n_rows)
        mc = _new_hgb()
        mc.fit(np.asarray(X_b)[perm], y_b[perm])
        pc_ = mc.predict_proba(np.asarray(X_b)[:2000])[:, 1]
        shuffled_diff = float(np.abs(pc_ - pb).max())
        print(f"\n[대조군] 행 순서를 섞으면 예측 최대 |차이|: {shuffled_diff:.3e}")
        if shuffled_diff < 1e-9:
            ok = False
            print("   ! 순서를 섞어도 같다면 이 테스트는 순서 문제를 못 잡는다")

        print("\n" + ("staged equivalence OK" if ok else "FAILED"))
        if not ok:
            raise SystemExit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
