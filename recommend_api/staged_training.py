"""저메모리 환경용 **단계 적재** 학습 경로.

왜 필요한가
-----------
`train_and_save` 는 `build_horizon_dataset(df, HORIZONS)` 로 7개 지평을 한 번에
전개한다. 2026-08-10 실측 외삽으로 약 21.5GB 가 필요했고(3지평 x 810k 행에서
4.94GB), 총 RAM 15.6GB 머신에서는 물리적으로 안 들어갔다.

게다가 이 요구량은 시간이 갈수록 는다. `prune_status.py --days 45` 로 보존이
45일인데 그 시점 데이터는 19일치였다. 45일이 차면 base 행이 약 2.4배가 되어
50GB 대가 된다 — "RAM 큰 머신"으로 피할 수 있는 문제가 아니다.

무엇을 보장하는가 — 정본과 **같은 모델**
----------------------------------------
같은 데이터라면 이 경로와 `train_and_save` 는 동일한 모델을 낸다.
등가를 깨뜨리는 요인 네 가지를 모두 맞췄다.

1. **행 순서.** HGB 의 `_BinMapper` 는 구간 경계를 정할 때 20만 행을
   표본추출한다(기본 `subsample=200_000`). 행 순서가 바뀌면 뽑히는 행이 달라져
   경계가 달라지고 **다른 모델이 나온다** — 실측으로 같은 데이터를 순서만 섞어
   적합했을 때 예측이 최대 0.086 벌어졌다. 그래서 지평 블록을 `HORIZONS` 순서로
   이어 붙여 `build_horizon_dataset` 의 출력 순서를 그대로 재현한다.
2. **더미.** `make_xy` 는 `fillna("UNK")` 후 `get_dummies` 한다. UNK 수준을
   빠뜨리면 열이 어긋나므로 동일하게 만들고, 열 순서도 `get_dummies` 의 사전순을
   따른다.
3. **중앙값.** 표본 근사가 아니라 정확값을 쓴다. parquet 이 컬럼 저장이라
   한 열씩 읽으면 열당 수십 MB 로 싸다. 더미는 1의 비율에서 바로 유도한다.
4. **dtype.** 값은 어느 쪽이든 같다. parts 를 내릴 때 이미 float64 컬럼을
   float32 로 낮췄으므로(`write_parts`), memmap 을 float64 로 넓혀도 되살아나는
   정밀도는 없다. 중앙값 대체값과 2^24 를 넘는 정수만 float64 쪽이 더 정확한데,
   전자는 255 분위 구간 경계에 묻히고 후자는 이 피처 집합에 존재하지 않는다.

**memmap dtype 은 float64 가 기본이다 (2026-08-18 변경, float32 -> float64)**
---------------------------------------------------------------------------
float32 는 디스크를 절반으로 줄이지만 **적합 피크는 오히려 늘린다.** sklearn
1.9.0 의 HGB 는 `check_array(dtype=[X_DTYPE])` 이고 `X_DTYPE` 이 `float64` 라,
float32 를 넘기면 RAM 에 float64 로 통째 복사한다. 직접 재본 결과:

    입력 memmap    출력 dtype   memmap 유지   메모리 공유
    float32        float64      X             X  (전량 복사)
    float64        float64      O             O  (제로카피)

float64 memmap 은 `np.asarray` 가 뷰를 돌려주므로 파일 기반으로 남고, 비닝이
디스크에서 페이지 단위로 읽는다. 2026-08-18 재학습(14,498,475 x 105)에서:

    float32 memmap:  업캐스트 복사 12.18GB + train_test_split 복사 10.96GB ~= 23GB
    float64 memmap:  업캐스트 없음        + train_test_split 복사 10.96GB ~= 11GB

실제로 float32 로 돌던 프로세스의 private 가 22.9GB 였다(계산값 23.1GB 와 일치).
그때 시스템 커밋이 40.89GB / 한도 41.85GB 까지 차서 전체 적합 중에 가용 RAM 이
36MB 로 떨어졌다. 디스크는 6.09GB -> 12.18GB 로 늘지만 작업 디렉터리를 여유 있는
드라이브에 두면 그만이다.

**남은 피크는 `train_test_split` 이다.** `early_stopping='auto'` 는 n>10,000 에서
켜지므로(현 모델도 `do_early_stopping_=True`) sklearn 이 비닝 **전에**
`train_test_split(X, ...)` 으로 90% 를 복사한다. 이건 memmap dtype 과 무관하다.
`early_stopping=False` 로 없앨 수 있지만 그러면 학습 데이터가 90% -> 100% 로
바뀌어 **다른 모델**이 된다(현 모델은 `n_iter_=250 == max_iter` 라 조기종료가
실제로 발동한 적은 없다). 정본 등가를 깨는 변경이라 여기서는 손대지 않았다.

한계
----
- 평가는 날짜 홀드아웃까지만 한다. rolling folds / logistic baseline 은
  적합을 여러 번 더 요구해서 이 경로의 목적(메모리)과 어긋난다.
  정본 metrics 의 `date_holdout` 블록과는 같은 눈금이라 비교는 된다.
- 중간 산출로 parts(수십 MB) 와 memmap(행수 x 열수 x itemsize) 을 디스크에 쓴다.
  기본은 임시 디렉터리이고 끝나면 지운다.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import (
    ARTIFACTS_DIR,
    DURATION_CAP_MIN,
    FEATURE_COLS,
    HORIZONS,
    LABEL_MAX_STALENESS_MIN,
    LABEL_METHOD,
    LABELABLE_FUTURE_STATS,
    LONG_STATE_DURATION_MIN,
    MATCH_TOLERANCE_MIN,
    METRICS_PATH,
    MODEL_PATH,
    STALE_UPDATE_AGE_MIN,
)
from .eval_metrics import calibration_table, evaluate_classifier
from .holdout_eval import save_metrics
from .model_store import (
    build_horizon_dataset,
    compute_feature_schema_hash,
    get_git_commit,
    load_joined,
    load_label_series,
)

CAT_SPECS = (("chger_type", "ctype"), ("kind", "kind"), ("busi_id", "busi"))
# 설계행렬 memmap 의 dtype. float64 여야 sklearn 이 제로카피로 읽는다 — 근거는
# 모듈 독스트링 "memmap dtype 은 float64 가 기본이다" 절.
MATRIX_DTYPE = np.float64
# dtype 별 memmap 파일명. 파일명이 같으면 이전 실행이 남긴 다른 dtype 의 memmap 을
# 그대로 덮어써도 크기가 어긋나 조용히 깨진다. dtype 을 이름에 박아 섞이지 않게 한다.
MATRIX_MEMMAP_NAME = {"float32": "X_f32.memmap", "float64": "X_f64.memmap"}
# make_xy 가 meta 로 남기는 것 중 이 경로에서 평가에 쓰는 집합.
#
# `stat_id`/`chger_id` 는 **피처가 아니다**(FEATURE_COLS 에 없다 — 충전기 정체성은
# 2026-08-03 요일 prior·2026-08-12 세션 해저드 계층에서 두 번 기각됐다). 그런데도
# parts 에 담는 이유는 **세그먼트 평가** 때문이다. 2026-08-18 에 "h30 이상 불가
# Recall 0.74~0.84 가 전 충전소에 고른 문제인지, 특정 충전소가 끌어내리는지" 를
# 보려다 parts 에 식별자가 없어 DB 재전개(약 10분)를 다시 해야 했다. 풀링 지표만으로는
# 그 질문에 답할 수 없고, 식별자는 zstd 사전 인코딩이라 용량도 사실상 안 는다.
# make_xy 도 같은 두 컬럼을 meta 로 남긴다 — 그쪽과 맞춘 것이다.
META_COLS = ["y", "created_at", "eta_minutes", "is_fast", "stat_id", "chger_id"] + [
    c for c, _ in CAT_SPECS
]


def _arrow_num(table: pa.Table, name: str) -> np.ndarray:
    """arrow 컬럼 -> **쓰기 가능한** float64 ndarray. bool/int/null 을 한 번에 처리한다.

    to_numpy() 는 zero-copy 가 가능하면 arrow 버퍼를 그대로 노출해 read-only 배열을
    돌려준다. 호출부에서 inf -> nan 치환을 in-place 로 하므로 쓰기 가능해야 한다.
    """
    col = table.column(name)
    try:
        arr = col.cast(pa.float64()).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        arr = np.asarray(col.to_pylist(), dtype=np.float64)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def write_parts(work: Path, max_rows: int | None = None) -> dict[str, Any]:
    """지평을 하나씩 전개해 parquet 으로 내린다. 피크가 전량의 1/len(HORIZONS) 이 된다."""
    work.mkdir(parents=True, exist_ok=True)
    print("JOIN 로드 중...")
    df = load_joined(limit=max_rows)
    print(f"JOIN: {len(df):,}행")
    if df.empty:
        raise RuntimeError("학습 데이터가 없습니다. features/status를 먼저 적재하세요.")
    meta = {
        "data_start": str(pd.to_datetime(df["created_at"]).min()),
        "data_end": str(pd.to_datetime(df["created_at"]).max()),
        "n_base": int(len(df)),
    }
    print("라벨 시계열 로드 중 (delta+snapshot)...")
    label_df = load_label_series()
    print(f"라벨 원천: {len(label_df):,}행")

    keep = list(dict.fromkeys(list(FEATURE_COLS) + META_COLS))
    total = 0
    for h in HORIZONS:
        hz = build_horizon_dataset(df, [h], label_df=label_df)
        if "future_stat" in hz.columns:
            bad = ~hz["future_stat"].isin(LABELABLE_FUTURE_STATS)
            if bad.any():
                raise RuntimeError(
                    f"라벨 필터 실패: future_stat not in "
                    f"{sorted(LABELABLE_FUTURE_STATS)} {int(bad.sum())}건"
                )
        hz = hz[[c for c in keep if c in hz.columns]]
        for c in hz.columns:
            if hz[c].dtype == "float64":
                hz[c] = hz[c].astype("float32")
        hz.to_parquet(work / f"h{h:04d}.parquet", index=False, compression="zstd")
        total += len(hz)
        print(f"  h{h}: {len(hz):,}행")
        del hz
        gc.collect()
    del df, label_df
    gc.collect()
    meta["n_rows"] = total
    (work / "parts_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return meta


def part_files(work: Path) -> list[Path]:
    """HORIZONS 순서 = 파일명 정렬 순서. build_horizon_dataset 의 블록 순서와 같다."""
    files = [work / f"h{h:04d}.parquet" for h in HORIZONS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise RuntimeError(f"지평 parquet 누락: {missing}")
    return files


def dummy_levels(files: list[Path]) -> dict[str, list[str]]:
    """make_xy 와 동일하게 결측을 'UNK' 로 채운 뒤의 수준 목록(사전순)."""
    levels: dict[str, set[str]] = {c: set() for c, _ in CAT_SPECS}
    for f in files:
        t = pq.read_table(f, columns=[c for c, _ in CAT_SPECS])
        for c, _ in CAT_SPECS:
            col = t.column(c)
            vals = {v for v in pc_unique(col) if v is not None}
            if col.null_count:
                vals.add("UNK")
            levels[c] |= vals
        del t
        gc.collect()
    return {c: sorted(v) for c, v in levels.items()}


def pc_unique(col) -> list:
    import pyarrow.compute as pc

    return pc.unique(col).to_pylist()


def feature_columns_of(levels: dict[str, list[str]]) -> list[str]:
    """make_xy 의 concat 순서: FEATURE_COLS + ctype + kind + busi (각각 사전순)."""
    cols = list(FEATURE_COLS)
    for c, prefix in CAT_SPECS:
        cols += [f"{prefix}_{v}" for v in levels[c]]
    return cols


def exact_medians(
    files: list[Path], levels: dict[str, list[str]], n_rows: int
) -> dict[str, float]:
    """X.median(numeric_only=True) 와 같은 값. NaN 은 건너뛴다(pandas 기본).

    수치 열은 열 단위로 모아 nanmedian, 더미는 1의 개수에서 유도한다.
    0/1 배열의 중앙값은 1의 비율이 0.5 보다 크면 1, 작으면 0, 정확히 0.5 면 0.5다.
    """
    med: dict[str, float] = {}
    for c in FEATURE_COLS:
        chunks = []
        for f in files:
            t = pq.read_table(f, columns=[c])
            chunks.append(_arrow_num(t, c))
            del t
        v = np.concatenate(chunks)
        del chunks
        v[~np.isfinite(v)] = np.nan  # make_xy 의 replace([inf,-inf], nan)
        med[c] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
        del v
        gc.collect()

    counts = {c: {lv: 0 for lv in levels[c]} for c, _ in CAT_SPECS}
    for f in files:
        t = pq.read_table(f, columns=[c for c, _ in CAT_SPECS])
        for c, _ in CAT_SPECS:
            vals = np.asarray(t.column(c).to_pylist(), dtype=object)
            vals = np.where(pd.isna(vals), "UNK", vals)
            u, n = np.unique(vals, return_counts=True)
            for lv, cnt in zip(u, n):
                if lv in counts[c]:
                    counts[c][lv] += int(cnt)
        del t
        gc.collect()
    for c, prefix in CAT_SPECS:
        for lv in levels[c]:
            frac = counts[c][lv] / n_rows
            med[f"{prefix}_{lv}"] = 1.0 if frac > 0.5 else (0.5 if frac == 0.5 else 0.0)
    return med


def build_matrix(
    files: list[Path],
    feature_columns: list[str],
    levels: dict[str, list[str]],
    med: dict[str, float],
    n_rows: int,
    mm_path: Path,
    dtype: np.dtype | type = MATRIX_DTYPE,
) -> tuple[np.memmap, np.ndarray, pd.DataFrame]:
    """memmap 을 **정본과 같은 행 순서**로 채우고 중앙값으로 결측을 메운다.

    dtype 기본값이 float64 인 이유는 모듈 독스트링 참고 — float32 는 디스크만
    절반이고 sklearn 이 적합 때 float64 로 전량 복사해 RAM 피크를 두 배로 만든다.
    """
    import pyarrow.compute as pc

    dtype = np.dtype(dtype)
    k = len(feature_columns)
    col_ix = {c: i for i, c in enumerate(feature_columns)}
    X = np.memmap(mm_path, dtype=dtype, mode="w+", shape=(n_rows, k))
    y = np.empty(n_rows, dtype=np.int8)
    created = np.empty(n_rows, dtype="datetime64[ns]")
    eta = np.empty(n_rows, dtype=np.int16)
    is_fast = np.empty(n_rows, dtype=np.int8)

    off = 0
    for f in files:
        t = pq.read_table(f)
        m = t.num_rows
        have = set(t.schema.names)
        X[off : off + m, :] = 0.0
        for c in FEATURE_COLS:
            if c in have:
                v = _arrow_num(t, c)
                v[~np.isfinite(v)] = np.nan
                np.nan_to_num(v, copy=False, nan=med[c])
            else:
                v = np.full(m, med[c], dtype=np.float64)
            X[off : off + m, col_ix[c]] = v.astype(dtype)
        for c, prefix in CAT_SPECS:
            col = t.column(c)
            filled = pc.fill_null(col, "UNK")
            code = pc.index_in(filled, value_set=pa.array(levels[c])).to_numpy(
                zero_copy_only=False
            ).astype("float64")
            ok = ~np.isnan(code)
            if ok.any():
                base = col_ix[f"{prefix}_{levels[c][0]}"]
                X[off + np.flatnonzero(ok), base + code[ok].astype(np.int64)] = 1.0
        y[off : off + m] = _arrow_num(t, "y").astype(np.int8)
        created[off : off + m] = t.column("created_at").to_numpy(
            zero_copy_only=False
        ).astype("datetime64[ns]")
        eta[off : off + m] = _arrow_num(t, "eta_minutes").astype(np.int16)
        is_fast[off : off + m] = np.nan_to_num(
            _arrow_num(t, "is_fast"), nan=0.0
        ).astype(np.int8)
        off += m
        del t
        gc.collect()
    if off != n_rows:
        raise RuntimeError(f"적재 행수 불일치 {off} != {n_rows}")
    X.flush()
    meta = pd.DataFrame({"created_at": created, "eta_minutes": eta, "is_fast": is_fast})
    return X, y, meta


def _date_holdout(X: np.memmap, y: np.ndarray, meta: pd.DataFrame, work: Path) -> dict:
    """날짜 8:2 홀드아웃. 행 순서를 바꾸지 않고 마스크로 부분집합을 만든다."""
    days = np.unique(meta["created_at"].to_numpy().astype("datetime64[D]"))
    if len(days) < 2:
        return {}
    split = max(1, int(len(days) * 0.8))
    cut = days[split].astype("datetime64[ns]")
    tr = np.flatnonzero(meta["created_at"].to_numpy() < cut)
    te = np.flatnonzero(meta["created_at"].to_numpy() >= cut)
    print(f"날짜 홀드아웃: train {split}일 {len(tr):,}행 / test {len(days)-split}일 {len(te):,}행")

    sub = work / "X_train_sub.memmap"
    # X 와 같은 dtype 이어야 한다. 여기서 float32 로 낮추면 sklearn 이 다시
    # float64 로 전량 복사해 이 함수가 피하려던 피크가 그대로 돌아온다.
    Xt = np.memmap(sub, dtype=X.dtype, mode="w+", shape=(len(tr), X.shape[1]))
    for s in range(0, len(tr), 400_000):
        e = min(s + 400_000, len(tr))
        Xt[s:e] = X[tr[s:e]]
    Xt.flush()
    model = _new_hgb()
    model.fit(Xt, y[tr])
    del Xt
    gc.collect()
    try:
        sub.unlink()
    except OSError:
        pass

    proba = model.predict_proba(X[te])[:, 1]
    del model
    gc.collect()
    y_te = pd.Series(y[te])
    out: dict[str, Any] = {
        "train_days": int(split),
        "test_days": int(len(days) - split),
        "train_dates": [str(d) for d in days[:split]],
        "test_dates": [str(d) for d in days[split:]],
        "overall": evaluate_classifier(y_te, proba),
    }
    eta_te = meta["eta_minutes"].to_numpy()[te]
    per = []
    for h in sorted(set(eta_te.tolist())):
        mk = eta_te == h
        mm = evaluate_classifier(pd.Series(y[te][mk]), proba[mk])
        per.append({"eta_minutes": int(h), **{k: mm[k] for k in (
            "n", "roc_auc", "unavailable_recall", "unavailable_precision",
            "brier_score", "pr_auc_unavailable")}})
    out["per_eta"] = per
    fm = meta["is_fast"].to_numpy()[te] == 1
    if fm.sum() > 200:
        out["rapid_only"] = evaluate_classifier(pd.Series(y[te][fm]), proba[fm])
    cal = calibration_table(y_te, proba)
    out["_calibration"] = cal.to_dict("records") if hasattr(cal, "to_dict") else None
    return out


def _new_hgb() -> HistGradientBoostingClassifier:
    """train_and_save 의 최종 적합 파라미터와 동일해야 한다."""
    return HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
    )


def train_and_save_staged(
    max_rows: int | None = None,
    skip_eval: bool = False,
    work_dir: str | os.PathLike | None = None,
    keep_work: bool = False,
    matrix_dtype: np.dtype | type | str = MATRIX_DTYPE,
) -> dict[str, Any]:
    """단계 적재로 학습하고 정본과 같은 형식의 아티팩트를 저장한다."""
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="staged_train_"))
    mdtype = np.dtype(matrix_dtype)
    if mdtype.name not in MATRIX_MEMMAP_NAME:
        raise ValueError(f"지원하지 않는 matrix_dtype: {mdtype.name} (float32/float64)")
    print(f"작업 디렉터리: {work}  (행렬 dtype={mdtype.name})")
    try:
        meta_p = work / "parts_meta.json"
        if meta_p.exists() and all((work / f"h{h:04d}.parquet").exists() for h in HORIZONS):
            print("기존 parts 재사용 (전개 건너뜀)")
            base_meta = json.loads(meta_p.read_text(encoding="utf-8"))
        else:
            for stale in work.glob("h*.parquet"):
                stale.unlink()
            base_meta = write_parts(work, max_rows=max_rows)

        files = part_files(work)
        n_rows = int(base_meta["n_rows"])
        levels = dummy_levels(files)
        feature_columns = feature_columns_of(levels)
        print(f"피처 {len(feature_columns)}개 (더미 {len(feature_columns)-len(FEATURE_COLS)})")
        med = exact_medians(files, levels, n_rows)
        X, y, meta = build_matrix(
            files,
            feature_columns,
            levels,
            med,
            n_rows,
            work / MATRIX_MEMMAP_NAME[mdtype.name],
            dtype=mdtype,
        )
        print(
            f"행렬 {n_rows:,} x {len(feature_columns)} "
            f"({mdtype.name} {n_rows*len(feature_columns)*mdtype.itemsize/1e9:.2f}GB)"
        )

        eval_block = {} if skip_eval else _date_holdout(X, y, meta, work)

        print(f"HistGradientBoosting 최종 학습(전체)... features={len(feature_columns)}")
        model = _new_hgb()
        model.fit(X, y)
        # 정본은 DataFrame 으로 적합해 피처 이름이 붙는다. 서빙의 align_features 가
        # 같은 순서의 DataFrame 을 넘기므로 이름을 맞춰 두어야 경고가 나지 않는다.
        model.feature_names_in_ = np.asarray(feature_columns, dtype=object)
        del X
        gc.collect()

        now = datetime.now(timezone.utc)
        version = now.strftime("%Y%m%dT%H%M%SZ")
        schema_hash = compute_feature_schema_hash(list(FEATURE_COLS), feature_columns)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        cal = (eval_block or {}).pop("_calibration", None)
        metrics = {
            "model_version": version,
            "trained_at": now.isoformat(),
            "git_commit": get_git_commit(),
            "training_data_start": base_meta["data_start"],
            "training_data_end": base_meta["data_end"],
            "training_row_count": int(base_meta["n_base"]),
            "n_horizon_samples": n_rows,
            "n_samples": n_rows,
            "positive_rate": float(y.mean()),
            "n_features": len(feature_columns),
            "feature_count": len(feature_columns),
            "base_feature_cols": list(FEATURE_COLS),
            "feature_schema_hash": schema_hash,
            "horizons": HORIZONS,
            "match_tolerance_min": MATCH_TOLERANCE_MIN,
            "label_method": LABEL_METHOD,
            "label_max_staleness_min": LABEL_MAX_STALENESS_MIN,
            "labelable_future_stats": sorted(LABELABLE_FUTURE_STATS),
            "model_name": "HistGradientBoosting",
            "build_method": "staged_per_horizon",
            "input_dtype": mdtype.name,
            "status_note": (
                "저메모리 단계 적재 경로로 학습. 모델은 train_and_save 와 등가이나 "
                "평가는 date_holdout 까지만 한다(rolling/logistic 생략)."
            ),
        }
        if eval_block:
            metrics["date_holdout"] = eval_block
        if cal:
            metrics["calibration"] = cal
        save_metrics(metrics, METRICS_PATH)
        print(f"지표 저장: {METRICS_PATH}")

        artifact = {
            "model": model,
            "feature_columns": feature_columns,
            "base_feature_cols": list(FEATURE_COLS),
            "feature_schema_hash": schema_hash,
            "feature_count": len(feature_columns),
            "medians": med,
            "trained_at": now.isoformat(),
            "model_version": version,
            "git_commit": get_git_commit(),
            "training_data_start": base_meta["data_start"],
            "training_data_end": base_meta["data_end"],
            "training_row_count": int(base_meta["n_base"]),
            "n_samples": n_rows,
            "positive_rate": float(y.mean()),
            "horizons": HORIZONS,
            "model_name": "HistGradientBoosting",
            "metrics_path": str(METRICS_PATH),
            "thresholds": {"rank_threshold": 0.5, "warn_threshold": 0.5},
            "long_state_duration_min": LONG_STATE_DURATION_MIN,
            "stale_update_age_min": STALE_UPDATE_AGE_MIN,
            "stale_threshold_min": STALE_UPDATE_AGE_MIN,
            "duration_cap_min": DURATION_CAP_MIN,
            "build_method": "staged_per_horizon",
            "input_dtype": mdtype.name,
        }
        joblib.dump(artifact, MODEL_PATH)
        print(f"모델 저장: {MODEL_PATH}  version={version} hash={schema_hash}")
        return {
            "path": str(MODEL_PATH),
            "metrics_path": str(METRICS_PATH),
            "model_version": version,
            "n_samples": n_rows,
            "positive_rate": artifact["positive_rate"],
            "n_features": len(feature_columns),
            "feature_schema_hash": schema_hash,
            "trained_at": artifact["trained_at"],
            "skip_eval": skip_eval,
            "build_method": "staged_per_horizon",
        }
    finally:
        if not keep_work and work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
            print(f"작업 디렉터리 정리: {work}")
