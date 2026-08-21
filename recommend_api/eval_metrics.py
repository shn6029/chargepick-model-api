"""분류 모델 평가 공용 유틸.

기존 analyze_*.py / train_availability.py가 각자 accuracy/f1/roc_auc만
반복 계산하던 것을 대체: confusion matrix, 사용불가 클래스 recall/precision,
balanced accuracy, PR-AUC(사용불가 기준), Brier score, log loss, calibration을
한 곳에서 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) > 1

    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_available": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_both_classes else None,
        # 사용불가(0)를 양성으로 뒤집어 PR-AUC 계산 — 희귀 클래스 탐지 성능 지표
        "pr_auc_unavailable": float(average_precision_score(1 - y_true, 1 - proba))
        if has_both_classes
        else None,
        "brier_score": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "unavailable_precision": float(
            precision_score(y_true, pred, pos_label=0, zero_division=0)
        ),
        "unavailable_recall": float(
            recall_score(y_true, pred, pos_label=0, zero_division=0)
        ),
        "unavailable_f1": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
    }


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """예측확률을 n_bins개 분위 구간으로 나눠 평균예측확률 vs 실제양성비율을 비교."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    df = pd.DataFrame({"y": y_true, "p": proba})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=n_bins)
    table = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), fraction_positive=("y", "mean"))
        .reset_index(drop=True)
    )
    return table


def always_available_baseline(y_true, proba_value: float = 0.999, threshold: float = 0.5) -> dict:
    """'항상 사용 가능'으로만 예측하는 기준모델. proba_value<1로 log_loss 발산을 방지."""
    y_true = np.asarray(y_true)
    proba = np.full(len(y_true), proba_value)
    return evaluate_classifier(y_true, proba, threshold=threshold)


def configure_matplotlib_kr() -> str:
    """Windows/macOS 한글 폰트 우선 적용. 선택된 폰트명 반환(없으면 '')."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = (
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans KR",
        "AppleGothic",
        "Hiragino Sans",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "")
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def plot_calibration_from_bins(
    curves: dict[str, list[dict] | pd.DataFrame],
    out_path,
) -> None:
    """저장된 calibration bin({mean_predicted, fraction_positive})으로 PNG 저장."""
    import matplotlib.pyplot as plt

    configure_matplotlib_kr()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for name, bins in curves.items():
        table = bins if isinstance(bins, pd.DataFrame) else pd.DataFrame(bins)
        ax.plot(
            table["mean_predicted"],
            table["fraction_positive"],
            marker="o",
            label=name,
        )
    ax.set_xlabel("평균 예측 확률")
    ax.set_ylabel("실제 양성(사용가능) 비율")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path, n_bins: int = 10) -> None:
    """curves: {모델명: (y_true, proba)} -> calibration curve PNG 저장."""
    configure_matplotlib_kr()
    plotted = {
        name: calibration_table(y_true, proba, n_bins=n_bins)
        for name, (y_true, proba) in curves.items()
    }
    plot_calibration_from_bins(plotted, out_path)


def split_by_date(meta: pd.DataFrame, date_col: str = "created_at", ratio: float = 0.8):
    """행 단위 80/20 대신 '날짜' 단위로 train/test 날짜 집합을 분리한 boolean mask 반환.

    반환되는 mask는 meta.index에 정렬된 pd.Series이므로 X[mask]/y[mask]로 바로 사용 가능.
    """
    dates = pd.to_datetime(meta[date_col]).dt.date
    unique_dates = np.sort(dates.unique())
    split = max(1, int(len(unique_dates) * ratio))
    train_dates = set(unique_dates[:split])
    is_train = dates.isin(train_dates)
    return is_train, unique_dates[:split], unique_dates[split:]


def iter_expanding_date_folds(
    meta: pd.DataFrame,
    date_col: str = "created_at",
    min_train_days: int = 2,
):
    """Expanding-window 날짜 fold: train D0..Dk-1 → test Dk.

    Yields (is_train: Series[bool], train_dates, test_date).
    """
    dates = pd.to_datetime(meta[date_col]).dt.date
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < min_train_days + 1:
        return
    for i in range(min_train_days, len(unique_dates)):
        train_dates = unique_dates[:i]
        test_date = unique_dates[i]
        is_train = dates.isin(set(train_dates))
        is_test = dates == test_date
        if not is_test.any() or not is_train.any():
            continue
        yield is_train, train_dates, test_date
