"""
evaluation/threshold.py - Threshold sweep and selection strategies.

Sweeps thresholds 0.10-0.90 and finds:
  - F2-max threshold
  - Recall-target threshold (min recall achieved)
  - Capacity-aware threshold (top-K% flagged)
"""
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, recall_score, precision_score

from config.settings import (
    THRESHOLD_MIN,
    THRESHOLD_MAX,
    THRESHOLD_STEP,
    F_BETA,
    MIN_RECALL_TARGET,
    CAPACITY_FLAG_PERCENT,
    PLOTS_DIR,
    REPORTS_DIR,
    FIGSIZE_WIDE,
    PLOT_DPI,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """Compute recall, precision, F2, and flagged% at each threshold step."""
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    records = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        records.append({
            "threshold": round(t, 4),
            "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=F_BETA, pos_label=1, zero_division=0),
            "flagged_pct": y_pred.sum() / len(y_pred),
        })

    return pd.DataFrame(records)


def find_f2_max_threshold(sweep_df: pd.DataFrame) -> float:
    """Return the threshold with the highest F2 score."""
    idx = sweep_df["f2"].idxmax()
    best_t = sweep_df.loc[idx, "threshold"]
    logger.info(f"F2-max threshold: {best_t:.3f} (F2={sweep_df.loc[idx, 'f2']:.4f})")
    return best_t


def find_recall_target_threshold(sweep_df: pd.DataFrame, min_recall: float = None) -> float:
    """Return the highest threshold that still meets the minimum recall target."""
    target = min_recall or MIN_RECALL_TARGET
    candidates = sweep_df[sweep_df["recall"] >= target]
    if candidates.empty:
        logger.warning(f"No threshold hits recall >= {target}. Falling back to F2-max.")
        return find_f2_max_threshold(sweep_df)
    # take the highest threshold (fewest false positives) that meets the recall bar
    best_t = candidates["threshold"].max()
    row = candidates[candidates["threshold"] == best_t].iloc[0]
    logger.info(f"Recall-target threshold: {best_t:.3f} (recall={row['recall']:.4f}, precision={row['precision']:.4f})")
    return best_t


def find_capacity_threshold(y_proba: np.ndarray, capacity_pct: float = None) -> float:
    """Return the threshold that flags approximately the top-K% of students."""
    pct = capacity_pct or CAPACITY_FLAG_PERCENT
    threshold = np.percentile(y_proba, (1 - pct) * 100)
    logger.info(f"Capacity threshold (top {pct*100:.0f}%): {threshold:.4f}")
    return threshold


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    model_name: str = "model",
    f2_threshold: float = None,
    capacity_threshold: float = None,
) -> None:
    """Plot recall/precision/F2 and flagged% vs threshold, marking key thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0]
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2", linewidth=2, linestyle="--")
    if f2_threshold is not None:
        ax.axvline(f2_threshold, color="red", linestyle=":", label=f"F2-max={f2_threshold:.2f}")
    if capacity_threshold is not None:
        ax.axvline(capacity_threshold, color="green", linestyle=":", label=f"Capacity={capacity_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(sweep_df["threshold"], sweep_df["flagged_pct"], color="purple", linewidth=2)
    if f2_threshold is not None:
        ax2.axvline(f2_threshold, color="red", linestyle=":", label=f"F2-max={f2_threshold:.2f}")
    if capacity_threshold is not None:
        ax2.axvline(capacity_threshold, color="green", linestyle=":", label=f"Capacity={capacity_threshold:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Flagged %")
    ax2.set_title(f"{model_name} - Students Flagged")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"threshold_sweep_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Threshold sweep plot saved to {path}")


def run_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """Run the full threshold analysis and return key threshold values."""
    sweep_df = threshold_sweep(y_true, y_proba, model_name)
    f2_t = find_f2_max_threshold(sweep_df)
    recall_t = find_recall_target_threshold(sweep_df)
    capacity_t = find_capacity_threshold(y_proba)

    plot_threshold_sweep(sweep_df, model_name, f2_threshold=f2_t, capacity_threshold=capacity_t)

    sweep_path = REPORTS_DIR / f"threshold_sweep_{model_name.lower().replace(' ', '_')}.csv"
    sweep_df.to_csv(sweep_path, index=False)

    return {
        "f2_max": f2_t,
        "recall_target": recall_t,
        "capacity_aware": capacity_t,
    }
