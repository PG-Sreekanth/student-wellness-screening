"""
evaluation/calibration.py - Calibration curve plotting and model calibration.
"""
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from config.settings import PLOTS_DIR, FIGSIZE_STANDARD, PLOT_DPI
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> float:
    """Plot calibration curve and return Brier score."""
    brier = brier_score_loss(y_true, y_proba)

    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_predicted, fraction_pos, "s-", label=f"{model_name} (Brier={brier:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"{model_name} Brier: {brier:.4f}  saved to {path}")
    return brier


def calibrate_model(
    base_model,
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wrap a fitted model with Platt (sigmoid) or Isotonic calibration."""
    logger.info(f"Calibrating with method='{method}', cv={cv}...")
    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    calibrated.fit(X_train_transformed, y_train)
    logger.info("Calibration complete.")
    return calibrated


def compare_calibration(
    y_true: np.ndarray,
    proba_original: np.ndarray,
    proba_calibrated: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Plot original vs calibrated curves side by side."""
    brier_orig = brier_score_loss(y_true, proba_original)
    brier_cal = brier_score_loss(y_true, proba_calibrated)

    frac_orig, mean_orig = calibration_curve(y_true, proba_original, n_bins=n_bins, strategy="uniform")
    frac_cal, mean_cal = calibration_curve(y_true, proba_calibrated, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_orig, frac_orig, "s-", label=f"Original (Brier={brier_orig:.4f})")
    ax.plot(mean_cal, frac_cal, "o-", label=f"Calibrated (Brier={brier_cal:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Comparison - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / f"calibration_comparison_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"{model_name} Brier: {brier_orig:.4f} → {brier_cal:.4f} (Δ={brier_orig - brier_cal:+.4f})")
    return brier_orig, brier_cal
