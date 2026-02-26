"""
final_model.py - Final model evaluation and deployment artifact saving.

Handles one-time test-set evaluation, confusion matrix, and serializing
the model + preprocessor + config for use in the Streamlit app.
"""
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config.settings import (
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    PLOT_DPI,
    FIGSIZE_STANDARD,
)
from utils.metrics import compute_all_metrics
from evaluation.threshold import run_threshold_analysis
from evaluation.calibration import plot_calibration_curve
from utils.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels up from ml_pipeline/pipeline/
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def select_final_model(model_results: Dict[str, Dict]):
    """Pick the model with the best F2 score on validation."""
    best_name, best_f2 = None, -1
    for name, info in model_results.items():
        f2 = info["val_metrics"]["f2"]
        logger.info(f"  {name}: F2={f2:.4f}")
        if f2 > best_f2:
            best_f2 = f2
            best_name = name
    logger.info(f"\n>>> SELECTED: {best_name} (F2={best_f2:.4f})")
    return best_name, model_results[best_name]


def evaluate_on_test(
    model,
    preprocessor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
    cat_indices: list = None,
) -> Dict[str, float]:
    """Run final evaluation on the held-out test set.

    This is called exactly once - never used for model selection.
    Also saves a confusion matrix plot.
    """
    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION (one-time)")
    logger.info("=" * 60)

    X_test_t = preprocessor.transform(X_test)

    if cat_indices:
        X_test_t = pd.DataFrame(X_test_t)
        for idx in cat_indices:
            X_test_t.iloc[:, idx] = X_test_t.iloc[:, idx].astype(str)

    y_test_proba = model.predict_proba(X_test_t)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    metrics = compute_all_metrics(y_test, y_test_proba, threshold=threshold, prefix="test_")

    logger.info(f"\nTest results for {model_name} @ threshold={threshold:.3f}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Depressed", "Depressed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = PLOTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(cm_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Calibration plot for test set
    plot_calibration_curve(y_test, y_test_proba, f"{model_name}_test")

    return metrics


def save_deployment_artifacts(
    model,
    preprocessor,
    threshold: float,
    metrics: Dict,
    model_name: str,
    feature_set: str = "A",
    cat_indices: list = None,
) -> None:
    """Save all artifacts needed by the Streamlit app.

    Saves: model .joblib, preprocessor .joblib, threshold config JSON,
    metrics JSON, and a clean model_metrics.csv at the project root.
    """
    logger.info(f"\nSaving deployment artifacts for {model_name}...")

    # Model
    model_path = MODELS_DIR / f"final_model_{model_name.lower()}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"  Model → {model_path}")

    # Preprocessor
    prep_path = MODELS_DIR / f"final_preprocessor_{model_name.lower()}.joblib"
    joblib.dump(preprocessor, prep_path)
    logger.info(f"  Preprocessor → {prep_path}")

    # Threshold + config
    config = {
        "model_name": model_name,
        "feature_set": feature_set,
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "cat_indices": cat_indices,
    }
    config_path = MODELS_DIR / "threshold_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Config → {config_path}")

    # Metrics JSON (for the Streamlit app and reporting)
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    metrics_json_path = REPORTS_DIR / "final_metrics_summary.json"
    with open(metrics_json_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    logger.info(f"  Metrics JSON → {metrics_json_path}")

    # Clean model summary CSV at project root (replaces old model_metrics.csv)
    tp = clean_metrics.get("test_tp", 0)
    tn = clean_metrics.get("test_tn", 0)
    fp = clean_metrics.get("test_fp", 0)
    fn = clean_metrics.get("test_fn", 0)
    total = tp + tn + fp + fn
    accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0

    summary_row = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": round(clean_metrics.get("test_precision", 0), 4),
        "Recall": round(clean_metrics.get("test_recall", 0), 4),
        "F1": round(clean_metrics.get("test_f1", 0), 4),
        "F2": round(clean_metrics.get("test_f2", 0), 4),
        "ROC-AUC": round(clean_metrics.get("test_roc_auc", 0), 4),
        "PR-AUC": round(clean_metrics.get("test_pr_auc", 0), 4),
        "Threshold": round(threshold, 4),
    }
    csv_path = PROJECT_ROOT / "model_metrics.csv"
    pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
    logger.info(f"  Model summary CSV → {csv_path}")

    logger.info("All artifacts saved.")
