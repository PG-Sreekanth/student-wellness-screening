"""
evaluation/metrics.py - Compare multiple models at a given threshold.
"""
from typing import Dict

import numpy as np
import pandas as pd

from config.settings import REPORTS_DIR
from utils.metrics import compute_all_metrics, metrics_to_row
from utils.logger import get_logger

logger = get_logger(__name__)


def compare_models(
    model_results: Dict[str, np.ndarray],
    y_val: np.ndarray,
    threshold: float = 0.5,
    save: bool = True,
) -> pd.DataFrame:
    """Score all models at the given threshold and print a comparison table."""
    rows = []
    for name, y_proba in model_results.items():
        metrics = compute_all_metrics(y_val, y_proba, threshold=threshold)
        rows.append(metrics_to_row(name, metrics))

    df = pd.DataFrame(rows).sort_values("f2", ascending=False).reset_index(drop=True)

    display_cols = ["model", "recall", "precision", "f1", "f2", "roc_auc", "pr_auc", "brier"]
    logger.info(f"\n{'='*80}\nModel Comparison @ threshold={threshold}\n{'='*80}")
    logger.info(f"\n{df[display_cols].to_string(index=False)}")

    if save:
        path = REPORTS_DIR / "model_comparison.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved to {path}")

    return df
