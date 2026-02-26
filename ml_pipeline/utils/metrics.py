"""
utils/metrics.py - Shared metric helpers used across the pipeline.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from config.settings import F_BETA


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute recall, precision, F1, F2, ROC-AUC, PR-AUC, Brier, and CM counts."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        f"{prefix}threshold": threshold,
        f"{prefix}recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f"{prefix}precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        f"{prefix}f2": fbeta_score(y_true, y_pred, beta=F_BETA, pos_label=1, zero_division=0),
        f"{prefix}roc_auc": roc_auc_score(y_true, y_proba),
        f"{prefix}pr_auc": average_precision_score(y_true, y_proba, pos_label=1),
        f"{prefix}brier": brier_score_loss(y_true, y_proba),
        f"{prefix}tp": int(tp),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tn": int(tn),
        f"{prefix}flagged_pct": float(y_pred.sum()) / len(y_pred),
    }


def metrics_to_row(model_name: str, metrics: Dict[str, float]) -> Dict[str, object]:
    """Prepend model name to a metrics dict for use in comparison tables."""
    return {"model": model_name, **metrics}
