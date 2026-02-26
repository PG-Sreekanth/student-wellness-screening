"""
explainability/importance.py - Feature and permutation importance plots.
"""
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from config.settings import PLOTS_DIR, REPORTS_DIR, FIGSIZE_STANDARD, PLOT_DPI, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_model_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot built-in feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} has no .feature_importances_ - skipping.")
        return pd.DataFrame()

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = df_imp.head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.barh(range(len(top)), top["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} - Feature Importance (top {top_n})")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = PLOTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {path}")

    return df_imp


def compute_permutation_importance(
    model,
    X_val_transformed: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_repeats: int = 10,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute and plot permutation importance on the validation set.

    More trustworthy than tree-based importance since it measures actual
    performance impact of each feature.
    """
    logger.info(f"Computing permutation importance for {model_name} ({n_repeats} repeats)...")

    result = permutation_importance(
        model,
        X_val_transformed,
        y_val,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": feature_names[:len(result.importances_mean)],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    top = df_perm.head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.barh(
        range(len(top)),
        top["importance_mean"].values,
        xerr=top["importance_std"].values,
        color="darkorange",
        capsize=3,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (decrease in F1)")
    ax.set_title(f"{model_name} - Permutation Importance (top {top_n})")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = PLOTS_DIR / f"permutation_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Permutation importance plot saved to {path}")

    csv_path = REPORTS_DIR / f"permutation_importance_{model_name.lower().replace(' ', '_')}.csv"
    df_perm.to_csv(csv_path, index=False)

    return df_perm
