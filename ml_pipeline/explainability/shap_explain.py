"""
shap_explain.py - SHAP explanations for tree-based models.

Generates global summary/bar plots, dependence plots for top features,
and local waterfall plots for example high/moderate/low risk cases.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import PLOTS_DIR, PLOT_DPI
from utils.logger import get_logger

logger = get_logger(__name__)


def run_shap_analysis(
    model,
    X_val_transformed: np.ndarray,
    feature_names: List[str],
    model_name: str,
    y_val_proba: np.ndarray = None,
    top_n: int = 15,
) -> Optional[np.ndarray]:
    """Run full SHAP analysis for a tree-based model.

    Returns the SHAP values array (positive class), or None if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Install with: pip install shap")
        return None

    logger.info(f"Running SHAP analysis for {model_name}...")

    # pick the right explainer
    model_type = type(model).__name__
    if model_type in ("CatBoostClassifier", "RandomForestClassifier",
                       "LGBMClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        logger.info(f"Using KernelExplainer for {model_type} (may be slow)...")
        bg = shap.sample(X_val_transformed, min(100, len(X_val_transformed)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)

    # compute SHAP values on a subsample
    sample_size = min(1000, len(X_val_transformed))
    X_sample = X_val_transformed[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # take positive class (index 1) for binary classifiers
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # global summary plot (beeswarm)
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values_pos, X_sample,
            feature_names=feature_names[:shap_values_pos.shape[1]],
            show=False, max_display=top_n,
        )
        path = PLOTS_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved SHAP summary plot to {path}")
    except Exception as e:
        logger.error(f"SHAP summary plot failed: {e}")
        plt.close("all")

    # bar chart of mean |SHAP|
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values_pos, X_sample,
            feature_names=feature_names[:shap_values_pos.shape[1]],
            plot_type="bar", show=False, max_display=top_n,
        )
        path = PLOTS_DIR / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved SHAP bar plot to {path}")
    except Exception as e:
        logger.error(f"SHAP bar plot failed: {e}")
        plt.close("all")

    # dependence plots for top 5 features
    try:
        mean_abs = np.abs(shap_values_pos).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:5]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, feat_idx in enumerate(top_indices):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            ax = axes[i]
            ax.scatter(
                X_sample[:, feat_idx], shap_values_pos[:, feat_idx],
                alpha=0.3, s=10, c=shap_values_pos[:, feat_idx], cmap="coolwarm",
            )
            ax.set_xlabel(feat_name, fontsize=9)
            ax.set_ylabel("SHAP value" if i == 0 else "")
            ax.set_title(feat_name, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{model_name} - SHAP Dependence (top 5)", fontsize=13)
        plt.tight_layout()
        path = PLOTS_DIR / f"shap_dependence_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
        logger.info(f"Saved SHAP dependence plots to {path}")
    except Exception as e:
        logger.error(f"SHAP dependence plots failed: {e}")
        plt.close("all")

    # local waterfall plots (high / moderate / low risk examples)
    if y_val_proba is not None:
        try:
            proba_sample = y_val_proba[:sample_size]
            examples = {
                "high_risk": np.argmax(proba_sample),
                "moderate_risk": np.argmin(np.abs(proba_sample - 0.5)),
                "low_risk": np.argmin(proba_sample),
            }

            for label, idx in examples.items():
                try:
                    explanation = shap.Explanation(
                        values=shap_values_pos[idx],
                        base_values=explainer.expected_value[1]
                        if isinstance(explainer.expected_value, (list, np.ndarray))
                        else explainer.expected_value,
                        data=X_sample[idx],
                        feature_names=feature_names[:shap_values_pos.shape[1]],
                    )
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(explanation, show=False, max_display=12)
                    path = PLOTS_DIR / f"shap_waterfall_{label}_{model_name.lower().replace(' ', '_')}.png"
                    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
                    plt.close("all")
                    logger.info(f"Saved SHAP waterfall ({label}) to {path}")
                except Exception as e:
                    logger.warning(f"Waterfall for {label} failed: {e}")
                    plt.close("all")
        except Exception as e:
            logger.error(f"Local SHAP explanations failed: {e}")
            plt.close("all")

    return shap_values_pos
