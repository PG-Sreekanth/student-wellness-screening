"""
pdp_ice.py - Partial Dependence Plots and ICE curves.

Generates PDP/ICE plots for the top numeric/ordinal features to show
how each feature affects the model's predicted probability.
"""
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from config.settings import PLOTS_DIR, PLOT_DPI
from config.feature_schema import NUMERIC_FEATURES, ORDINAL_FEATURES
from utils.logger import get_logger

logger = get_logger(__name__)

# Default features for PDP analysis (top numeric + ordinal)
DEFAULT_PDP_FEATURES = [
    "academic_pressure",
    "financial_stress",
    "study_satisfaction",
    "work_study_hours",
    "age",
    "cgpa",
]


def plot_pdp(
    model,
    X_transformed: np.ndarray,
    feature_names: List[str],
    model_name: str,
    features_to_plot: List[str] = None,
) -> None:
    """Generate PDP/ICE plots for selected features.

    Uses sklearn's PartialDependenceDisplay. Only plots features that
    exist in the transformed feature set.

    Args:
        model: Fitted classifier.
        X_transformed: Preprocessed feature array.
        feature_names: Feature names after transformation.
        model_name: Label for plots.
        features_to_plot: Feature names to generate PDPs for.
    """
    target_features = features_to_plot or DEFAULT_PDP_FEATURES
    logger.info(
        f"Generating PDP plots for {model_name}: {target_features}"
    )

    # Map feature names to column indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    valid_features = []
    valid_indices = []
    for f in target_features:
        # Check for exact match or prefix match (due to pipeline naming)
        for fname, idx in name_to_idx.items():
            if fname == f or fname.endswith(f"__{f}") or fname.startswith(f"{f}"):
                # Only PDP for numeric-like features
                valid_features.append(fname)
                valid_indices.append(idx)
                break

    if not valid_indices:
        logger.warning(f"No valid features found for PDP in {model_name}. Skipping.")
        return

    logger.info(f"Plotting PDP for {len(valid_indices)} features: {valid_features}")

    try:
        fig, axes = plt.subplots(
            2, 3,
            figsize=(18, 10),
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        for i, (feat_idx, feat_name) in enumerate(zip(valid_indices[:6], valid_features[:6])):
            ax = axes_flat[i]
            PartialDependenceDisplay.from_estimator(
                model,
                X_transformed,
                features=[feat_idx],
                feature_names=feature_names,
                kind="both",  # PDP + ICE
                ax=ax,
                ice_lines_kw={"alpha": 0.1, "linewidth": 0.5},
                pd_line_kw={"linewidth": 2, "color": "red"},
                subsample=500,
                random_state=42,
            )
            ax.set_title(feat_name, fontsize=10)

        # Hide unused axes
        for j in range(len(valid_indices), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"{model_name} - Partial Dependence Plots (PDP + ICE)", fontsize=14)

        path = PLOTS_DIR / f"pdp_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
        logger.info(f"Saved PDP plot to {path}")

    except Exception as e:
        logger.error(f"PDP plot failed for {model_name}: {e}")
        plt.close("all")
