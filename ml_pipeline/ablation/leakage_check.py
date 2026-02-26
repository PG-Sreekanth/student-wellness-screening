"""
ablation/leakage_check.py - Ablation experiments to check for leakage and proxy features.

Two experiments:
  1. suicidal_thoughts: compare model with vs without this feature
  2. degree vs degree_category: compare feature set A vs B
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config.settings import REPORTS_DIR, RANDOM_STATE, CATBOOST_DEFAULTS
from config.feature_schema import (
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    ORDERED_CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    SLEEP_DURATION_ORDER,
    get_nominal_features,
)
from preprocessing.transformers import OrdinalMapper, BinaryMapper
from utils.metrics import compute_all_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def _train_catboost_custom(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    nominal_features: List[str],
    binary_features: List[str],
    label: str = "",
) -> np.ndarray:
    """Train CatBoost with an arbitrary feature list - used for ablations.

    Rebuilds the preprocessor based on whichever columns are actually present,
    so we can safely drop features without breaking the pipeline.
    """
    from catboost import CatBoostClassifier, Pool

    num_feats = [f for f in NUMERIC_FEATURES + ORDINAL_FEATURES if f in X_train.columns]
    sleep_feats = [f for f in ORDERED_CATEGORICAL_FEATURES if f in X_train.columns]
    nom_feats = [f for f in nominal_features if f in X_train.columns]
    bin_feats = [f for f in binary_features if f in X_train.columns]

    transformers = []
    if num_feats:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats))
    if sleep_feats:
        transformers.append(("sleep", Pipeline([("ordinal_map", OrdinalMapper(mapping=SLEEP_DURATION_ORDER))]), sleep_feats))
    if nom_feats:
        transformers.append(("nominal", "passthrough", nom_feats))
    if bin_feats:
        transformers.append(("binary", Pipeline([("binary_map", BinaryMapper(positive_label="yes"))]), bin_feats))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_t = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_val_t = pd.DataFrame(preprocessor.transform(X_val))

    # CatBoost needs categorical columns as strings
    n_before_nominal = len(num_feats) + len(sleep_feats)
    cat_indices = list(range(n_before_nominal, n_before_nominal + len(nom_feats)))
    for idx in cat_indices:
        X_train_t.iloc[:, idx] = X_train_t.iloc[:, idx].astype(str)
        X_val_t.iloc[:, idx] = X_val_t.iloc[:, idx].astype(str)

    model = CatBoostClassifier(**CATBOOST_DEFAULTS)
    model.fit(Pool(X_train_t, y_train, cat_features=cat_indices),
              eval_set=Pool(X_val_t, y_val, cat_features=cat_indices), verbose=0)

    y_val_proba = model.predict_proba(X_val_t)[:, 1]
    logger.info(f"  {label} - best iter: {model.get_best_iteration()}, proba: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
    return y_val_proba


def ablation_suicidal_thoughts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set: str = "A",
) -> pd.DataFrame:
    """Check how much suicidal_thoughts drives the model.

    If removing it tanks performance, the model is too reliant on a single
    signal instead of the broader risk picture.
    """
    logger.info("ABLATION: suicidal_thoughts inclusion vs exclusion")

    nominal_feats = get_nominal_features(feature_set)

    proba_with = _train_catboost_custom(
        X_train, y_train, X_val, y_val,
        nominal_features=nominal_feats,
        binary_features=BINARY_FEATURES,
        label="WITH suicidal_thoughts",
    )
    metrics_with = compute_all_metrics(y_val, proba_with, threshold=0.5)
    metrics_with["variant"] = "with_suicidal_thoughts"

    # drop suicidal_thoughts from both splits
    binary_no_st = [f for f in BINARY_FEATURES if f != "suicidal_thoughts"]
    proba_without = _train_catboost_custom(
        X_train.drop(columns=["suicidal_thoughts"], errors="ignore"),
        y_train,
        X_val.drop(columns=["suicidal_thoughts"], errors="ignore"),
        y_val,
        nominal_features=nominal_feats,
        binary_features=binary_no_st,
        label="WITHOUT suicidal_thoughts",
    )
    metrics_without = compute_all_metrics(y_val, proba_without, threshold=0.5)
    metrics_without["variant"] = "without_suicidal_thoughts"

    cols = ["variant", "recall", "precision", "f2", "roc_auc", "pr_auc", "brier"]
    df = pd.DataFrame([metrics_with, metrics_without])[cols]
    delta = {c: metrics_with[c] - metrics_without[c] for c in cols if c != "variant"}
    delta["variant"] = "DELTA (with - without)"
    df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)

    logger.info(f"\nSuicidal Thoughts Ablation:\n{df.to_string(index=False)}")
    path = REPORTS_DIR / "ablation_suicidal_thoughts.csv"
    df.to_csv(path, index=False)
    return df


def ablation_degree_vs_degree_category(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Compare Feature Set A (degree_category) vs B (degree)."""
    logger.info("ABLATION: Feature Set A (degree_category) vs B (degree)")

    proba_a = _train_catboost_custom(
        X_train, y_train, X_val, y_val,
        nominal_features=get_nominal_features("A"),
        binary_features=BINARY_FEATURES,
        label="Feature Set A",
    )
    metrics_a = compute_all_metrics(y_val, proba_a, threshold=0.5)
    metrics_a["variant"] = "feature_set_A (degree_category)"

    proba_b = _train_catboost_custom(
        X_train, y_train, X_val, y_val,
        nominal_features=get_nominal_features("B"),
        binary_features=BINARY_FEATURES,
        label="Feature Set B",
    )
    metrics_b = compute_all_metrics(y_val, proba_b, threshold=0.5)
    metrics_b["variant"] = "feature_set_B (degree)"

    cols = ["variant", "recall", "precision", "f2", "roc_auc", "pr_auc", "brier"]
    df = pd.DataFrame([metrics_a, metrics_b])[cols]
    delta = {c: metrics_a[c] - metrics_b[c] for c in cols if c != "variant"}
    delta["variant"] = "DELTA (A - B)"
    df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)

    logger.info(f"\nDegree Ablation:\n{df.to_string(index=False)}")
    df.to_csv(REPORTS_DIR / "ablation_degree_vs_degree_category.csv", index=False)
    return df


def run_all_ablations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Run both ablation tests and return results."""
    return {
        "suicidal_thoughts": ablation_suicidal_thoughts(X_train, y_train, X_val, y_val),
        "degree": ablation_degree_vs_degree_category(X_train, y_train, X_val, y_val),
    }
