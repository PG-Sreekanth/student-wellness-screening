"""
run_explain.py - Standalone explainability runner.

Trains RF and CatBoost, then generates feature importance,
permutation importance, PDP/ICE, and SHAP plots.
Usage:
    cd ml_pipeline
    python run_explain.py
"""
import sys
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from data.loader import load_data, stratified_split
from models.trainer import train_random_forest, train_catboost
from preprocessing.pipelines import (
    build_rf_preprocessor,
    build_catboost_preprocessor,
    get_feature_names,
)
from explainability.importance import plot_model_feature_importance, compute_permutation_importance
from explainability.pdp_ice import plot_pdp
from explainability.shap_explain import run_shap_analysis
from utils.logger import get_logger

logger = get_logger("run_explain")


def main():
    logger.info("=== EXPLAINABILITY ===")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    feature_set = "A"

    rf_model, rf_prep, rf_proba = train_random_forest(X_train, y_train, X_val, feature_set)
    cb_model, cb_prep, cb_proba = train_catboost(X_train, y_train, X_val, y_val, feature_set)

    # transformed data for RF
    rf_prep_fresh = build_rf_preprocessor(feature_set)
    X_train_rf = rf_prep_fresh.fit_transform(X_train)
    X_val_rf = rf_prep_fresh.transform(X_val)
    rf_names = get_feature_names(rf_prep_fresh)

    # transformed data for CatBoost
    cb_prep_fresh, cb_cat_idx = build_catboost_preprocessor(feature_set)
    cb_prep_fresh.fit(X_train)
    X_val_cb = cb_prep_fresh.transform(X_val)
    cb_names = get_feature_names(cb_prep_fresh)

    plot_model_feature_importance(rf_model, rf_names, "RandomForest")
    plot_model_feature_importance(cb_model, cb_names, "CatBoost")
    compute_permutation_importance(rf_model, X_val_rf, y_val, rf_names, "RandomForest")
    plot_pdp(rf_model, X_val_rf, rf_names, "RandomForest")
    run_shap_analysis(cb_model, X_val_cb, cb_names, "CatBoost", y_val_proba=cb_proba)

    logger.info("Done. Check artifacts/plots/.")


if __name__ == "__main__":
    main()
