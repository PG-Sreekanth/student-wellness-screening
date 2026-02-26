"""
run_pipeline.py - Full ML pipeline entry point.

Runs all stages: load → train → evaluate → explain → tune → final model.
Usage:
    cd ml_pipeline
    python run_pipeline.py
"""
import sys
import time
import warnings
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from config.settings import RANDOM_STATE, REPORTS_DIR, MODELS_DIR
from config.feature_schema import get_feature_set_a
from data.loader import load_data, stratified_split
from models.trainer import train_logistic_regression, train_random_forest, train_catboost
from evaluation.metrics import compare_models
from evaluation.threshold import run_threshold_analysis
from evaluation.calibration import plot_calibration_curve
from explainability.importance import plot_model_feature_importance, compute_permutation_importance
from explainability.pdp_ice import plot_pdp
from explainability.shap_explain import run_shap_analysis
from ablation.leakage_check import run_all_ablations
from tuning.optuna_tuner import run_all_tuning
from pipeline.final_model import evaluate_on_test, save_deployment_artifacts
from preprocessing.pipelines import (
    build_lr_preprocessor,
    build_rf_preprocessor,
    build_catboost_preprocessor,
    get_feature_names,
)
from utils.metrics import compute_all_metrics
from utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("run_pipeline")


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("  STUDENT DEPRESSION SCREENING - ML PIPELINE")
    logger.info("=" * 70)

    # Stage 1: load data and create train/val/test splits
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    feature_set = "A"

    # Stage 2: train baseline models
    lr_model, lr_prep, lr_val_proba = train_logistic_regression(
        X_train, y_train, X_val, feature_set=feature_set
    )
    rf_model, rf_prep, rf_val_proba = train_random_forest(
        X_train, y_train, X_val, feature_set=feature_set
    )
    cb_model, cb_prep, cb_val_proba = train_catboost(
        X_train, y_train, X_val, y_val, feature_set=feature_set
    )

    # Stage 3: compare all baseline models
    model_probas = {
        "LogisticRegression": lr_val_proba,
        "RandomForest": rf_val_proba,
        "CatBoost": cb_val_proba,
    }
    comparison_df = compare_models(model_probas, y_val, threshold=0.5)

    # Stage 4: threshold analysis per model
    thresholds = {}
    for name, proba in model_probas.items():
        thresholds[name] = run_threshold_analysis(y_val, proba, name)

    # Stage 5: calibration curves
    for name, proba in model_probas.items():
        plot_calibration_curve(y_val, proba, name)

    # Stage 6: feature importance
    # rebuild RF preprocessor to get transformed data and feature names
    rf_prep_fresh = build_rf_preprocessor(feature_set)
    X_train_rf = rf_prep_fresh.fit_transform(X_train)
    X_val_rf = rf_prep_fresh.transform(X_val)
    rf_feature_names = get_feature_names(rf_prep_fresh)

    plot_model_feature_importance(rf_model, rf_feature_names, "RandomForest")

    _, cb_cat_indices = build_catboost_preprocessor(feature_set)
    cb_prep_fresh = build_catboost_preprocessor(feature_set)[0]
    X_train_cb = cb_prep_fresh.fit_transform(X_train)
    X_val_cb = cb_prep_fresh.transform(X_val)
    cb_feature_names = get_feature_names(cb_prep_fresh)
    plot_model_feature_importance(cb_model, cb_feature_names, "CatBoost")

    compute_permutation_importance(
        rf_model, X_val_rf, y_val, rf_feature_names, "RandomForest"
    )

    # Stage 7: PDP/ICE and SHAP
    plot_pdp(rf_model, X_val_rf, rf_feature_names, "RandomForest")
    run_shap_analysis(
        cb_model, X_val_cb, cb_feature_names, "CatBoost",
        y_val_proba=cb_val_proba,
    )

    # Stage 8: ablation - check for leaky or proxy features
    ablation_results = run_all_ablations(X_train, y_train, X_val, y_val)

    # Stage 9: tune all models with Optuna
    tuning_results = run_all_tuning(
        X_train, y_train,
        feature_set=feature_set,
        n_trials=10,
    )

    # Stage 10: retrain with best tuned params
    lr_best_params = {
        **tuning_results["LogisticRegression"]["best_params"],
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }
    lr_tuned, lr_tuned_prep, lr_tuned_proba = train_logistic_regression(
        X_train, y_train, X_val, feature_set=feature_set, params=lr_best_params
    )

    rf_best_params = {
        **tuning_results["RandomForest"]["best_params"],
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_tuned, rf_tuned_prep, rf_tuned_proba = train_random_forest(
        X_train, y_train, X_val, feature_set=feature_set, params=rf_best_params
    )

    cb_best_params = {
        **tuning_results["CatBoost"]["best_params"],
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
    }
    cb_tuned, cb_tuned_prep, cb_tuned_proba = train_catboost(
        X_train, y_train, X_val, y_val, feature_set=feature_set, params=cb_best_params
    )

    tuned_probas = {
        "LR_tuned": lr_tuned_proba,
        "RF_tuned": rf_tuned_proba,
        "CatBoost_tuned": cb_tuned_proba,
    }
    compare_models(tuned_probas, y_val, threshold=0.5)

    tuned_thresholds = {}
    for name, proba in tuned_probas.items():
        tuned_thresholds[name] = run_threshold_analysis(y_val, proba, name)

    # Stage 11: pick the best tuned model and evaluate on test set (only once)
    final_candidates = {
        "LR_tuned": {
            "model": lr_tuned,
            "preprocessor": lr_tuned_prep,
            "val_proba": lr_tuned_proba,
            "val_metrics": compute_all_metrics(y_val, lr_tuned_proba,
                                               threshold=tuned_thresholds["LR_tuned"]["f2_max"]),
            "best_threshold": tuned_thresholds["LR_tuned"]["f2_max"],
            "cat_indices": None,
        },
        "RF_tuned": {
            "model": rf_tuned,
            "preprocessor": rf_tuned_prep,
            "val_proba": rf_tuned_proba,
            "val_metrics": compute_all_metrics(y_val, rf_tuned_proba,
                                               threshold=tuned_thresholds["RF_tuned"]["f2_max"]),
            "best_threshold": tuned_thresholds["RF_tuned"]["f2_max"],
            "cat_indices": None,
        },
        "CatBoost_tuned": {
            "model": cb_tuned,
            "preprocessor": cb_tuned_prep,
            "val_proba": cb_tuned_proba,
            "val_metrics": compute_all_metrics(y_val, cb_tuned_proba,
                                               threshold=tuned_thresholds["CatBoost_tuned"]["f2_max"]),
            "best_threshold": tuned_thresholds["CatBoost_tuned"]["f2_max"],
            "cat_indices": cb_cat_indices,
        },
    }

    best_name = max(
        final_candidates,
        key=lambda k: final_candidates[k]["val_metrics"]["f2"],
    )
    best_info = final_candidates[best_name]
    logger.info(f"\n>>> FINAL MODEL: {best_name}")

    test_metrics = evaluate_on_test(
        best_info["model"],
        best_info["preprocessor"],
        X_test,
        y_test,
        threshold=best_info["best_threshold"],
        model_name=best_name,
        cat_indices=best_info.get("cat_indices"),
    )

    save_deployment_artifacts(
        model=best_info["model"],
        preprocessor=best_info["preprocessor"],
        threshold=best_info["best_threshold"],
        metrics=test_metrics,
        model_name=best_name,
        feature_set=feature_set,
        cat_indices=best_info.get("cat_indices"),
    )

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"  PIPELINE COMPLETE - {elapsed:.1f}s")
    logger.info(f"  Final model : {best_name}")
    logger.info(f"  Test F2     : {test_metrics.get('test_f2', 'N/A')}")
    logger.info(f"  Test Recall : {test_metrics.get('test_recall', 'N/A')}")
    logger.info(f"  Test PR-AUC : {test_metrics.get('test_pr_auc', 'N/A')}")
    logger.info(f"  Threshold   : {best_info['best_threshold']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
