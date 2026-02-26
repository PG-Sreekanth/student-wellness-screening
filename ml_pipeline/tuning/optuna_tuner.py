"""
tuning/optuna_tuner.py - Hyperparameter tuning with Optuna.

Tunes LR, RF, and CatBoost via stratified 5-fold CV using F2 as the objective.
All trials are logged to CSV for experiment tracking.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score

from config.settings import (
    RANDOM_STATE,
    CV_FOLDS,
    F_BETA,
    OPTUNA_N_TRIALS,
    REPORTS_DIR,
    LOGS_DIR,
)
from preprocessing.pipelines import (
    build_lr_preprocessor,
    build_rf_preprocessor,
    build_catboost_preprocessor,
)
from utils.logger import get_logger

logger = get_logger(__name__)

f2_scorer = make_scorer(fbeta_score, beta=F_BETA, pos_label=1)


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "A",
    n_trials: int = None,
) -> Dict:
    """Tune LR with Optuna (search: C, penalty). Objective: maximize F2 CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        raise

    n_trials = n_trials or OPTUNA_N_TRIALS
    preprocessor = build_lr_preprocessor(feature_set)
    X_transformed = preprocessor.fit_transform(X_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        model = LogisticRegression(
            C=C, penalty=penalty, solver="liblinear",
            max_iter=1000, random_state=RANDOM_STATE,
        )
        return cross_val_score(model, X_transformed, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", study_name="LR_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"LR best F2: {study.best_value:.4f}  params: {study.best_params}")

    study.trials_dataframe().to_csv(LOGS_DIR / "optuna_lr_trials.csv", index=False)
    return {"best_params": study.best_params, "best_score": study.best_value}


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "A",
    n_trials: int = None,
) -> Dict:
    """Tune RF with Optuna (search: n_estimators, max_depth, min_samples, max_features)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise

    n_trials = n_trials or OPTUNA_N_TRIALS
    preprocessor = build_rf_preprocessor(feature_set)
    X_transformed = preprocessor.fit_transform(X_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_transformed, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", study_name="RF_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"RF best F2: {study.best_value:.4f}  params: {study.best_params}")

    study.trials_dataframe().to_csv(LOGS_DIR / "optuna_rf_trials.csv", index=False)
    return {"best_params": study.best_params, "best_score": study.best_value}


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "A",
    n_trials: int = None,
) -> Dict:
    """Tune CatBoost with Optuna (search: iterations, lr, depth, l2, subsample)."""
    try:
        import optuna
        from catboost import CatBoostClassifier, Pool
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise

    n_trials = n_trials or OPTUNA_N_TRIALS
    preprocessor, cat_indices = build_catboost_preprocessor(feature_set)
    X_transformed = pd.DataFrame(preprocessor.fit_transform(X_train))
    for idx in cat_indices:
        X_transformed.iloc[:, idx] = X_transformed.iloc[:, idx].astype(str)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        f2_scores = []
        for train_idx, val_idx in cv.split(X_transformed, y_train):
            X_f_tr, X_f_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
            y_f_tr, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_f_tr, y_f_tr, cat_features=cat_indices),
                eval_set=Pool(X_f_val, y_f_val, cat_features=cat_indices),
                verbose=0,
            )
            y_pred = model.predict(X_f_val)
            f2_scores.append(fbeta_score(y_f_val, y_pred, beta=F_BETA, pos_label=1))
        return np.mean(f2_scores)

    study = optuna.create_study(direction="maximize", study_name="CatBoost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"CatBoost best F2: {study.best_value:.4f}  params: {study.best_params}")

    study.trials_dataframe().to_csv(LOGS_DIR / "optuna_catboost_trials.csv", index=False)
    return {"best_params": study.best_params, "best_score": study.best_value}


def run_all_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "A",
    n_trials: int = None,
) -> Dict[str, Dict]:
    """Run Optuna tuning for all three models and save a summary CSV."""
    logger.info("=== HYPERPARAMETER TUNING (Optuna) ===")

    results = {
        "LogisticRegression": tune_logistic_regression(X_train, y_train, feature_set, n_trials),
        "RandomForest": tune_random_forest(X_train, y_train, feature_set, n_trials),
        "CatBoost": tune_catboost(X_train, y_train, feature_set, n_trials),
    }

    summary_rows = [
        {"model": name, "best_f2_cv": res["best_score"], **{f"param_{k}": v for k, v in res["best_params"].items()}}
        for name, res in results.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "tuning_summary.csv", index=False)
    logger.info(f"\nTuning summary:\n{summary_df.to_string(index=False)}")

    return results
