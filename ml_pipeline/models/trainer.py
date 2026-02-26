"""
models/trainer.py - Train baseline LR, RF, and CatBoost models.

Each function: builds preprocessor → fits on train → trains model → returns
(model, preprocessor, val_probabilities). No tuning happens here.
"""
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config.settings import RANDOM_STATE, LR_DEFAULTS, RF_DEFAULTS, CATBOOST_DEFAULTS
from preprocessing.pipelines import (
    build_lr_preprocessor,
    build_rf_preprocessor,
    build_catboost_preprocessor,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    feature_set: str = "A",
    params: Dict = None,
) -> Tuple[LogisticRegression, object, np.ndarray]:
    """Train a Logistic Regression model and return val probabilities."""
    logger.info("Training Logistic Regression...")

    preprocessor = build_lr_preprocessor(feature_set)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model_params = {**LR_DEFAULTS, **(params or {})}
    model = LogisticRegression(**model_params)
    model.fit(X_train_t, y_train)

    y_val_proba = model.predict_proba(X_val_t)[:, 1]
    logger.info(f"LR done. Val proba range: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
    return model, preprocessor, y_val_proba


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    feature_set: str = "A",
    params: Dict = None,
) -> Tuple[RandomForestClassifier, object, np.ndarray]:
    """Train a Random Forest model and return val probabilities."""
    logger.info("Training Random Forest...")

    preprocessor = build_rf_preprocessor(feature_set)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model_params = {**RF_DEFAULTS, **(params or {})}
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_t, y_train)

    y_val_proba = model.predict_proba(X_val_t)[:, 1]
    logger.info(f"RF done. n_estimators={model.n_estimators}. Val proba range: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
    return model, preprocessor, y_val_proba


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set: str = "A",
    params: Dict = None,
) -> Tuple[object, object, np.ndarray]:
    """Train a CatBoost model with early stopping on the val set."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.error("CatBoost not installed. Run: pip install catboost")
        raise

    logger.info("Training CatBoost...")

    preprocessor, cat_indices = build_catboost_preprocessor(feature_set)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # CatBoost needs categorical columns as strings
    X_train_t = pd.DataFrame(X_train_t)
    X_val_t = pd.DataFrame(X_val_t)
    for idx in cat_indices:
        X_train_t.iloc[:, idx] = X_train_t.iloc[:, idx].astype(str)
        X_val_t.iloc[:, idx] = X_val_t.iloc[:, idx].astype(str)

    model_params = {**CATBOOST_DEFAULTS, **(params or {})}
    model = CatBoostClassifier(**model_params)

    train_pool = Pool(X_train_t, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val_t, y_val, cat_features=cat_indices)
    model.fit(train_pool, eval_set=val_pool, verbose=0)

    y_val_proba = model.predict_proba(X_val_t)[:, 1]
    logger.info(f"CatBoost done. Best iter: {model.get_best_iteration()}. Val proba range: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
    return model, preprocessor, y_val_proba
