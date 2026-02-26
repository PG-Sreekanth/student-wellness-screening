"""
preprocessing/pipelines.py - Build sklearn ColumnTransformer pipelines for LR, RF, and CatBoost.

LR pipeline: impute + scale numeric, ordinal-map sleep_duration + scale, OHE nominals, binary-map yes/no.
RF pipeline: same but no scaling (trees don't need it).
CatBoost pipeline: minimal - numeric imputed, sleep mapped, nominals passed through as strings.
"""
from typing import List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config.feature_schema import (
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    ORDERED_CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    SLEEP_DURATION_ORDER,
    get_nominal_features,
)
from preprocessing.transformers import OrdinalMapper, BinaryMapper
from utils.logger import get_logger

logger = get_logger(__name__)


def build_lr_preprocessor(feature_set: str = "A") -> ColumnTransformer:
    """Build LR preprocessor: impute + scale numerics, OHE nominals, map binary."""
    nominal_features = get_nominal_features(feature_set)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    ordinal_map_pipeline = Pipeline([
        ("ordinal_map", OrdinalMapper(mapping=SLEEP_DURATION_ORDER)),
        ("scaler", StandardScaler()),
    ])
    nominal_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary_pipeline = Pipeline([
        ("binary_map", BinaryMapper(positive_label="yes")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ORDINAL_FEATURES),
            ("sleep", ordinal_map_pipeline, ORDERED_CATEGORICAL_FEATURES),
            ("nominal", nominal_pipeline, nominal_features),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )
    logger.info(f"Built LR preprocessor (feature_set={feature_set}).")
    return preprocessor


def build_rf_preprocessor(feature_set: str = "A") -> ColumnTransformer:
    """Build RF preprocessor: same as LR but without StandardScaler."""
    nominal_features = get_nominal_features(feature_set)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    ordinal_map_pipeline = Pipeline([
        ("ordinal_map", OrdinalMapper(mapping=SLEEP_DURATION_ORDER)),
    ])
    nominal_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary_pipeline = Pipeline([
        ("binary_map", BinaryMapper(positive_label="yes")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ORDINAL_FEATURES),
            ("sleep", ordinal_map_pipeline, ORDERED_CATEGORICAL_FEATURES),
            ("nominal", nominal_pipeline, nominal_features),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )
    logger.info(f"Built RF preprocessor (feature_set={feature_set}).")
    return preprocessor


def build_catboost_preprocessor(feature_set: str = "A") -> Tuple[ColumnTransformer, List[str]]:
    """Build CatBoost preprocessor.

    CatBoost handles categoricals natively, so nominals are passed through as-is.
    Returns the preprocessor AND the list of categorical column indices.
    """
    nominal_features = get_nominal_features(feature_set)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    ordinal_map_pipeline = Pipeline([
        ("ordinal_map", OrdinalMapper(mapping=SLEEP_DURATION_ORDER)),
    ])
    binary_pipeline = Pipeline([
        ("binary_map", BinaryMapper(positive_label="yes")),
    ])

    # pass nominal categoricals through without encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ORDINAL_FEATURES),
            ("sleep", ordinal_map_pipeline, ORDERED_CATEGORICAL_FEATURES),
            ("nominal", "passthrough", nominal_features),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )

    # compute which column indices are categorical after the transform
    n_numeric = len(NUMERIC_FEATURES) + len(ORDINAL_FEATURES)
    n_sleep = len(ORDERED_CATEGORICAL_FEATURES)
    n_nominal = len(nominal_features)
    cat_indices = list(range(n_numeric + n_sleep, n_numeric + n_sleep + n_nominal))

    logger.info(f"Built CatBoost preprocessor (feature_set={feature_set}). Cat indices: {cat_indices}")
    return preprocessor, cat_indices


def get_feature_names(preprocessor: ColumnTransformer, X_sample=None) -> List[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # manual fallback if get_feature_names_out isn't available
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if trans == "passthrough":
                names.extend(cols if isinstance(cols, list) else [str(cols)])
            elif hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(trans.get_feature_names_out())
                except Exception:
                    names.extend(cols if isinstance(cols, list) else [str(cols)])
            elif isinstance(cols, list):
                names.extend(cols)
            else:
                names.append(str(cols))
        return names
