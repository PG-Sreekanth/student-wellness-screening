"""
data/loader.py - Load the dataset and split it into train/val/test.
"""
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config.settings import (
    DATA_PATH,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TARGET_COLUMN,
)
from config.feature_schema import ALL_EXPECTED_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all expected columns are present and warn on nulls."""
    missing = set(ALL_EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        logger.warning(f"Columns with nulls:\n{cols_with_nulls}")
    else:
        logger.info("Schema OK - no missing columns or nulls.")


def load_data(path: str = None) -> pd.DataFrame:
    """Load CSV and validate schema."""
    data_path = path or str(DATA_PATH)
    logger.info(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    validate_schema(df)
    return df


def stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 split that preserves class balance in each set."""
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # train vs (val + test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # split the temp set into val and test (50/50)
    relative_test = TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"Split - Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}"
    )
    for name, y_part in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_part.value_counts(normalize=True)
        logger.info(
            f"  {name}: dep=1 {dist.get(1, 0):.3f}  dep=0 {dist.get(0, 0):.3f}"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
