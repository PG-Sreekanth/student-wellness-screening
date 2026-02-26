"""
transformers.py - Custom sklearn-compatible transformers for preprocessing.

OrdinalMapper: Maps ordered categories (e.g. sleep_duration) to integer ordinals.
BinaryMapper:  Maps binary string columns (yes/no) to 1/0.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Map ordered categorical values to integer ordinals.

    Parameters:
        mapping: Dictionary mapping category strings to integer ranks.
                 e.g. {'less than 5 hours': 0, '5-6 hours': 1, ...}
    """

    def __init__(self, mapping: Dict[str, int] = None):
        self.mapping = mapping or {}

    def fit(self, X, y=None):
        """No fitting needed - mapping is provided at init."""
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else 1
        return self

    def transform(self, X) -> np.ndarray:
        """Apply the ordinal mapping.

        Args:
            X: 2-D array-like (n_samples, n_features). Typically a single column.

        Returns:
            np.ndarray of integers with the same shape.
        """
        X_df = pd.DataFrame(X)
        result = X_df.apply(
            lambda col: col.map(self.mapping).fillna(-1).astype(int)
        )
        return result.values


class BinaryMapper(BaseEstimator, TransformerMixin):
    """Map binary string columns (yes/no) to 1/0.

    Parameters:
        positive_label: The string that maps to 1. Default is 'yes'.
    """

    def __init__(self, positive_label: str = "yes"):
        self.positive_label = positive_label

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else 1
        return self

    def transform(self, X) -> np.ndarray:
        """Convert to binary integers.

        Args:
            X: 2-D array-like of string values ('yes'/'no').

        Returns:
            np.ndarray of 0/1 integers.
        """
        X_df = pd.DataFrame(X)
        result = X_df.apply(
            lambda col: (col.str.lower().str.strip() == self.positive_label)
            .astype(int)
        )
        return result.values
