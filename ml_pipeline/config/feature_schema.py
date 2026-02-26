"""
feature_schema.py - Feature type definitions, ordinal mappings, and feature-set contracts.

Defines which columns are numeric, ordinal, ordered-categorical, nominal, and binary.
Also defines Feature Set A (degree_category) vs B (degree).
"""
from typing import Dict, List

TARGET = "depression"

# Continuous / quasi-continuous numeric features
NUMERIC_FEATURES: List[str] = ["age", "cgpa", "work_study_hours"]

# Ordinal features (integer scales 1-5, treated as numeric)
ORDINAL_FEATURES: List[str] = [
    "academic_pressure",
    "study_satisfaction",
    "financial_stress",
]

# Ordered categorical → will be mapped to ordinal ints
ORDERED_CATEGORICAL_FEATURES: List[str] = ["sleep_duration"]

# Nominal categorical (unordered)
NOMINAL_FEATURES_BASE: List[str] = [
    "city",
    "dietary_habits",
    "gender",
]

# Binary text features (yes/no → 1/0)
BINARY_FEATURES: List[str] = [
    "suicidal_thoughts",
    "family_history_mental_illness",
]

# Ordinal order maps
SLEEP_DURATION_ORDER: Dict[str, int] = {
    "less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "more than 8 hours": 3,
}

BINARY_MAP: Dict[str, int] = {
    "yes": 1,
    "no": 0,
}


# Feature-set A vs B (degree handling)
def get_feature_set_a() -> List[str]:
    """Feature Set A - uses degree_category (simpler, interpretable). Drops raw degree."""
    return (
        NUMERIC_FEATURES
        + ORDINAL_FEATURES
        + ORDERED_CATEGORICAL_FEATURES
        + NOMINAL_FEATURES_BASE
        + ["degree_category"]
        + BINARY_FEATURES
    )


def get_feature_set_b() -> List[str]:
    """Feature Set B - uses raw degree (richer, more granular). Drops degree_category."""
    return (
        NUMERIC_FEATURES
        + ORDINAL_FEATURES
        + ORDERED_CATEGORICAL_FEATURES
        + NOMINAL_FEATURES_BASE
        + ["degree"]
        + BINARY_FEATURES
    )


def get_nominal_features(feature_set: str = "A") -> List[str]:
    """Return nominal features for the given feature set."""
    if feature_set.upper() == "A":
        return NOMINAL_FEATURES_BASE + ["degree_category"]
    else:
        return NOMINAL_FEATURES_BASE + ["degree"]


# All expected columns in the clean CSV (for schema validation)
ALL_EXPECTED_COLUMNS: List[str] = [
    "gender",
    "age",
    "city",
    "academic_pressure",
    "cgpa",
    "study_satisfaction",
    "sleep_duration",
    "dietary_habits",
    "degree",
    "suicidal_thoughts",
    "work_study_hours",
    "financial_stress",
    "family_history_mental_illness",
    "depression",
    "degree_category",
]
