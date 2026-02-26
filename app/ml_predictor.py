"""
ml_predictor.py - Load the trained ML model and make predictions.

Handles: depression prediction, red_flag calculation, wellness tier,
support_priority assignment, and degree_category mapping.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

# Paths to ML artifacts
ML_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "ml_pipeline"

# Add ml_pipeline to sys.path so joblib can find the custom transformers
# (the pickled preprocessor references preprocessing.transformers.OrdinalMapper etc.)
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))
MODELS_DIR = ML_PIPELINE_DIR / "artifacts" / "models"
MODEL_PATH = MODELS_DIR / "final_model_rf_tuned.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "final_preprocessor_rf_tuned.joblib"
CONFIG_PATH = MODELS_DIR / "threshold_config.json"

# Load once at startup
_model = None
_preprocessor = None
_threshold = None


def _patch_fitted_status(obj):
    """
    Recursively look for objects and ensure they appear 'fitted' to scikit-learn.
    Fixes:
    1. 'SimpleImputer' object has no attribute '_fill_dtype' (version mismatch)
    2. 'This Pipeline instance is not fitted yet' (missing fitted attributes)
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # Custom transformers
    try:
        from preprocessing.transformers import OrdinalMapper, BinaryMapper
    except ImportError:
        OrdinalMapper, BinaryMapper = type(None), type(None)

    # 1. Fix SimpleImputer
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = getattr(obj, "dtype", np.float64)
        if not hasattr(obj, "statistics_"): # Another potential 'not fitted' trigger
            obj.statistics_ = np.array([]) 

    # 2. Fix Custom Transformers (ensure they look fitted)
    if isinstance(obj, (OrdinalMapper, BinaryMapper)):
        if not hasattr(obj, "n_features_in_"):
            obj.n_features_in_ = 1

    # 3. Recursive traversal
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            _patch_fitted_status(step)
    elif isinstance(obj, ColumnTransformer):
        if hasattr(obj, "transformers_"):
            for _, transformer, _ in obj.transformers_:
                _patch_fitted_status(transformer)
        if hasattr(obj, "transformers"):
            for item in obj.transformers:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _patch_fitted_status(item[1])


def _load_artifacts():
    """Lazy-load model artifacts on first call."""
    global _model, _preprocessor, _threshold
    if _model is None:
        _ensure_numpy_pickle_compat()
        _model = joblib.load(MODEL_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Apply patches to ensure compatibility and 'fitted' status
        try:
            _patch_fitted_status(_preprocessor)
            _patch_fitted_status(_model)
        except Exception as e:
            print(f"Non-critical: Failed to apply patches: {e}")

        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        _threshold = config["threshold"]
    return _model, _preprocessor, _threshold


def _ensure_numpy_pickle_compat():
    """Best-effort compatibility for artifacts pickled with NumPy 2.x."""
    try:
        import importlib
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)
        for name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            try:
                mod = importlib.import_module(f"numpy.core.{name}")
                sys.modules.setdefault(f"numpy._core.{name}", mod)
            except Exception:
                pass
    except Exception:
        pass


# Degree → degree_category mapping

DEGREE_CATEGORY_MAP = {
    # School
    "class 12": "school", "class 12th": "school",
    "class 10": "school", "class 10th": "school",
    # UG
    "btech": "ug", "b.tech": "ug",
    "bcom":  "ug", "b.com":  "ug",
    "bca":   "ug",
    "bsc":   "ug", "b.sc":   "ug",
    "bed":   "ug", "b.ed":   "ug",
    "ba":    "ug",
    "bba":   "ug",
    "barch": "ug", "b.arch": "ug",
    "bpharm":"ug",
    "bhm":   "ug",
    "bdes":  "ug", "b.des":  "ug",
    "llb":   "ug",
    "be":    "ug",
    "bds":   "ug",
    "mbbs":  "ug",
    # PG
    "mtech": "pg", "m.tech": "pg",
    "msc":   "pg", "m.sc":   "pg",
    "mca":   "pg",
    "mcom":  "pg", "m.com":  "pg",
    "med":   "pg", "m.ed":   "pg",
    "ma":    "pg",
    "mba":   "pg",
    "mdes":  "pg", "m.des":  "pg",
    "march": "pg", "m.arch": "pg",
    "mpharm":"pg",
    "md":    "pg",
    "llm":   "pg",
    "me":    "pg",
    "ms":    "pg",
    "mphil": "pg", "m.phil": "pg",
    "pgdm":  "pg",
    # PhD
    "phd":   "phd",
    "doctorate": "phd",
    # Others
    "diploma": "others",
}


def map_degree_category(degree: str) -> str:
    """Map a degree string to its category."""
    key = degree.strip().lower()
    # Normalize by removing dots for better matching
    key_clean = key.replace(".", "")
    return DEGREE_CATEGORY_MAP.get(key_clean, DEGREE_CATEGORY_MAP.get(key, "others"))


# Red flag computation

def compute_red_flag(data: Dict[str, Any]) -> int:
    """
    Count triggered risk conditions (0-7).

    Conditions:
      1. academic_pressure >= 4
      2. financial_stress >= 4
      3. work_study_hours >= 10
      4. study_satisfaction <= 2
      5. sleep_duration == "less than 5 hours"
      6. dietary_habits == "unhealthy"
      7. suicidal_thoughts == "yes"
    """
    flags = 0
    if data.get("academic_pressure", 0) >= 4:
        flags += 1
    if data.get("financial_stress", 0) >= 4:
        flags += 1
    if data.get("work_study_hours", 0) >= 10:
        flags += 1
    if data.get("study_satisfaction", 5) <= 2:
        flags += 1
    if str(data.get("sleep_duration", "")).strip().lower() == "less than 5 hours":
        flags += 1
    if str(data.get("dietary_habits", "")).strip().lower() == "unhealthy":
        flags += 1
    if str(data.get("suicidal_thoughts", "")).strip().lower() == "yes":
        flags += 1
    return flags


def compute_wellness(red_flag: int) -> str:
    """Map red_flag count to wellness tier."""
    if red_flag <= 2:
        return "high"
    elif red_flag <= 5:
        return "moderate"
    else:
        return "low"


def compute_support_priority(depression: int, red_flag: int, form_data: dict = None) -> str:
    """
    SQL-aligned support_priority mapping (matches the cleaned dataset labels).

    Mapping uses only (depression, red_flag) so the app remains consistent with
    the SQL-cleaned dataset, EDA outputs, and Power BI label definitions.
    """
    if depression == 1:
        if red_flag >= 6:
            return "critical"
        elif red_flag >= 5:
            return "high priority"
        else:
            return "moderate priority"
    else:
        if red_flag >= 5:
            return "preventive high risk"
        elif red_flag >= 3:
            return "preventive watchlist"
        else:
            return "stable"

def predict(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full prediction pipeline on a single student submission.

    Returns a dict with: depression, depression_probability, degree_category,
    red_flag, wellness, support_priority.
    """
    model, preprocessor, threshold = _load_artifacts()

    # Build a single-row DataFrame matching the model's expected features
    degree_category = map_degree_category(form_data["degree"])

    row = {
        "gender": form_data["gender"].strip().lower(),
        "age": form_data["age"],
        "city": form_data["city"].strip().lower(),
        "academic_pressure": form_data["academic_pressure"],
        "cgpa": form_data["cgpa"],
        "study_satisfaction": form_data["study_satisfaction"],
        "sleep_duration": form_data["sleep_duration"].strip().lower(),
        "dietary_habits": form_data["dietary_habits"].strip().lower(),
        "degree": form_data["degree"].strip().lower(),
        "suicidal_thoughts": form_data["suicidal_thoughts"].strip().lower(),
        "work_study_hours": form_data["work_study_hours"],
        "financial_stress": form_data["financial_stress"],
        "family_history_mental_illness": form_data["family_history_mental_illness"].strip().lower(),
        "degree_category": degree_category,
    }

    df = pd.DataFrame([row])

    # Transform and predict
    X_transformed = preprocessor.transform(df)
    probability = float(model.predict_proba(X_transformed)[0, 1])
    depression = 1 if probability >= threshold else 0

    # Compute derived fields
    red_flag = compute_red_flag(form_data)
    wellness = compute_wellness(red_flag)
    support_priority = compute_support_priority(depression, red_flag, form_data)

    return {
        "depression": depression,
        "depression_probability": round(probability, 4),
        "degree_category": degree_category,
        "red_flag": red_flag,
        "wellness": wellness,
        "support_priority": support_priority,
    }
