"""
config/settings.py - All project constants in one place.

Paths, split ratios, model defaults, and plotting config all live here
so every module reads from a single source of truth.
"""
from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_PIPELINE_ROOT = Path(__file__).resolve().parent.parent

DATA_FILENAME = "student_depression_final_age_degree_outliers_removed.csv"
DATA_PATH = PROJECT_ROOT / "datasets" / "ml_ready" / DATA_FILENAME

ARTIFACTS_DIR = ML_PIPELINE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Create dirs if they don't exist
for _d in [ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# Train / val / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TARGET_COLUMN = "depression"

# Threshold selection targets
MIN_RECALL_TARGET = 0.85       # minimum recall we want for triage
CAPACITY_FLAG_PERCENT = 0.25   # flag top 25% of students
F_BETA = 2.0                   # beta for F-beta score

# Threshold sweep range
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

# Baseline model defaults
LR_DEFAULTS = {
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RF_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CATBOOST_DEFAULTS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}

# Optuna
OPTUNA_N_TRIALS = 50
CV_FOLDS = 5

# Plotting
PLOT_DPI = 150
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (14, 6)
