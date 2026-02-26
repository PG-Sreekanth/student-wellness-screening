"""
run_evaluate.py - Evaluate trained models: threshold analysis and calibration.

Usage:
    cd ml_pipeline
    python run_evaluate.py
"""
import sys
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from data.loader import load_data, stratified_split
from models.trainer import train_logistic_regression, train_random_forest, train_catboost
from evaluation.metrics import compare_models
from evaluation.threshold import run_threshold_analysis
from evaluation.calibration import plot_calibration_curve
from utils.logger import get_logger

logger = get_logger("run_evaluate")


def main():
    logger.info("=== MODEL EVALUATION ===")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    feature_set = "A"

    # Train models
    _, _, lr_proba = train_logistic_regression(X_train, y_train, X_val, feature_set)
    _, _, rf_proba = train_random_forest(X_train, y_train, X_val, feature_set)
    _, _, cb_proba = train_catboost(X_train, y_train, X_val, y_val, feature_set)

    model_probas = {
        "LogisticRegression": lr_proba,
        "RandomForest": rf_proba,
        "CatBoost": cb_proba,
    }

    # Comparison
    compare_models(model_probas, y_val)

    # Threshold analysis
    for name, proba in model_probas.items():
        run_threshold_analysis(y_val, proba, name)

    # Calibration
    for name, proba in model_probas.items():
        plot_calibration_curve(y_val, proba, name)

    logger.info("Evaluation complete. Check artifacts/plots and artifacts/reports.")


if __name__ == "__main__":
    main()
