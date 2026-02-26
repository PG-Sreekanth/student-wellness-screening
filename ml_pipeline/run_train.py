"""
run_train.py - Train baseline models only.

Usage:
    cd ml_pipeline
    python run_train.py
"""
import sys
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from data.loader import load_data, stratified_split
from models.trainer import train_logistic_regression, train_random_forest, train_catboost
from evaluation.metrics import compare_models
from utils.logger import get_logger

logger = get_logger("run_train")


def main():
    logger.info("=== TRAINING BASELINE MODELS ===")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    feature_set = "A"

    lr_model, lr_prep, lr_proba = train_logistic_regression(
        X_train, y_train, X_val, feature_set=feature_set
    )
    rf_model, rf_prep, rf_proba = train_random_forest(
        X_train, y_train, X_val, feature_set=feature_set
    )
    cb_model, cb_prep, cb_proba = train_catboost(
        X_train, y_train, X_val, y_val, feature_set=feature_set
    )

    compare_models(
        {"LogisticRegression": lr_proba, "RandomForest": rf_proba, "CatBoost": cb_proba},
        y_val,
    )

    logger.info("Training complete. See artifacts/reports/model_comparison.csv")


if __name__ == "__main__":
    main()
