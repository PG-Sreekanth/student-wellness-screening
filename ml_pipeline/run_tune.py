"""
run_tune.py - Hyperparameter tuning with Optuna.

Usage:
    cd ml_pipeline
    python run_tune.py
"""
import sys
import warnings

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from data.loader import load_data, stratified_split
from tuning.optuna_tuner import run_all_tuning
from utils.logger import get_logger

logger = get_logger("run_tune")


def main():
    logger.info("=== HYPERPARAMETER TUNING ===")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    results = run_all_tuning(X_train, y_train, feature_set="A", n_trials=30)

    for name, res in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Best F2 (CV): {res['best_score']:.4f}")
        logger.info(f"  Best params: {res['best_params']}")

    logger.info("Tuning complete. Check artifacts/logs for trial details.")


if __name__ == "__main__":
    main()
