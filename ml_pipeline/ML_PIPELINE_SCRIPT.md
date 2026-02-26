# ML Pipeline Script and Methodology Guide

This file is a presentation-ready and implementation-ready explanation of the machine learning workflow used in the Student Wellness Early Warning and Triage project.

It is written to match the actual code structure inside `ml_pipeline/` and to explain the logic in a clear, human tone.

## ML Goal in This Project

The ML pipeline supports **screening and prioritization**, not diagnosis.

The model estimates whether a student is likely to be in the `depression = 1` class based on survey, lifestyle, and stress-related features. The output is then combined with the engineered `red_flag` score to produce actionable support tiers such as critical, high priority, and preventive watchlist.

## Why ML Is Used Here

The dashboard explains burden, drivers, and capacity planning at an aggregate level.

The ML pipeline adds the ability to score new student submissions consistently so the support team can:
- identify potentially high-risk cases earlier
- prioritize reviews at scale
- keep screening decisions repeatable
- support a triage workflow with measurable thresholds

## Dataset Used for ML

### Input dataset
The ML pipeline uses the **ML-ready dataset** after outlier filtering:

- Path: `datasets/ml_ready/student_depression_final_age_degree_outliers_removed.csv`
- Rows: **27,129** (based on the project artifact package)

### Why not use the raw dataset directly?
Because the pipeline needs:
- cleaned categorical values
- stable feature names
- engineered risk indicators (`red_flag`, `degree_category`, triage labels for downstream logic)
- outlier filtering for more stable learning and evaluation

## Target and Features

### Target variable
- `depression` (binary classification target: `0` or `1`)

### Feature groups used by the ML code
The pipeline supports feature-set contracts in `config/feature_schema.py`.
The default workflow uses **Feature Set A**, which favors interpretability.

Typical feature categories include:
- Demographics: gender, age, city, degree category
- Academic signals: academic pressure, study satisfaction, CGPA
- Lifestyle signals: sleep duration, dietary habits, work/study hours
- Stress and background signals: financial stress, family history, suicidal thoughts

## Metric Choice (Why F2 and Recall Matter)

### Business context
In a triage workflow, missing a student who may need support is more costly than reviewing a false positive.

### Metric priorities
- **Primary focus:** Recall and F2 score
- **Secondary checks:** Precision, F1, ROC-AUC, PR-AUC, accuracy

### Why F2 instead of only F1?
F2 gives more weight to recall. That better reflects the project goal of early support prioritization.

The pipeline still reports a range of metrics so trade-offs remain visible.

## Train / Validation / Test Strategy

The pipeline uses a **stratified split** so the class distribution is preserved across subsets.

Typical purpose of each split:
- **Train set:** fit preprocessors and models
- **Validation set:** compare models, tune thresholds, check calibration, run explainability and ablations
- **Test set:** final one-time evaluation of the selected model

This keeps model selection separate from final reporting and reduces leakage risk.

## Preprocessing Strategy

The pipeline uses modular preprocessing in `preprocessing/`.

### Why preprocessing matters here
The dataset contains mixed feature types:
- numeric
- ordinal scales (1 to 5)
- categorical strings
- binary yes/no values

A single preprocessing approach for all columns would be less reliable and harder to interpret.

### What the code does
The pipeline builds different preprocessors depending on model type:

- **Logistic Regression / Random Forest preprocessors**
  - map ordered categories to ordinal values where appropriate
  - map yes/no values to binary integers
  - encode categorical features where needed
  - keep numeric features in a model-friendly format

- **CatBoost preprocessor**
  - preserves categorical handling in a CatBoost-friendly way
  - tracks categorical feature indices used by CatBoost

### Why this is a strong design choice
Different models benefit from different preprocessing requirements. The pipeline keeps that logic explicit and reusable.

## Baseline Models Trained

The code trains three baseline models:

1. **Logistic Regression**
   - strong baseline
   - interpretable coefficients (after encoding)
   - fast to train

2. **Random Forest**
   - strong tabular baseline
   - handles nonlinear relationships
   - useful for feature importance and stable deployment in this project

3. **CatBoost**
   - excellent for mixed tabular data with categorical features
   - often strong predictive performance on survey-style datasets
   - useful for SHAP-based explanations in the project

## Full ML Workflow (Mapped to `run_pipeline.py`)

Below is the exact flow implemented by the full pipeline runner.

### Stage 1: Load data and split
- Load ML-ready dataset
- Validate schema
- Create stratified train/validation/test split

### Stage 2: Train baseline models
- Train Logistic Regression
- Train Random Forest
- Train CatBoost
- Generate validation probabilities for each model

### Stage 3: Compare baseline models
- Compute common metrics at a default threshold (typically 0.5)
- Save comparison outputs to `artifacts/reports`

### Stage 4: Threshold analysis
- Sweep probability thresholds (for example, 0.10 to 0.90)
- Track recall, precision, F1, F2, and number of students flagged
- Select thresholds based on screening priorities (F2-max and recall-friendly options)

### Stage 5: Calibration analysis
- Generate calibration curves
- Compare original vs calibrated probabilities where applicable
- Review Brier score improvements

### Stage 6: Feature importance and permutation importance
- Generate model feature importance charts
- Run permutation importance on validation data
- Save plots for reporting and review

### Stage 7: PDP / ICE and SHAP explainability
- Partial dependence plots (PDP) and ICE curves for selected features
- SHAP explanations for tree-based model behavior

This supports stakeholder explainability and helps verify that the model behavior is directionally sensible.

### Stage 8: Ablation checks (leakage / proxy risk checks)
- Remove or alter feature groups in controlled experiments
- Compare performance changes
- Identify suspicious performance jumps or proxy-heavy dependence

### Stage 9: Hyperparameter tuning (Optuna)
- Run tuning trials for candidate models
- Use cross-validation to reduce overfitting to one split
- Save best parameters and tuning summaries

### Stage 10: Final model selection
- Select the best model based on validation evidence and screening priorities
- Refit or finalize artifacts with the selected configuration

### Stage 11: Held-out test evaluation and artifact save
- Evaluate exactly once on the test set
- Save final model, preprocessor, threshold config, and metrics JSON
- Export deployment-ready artifacts used by the Streamlit app

## Why the Final Deployment Uses Thresholds (Not Just 0.5)

A default probability threshold of 0.5 is not always the best choice for triage.

In screening workflows, a lower threshold can improve recall and help catch more students who may need follow-up. The trade-off is more false positives, which is acceptable when the system is positioned as a review aid rather than a diagnosis tool.

That is why the pipeline saves a threshold configuration file and the app uses it during prediction.

## Explainability Strategy (What to Say in a Review)

The project uses multiple explainability methods because no single method is enough:

- **Feature importance** for quick model-level ranking
- **Permutation importance** for validation-based importance checks
- **PDP / ICE** to understand response patterns across feature ranges
- **SHAP** for richer local and global interpretation on tree-based models

This combination helps communicate the model responsibly to non-technical stakeholders.

## Leakage and Governance Notes

This is an early warning screening system, so governance matters.

The pipeline includes ablation checks and split discipline to reduce leakage risk, but operational governance is still required:
- keep diagnosis claims out of the product
- ensure human review for sensitive flags
- monitor drift over time
- retrain and revalidate periodically
- document threshold choices and rationale

## How to Run the ML Pipeline

### Full pipeline
```bash
cd ml_pipeline
pip install -r requirements.txt
python run_pipeline.py
```

### Individual stages
```bash
python run_train.py
python run_evaluate.py
python run_explain.py
python run_tune.py
```

## What Gets Saved (Artifacts)

The pipeline writes artifacts under `ml_pipeline/artifacts/`, including:
- `models/` (final model and preprocessor joblib files)
- `plots/` (threshold curves, calibration curves, importance plots, SHAP, PDP)
- `reports/` (model comparisons, tuning summaries, metrics outputs)
- `logs/` (pipeline logs and tuning logs)

These artifacts are the bridge between experimentation and the deployed app.

## How the App Uses ML Output

The app (`app/ml_predictor.py`) loads:
- final model artifact
- final preprocessor artifact
- threshold config

It predicts depression probability, applies the saved threshold, and then combines the prediction with `red_flag` to compute:
- `wellness`
- `support_priority`

This keeps the app aligned with the project dataset logic and Power BI triage labels.

## Short Presentation Script (ML Section)

Here is a concise script you can use while presenting the ML part of the project:

We use machine learning here for screening support prioritization, not diagnosis. The model predicts the probability that a student falls into the depression risk class using demographic, academic, lifestyle, and stress-related signals. We train and compare Logistic Regression, Random Forest, and CatBoost on the ML-ready dataset after cleaning and outlier filtering. Because this is a triage use case, we prioritize recall and F2 score so we do not miss students who may need support. We then run threshold analysis, calibration checks, explainability, and leakage ablation checks before selecting the final model. The final model, preprocessor, and threshold are saved as deployment artifacts and used directly in the Streamlit screening app.

## Final Reminder

The ML pipeline improves consistency and scalability of screening, but it does not replace counselor judgment. It is one part of a broader decision-support workflow that includes EDA findings, dashboard monitoring, and human review.
