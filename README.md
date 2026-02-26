# Student Wellness Early Warning and Triage System

A complete analytics and machine learning project for student wellness risk screening and support prioritization.

This project helps a university wellness team move from reactive support to earlier, data-informed follow-up. It uses student survey and lifestyle indicators to estimate depression risk, organize students into actionable support tiers, and support capacity planning through a Power BI dashboard.

**Important boundary:** This is a screening and prioritization system for support planning. It is not a diagnosis tool and it should not be used to make clinical claims.

## Project Summary

The project is built as an end-to-end workflow:

**SQL cleaning -> EDA -> Power BI -> ML pipeline -> Streamlit app**

It includes:
- Raw, cleaned, and ML-ready datasets
- SQL cleaning and feature engineering logic in notebooks
- EDA notebook for driver analysis and segmentation
- 4-page Power BI dashboard for executive decision support
- Modular ML pipeline for training, evaluation, explainability, and tuning
- Streamlit app for student screening and admin analytics
- Optional FastAPI backend endpoints for API-based integration

## Backstory

At a mid-sized university, the Student Wellness Office started noticing a pattern that did not show up in grades or attendance at first. Students were reporting constant stress, poor sleep, low motivation, unhealthy routines, and emotional exhaustion, but many never asked for help. By the time the university reacted through missed classes, complaints, or a visible crisis, the situation had often escalated beyond early support into urgent intervention.

The issue was not a lack of care. The issue was a lack of visibility.

The counseling team had limited capacity and could not proactively check in with everyone. Support still depended heavily on self-referrals, which missed students who were struggling quietly or felt hesitant to reach out. The team needed a practical way to surface risk earlier, prioritize outreach, and use limited support time where it matters most.

The university therefore ran a structured voluntary wellness pulse using anonymized survey responses plus basic lifestyle and workload indicators. The goal was to build an early warning and triage system that estimates who may be at elevated risk and helps the team decide who to review first.

## Problem Statement

Create a data-driven early warning and triage system that helps a mid-sized university identify and prioritize students who may be at elevated mental health risk so the Wellness Office can allocate limited support capacity effectively.

### What the system must solve

- **Early detection:** Surface risk before visible academic or behavioral breakdown happens.
- **Prioritization at scale:** Convert many responses into clear priority tiers so staff know who to contact first.
- **Explainability:** Show which signals are most strongly linked with higher risk so decisions are defensible.
- **Actionability:** Translate insights into workload planning, backlog, capacity, and timeline scenarios.
- **Consistency across segments:** Support slicing by age, gender, degree category, and city with stable measures.
- **Safety and boundaries:** Keep outputs framed as screening and support prioritization, not diagnosis.

## Target Audience

Core stakeholders:
- Student Wellness and Counseling Team (triage and follow-ups)
- Student Affairs and Administrative Leadership (resource planning and policy)

## Core Project Objective

Build a data-driven early warning and prioritization system using student survey and lifestyle data to:

- Estimate depression risk (binary 0 or 1) from academic, lifestyle, and stress-related factors
- Identify the strongest risk drivers and protective factors through EDA
- Segment risk across groups such as age, gender, degree category, and city
- Deliver an executive-friendly Power BI dashboard for quick decision support
- Prepare a robust ML-ready dataset and pipeline for screening support prioritization

## End-to-End Workflow

1. **SQL Clean and Feature Engineering** (`notebooks/load_clean.ipynb`)
2. **EDA** (`notebooks/eda.ipynb`)
3. **Power BI Dashboard** (`powerbi/Final Student Wellness.pbix`)
4. **ML Pipeline** (`ml_pipeline/`)
5. **Streamlit App** (`Home.py`, `pages/`) and optional FastAPI backend (`app/main.py`)

## Repository Structure

```text
.
├── Home.py                         # Streamlit landing page
├── pages/                          # Streamlit pages (student form + admin dashboard)
├── app/                            # FastAPI app and ML prediction utility
├── standalone_db.py                # SQLite helper used directly by Streamlit
├── datasets/
│   ├── raw/                        # Original survey CSV
│   ├── cleaned/                    # SQL-cleaned + engineered dataset
│   └── ml_ready/                   # Outlier-filtered dataset used for ML pipeline
├── notebooks/
│   ├── load_clean.ipynb            # SQL cleaning and feature engineering workflow
│   └── eda.ipynb                   # Exploratory data analysis
├── powerbi/
│   ├── Final Student Wellness.pbix
│   ├── final_powerbi_dashboard.pdf
│   └── power_bi_explanation.md     # Page-by-page dashboard walkthrough and speaking script
├── ml_pipeline/
│   ├── run_pipeline.py             # Full ML workflow entry point
│   ├── run_train.py                # Baseline model training
│   ├── run_evaluate.py             # Threshold and calibration analysis
│   ├── run_tune.py                 # Optuna tuning
│   ├── run_explain.py              # Explainability plots
│   ├── ML_PIPELINE_SCRIPT.md       # ML presentation and methodology script
│   └── ...                         # Modular config, preprocessing, models, evaluation, tuning
├── data_info.txt                   # Dataset reference card
├── requirements.txt                # App dependencies (pinned core versions)
└── README.md
```

## Dataset Versions and Counts

### 1) Raw dataset (`datasets/raw/Student Depression Dataset.csv`)
- Rows: **27,870**
- Columns: **15**
- Each row represents one student survey response.

### 2) Clean + engineered dataset (`datasets/cleaned/mental_health.csv`)
- Rows: **27,853**
- Columns: **19**
- Built after SQL cleaning and feature engineering
- **17 rows removed** during cleaning

### 3) ML-ready dataset (`datasets/ml_ready/student_depression_final_age_degree_outliers_removed.csv`)
- Rows: **27,129**
- Used for machine learning training and evaluation
- Built after applying the age-degree outlier exclusion layer

### Power BI row count note

Power BI Page 2 KPI uses the dashboard filter logic that excludes age-degree outliers. In the dashboard version used for presentation, the screened count is **27,197**. This is expected to differ from the cleaned dataset row count (27,853) because the dashboard applies an outlier filtering layer for KPI stability.

## Cleaning and Feature Engineering Logic

### Raw data cleanup (SQL stage)

Main cleaning steps include:
- Standardize text values (lowercase and trim spaces)
- Rename columns to snake_case
- Clean and standardize city names for reliable slicing
- Remove invalid rows where core survey values are missing or zero in key fields

Rows removed during cleaning (17 total) include records with invalid values such as:
- `academic_pressure = 0`
- `cgpa = 0`
- `study_satisfaction = 0`
- `financial_stress IS NULL`

### Engineered features

#### 1) `degree_category`
A stable grouping of degree names into:
- `school`
- `ug`
- `pg`
- `phd`
- `others`

#### 2) `red_flag` (0 to 7)
A count of triggered risk conditions. One point is added for each condition met:
- academic pressure >= 4
- financial stress >= 4
- work or study hours >= 10
- study satisfaction <= 2
- sleep duration = less than 5 hours
- dietary habits = unhealthy
- suicidal thoughts = yes

#### 3) `wellness`
Broad wellness bands derived from `red_flag`:

```sql
CASE
  WHEN red_flag <= 2 THEN 'high'
  WHEN red_flag <= 5 THEN 'moderate'
  ELSE 'low'
END
```

#### 4) `support_priority`
Operational triage tier used for action planning:
- `critical`
- `high priority`
- `moderate priority`
- `preventive high risk`
- `preventive watchlist`
- `stable`

This feature is the main action layer used by the dashboard and the app.

## EDA Summary

The EDA notebook validates the signal quality and helps explain risk patterns.

Key outcomes:
- Depression risk and triage burden by demographics and segments
- Strong monotonic rise in depression rate as `red_flag` increases from 0 to 7
- Academic pressure is the strongest observed risk driver in the dataset
- Financial stress is another strong driver
- Study satisfaction behaves as a protective factor
- Sleep and diet are practical lifestyle levers for prevention planning

## Power BI Dashboard (4 Pages)

The dashboard is designed for decision support and workload planning. It does not include an ML page. The ML work is documented separately and used for the screening application.

### Page 1 - Introduction
Explains the project purpose, triage meaning, signal types, and the non-diagnostic boundary.

### Page 2 - Executive Overview
Focuses on current burden and action queues:
- Students screened
- Depressed students
- Suicidal thoughts (yes)
- Depressed + suicidal overlap
- Support priority distribution
- Urgent vs prevention workload split
- Red flag ladder validation table

### Page 3 - Drivers
Focuses on factor associations and intervention levers:
- Driver ranking by depression rate gap
- Academic pressure and financial stress as major risk-linked factors
- Study satisfaction as a protective factor
- Sleep and diet as prevention-friendly levers

### Page 4 - Capacity and Action Plan
Translates burden into staffing and timeline planning:
- Scenario controls for staff count, follow-ups per staff, and target months
- Monthly capacity calculation
- Urgent and prevention backlog
- Months to clear urgent backlog
- Capacity gap and additional staff needed

See `powerbi/power_bi_explanation.md` for the complete page-by-page walkthrough and speaking script.

## ML Pipeline Overview

The ML pipeline is modular and designed for repeatable experimentation and deployment artifact generation.

### What it does
- Loads the ML-ready dataset
- Validates schema and performs stratified train/validation/test split
- Trains baseline models (Logistic Regression, Random Forest, CatBoost)
- Compares metrics and performs threshold analysis
- Runs calibration checks
- Generates explainability outputs (feature importance, permutation importance, PDP, SHAP)
- Runs ablation checks for leakage and proxy features
- Tunes models with Optuna
- Evaluates final model on held-out test set
- Saves deployment artifacts used by the app

### Primary metric choice
This project prioritizes **recall** and **F2 score** because missing a high-risk student is more costly than reviewing a false positive in a screening context.

### ML documentation and script
See `ml_pipeline/ML_PIPELINE_SCRIPT.md` for a complete, presentation-ready explanation of the ML workflow, metric choices, modeling decisions, and outputs.

## Streamlit App and API

### Streamlit app
- `Home.py` provides the landing page
- `pages/1_Student_Form.py` handles student screening submissions
- `pages/2_Admin_Dashboard.py` provides admin analytics and model metrics views
- `standalone_db.py` supports SQLite-based standalone deployment without a separate API server

### FastAPI backend (optional)
`app/main.py` exposes REST endpoints for form submission and admin analytics.

Common endpoints:
- `POST /api/submit`
- `GET /api/students`
- `GET /api/student/{id}`
- `GET /api/stats`
- `GET /api/insights`
- `GET /api/model-metrics`

## Dependency and Artifact Compatibility

This project now pins core serialization-sensitive packages to reduce model artifact loading issues across environments:
- `numpy==2.1.3`
- `scikit-learn==1.7.2`
- `joblib==1.4.2`

The app also includes a NumPy compatibility shim to improve loading behavior when artifacts were created in a slightly different environment. For deployment notes, see:

- `ml_pipeline/artifacts/models/DEPLOYMENT_COMPATIBILITY_NOTE.md`

## Quick Start

### Option A: Run the Streamlit app (standalone mode)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py
```

### Option B: Run the FastAPI backend

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run the ML pipeline

```bash
cd ml_pipeline
pip install -r requirements.txt
python run_pipeline.py
```

## Project Notes and Boundaries

- This system is meant for screening, triage, and support prioritization.
- It should support human review, not replace it.
- Outputs must be interpreted with context, especially for sensitive safety signals.
- The dashboard and app are designed to support early outreach, prevention planning, and capacity decisions.

## License and Usage

Use this project for portfolio, learning, internal demonstrations, and academic discussion with appropriate safety framing. If deployed in a real institutional setting, involve qualified mental health and policy stakeholders in governance and review.
