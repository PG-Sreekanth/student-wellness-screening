"""
main.py - FastAPI application with REST endpoints.

Endpoints:
  POST /api/submit         - Submit a student screening form
  GET  /api/students       - List all students (with optional filters)
  GET  /api/student/{id}   - Get a single student
  GET  /api/stats          - Aggregated statistics for admin dashboard
  GET  /api/model-metrics  - ML model test-set performance metrics
"""
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import engine, get_db, Base
from app.models import Student
from app.schemas import StudentSubmission, StudentResponse, StatsResponse
from app.ml_predictor import predict

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Student Depression Screening API",
    description="API for student mental wellness screening and triage",
    version="1.0.0",
)

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Student Depression Screening API is running."}


def _json_safe(value):
    """Convert NaN/Inf values to JSON-safe None recursively."""
    import math

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if np is not None and isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@app.post("/api/submit", response_model=StudentResponse)
def submit_student(submission: StudentSubmission, db: Session = Depends(get_db)):
    """Submit a student screening form. Runs ML prediction and saves to DB."""
    # Run prediction
    form_data = submission.model_dump()
    prediction = predict(form_data)

    # Create DB record
    student = Student(
        name=submission.name,
        gender=submission.gender.strip().lower(),
        age=submission.age,
        city=submission.city.strip().lower(),
        degree=submission.degree.strip().lower(),
        academic_pressure=submission.academic_pressure,
        cgpa=submission.cgpa,
        study_satisfaction=submission.study_satisfaction,
        sleep_duration=submission.sleep_duration.strip().lower(),
        dietary_habits=submission.dietary_habits.strip().lower(),
        suicidal_thoughts=submission.suicidal_thoughts.strip().lower(),
        work_study_hours=submission.work_study_hours,
        financial_stress=submission.financial_stress,
        family_history_mental_illness=submission.family_history_mental_illness.strip().lower(),
        degree_category=prediction["degree_category"],
        depression=prediction["depression"],
        depression_probability=prediction["depression_probability"],
        red_flag=prediction["red_flag"],
        wellness=prediction["wellness"],
        support_priority=prediction["support_priority"],
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return student


@app.get("/api/students", response_model=List[StudentResponse])
def list_students(
    support_priority: Optional[str] = Query(None, description="Filter by support priority tier"),
    depression: Optional[int] = Query(None, description="Filter by depression (0 or 1)"),
    wellness: Optional[str] = Query(None, description="Filter by wellness tier"),
    db: Session = Depends(get_db),
):
    """List all students with optional filters."""
    query = db.query(Student)

    if support_priority:
        query = query.filter(Student.support_priority == support_priority.lower())
    if depression is not None:
        query = query.filter(Student.depression == depression)
    if wellness:
        query = query.filter(Student.wellness == wellness.lower())

    return query.order_by(Student.submitted_at.desc()).all()


@app.get("/api/student/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, db: Session = Depends(get_db)):
    """Get a single student by ID."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get aggregated statistics for the admin dashboard."""
    total = db.query(func.count(Student.id)).scalar() or 0

    if total == 0:
        return StatsResponse(
            total_students=0, depressed_count=0, depressed_pct=0.0,
            suicidal_count=0, suicidal_pct=0.0,
            critical_count=0, high_priority_count=0, moderate_priority_count=0,
            preventive_high_count=0, preventive_watchlist_count=0, stable_count=0,
            avg_red_flag=0.0,
            wellness_high_count=0, wellness_moderate_count=0, wellness_low_count=0,
        )

    depressed = db.query(func.count(Student.id)).filter(Student.depression == 1).scalar() or 0
    suicidal = db.query(func.count(Student.id)).filter(Student.suicidal_thoughts == "yes").scalar() or 0
    avg_rf = db.query(func.avg(Student.red_flag)).scalar() or 0.0

    # Support priority counts
    def priority_count(tier: str) -> int:
        return db.query(func.count(Student.id)).filter(Student.support_priority == tier).scalar() or 0

    # Wellness counts
    def wellness_count(tier: str) -> int:
        return db.query(func.count(Student.id)).filter(Student.wellness == tier).scalar() or 0

    return StatsResponse(
        total_students=total,
        depressed_count=depressed,
        depressed_pct=round(depressed / total * 100, 2) if total > 0 else 0.0,
        suicidal_count=suicidal,
        suicidal_pct=round(suicidal / total * 100, 2) if total > 0 else 0.0,
        critical_count=priority_count("critical"),
        high_priority_count=priority_count("high priority"),
        moderate_priority_count=priority_count("moderate priority"),
        preventive_high_count=priority_count("preventive high risk"),
        preventive_watchlist_count=priority_count("preventive watchlist"),
        stable_count=priority_count("stable"),
        avg_red_flag=round(float(avg_rf), 2),
        wellness_high_count=wellness_count("high"),
        wellness_moderate_count=wellness_count("moderate"),
        wellness_low_count=wellness_count("low"),
    )


@app.get("/api/insights")
def get_insights(db: Session = Depends(get_db)):
    """Get detailed insights for the admin dashboard charts."""
    students = db.query(Student).all()
    if not students:
        return {"message": "No data available", "data": {}}

    import pandas as pd

    records = [{
        "age": s.age, "gender": s.gender, "degree_category": s.degree_category,
        "academic_pressure": s.academic_pressure, "financial_stress": s.financial_stress,
        "study_satisfaction": s.study_satisfaction, "work_study_hours": s.work_study_hours,
        "sleep_duration": s.sleep_duration, "dietary_habits": s.dietary_habits,
        "suicidal_thoughts": s.suicidal_thoughts, "depression": s.depression,
        "red_flag": s.red_flag, "wellness": s.wellness,
        "support_priority": s.support_priority, "cgpa": s.cgpa,
        "family_history_mental_illness": s.family_history_mental_illness,
    } for s in students]
    df = pd.DataFrame(records)

    # Depression rate by degree category
    deg_dep = df.groupby("degree_category")["depression"].agg(["mean", "count"]).reset_index()
    deg_dep.columns = ["degree_category", "depression_rate", "count"]
    deg_dep["depression_rate"] = (deg_dep["depression_rate"] * 100).round(2)

    # Depression by age group
    bins = [0, 18, 21, 25, 30, 100]
    labels = ["≤18", "19-21", "22-25", "26-30", "30+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    age_dep = df.groupby("age_group", observed=True)["depression"].agg(["mean", "count"]).reset_index()
    age_dep.columns = ["age_group", "depression_rate", "count"]
    age_dep["depression_rate"] = (age_dep["depression_rate"] * 100).round(2)

    # Red flag ladder
    rf_ladder = df.groupby("red_flag")["depression"].agg(["mean", "count"]).reset_index()
    rf_ladder.columns = ["red_flag", "depression_rate", "count"]
    rf_ladder["depression_rate"] = (rf_ladder["depression_rate"] * 100).round(2)

    # Risk driver comparison (depressed vs not)
    drivers = ["academic_pressure", "financial_stress", "study_satisfaction",
               "work_study_hours", "cgpa"]
    dep_subset = df[df["depression"] == 1][drivers]
    nondep_subset = df[df["depression"] == 0][drivers]
    dep_means = dep_subset.mean().round(2).to_dict() if not dep_subset.empty else {c: None for c in drivers}
    nondep_means = nondep_subset.mean().round(2).to_dict() if not nondep_subset.empty else {c: None for c in drivers}

    # Support priority distribution
    sp_dist = df["support_priority"].value_counts().to_dict()

    # Gender distribution
    gender_dep = df.groupby("gender")["depression"].agg(["mean", "count"]).reset_index()
    gender_dep.columns = ["gender", "depression_rate", "count"]
    gender_dep["depression_rate"] = (gender_dep["depression_rate"] * 100).round(2)

    return _json_safe({
        "depression_by_degree": deg_dep.to_dict(orient="records"),
        "depression_by_age": age_dep.to_dict(orient="records"),
        "red_flag_ladder": rf_ladder.to_dict(orient="records"),
        "risk_drivers_depressed": dep_means,
        "risk_drivers_not_depressed": nondep_means,
        "support_priority_distribution": sp_dist,
        "depression_by_gender": gender_dep.to_dict(orient="records"),
    })


@app.get("/api/model-metrics")
def get_model_metrics():
    """Return ML model test-set performance metrics."""
    import json
    from pathlib import Path

    metrics_path = Path(__file__).resolve().parent.parent / "ml_pipeline" / "artifacts" / "reports" / "final_metrics_summary.json"
    config_path = Path(__file__).resolve().parent.parent / "ml_pipeline" / "artifacts" / "models" / "threshold_config.json"

    result = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        tp = raw.get("test_tp", 0)
        fp = raw.get("test_fp", 0)
        fn = raw.get("test_fn", 0)
        tn = raw.get("test_tn", 0)
        total = tp + fp + fn + tn
        result = {
            "accuracy": round((tp + tn) / total * 100, 2) if total else 0,
            "recall": round(raw.get("test_recall", 0) * 100, 2),
            "precision": round(raw.get("test_precision", 0) * 100, 2),
            "f1_score": round(raw.get("test_f1", 0) * 100, 2),
            "f2_score": round(raw.get("test_f2", 0) * 100, 2),
            "roc_auc": round(raw.get("test_roc_auc", 0) * 100, 2),
            "pr_auc": round(raw.get("test_pr_auc", 0) * 100, 2),
            "brier_score": round(raw.get("test_brier", 0), 4),
            "threshold": raw.get("test_threshold", 0.36),
            "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
            "test_total": total,
        }

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        result["model_name"] = cfg.get("model_name", "Unknown")
        result["feature_set"] = cfg.get("feature_set", "Unknown")

    return result
