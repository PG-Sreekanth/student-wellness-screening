"""
standalone_db.py - Lightweight SQLite database helper for standalone Streamlit deployment.
No FastAPI needed. Works locally and on Streamlit Cloud.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# SQLAlchemy setup
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Use a local SQLite file; Streamlit Cloud will auto-create it
DB_PATH = Path(__file__).resolve().parent / "students.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ORM Model
class Student(Base):
    __tablename__ = "students"

    id                          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name                        = Column(String(100), nullable=False)
    gender                      = Column(String(20),  nullable=False)
    age                         = Column(Integer,     nullable=False)
    city                        = Column(String(50),  nullable=False)
    degree                      = Column(String(50),  nullable=False)
    academic_pressure           = Column(Integer,     nullable=False)
    cgpa                        = Column(Float,       nullable=False)
    study_satisfaction          = Column(Integer,     nullable=False)
    sleep_duration              = Column(String(30),  nullable=False)
    dietary_habits              = Column(String(20),  nullable=False)
    suicidal_thoughts           = Column(String(5),   nullable=False)
    work_study_hours            = Column(Float,       nullable=False)
    financial_stress            = Column(Integer,     nullable=False)
    family_history_mental_illness = Column(String(5), nullable=False)

    # Qualitative feedback
    notes                     = Column(String(500), nullable=True)

    # Derived
    degree_category           = Column(String(20), nullable=False)
    depression                = Column(Integer,    nullable=False)
    depression_probability    = Column(Float,      nullable=False)
    red_flag                  = Column(Integer,    nullable=False)
    wellness                  = Column(String(20), nullable=False)
    support_priority          = Column(String(30), nullable=False)
    submitted_at              = Column(DateTime,   default=lambda: datetime.now(timezone.utc))


# Create tables on first import
Base.metadata.create_all(bind=engine)

# Auto-migration: add missing columns to existing tables
def _migrate_db():
    """Add any new columns that don't exist yet in old databases.
    Uses the same SQLAlchemy engine to ensure consistent connection."""
    from sqlalchemy import inspect, text as sa_text
    inspector = inspect(engine)
    
    if "students" in inspector.get_table_names():
        existing_cols = {col["name"] for col in inspector.get_columns("students")}
        
        if "notes" not in existing_cols:
            with engine.begin() as conn:
                conn.execute(sa_text("ALTER TABLE students ADD COLUMN notes VARCHAR(500)"))
            print("Migration: added 'notes' column to students table.")

try:
    _migrate_db()
except Exception as e:
    print(f"Migration note: {e}")


# DB helper functions (used directly by Streamlit pages)

def save_student(data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a student record and return the saved dict."""
    db = SessionLocal()
    try:
        s = Student(**data)
        db.add(s)
        db.commit()
        db.refresh(s)
        return _to_dict(s)
    finally:
        db.close()


def get_all_students(
    support_priority: Optional[str] = None,
    depression: Optional[int] = None,
    wellness: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        q = db.query(Student)
        if support_priority:
            q = q.filter(Student.support_priority == support_priority.lower())
        if depression is not None:
            q = q.filter(Student.depression == depression)
        if wellness:
            q = q.filter(Student.wellness == wellness.lower())
        return [_to_dict(s) for s in q.order_by(Student.submitted_at.desc()).all()]
    finally:
        db.close()


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        s = db.query(Student).filter(Student.id == student_id).first()
        return _to_dict(s) if s else None
    finally:
        db.close()


def get_stats() -> Dict[str, Any]:
    from sqlalchemy import func
    db = SessionLocal()
    try:
        total = db.query(func.count(Student.id)).scalar() or 0
        if total == 0:
            return _empty_stats()

        depressed  = db.query(func.count(Student.id)).filter(Student.depression == 1).scalar() or 0
        suicidal   = db.query(func.count(Student.id)).filter(Student.suicidal_thoughts == "yes").scalar() or 0
        avg_rf     = db.query(func.avg(Student.red_flag)).scalar() or 0.0

        def pc(tier):
            return db.query(func.count(Student.id)).filter(Student.support_priority == tier).scalar() or 0

        def wc(tier):
            return db.query(func.count(Student.id)).filter(Student.wellness == tier).scalar() or 0

        return {
            "total_students":           total,
            "depressed_count":          depressed,
            "depressed_pct":            round(depressed / total * 100, 2),
            "suicidal_count":           suicidal,
            "suicidal_pct":             round(suicidal / total * 100, 2),
            "critical_count":           pc("critical"),
            "high_priority_count":      pc("high priority"),
            "moderate_priority_count":  pc("moderate priority"),
            "preventive_high_count":    pc("preventive high risk"),
            "preventive_watchlist_count": pc("preventive watchlist"),
            "stable_count":             pc("stable"),
            "avg_red_flag":             round(float(avg_rf), 2),
            "wellness_high_count":      wc("high"),
            "wellness_moderate_count":  wc("moderate"),
            "wellness_low_count":       wc("low"),
        }
    finally:
        db.close()


def get_insights() -> Optional[Dict[str, Any]]:
    import pandas as pd
    students = get_all_students()
    if not students:
        return None

    df = pd.DataFrame(students)

    # Depression by degree category
    deg_dep = df.groupby("degree_category")["depression"].agg(["mean", "count"]).reset_index()
    deg_dep.columns = ["degree_category", "depression_rate", "count"]
    deg_dep["depression_rate"] = (deg_dep["depression_rate"] * 100).round(2)

    # Depression by age group
    bins   = [0, 18, 21, 25, 30, 100]
    labels = ["≤18", "19-21", "22-25", "26-30", "30+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    age_dep = df.groupby("age_group", observed=True)["depression"].agg(["mean", "count"]).reset_index()
    age_dep.columns = ["age_group", "depression_rate", "count"]
    age_dep["depression_rate"] = (age_dep["depression_rate"] * 100).round(2)

    # Red flag ladder
    rf_ladder = df.groupby("red_flag")["depression"].agg(["mean", "count"]).reset_index()
    rf_ladder.columns = ["red_flag", "depression_rate", "count"]
    rf_ladder["depression_rate"] = (rf_ladder["depression_rate"] * 100).round(2)

    # Risk drivers
    drivers   = ["academic_pressure", "financial_stress", "study_satisfaction", "work_study_hours", "cgpa"]
    dep_means = df[df["depression"] == 1][drivers].mean().round(2).to_dict()   if (df["depression"] == 1).any() else {}
    nondep    = df[df["depression"] == 0][drivers].mean().round(2).to_dict()   if (df["depression"] == 0).any() else {}

    # Support priority dist
    sp_dist = df["support_priority"].value_counts().to_dict()

    # Gender depression
    gender_dep = df.groupby("gender")["depression"].agg(["mean", "count"]).reset_index()
    gender_dep.columns = ["gender", "depression_rate", "count"]
    gender_dep["depression_rate"] = (gender_dep["depression_rate"] * 100).round(2)

    # City depression (top 10)
    city_dep = df.groupby("city")["depression"].agg(["mean", "count"]).reset_index()
    city_dep.columns = ["city", "depression_rate", "count"]
    city_dep["depression_rate"] = (city_dep["depression_rate"] * 100).round(2)
    city_dep = city_dep.sort_values("count", ascending=False).head(10)

    return {
        "depression_by_degree":     deg_dep.to_dict(orient="records"),
        "depression_by_age":        age_dep.to_dict(orient="records"),
        "red_flag_ladder":          rf_ladder.to_dict(orient="records"),
        "risk_drivers_depressed":   dep_means,
        "risk_drivers_not_depressed": nondep,
        "support_priority_distribution": sp_dist,
        "depression_by_gender":     gender_dep.to_dict(orient="records"),
        "depression_by_city":       city_dep.to_dict(orient="records"),
    }


# Model metrics
def get_model_metrics() -> Dict[str, Any]:
    base_dir     = Path(__file__).resolve().parent / "ml_pipeline" / "artifacts"
    metrics_path = base_dir / "reports" / "final_metrics_summary.json"
    config_path  = base_dir / "models"  / "threshold_config.json"

    result = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = json.load(f)
        tp    = raw.get("test_tp", 0)
        fp    = raw.get("test_fp", 0)
        fn    = raw.get("test_fn", 0)
        tn    = raw.get("test_tn", 0)
        total = tp + fp + fn + tn
        result = {
            "accuracy":    round((tp + tn) / total * 100, 2) if total else 0,
            "recall":      round(raw.get("test_recall",    0) * 100, 2),
            "precision":   round(raw.get("test_precision", 0) * 100, 2),
            "f1_score":    round(raw.get("test_f1",        0) * 100, 2),
            "f2_score":    round(raw.get("test_f2",        0) * 100, 2),
            "roc_auc":     round(raw.get("test_roc_auc",   0) * 100, 2),
            "pr_auc":      round(raw.get("test_pr_auc",    0) * 100, 2),
            "brier_score": round(raw.get("test_brier",     0), 4),
            "threshold":   raw.get("test_threshold", 0.36),
            "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
            "test_total": total,
        }
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        result["model_name"]  = cfg.get("model_name",  "RF Tuned")
        result["feature_set"] = cfg.get("feature_set", "full")

    return result


# Internals
def _to_dict(s: Student) -> Dict[str, Any]:
    return {
        "id":                           s.id,
        "name":                         s.name,
        "gender":                       s.gender,
        "age":                          s.age,
        "city":                         s.city,
        "degree":                       s.degree,
        "academic_pressure":            s.academic_pressure,
        "cgpa":                         s.cgpa,
        "study_satisfaction":           s.study_satisfaction,
        "sleep_duration":               s.sleep_duration,
        "dietary_habits":               s.dietary_habits,
        "suicidal_thoughts":            s.suicidal_thoughts,
        "work_study_hours":             s.work_study_hours,
        "financial_stress":             s.financial_stress,
        "family_history_mental_illness": s.family_history_mental_illness,
        "notes":                        s.notes,
        "degree_category":              s.degree_category,
        "depression":                   s.depression,
        "depression_probability":       s.depression_probability,
        "red_flag":                     s.red_flag,
        "wellness":                     s.wellness,
        "support_priority":             s.support_priority,
        "submitted_at":                 str(s.submitted_at) if s.submitted_at else None,
    }


def _empty_stats() -> Dict[str, Any]:
    return {
        "total_students": 0, "depressed_count": 0, "depressed_pct": 0.0,
        "suicidal_count": 0, "suicidal_pct": 0.0,
        "critical_count": 0, "high_priority_count": 0, "moderate_priority_count": 0,
        "preventive_high_count": 0, "preventive_watchlist_count": 0, "stable_count": 0,
        "avg_red_flag": 0.0,
        "wellness_high_count": 0, "wellness_moderate_count": 0, "wellness_low_count": 0,
    }
