"""
models.py - SQLAlchemy ORM model for students.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime, timezone

from app.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)

    # Survey / form fields
    gender = Column(String(20), nullable=False)
    age = Column(Integer, nullable=False)
    city = Column(String(50), nullable=False)
    degree = Column(String(50), nullable=False)
    academic_pressure = Column(Integer, nullable=False)       # 1-5
    cgpa = Column(Float, nullable=False)
    study_satisfaction = Column(Integer, nullable=False)       # 1-5
    sleep_duration = Column(String(30), nullable=False)
    dietary_habits = Column(String(20), nullable=False)
    suicidal_thoughts = Column(String(5), nullable=False)      # yes / no
    work_study_hours = Column(Float, nullable=False)
    financial_stress = Column(Integer, nullable=False)         # 1-5
    family_history_mental_illness = Column(String(5), nullable=False)  # yes / no

    # Derived / computed
    degree_category = Column(String(20), nullable=False)
    depression = Column(Integer, nullable=False)               # 0 or 1
    depression_probability = Column(Float, nullable=False)
    red_flag = Column(Integer, nullable=False)                 # 0-7
    wellness = Column(String(20), nullable=False)              # high / moderate / low
    support_priority = Column(String(30), nullable=False)      # 6 tiers

    submitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
