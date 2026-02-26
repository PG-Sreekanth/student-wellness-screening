"""
schemas.py - Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class StudentSubmission(BaseModel):
    """Form data submitted by a student."""
    name: str = Field(..., min_length=1, max_length=100)
    gender: str
    age: int = Field(..., ge=15, le=60)
    city: str
    degree: str
    academic_pressure: int = Field(..., ge=1, le=5)
    cgpa: float = Field(..., ge=0.0, le=10.0)
    study_satisfaction: int = Field(..., ge=1, le=5)
    sleep_duration: str
    dietary_habits: str
    suicidal_thoughts: str
    work_study_hours: float = Field(..., ge=0.0, le=24.0)
    financial_stress: int = Field(..., ge=1, le=5)
    family_history_mental_illness: str


class StudentResponse(BaseModel):
    """Full student record returned by the API."""
    id: int
    name: str
    gender: str
    age: int
    city: str
    degree: str
    degree_category: str
    academic_pressure: int
    cgpa: float
    study_satisfaction: int
    sleep_duration: str
    dietary_habits: str
    suicidal_thoughts: str
    work_study_hours: float
    financial_stress: int
    family_history_mental_illness: str
    depression: int
    depression_probability: float
    red_flag: int
    wellness: str
    support_priority: str
    submitted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    """Aggregated statistics for the admin dashboard."""
    total_students: int
    depressed_count: int
    depressed_pct: float
    suicidal_count: int
    suicidal_pct: float
    critical_count: int
    high_priority_count: int
    moderate_priority_count: int
    preventive_high_count: int
    preventive_watchlist_count: int
    stable_count: int
    avg_red_flag: float
    wellness_high_count: int
    wellness_moderate_count: int
    wellness_low_count: int
