"""
1_Student_Form.py - Standalone student screening form.
Calls ML model directly - no FastAPI needed.
Works on Streamlit Cloud.
"""
import sys
from pathlib import Path
import streamlit as st

# Make sure ml_pipeline custom transformers are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR   = ROOT_DIR / "ml_pipeline"
for p in [str(ROOT_DIR), str(ML_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.ml_predictor import predict
from standalone_db import save_student

st.set_page_config(page_title="Student Screening", page_icon="📋", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background: #f8f9fa; color: #1a1a2e; }
    section[data-testid="stSidebar"] {
        background: #ffffff; border-right: 1px solid #e8e8e8;
    }
    .block-container { padding-top: 2rem; max-width: 860px; }
    h1, h2, h3, h4 { font-family: 'Inter', -apple-system, sans-serif !important; }

    .page-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 18px; margin-bottom: 28px;
    }
    .page-header h1 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 1.6rem; font-weight: 700; color: #1a1a2e;
        margin: 0 0 6px 0; letter-spacing: -0.4px;
    }
    .page-header p { color: #6b7280; font-size: 0.9rem; margin: 0; }

    .section-label {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.7px; color: #6b7280; margin: 24px 0 12px 0;
    }

    .result-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 28px 32px; margin-top: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .result-card.elevated  { border-left: 3px solid #ef4444; }
    .result-card.low-risk  { border-left: 3px solid #22c55e; }
    .result-card h2 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 1.15rem; font-weight: 700; color: #1a1a2e;
        margin: 0 0 20px 0;
    }

    .metric-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    }
    .metric-item { text-align: center; }
    .metric-item .label {
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; color: #9ca3af; margin-bottom: 6px;
    }
    .metric-item .value {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
    }

    .priority-badge {
        display: inline-block; padding: 5px 14px; border-radius: 6px;
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.4px;
    }

    .explanation-box {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 18px 22px; margin-top: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .explanation-box h4 {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #1a1a2e; font-size: 0.82rem; font-weight: 600;
        margin: 0 0 8px 0;
    }
    .explanation-box p {
        color: #6b7280; font-size: 0.8rem; line-height: 1.6; margin: 0;
    }

    .disclaimer {
        background: #eff6ff; border: 1px solid #dbeafe;
        border-radius: 8px; padding: 14px 18px; margin-top: 14px;
        color: #6b7280; font-size: 0.78rem; line-height: 1.5;
    }

    /* city note box */
    .city-note {
        background: #fefce8; border: 1px solid #fde68a;
        border-radius: 6px; padding: 8px 12px; margin-top: 6px;
        font-size: 0.75rem; color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DEGREE_OPTIONS = [
    "B.Tech", "M.Tech", "B.Sc", "M.Sc", "B.Com", "M.Com", "BCA", "MCA", 
    "BA", "MA", "BBA", "MBA", "Diploma", "B.Ed", "M.Ed",
    "MBBS", "MD", "BDS", "LL.B", "LL.M", "B.Arch", "M.Arch", 
    "B.Pharm", "M.Pharm", "Ph.D", "Class 12", "Class 10", "Other (Type Below)"
]

CITY_OPTIONS = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Bhopal", "Patna", "Indore", "Nagpur", "Chandigarh",
    "Kochi", "Guwahati", "Coimbatore", "Visakhapatnam", "Varanasi",
    "Other (Type Below)",
]

PRIORITY_STYLES = {
    "critical":             ("CRITICAL",             "#dc2626", "rgba(220,38,38,.08)"),
    "high priority":        ("HIGH PRIORITY",        "#ea580c", "rgba(234,88,12,.08)"),
    "moderate priority":    ("MODERATE PRIORITY",    "#ca8a04", "rgba(202,138,4,.08)"),
    "preventive high risk": ("PREVENTIVE HIGH RISK", "#7c3aed", "rgba(124,58,237,.08)"),
    "preventive watchlist": ("PREVENTIVE WATCHLIST", "#2563eb", "rgba(37,99,235,.08)"),
    "stable":               ("STABLE",               "#16a34a", "rgba(22,163,74,.08)"),
}

RESULT_TITLES = {
    "critical":             ("Immediate Attention Required",         "elevated"),
    "high priority":        ("High Risk - Action Needed",            "elevated"),
    "moderate priority":    ("Moderate Risk - Follow Up Recommended","elevated"),
    "preventive watchlist": ("Watchlist - Monitor and Follow Up",    "low-risk"),
    "preventive high risk": ("Preventive Alert - Risk Factors Present","elevated"),
    "stable":               ("Low Risk - No Immediate Concern",      "low-risk"),
}

# Page Header
st.markdown("""
<div class="page-header">
    <h1>Student Screening</h1>
    <p>Complete the form below to assess a student's mental wellness risk profile</p>
</div>
""", unsafe_allow_html=True)

# Form
with st.form("screening_form"):

    # Personal Information
    st.markdown('<p class="section-label">Personal Information</p>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        name = st.text_input("Full Name", placeholder="Enter student name")
    with pc2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with pc3:
        age = st.number_input("Age", min_value=15, max_value=60, value=20)

    pc4, pc5 = st.columns(2)
    with pc4:
        city_selection = st.selectbox("City", CITY_OPTIONS)
        if city_selection == "Other (Type Below)":
            city_manual = st.text_input("Enter your city", placeholder="e.g., Mysore, Surat, Tiruchirappalli...")
        else:
            city_manual = ""
    with pc5:
        degree_selection = st.selectbox("Degree", DEGREE_OPTIONS)
        if degree_selection == "Other (Type Below)":
            degree_manual = st.text_input("Enter your degree", placeholder="e.g., B.Sc Physics, B.Arch, Diploma...")
        else:
            degree_manual = ""

    # Academic & Workload
    st.markdown('<p class="section-label">Academic &amp; Workload</p>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
    with ac2:
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
    with ac3:
        financial_stress = st.slider("Financial Stress", 1, 5, 3)

    ac4, ac5 = st.columns(2)
    with ac4:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    with ac5:
        work_study_hours = st.number_input(
            "Avg. Daily Study/Work Hours", min_value=0.0, max_value=24.0, value=6.0, step=0.5
        )

    # Lifestyle & Health
    st.markdown('<p class="section-label">Lifestyle &amp; Health</p>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
        )
    with lc2:
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    lc3, lc4 = st.columns(2)
    with lc3:
        suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["No", "Yes"])
    with lc4:
        family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

    # Qualitative Notes
    st.markdown('<p class="section-label">Additional Context</p>', unsafe_allow_html=True)
    notes = st.text_area(
        "Notes/Qualitative Observations", 
        placeholder="Describe any specific behavioral observations or qualitative feedback...",
        help="Optional: This data is saved for clinical reference but not used by the ML model."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "Submit for Screening", use_container_width=True, type="primary"
    )


# Handle Submission
if submitted:
    # Resolve city & degree
    city   = city_manual.strip()   if city_selection   == "Other (Type Below)" else city_selection
    degree = degree_manual.strip() if degree_selection == "Other (Type Below)" else degree_selection

    if not name.strip():
        st.error("Please enter the student's name.")
    elif not city:
        st.error("Please enter a city name in the text box below the City dropdown.")
    elif degree_selection == "Other (Type Below)" and not degree:
        st.error("Please enter your degree name in the text box below the Degree dropdown.")
    else:
        form_data = {
            "name":                          name.strip(),
            "gender":                        gender,
            "age":                           age,
            "city":                          city,
            "degree":                        degree,
            "academic_pressure":             academic_pressure,
            "cgpa":                          cgpa,
            "study_satisfaction":            study_satisfaction,
            "sleep_duration":                sleep_duration,
            "dietary_habits":                dietary_habits,
            "suicidal_thoughts":             suicidal_thoughts,
            "work_study_hours":              work_study_hours,
            "financial_stress":              financial_stress,
            "family_history_mental_illness": family_history,
        }

        with st.spinner("Running ML prediction..."):
            try:
                # ML prediction (direct call, no API needed)
                result = predict(form_data)

                prob       = result["depression_probability"]
                red_flags  = result["red_flag"]
                wellness   = result["wellness"]
                priority   = result["support_priority"]
                dep_flag   = result["depression"]
                deg_cat    = result["degree_category"]

                # Save to SQLite
                db_record = {**form_data, **result}
                db_record["city"]              = city.strip().lower()
                db_record["gender"]            = gender.strip().lower()
                db_record["degree"]            = degree.strip().lower()
                db_record["sleep_duration"]    = sleep_duration.strip().lower()
                db_record["dietary_habits"]    = dietary_habits.strip().lower()
                db_record["suicidal_thoughts"] = suicidal_thoughts.strip().lower()
                db_record["family_history_mental_illness"] = family_history.strip().lower()
                # remove keys not in DB model
                db_record.pop("name", None)
                saved = save_student({
                    "name":                           name.strip(),
                    "gender":                         db_record["gender"],
                    "age":                            age,
                    "city":                           db_record["city"],
                    "degree":                         db_record["degree"],
                    "academic_pressure":              academic_pressure,
                    "cgpa":                           cgpa,
                    "study_satisfaction":             study_satisfaction,
                    "sleep_duration":                 db_record["sleep_duration"],
                    "dietary_habits":                 db_record["dietary_habits"],
                    "suicidal_thoughts":              db_record["suicidal_thoughts"],
                    "work_study_hours":               work_study_hours,
                    "financial_stress":               financial_stress,
                    "family_history_mental_illness":  db_record["family_history_mental_illness"],
                    "degree_category":                deg_cat,
                    "depression":                     dep_flag,
                    "depression_probability":         prob,
                    "red_flag":                       red_flags,
                    "wellness":                       wellness,
                    "support_priority":               priority,
                    "notes":                          notes.strip() if notes else None,
                })

                # Result card
                title, card_class = RESULT_TITLES.get(priority, ("Screening Complete", "low-risk"))
                p_label, p_color, p_bg = PRIORITY_STYLES.get(
                    priority, (priority.upper(), "#6b7280", "rgba(107,114,128,.08)")
                )
                badge_html = (
                    f'<span class="priority-badge" '
                    f'style="color:{p_color}; background:{p_bg};">{p_label}</span>'
                )

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h2>{title}</h2>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="label">Depression Probability</div>
                            <div class="value">{prob*100:.1f}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="label">Red Flags</div>
                            <div class="value">{red_flags} / 7</div>
                        </div>
                        <div class="metric-item">
                            <div class="label">Wellness Tier</div>
                            <div class="value">{wellness.upper()}</div>
                        </div>
                        <div class="metric-item">
                            <div class="label">Support Priority</div>
                            {badge_html}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # City note - inform user about city's role in ML
                city_display = city.title()
                st.markdown(f"""
                <div class="city-note">
                    📍 <b>City recorded:</b> {city_display} &nbsp;·&nbsp;
                    City is passed to the ML model as a feature. Its influence depends on
                    patterns learned during training. The primary drivers are
                    academic pressure, financial stress, sleep, and suicidal ideation.
                </div>
                """, unsafe_allow_html=True)

                # Explanation
                st.markdown(f"""
                <div class="explanation-box">
                    <h4>What does "Depression Probability" mean?</h4>
                    <p>
                        The depression probability ({prob*100:.1f}%) is the confidence score from
                        our tuned Random Forest model, indicating how likely the student's
                        profile matches patterns associated with depression in the training data.
                        A threshold of <b>36%</b> is used for classification - scores at or above
                        this threshold are flagged as elevated risk. This is <b>not</b> a clinical
                        diagnosis; it is a statistical screening signal calibrated to maximize
                        recall (catch rate of 97.2%) while maintaining acceptable precision.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer">
                    <b>Important:</b> This is a screening tool, not a diagnosis.
                    If a student is in distress, please refer them to your institution's
                    counseling services or a qualified mental health professional.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info(
                    "Make sure the model artifacts exist at:\n"
                    "`ml_pipeline/artifacts/models/final_model_rf_tuned.joblib`"
                )
