"""
Home.py - Landing page for Student Wellness Screening.
Streamlit Cloud entry point.
"""
import streamlit as st

st.set_page_config(
    page_title="Student Wellness Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background: #f8f9fa; color: #1a1a2e; }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8e8e8;
    }
    .block-container { padding-top: 2.5rem; max-width: 960px; }
    h1, h2, h3, h4 { font-family: 'Inter', -apple-system, sans-serif !important; }

    .hero {
        text-align: center;
        padding: 72px 24px 56px;
    }
    .hero h1 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.8px;
        margin: 0 0 12px 0;
        line-height: 1.15;
    }
    .hero p {
        color: #6b7280;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 540px;
        margin: 0 auto;
    }

    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 40px;
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 28px 24px;
        transition: box-shadow .2s ease;
    }
    .feature-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,.06); }
    .feature-card .icon {
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; margin-bottom: 16px; font-weight: 600;
    }
    .feature-card h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.95rem; font-weight: 600; color: #1a1a2e;
        margin: 0 0 8px 0;
    }
    .feature-card p {
        color: #6b7280; font-size: 0.85rem; line-height: 1.55; margin: 0;
    }

    .footer-note {
        text-align: center; margin-top: 48px; padding: 24px;
        color: #9ca3af; font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Student Wellness Screening</h1>
    <p>
        An ML-powered screening tool that identifies students who may benefit
        from mental health support, using validated risk indicators and
        a tuned Random Forest classifier.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features">
    <div class="feature-card">
        <div class="icon" style="background:#eff6ff; color:#3b82f6;">S</div>
        <h3>Student Screening</h3>
        <p>
            Submit a screening form with academic, lifestyle, and health indicators.
            Receive an instant risk assessment with depression probability and priority level.
        </p>
    </div>
    <div class="feature-card">
        <div class="icon" style="background:#f0fdf4; color:#22c55e;">A</div>
        <h3>Analytics Dashboard</h3>
        <p>
            View aggregate insights, support priority distributions, wellness metrics,
            and drill into individual student profiles with detailed breakdowns.
        </p>
    </div>
    <div class="feature-card">
        <div class="icon" style="background:#fefce8; color:#eab308;">M</div>
        <h3>Model Validation</h3>
        <p>
            Review test-set performance metrics including accuracy, recall, precision,
            ROC-AUC, and a confusion matrix with plain-language interpretation.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Use the sidebar to navigate between Student Screening and Admin Dashboard.<br>
    This tool is for screening purposes only and is not a clinical diagnosis.
</div>
""", unsafe_allow_html=True)
