"""
2_Admin_Dashboard.py - Standalone admin analytics dashboard.
Reads directly from SQLite via standalone_db.py - no FastAPI needed.
Works on Streamlit Cloud.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from standalone_db import get_stats, get_all_students, get_insights, get_model_metrics

st.set_page_config(page_title="Admin Dashboard", page_icon="📊", layout="wide")

# Palette
PRIORITY_COLORS = {
    "critical":             "#dc2626",
    "high priority":        "#ea580c",
    "moderate priority":    "#ca8a04",
    "preventive high risk": "#7c3aed",
    "preventive watchlist": "#2563eb",
    "stable":               "#16a34a",
}
PRIORITY_ORDER = [
    "critical", "high priority", "moderate priority",
    "preventive high risk", "preventive watchlist", "stable",
]
WELLNESS_COLORS = {"Low": "#ef4444", "Moderate": "#f59e0b", "High": "#22c55e"}

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { background: #f4f5f7; }
    section[data-testid="stSidebar"] {
        background: #ffffff; border-right: 1px solid #eaedf0;
    }
    .block-container { padding: 2rem 2.5rem; max-width: 1200px; }
    h1,h2,h3,h4,h5,h6,.stMetricLabel,.stMetricValue {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .dash-header { margin-bottom: 32px; }
    .dash-header h1 {
        font-size: 1.55rem; font-weight: 700; color: #111827;
        margin: 0 0 4px 0; letter-spacing: -0.4px;
    }
    .dash-header p { color: #6b7280; font-size: 0.88rem; margin: 0; }

    .sec-title {
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.6px; color: #9ca3af; margin: 32px 0 12px 0;
        padding-bottom: 8px;
    }

    .kpi-row {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px; margin-bottom: 10px;
    }
    .kpi {
        background: #fff; border: 1px solid #eaedf0; border-radius: 10px;
        padding: 18px 16px; position: relative; overflow: hidden;
    }
    .kpi::before {
        content: ''; position: absolute; top: 0; left: 0;
        width: 3px; height: 100%; border-radius: 10px 0 0 10px;
    }
    .kpi.blue::before  { background: #2563eb; }
    .kpi.red::before   { background: #ef4444; }
    .kpi.amber::before { background: #f59e0b; }
    .kpi.green::before { background: #22c55e; }
    .kpi .kpi-label {
        font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; color: #9ca3af; margin-bottom: 5px;
    }
    .kpi .kpi-val { font-size: 1.5rem; font-weight: 700; color: #111827; line-height: 1; }
    .kpi .kpi-sub { font-size: 0.72rem; color: #9ca3af; margin-top: 3px; }

    .card {
        background: #fff; border: 1px solid #eaedf0; border-radius: 10px;
        padding: 20px 20px 12px 20px; margin-bottom: 8px;
    }
    .card-title { font-size: 0.8rem; font-weight: 600; color: #374151; margin: 0 0 4px 0; }
    .card-sub   { font-size: 0.7rem; color: #9ca3af; margin: 0 0 12px 0; }

    .mm-row {
        display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
    }
    .mm-box {
        background: #fff; border: 1px solid #eaedf0; border-radius: 8px;
        padding: 14px 10px; text-align: center;
    }
    .mm-box .mm-lbl {
        font-size: 0.6rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.4px; color: #9ca3af; margin-bottom: 4px;
    }
    .mm-box .mm-val { font-size: 1.15rem; font-weight: 700; color: #2563eb; }

    .cm-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
    .cm-c { border-radius: 6px; padding: 12px 8px; text-align: center; }
    .cm-c .cm-n { font-size: 1.1rem; font-weight: 700; color: #111827; }
    .cm-c .cm-l {
        font-size: 0.58rem; font-weight: 600; text-transform: uppercase;
        color: #9ca3af; letter-spacing: 0.3px; margin-top: 2px;
    }

    .pcard {
        background: #fff; border: 1px solid #eaedf0;
        border-radius: 10px; padding: 20px;
    }
    .pcard h3 { font-size: 1rem; font-weight: 700; color: #111827; margin: 0 0 4px 0; }
    .pcard .det { font-size: 0.82rem; color: #6b7280; margin: 2px 0; }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .explainer {
        background: #fff; border: 1px solid #eaedf0;
        border-radius: 8px; padding: 14px 16px;
    }
    .explainer h4 {
        font-size: 0.72rem; font-weight: 600; color: #374151;
        text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 6px 0;
    }
    .explainer p { font-size: 0.78rem; color: #6b7280; line-height: 1.65; margin: 0; }

    hr { border: none; border-top: 1px solid #eaedf0; margin: 28px 0; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #eaedf0; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; font-size: 0.82rem; font-weight: 500; color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important; border-bottom-color: #2563eb !important;
    }
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #eaedf0;
        border-radius: 8px; padding: 12px 14px;
    }
</style>
""", unsafe_allow_html=True)


# Helpers
def chart_layout(height=300, **kw):
    base = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#374151", family="Inter, -apple-system, sans-serif", size=12),
        height=height,
        margin=dict(l=14, r=14, t=28, b=14),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False),
    )
    base.update(kw)
    return base


def hex_to_rgba(hex_c, alpha=0.08):
    h = hex_c.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Page Header
st.markdown("""
<div class="dash-header">
    <h1>Admin Dashboard</h1>
    <p>Screening analytics, model performance, and student triage records</p>
</div>
""", unsafe_allow_html=True)

# Load data
stats = get_stats()

if stats["total_students"] == 0:
    st.info("No student submissions yet. Go to the **Student Screening** page to add records.")
    st.stop()


# Step 1: Kpi Row
urgent     = stats["critical_count"] + stats["high_priority_count"]
urgent_pct = round(urgent / stats["total_students"] * 100, 1) if stats["total_students"] else 0

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi blue">
        <div class="kpi-label">Total Screened</div>
        <div class="kpi-val">{stats['total_students']}</div>
        <div class="kpi-sub">students assessed</div>
    </div>
    <div class="kpi red">
        <div class="kpi-label">Depression Flagged</div>
        <div class="kpi-val">{stats['depressed_count']}</div>
        <div class="kpi-sub">{stats['depressed_pct']}% of total</div>
    </div>
    <div class="kpi amber">
        <div class="kpi-label">Suicidal Ideation</div>
        <div class="kpi-val">{stats['suicidal_count']}</div>
        <div class="kpi-sub">{stats['suicidal_pct']}% of total</div>
    </div>
    <div class="kpi green">
        <div class="kpi-label">Avg Red Flags</div>
        <div class="kpi-val">{stats['avg_red_flag']}</div>
        <div class="kpi-sub">out of 7</div>
    </div>
    <div class="kpi red">
        <div class="kpi-label">Urgent Cases</div>
        <div class="kpi-val">{urgent}</div>
        <div class="kpi-sub">{urgent_pct}% critical + high</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Step 2: Distributions
st.markdown('<p class="sec-title">Distributions</p>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="card">
        <p class="card-title">Support Priority Breakdown</p>
        <p class="card-sub">How students are triaged by the screening model</p>
    </div>
    """, unsafe_allow_html=True)

    sp_data = {
        "critical":             stats["critical_count"],
        "high priority":        stats["high_priority_count"],
        "moderate priority":    stats["moderate_priority_count"],
        "preventive high risk": stats["preventive_high_count"],
        "preventive watchlist": stats["preventive_watchlist_count"],
        "stable":               stats["stable_count"],
    }
    sp_rows = [{"Priority": k.title(), "Count": v, "key": k}
               for k, v in sp_data.items() if v > 0]
    if sp_rows:
        sp_df = pd.DataFrame(sp_rows)
        sp_df["order"] = sp_df["key"].map({p: i for i, p in enumerate(PRIORITY_ORDER)})
        sp_df = sp_df.sort_values("order")
        fig_sp = px.bar(
            sp_df, x="Count", y="Priority", orientation="h",
            color="key", color_discrete_map=PRIORITY_COLORS, text="Count",
        )
        layout = chart_layout(280)
        layout["yaxis"] = dict(autorange="reversed", showgrid=False)
        fig_sp.update_layout(**layout, showlegend=False)
        fig_sp.update_traces(textposition="outside", textfont=dict(size=11, color="#374151"))
        st.plotly_chart(fig_sp, use_container_width=True)

with col_b:
    st.markdown("""
    <div class="card">
        <p class="card-title">Wellness Tier</p>
        <p class="card-sub">Overall wellness distribution based on red flag count</p>
    </div>
    """, unsafe_allow_html=True)

    well_rows = [
        {"Tier": "High",     "Count": stats["wellness_high_count"]},
        {"Tier": "Moderate", "Count": stats["wellness_moderate_count"]},
        {"Tier": "Low",      "Count": stats["wellness_low_count"]},
    ]
    well_df = pd.DataFrame([r for r in well_rows if r["Count"] > 0])
    if not well_df.empty:
        fig_w = px.pie(
            well_df, names="Tier", values="Count", color="Tier",
            color_discrete_map=WELLNESS_COLORS, hole=0.52,
        )
        fig_w.update_layout(**chart_layout(280))
        fig_w.update_traces(
            textinfo="label+percent+value",
            textfont=dict(size=12),
            marker=dict(line=dict(color="#ffffff", width=2)),
        )
        st.plotly_chart(fig_w, use_container_width=True)


# Step 3: Insights Tabs
st.markdown("---")
st.markdown('<p class="sec-title">Insights &amp; Analytics</p>', unsafe_allow_html=True)

insights = get_insights()
if insights:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Red Flag Ladder", "By Degree", "By Age Group", "Risk Drivers", "By City"
    ])

    with tab1:
        rf_data = insights.get("red_flag_ladder", [])
        if rf_data:
            st.markdown("""
            <div class="card">
                <p class="card-title">Red Flag Score vs Depression Rate</p>
                <p class="card-sub">Validating the risk ladder - higher flags should correlate with higher depression rates</p>
            </div>
            """, unsafe_allow_html=True)
            rf_df = pd.DataFrame(rf_data)
            fig_rf = px.bar(
                rf_df, x="red_flag", y="depression_rate", text="count",
                color="depression_rate",
                color_continuous_scale=["#dcfce7","#fef9c3","#fee2e2","#ef4444"],
                labels={"red_flag": "Red Flag Score (0-7)", "depression_rate": "Depression Rate (%)"},
            )
            fig_rf.update_layout(**chart_layout(340),
                                 coloraxis_colorbar=dict(title="Dep %", thickness=12, len=0.6))
            fig_rf.update_traces(texttemplate="n=%{text}", textposition="outside",
                                 textfont=dict(size=10, color="#6b7280"))
            st.plotly_chart(fig_rf, use_container_width=True)

    with tab2:
        deg_data = insights.get("depression_by_degree", [])
        if deg_data:
            st.markdown("""
            <div class="card">
                <p class="card-title">Depression Rate by Degree Category</p>
                <p class="card-sub">Comparing risk across academic disciplines</p>
            </div>
            """, unsafe_allow_html=True)
            deg_df = pd.DataFrame(deg_data)
            fig_deg = px.bar(
                deg_df, x="degree_category", y="depression_rate", text="count",
                color="depression_rate",
                color_continuous_scale=["#dcfce7","#fef9c3","#fee2e2","#ef4444"],
                labels={"degree_category": "Degree Category", "depression_rate": "Depression Rate (%)"},
            )
            fig_deg.update_layout(**chart_layout(340))
            fig_deg.update_traces(texttemplate="n=%{text}", textposition="outside",
                                  textfont=dict(size=10, color="#6b7280"))
            st.plotly_chart(fig_deg, use_container_width=True)

    with tab3:
        age_data = insights.get("depression_by_age", [])
        if age_data:
            st.markdown("""
            <div class="card">
                <p class="card-title">Depression Rate by Age Group</p>
                <p class="card-sub">Identifying which age brackets carry higher risk</p>
            </div>
            """, unsafe_allow_html=True)
            age_df = pd.DataFrame(age_data)
            fig_age = px.bar(
                age_df, x="age_group", y="depression_rate", text="count",
                color="depression_rate",
                color_continuous_scale=["#dcfce7","#fef9c3","#fee2e2","#ef4444"],
                labels={"age_group": "Age Group", "depression_rate": "Depression Rate (%)"},
            )
            fig_age.update_layout(**chart_layout(340))
            fig_age.update_traces(texttemplate="n=%{text}", textposition="outside",
                                  textfont=dict(size=10, color="#6b7280"))
            st.plotly_chart(fig_age, use_container_width=True)

    with tab4:
        dep_d    = insights.get("risk_drivers_depressed", {})
        nondep_d = insights.get("risk_drivers_not_depressed", {})
        if dep_d and nondep_d:
            st.markdown("""
            <div class="card">
                <p class="card-title">Key Risk Drivers - Depressed vs Not Depressed</p>
                <p class="card-sub">Average values of academic and lifestyle indicators, grouped by depression status</p>
            </div>
            """, unsafe_allow_html=True)
            drivers = list(dep_d.keys())
            labels  = [d.replace("_", " ").title() for d in drivers]
            fig_dr  = go.Figure()
            fig_dr.add_trace(go.Bar(
                name="Depressed", x=labels,
                y=[dep_d.get(d, 0) for d in drivers],
                marker_color="#ef4444", marker_line_width=0,
            ))
            fig_dr.add_trace(go.Bar(
                name="Not Depressed", x=labels,
                y=[nondep_d.get(d, 0) for d in drivers],
                marker_color="#22c55e", marker_line_width=0,
            ))
            fig_dr.update_layout(
                **chart_layout(360), barmode="group",
                yaxis_title="Average Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.04,
                            xanchor="center", x=0.5, font=dict(size=11)),
            )
            st.plotly_chart(fig_dr, use_container_width=True)

    with tab5:
        city_data = insights.get("depression_by_city", [])
        if city_data:
            st.markdown("""
            <div class="card">
                <p class="card-title">Depression Rate by City (Top 10)</p>
                <p class="card-sub">Geographic distribution of mental health risk among screened students</p>
            </div>
            """, unsafe_allow_html=True)
            city_df = pd.DataFrame(city_data)
            city_df["city"] = city_df["city"].str.title()
            fig_city = px.bar(
                city_df.sort_values("depression_rate", ascending=True),
                x="depression_rate", y="city", orientation="h",
                text="count",
                color="depression_rate",
                color_continuous_scale=["#dcfce7","#fef9c3","#fee2e2","#ef4444"],
                labels={"city": "City", "depression_rate": "Depression Rate (%)"},
            )
            fig_city.update_layout(**chart_layout(360))
            fig_city.update_traces(texttemplate="n=%{text}", textposition="outside",
                                   textfont=dict(size=10, color="#6b7280"))
            st.plotly_chart(fig_city, use_container_width=True)
            st.caption("City is one of the ML model's input features. "
                       "Higher depression rates in certain cities may reflect local academic pressures.")
        else:
            st.info("Submit more records with varied cities to see geographic insights.")


# Step 4: Model Performance
st.markdown("---")
st.markdown('<p class="sec-title">Model Validation - Test Set Performance</p>', unsafe_allow_html=True)

metrics = get_model_metrics()
if metrics:
    threshold = metrics.get("threshold", 0.36)
    st.markdown(f"""
    <div class="mm-row">
        <div class="mm-box"><div class="mm-lbl">Accuracy</div><div class="mm-val">{metrics.get('accuracy', 0)}%</div></div>
        <div class="mm-box"><div class="mm-lbl">Recall</div><div class="mm-val">{metrics.get('recall', 0)}%</div></div>
        <div class="mm-box"><div class="mm-lbl">Precision</div><div class="mm-val">{metrics.get('precision', 0)}%</div></div>
        <div class="mm-box"><div class="mm-lbl">F1 Score</div><div class="mm-val">{metrics.get('f1_score', 0)}%</div></div>
        <div class="mm-box"><div class="mm-lbl">ROC-AUC</div><div class="mm-val">{metrics.get('roc_auc', 0)}%</div></div>
        <div class="mm-box"><div class="mm-lbl">Threshold</div><div class="mm-val">{threshold}</div></div>
    </div>
    """, unsafe_allow_html=True)

    tp = metrics.get("test_tp", 0)
    fp = metrics.get("test_fp", 0)
    fn = metrics.get("test_fn", 0)
    tn = metrics.get("test_tn", 0)
    if tp + fp + fn + tn > 0:
        cm_c1, cm_c2 = st.columns([1, 2])
        with cm_c1:
            st.markdown(f"""
            <div style="margin-top:14px;">
                <p style="font-size:0.68rem; font-weight:600; text-transform:uppercase;
                          letter-spacing:0.4px; color:#9ca3af; margin-bottom:6px;">
                    Confusion Matrix
                </p>
                <div class="cm-wrap">
                    <div class="cm-c" style="background:#f0fdf4;">
                        <div class="cm-n">{tn}</div><div class="cm-l">True Neg</div>
                    </div>
                    <div class="cm-c" style="background:#fef2f2;">
                        <div class="cm-n">{fp}</div><div class="cm-l">False Pos</div>
                    </div>
                    <div class="cm-c" style="background:#fff7ed;">
                        <div class="cm-n">{fn}</div><div class="cm-l">False Neg</div>
                    </div>
                    <div class="cm-c" style="background:#f0fdf4;">
                        <div class="cm-n">{tp}</div><div class="cm-l">True Pos</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cm_c2:
            st.markdown(f"""
            <div class="explainer" style="margin-top:14px;">
                <h4>How to Read These Metrics</h4>
                <p>
                    Evaluated on <b>{tp+fp+fn+tn:,}</b> held-out test samples.
                    The model correctly identifies <b>{metrics.get('recall',0)}%</b> of
                    depressed students (recall), missing only <b>{fn}</b> cases.
                    The trade-off is <b>{metrics.get('precision',0)}%</b> precision -
                    some non-depressed students get flagged, which is acceptable
                    in screening where missing a case is costlier than over-flagging.
                </p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Model metrics not found. Run the ML pipeline to generate `final_metrics_summary.json`.")


# Step 5: Student Records
st.markdown("---")
st.markdown('<p class="sec-title">Student Records</p>', unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    filter_priority = st.selectbox("Filter by Priority", ["All"] + [p.title() for p in PRIORITY_ORDER])
with fc2:
    filter_depression = st.selectbox("Filter by Depression", ["All", "Flagged", "Not Flagged"])
with fc3:
    filter_wellness = st.selectbox("Filter by Wellness", ["All", "High", "Moderate", "Low"])

kw = {}
if filter_priority   != "All":     kw["support_priority"] = filter_priority.lower()
if filter_depression == "Flagged": kw["depression"]       = 1
elif filter_depression == "Not Flagged": kw["depression"] = 0
if filter_wellness   != "All":     kw["wellness"]         = filter_wellness.lower()

students = get_all_students(**kw)

if students:
    df = pd.DataFrame(students)
    display_cols = ["id", "name", "age", "gender", "city", "degree", "degree_category",
                    "depression", "depression_probability", "red_flag",
                    "wellness", "support_priority", "submitted_at"]
    display_cols = [c for c in display_cols if c in df.columns]
    dfd = df[display_cols].copy()

    dfd["depression"]            = dfd["depression"].map({1: "Yes", 0: "No"})
    dfd["depression_probability"] = (dfd["depression_probability"] * 100).round(1).astype(str) + "%"
    dfd["support_priority"]      = dfd["support_priority"].str.title()
    dfd["wellness"]              = dfd["wellness"].str.title()
    dfd["gender"]                = dfd["gender"].str.title()
    dfd["degree"]                = dfd["degree"].str.title()
    dfd["city"]                  = dfd["city"].str.title()
    dfd["degree_category"]       = dfd["degree_category"].str.upper()

    dfd.columns = ["ID", "Name", "Age", "Gender", "City", "Degree", "Category",
                   "Depression", "Risk %", "Red Flags", "Wellness", "Priority", "Submitted"]

    st.dataframe(dfd, use_container_width=True, hide_index=True,
                 height=min(420, 40 + len(dfd) * 35))
    st.caption(f"Showing {len(dfd)} record(s)")
else:
    st.info("No students match the selected filters.")


# Step 6: Student Detail View
if students:
    st.markdown("---")
    st.markdown('<p class="sec-title">Student Detail View</p>', unsafe_allow_html=True)

    names_map = {s["id"]: f"#{s['id']}  {s['name']}" for s in students}
    sel_id    = st.selectbox("Select a student", list(names_map.keys()),
                              format_func=lambda x: names_map[x])

    if sel_id:
        stu = next((s for s in students if s["id"] == sel_id), None)
        if stu:
            pr       = stu["support_priority"]
            pr_color = PRIORITY_COLORS.get(pr, "#6b7280")

            dc1, dc2 = st.columns([1, 2])
            with dc1:
                st.markdown(f"""
                <div class="pcard">
                    <h3>{stu['name']}</h3>
                    <p class="det">{stu['age']} years · {stu['gender'].title()}</p>
                    <p class="det">📍 {stu['city'].title()}</p>
                    <p class="det">{stu['degree'].title()} ({stu['degree_category'].upper()})</p>
                    <hr style="border-color:#eaedf0; margin:12px 0;">
                    <p style="font-size:0.6rem; font-weight:600; text-transform:uppercase;
                              letter-spacing:0.4px; color:#9ca3af; margin-bottom:5px;">
                        Support Priority
                    </p>
                    <span class="badge"
                          style="color:{pr_color}; background:{hex_to_rgba(pr_color)};">
                        {pr.upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with dc2:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Depression Risk", f"{stu['depression_probability']*100:.1f}%")
                m2.metric("Red Flags",       f"{stu['red_flag']} / 7")
                m3.metric("Wellness",         stu["wellness"].title())
                m4.metric("CGPA",            f"{stu['cgpa']:.1f}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Academic Pressure",  f"{stu['academic_pressure']} / 5")
                m6.metric("Financial Stress",   f"{stu['financial_stress']} / 5")
                m7.metric("Study Satisfaction", f"{stu['study_satisfaction']} / 5")
                m8.metric("Work Hours",         f"{stu['work_study_hours']}h / day")

            if stu.get("notes"):
                st.markdown("---")
                st.markdown(f"""
                <div class="explainer">
                    <h4>Clinical Notes / Observations</h4>
                    <p>{stu['notes']}</p>
                </div>
                """, unsafe_allow_html=True)
