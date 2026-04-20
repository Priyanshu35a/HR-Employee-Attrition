import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* 1) Light Mode Background & Global Fonts */
    [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Inter', sans-serif !important;
        color: #1f2937 !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Target main text colors to ensure visibility on light bg */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText {
        color: #1f2937 !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #ec4899, #8b5cf6, #3b82f6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-animation 6s ease infinite;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        color: #4b5563 !important;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* GLASSMORPHISM CARDS - LIGHT THEME */
    [data-testid="metric-container"], .stExpander, div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(209, 213, 219, 0.5);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.5);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
    }
    
    /* Button */
    .stButton > button {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #8b5cf6 !important;
        color: #8b5cf6 !important;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-weight: 600;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px rgba(139, 92, 246, 0.1);
    }
    .stButton > button:hover {
        background: #8b5cf6 !important;
        color: #fff !important;
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.3);
    }
    
    .section-title {
        color: #374151;
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(236, 72, 153, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 1rem 0;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        border-bottom-color: #ec4899 !important;
    }
    
    /* Make inputs look glass-like too */
    .stSelectbox div[data-baseweb="select"], .stSlider div[data-baseweb="slider"] {
        background: transparent !important;
    }
    
    /* Enhance data tables for light mode */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(12px);
        border-radius: 8px;
        border: 1px solid rgba(209, 213, 219, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ─── 1. Load Model & Data ────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    model = joblib.load('attrition_model.pkl')
    df_raw = pd.read_csv('HR-Employee-Attrition.csv')

    # Same drop as original
    df = df_raw.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
    y = (df['Attrition'] == 'Yes').astype(int)
    X_raw = df.drop(columns=['Attrition'])

    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    expected_columns = X_encoded.columns

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_encoded)

    return model, pca, expected_columns, X_raw, X_pca, y, df_raw

model, pca, expected_columns, X_raw, X_pca, y_true, df_full = load_components()


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"> HR Employee Attrition Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict, explore, and understand employee attrition using Machine Learning.</div>', unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " Prediction",
    " EDA & Charts",
    " Model Performance",
    " About Project",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Employee Attrition Risk Predictor")
    st.write("Fill in the employee details below and click **Predict** to assess attrition risk.")
    st.divider()

    # Input layout – 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title"> Personal Info</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", X_raw['Gender'].unique())
        marital_status = st.selectbox("Marital Status", X_raw['MaritalStatus'].unique())
        distance_from_home = st.slider("Distance from Home (miles)", 1, 30, 5)
        education = st.slider("Education Level (1–5)", 1, 5, 3)

    with col2:
        st.markdown('<div class="section-title"> Job Details</div>', unsafe_allow_html=True)
        job_role = st.selectbox("Job Role", sorted(X_raw['JobRole'].unique()))
        department = st.selectbox("Department", X_raw['Department'].unique())
        job_level = st.slider("Job Level (1–5)", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)

    with col3:
        st.markdown('<div class="section-title"> Compensation & Time</div>', unsafe_allow_html=True)
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 20, 3)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        over_time = st.radio("Works Overtime?", ["Yes", "No"], horizontal=True)

    st.divider()

    # Predict button – centered
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_clicked = st.button(" Predict Attrition Risk", type="primary", use_container_width=True)

    if predict_clicked:
        # Build input dict with median/mode defaults
        input_dict = {}
        for col in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                input_dict[col] = X_raw[col].median()
            else:
                input_dict[col] = X_raw[col].mode()[0]

        # Override with user selections
        input_dict.update({
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'JobRole': job_role,
            'Department': department,
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'TotalWorkingYears': total_working_years,
            'OverTime': over_time,
        })

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=False)
        proba = model.predict_proba(input_aligned)[0] if hasattr(model, "predict_proba") else [0, 0]
        
        # Custom threshold logic for Imbalanced Attrition Risk (0.3 instead of 0.5)
        is_high_risk = (proba[1] >= 0.3) if hasattr(model, "predict_proba") else (model.predict(input_aligned)[0] == 1)

        st.divider()
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            if is_high_risk:
                st.error("###  HIGH RISK — Likely to Leave")
                st.write("Consider discussing **career progression**, **salary hike**, or **reducing overtime load**.")
            else:
                st.success("###  LOW RISK — Likely to Stay")
                st.write("This employee appears satisfied and engaged with their current role.")

            if proba is not None:
                risk_pct = round(proba[1] * 100, 1)
                st.metric("Attrition Probability", f"{risk_pct}%")

        with res_col2:
            st.info("** Evaluated Parameters**")
            summary = {
                "Age": age, "Gender": gender, "Job Role": job_role,
                "Department": department, "Monthly Income": f"${monthly_income:,}",
                "Job Satisfaction": job_satisfaction, "Overtime": over_time,
                "Years at Company": years_at_company, "Work-Life Balance": work_life_balance,
                "Distance from Home": f"{distance_from_home} mi",
            }
            s1, s2 = st.columns(2)
            items = list(summary.items())
            for k, v in items[:5]:
                s1.metric(k, v)
            for k, v in items[5:]:
                s2.metric(k, v)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA & CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    # Data preview
    st.markdown("####  Data Preview")
    st.dataframe(df_full.head(100), use_container_width=True, height=280)

    with st.expander(" View Summary Statistics"):
        st.dataframe(df_full.describe(), use_container_width=True)

    st.divider()

    # Charts row 1
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title"> Overall Attrition Rate</div>', unsafe_allow_html=True)
        attrition_counts = df_full['Attrition'].value_counts().reset_index()
        attrition_counts.columns = ['Attrition', 'Count']
        fig_donut = px.pie(
            attrition_counts, names='Attrition', values='Count',
            hole=0.5,
            color='Attrition',
            color_discrete_map={'Yes': '#EF4444', 'No': '#22C55E'},
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title"> Monthly Income vs. Attrition</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df_full, x='Attrition', y='MonthlyIncome',
            color='Attrition',
            color_discrete_map={'Yes': '#EF4444', 'No': '#22C55E'},
            points='outliers',
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title"> Attrition by Job Role</div>', unsafe_allow_html=True)
        role_attr = (
            df_full.groupby(['JobRole', 'Attrition'])
            .size().reset_index(name='Count')
        )
        fig_bar = px.bar(
            role_attr, x='JobRole', y='Count', color='Attrition',
            barmode='group',
            color_discrete_map={'Yes': '#EF4444', 'No': '#22C55E'},
        )
        fig_bar.update_layout(xaxis_tickangle=-35, margin=dict(t=20, b=80))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title"> Age Distribution by Attrition</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_full, x='Age', color='Attrition',
            nbins=25, barmode='overlay', opacity=0.75,
            color_discrete_map={'Yes': '#EF4444', 'No': '#22C55E'},
        )
        fig_hist.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Metrics")

    # Encode X_raw the same way it was encoded during training
    X_test_encoded = pd.get_dummies(X_raw, drop_first=True)
    # Ensure all expected columns are present
    X_test_encoded = X_test_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Use a custom 0.3 cutoff threshold to adjust for extreme class imbalance 
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_encoded)[:, 1]
        y_pred = (probs >= 0.3).astype(int)
    else:
        y_pred = model.predict(X_test_encoded)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)

    # Top-level metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(" Accuracy",  f"{acc:.1%}")
    m2.metric(" Precision", f"{prec:.1%}")
    m3.metric(" Recall",    f"{rec:.1%}")
    m4.metric(" F1 Score",  f"{f1:.1%}")

    st.divider()

    perf_col1, perf_col2 = st.columns(2)

    # Confusion Matrix
    with perf_col1:
        st.markdown('<div class="section-title"> Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)
        labels = ['No Attrition (0)', 'Attrition (1)']
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale='Blues',
            aspect='auto',
        )
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Classification Report
    with perf_col2:
        st.markdown('<div class="section-title"> Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_true, y_pred,
                                       target_names=['No Attrition', 'Attrition'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

    st.divider()

    # PCA 2D Scatter
    st.markdown('<div class="section-title"> PCA Component Space — Actual Labels</div>', unsafe_allow_html=True)
    st.caption("Shows how well the two PCA components separate the two classes.")
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Attrition': y_true.map({0: 'No', 1: 'Yes'}),
    })
    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='Attrition',
        opacity=0.6,
        color_discrete_map={'Yes': '#EF4444', 'No': '#22C55E'},
        title='PCA — 2D Projection of Employee Features',
    )
    fig_pca.update_traces(marker=dict(size=4))
    fig_pca.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig_pca, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – ABOUT PROJECT
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
##  About This Project

Welcome to the **HR Employee Attrition & Retention Dashboard**. This powerful analytical tool goes beyond standard reporting to provide actionable, predictive insights using advanced Machine Learning. 

---

###  The Core Mission

In today's competitive landscape, employee retention is critical. Replacing an employee often costs anywhere from 50% to 200% of their annual salary. Our dashboard empowers HR professionals and organizational leaders to proactively tackle attrition by understanding the *why* behind employee turnover.

By utilizing this tool, your organization can:
- **Identify Flight Risks Early:** Preemptively flag high-performing individuals who might be considering leaving.
- **Deep-Dive into Drivers:** Understand whether salary, commute, work-life balance, or management is driving dissatisfaction.
- **Formulate Retention Strategies:** Make data-backed decisions on salary bumps, role changes, or wellness initiatives to retain top talent.

---

###  Dataset Overview

This project is built atop the renowned **IBM HR Analytics Employee Attrition & Performance** dataset, originally devised by IBM data scientists. 

**Dataset Highlights:**
- **Volume:** 1,470 highly detailed employee records.
- **Breadth:** 35 comprehensive features encompassing multiple facets of an employee's professional and personal profile.
- **Target Variable:** `Attrition` (`Yes` or `No`), formulating a robust binary classification challenge.

**Key Feature Categories:**
| Category | Evaluated Dimensions | Impact Hypothesis |
|---|---|---|
| **Demographics** | Age, Gender, MaritalStatus, DistanceFromHome | Commute stress or life-stage needs often trigger departures. |
| **Job Setup** | JobRole, Department, JobLevel, JobInvolvement | Stagnation or lack of involvement strongly correlates with turnover. |
| **Compensation** | MonthlyIncome, HourlyRate, StockOptionLevel | Financial misalignment is a primary, solvable driver for quitting. |
| **Tenure & History**| YearsAtCompany, YearsInCurrentRole, TotalWorkingYears | Strong managerial relationships and growth paths foster loyalty. |
| **Well-being** | WorkLifeBalance, EnvironmentSatisfaction, OverTime | Chronic overtime and poor life balance are classic recipes for burnout. |

---

###  Machine Learning Pipeline Architecture

To achieve accurate forecasting, the data traverses a rigorous, state-of-the-art modeling pipeline:

**1. Data Cleansing & Feature Selection**
We meticulously purge noise (e.g., zero-variance columns like `EmployeeCount`, `StandardHours`) and identifiers (`EmployeeNumber`) that carry no predictive weight, preventing data leakage and overfitting.

**2. Categorical Encoding (One-Hot)**
Non-numeric data (such as `JobRole`, `Department`, `BusinessTravel`) is transformed into machine-readable binary logic. We utilize `drop_first=True` to sidestep the dummy-variable trap, ensuring optimal mathematical stability.

**3. Dimensionality Reduction via PCA (Principal Component Analysis)**
Operating on 50+ expanded dimensions is computationally intense and susceptible to the curse of dimensionality. We distil these myriad factors down into **Principal Components**. This achieves:
- Elimination of severe multi-collinearity (where features like Job Level and Income are tightly coupled).
- The ability to plot complex employee personas on an intuitive 2D plane (visible in our Model Performance tab).

**4. Predictive Modeling Engine**
Our core classifier, serialized as `attrition_model.pkl`, relies on historical patterns to score active employees. By feeding real-time user inputs through this identical pipeline, the model computes a live "Probability of Attrition," enabling targeted, strategic HR interventions.

---

###  Technology Stack & Tooling

We built this robust analytical product using industry-leading open-source technologies:

| Domain | Technology / Library | Role in Project |
|---|---|---|
| **Interactive UI** | `Streamlit` | Orchestrates the responsive, Glassmorphism-themed frontend. |
| **Data Engine** | `Pandas` & `NumPy` | Core data wrangling, aggregation, and mathematical operations. |
| **Machine Learning**| `Scikit-Learn` | Houses the PCA, preprocessing, and classification algorithms. |
| **Visual Analytics**| `Plotly Express` | Drives the interactive charts, donuts, and scatter plots. |
| **Model Persistence**| `Joblib` | Serializes and loads the trained ML pipeline safely for production. |

---
""")

    st.markdown("""
<div style="text-align:center; padding: 1.5rem; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(8px); border: 1px solid rgba(209, 213, 219, 0.5); border-radius: 12px; margin-top:2rem; color:#4b5563; font-size:0.95rem; box-shadow: 0 4px 20px rgba(31, 38, 135, 0.05);">
    <b>Developed by Priyanshu Yadav</b><br>
    <span style="font-size: 0.85rem; color: #6b7280;">IBM HR Attrition Capstone Project</span>
</div>
""", unsafe_allow_html=True)