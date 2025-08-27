ackage (regression)
- Also gives actionable recommendations.

How to run (locally)
1) pip install streamlit scikit-learn pandas numpy
2) streamlit run app.py

How to plug real models
- Place your trained models as:
  models/
    placement_clf.pkl   (Binary classifier: 0/1)
    next_cgpa_reg.pkl   (Regressor: float)
    package_reg.pkl     (Regressor: float, e.g., LPA)
- The app will auto-load these if present. If not, it uses a simple heuristic baseline.

Notes
- Replace the heuristic with real models for production.
- Feature engineering must match the training pipeline.
"""

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

# Optional: only if scikit-learn models are available
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Student Career Prediction", page_icon="🎯", layout="centered")

# -------------------------
# Feature schema
# -------------------------
FEATURES = [
    "current_cgpa",           # float 0-10
    "attendance_pct",         # float 0-100
    "backlogs",               # int >=0
    "projects_count",         # int >=0
    "internships_count",      # int >=0
    "tech_skill_score",       # float 0-100
    "soft_skill_score",       # float 0-100
    "certifications_count",   # int >=0
    "applied_companies",      # int >=0
    # One-hot for branch (CSE, ECE, ME, CE, IT, Other)
    "branch_CSE",
    "branch_ECE",
    "branch_ME",
    "branch_CE",
    "branch_IT",
    "branch_Other",
]

BRANCHES = ["CSE", "ECE", "ME", "CE", "IT", "Other"]

@dataclass
class Models:
    placement_clf: object | None
    next_cgpa_reg: object | None
    package_reg: object | None


def load_models() -> Models:
    """Load models from models/*.pkl if available; else return None placeholders."""
    placement, next_cgpa, package = None, None, None
    if joblib is not None:
        try:
            placement = joblib.load("models/placement_clf.pkl")
        except Exception:
            placement = None
        try:
            next_cgpa = joblib.load("models/next_cgpa_reg.pkl")
        except Exception:
            next_cgpa = None
        try:
            package = joblib.load("models/package_reg.pkl")
        except Exception:
            package = None
    return Models(placement, next_cgpa, package)


MODELS = load_models()

# -------------------------
# Heuristic baselines (used when models not found)
# -------------------------

def heuristic_proba_placement(x: dict) -> float:
    # A simple, transparent scoring translated to probability
    score = 0.0
    score += (x["current_cgpa"] - 6.0) * 8      # CGPA weight
    score += (x["attendance_pct"] - 70) * 0.4   # Attendance weight
    score -= x["backlogs"] * 6                  # Backlogs penalty
    score += x["projects_count"] * 2.5
    score += x["internships_count"] * 4.0
    score += (x["tech_skill_score"] - 50) * 0.3
    score += (x["soft_skill_score"] - 50) * 0.2
    score += x["certifications_count"] * 1.0
    score += min(x["applied_companies"], 20) * 0.6  # network effect (cap)
    # Branch slight effect
    if x["branch_CSE"] or x["branch_IT"]:
        score += 4
    elif x["branch_ECE"]:
        score += 2
    # Convert score to 0-1 via logistic
    proba = 1 / (1 + math.exp(-score/20))
    return float(np.clip(proba, 0, 1))


def heuristic_next_cgpa(x: dict) -> float:
    base = x["current_cgpa"]
    improvement = 0.15 * (x["attendance_pct"] - 75) / 25 + 0.1 * (x["tech_skill_score"] - 60) / 40
    penalty = 0.2 * x["backlogs"]
    pred = base + improvement - penalty
    return float(np.clip(pred, 5.0, 10.0))


def heuristic_package_lpa(x: dict, placed_proba: float) -> float:
    base = 2.5  # base LPA
    lift = 0.25 * x["projects_count"] + 0.6 * x["internships_count"] + 0.03 * x["tech_skill_score"] + 0.02 * x["soft_skill_score"] + 0.2 * x["certifications_count"]
    cgpa_lift = 0.5 * max(0, x["current_cgpa"] - 7.0)
    pred = (base + lift + cgpa_lift) * (0.6 + 0.8 * placed_proba)
    return float(np.clip(pred, 1.8, 30.0))


# -------------------------
# Recommendations
# -------------------------

def build_recommendations(x: dict, placed_proba: float) -> list[str]:
    recs = []
    if x["current_cgpa"] < 7.0:
        recs.append("Target +0.5 CGPA next term: do weekly study sprints, previous-year papers, and 2 high-weight assignments early.")
    if x["attendance_pct"] < 80:
        recs.append("Raise attendance above 85% to boost next CGPA and many companies' eligibility filters.")
    if x["backlogs"] > 0:
        recs.append("Clear existing backlogs first — prioritize easy-to-finish ones before placement season.")
    if x["projects_count"] < 2:
        recs.append("Build 2 outcome-driven projects (with README + demo). Focus on data + backend for stronger portfolios.")
    if x["internships_count"] == 0:
        recs.append("Do 1 short internship or freelance gig to get real-world exposure and references.")
    if x["tech_skill_score"] < 70:
        recs.append("Daily 1–2 hrs DSA/Python-SQL-ML practice; complete a structured roadmap and mock tests.")
    if x["soft_skill_score"] < 70:
        recs.append("Join weekly mock interviews + group discussions; record yourself to improve clarity & brevity.")
    if x["certifications_count"] < 2:
        recs.append("Add 2 relevant certifications (e.g., SQL, Python, Excel/PowerBI, ML) aligned with your target role.")
    if x["applied_companies"] < 10:
        recs.append("Expand applications to 10–20 companies; tailor resume keywords to JD with measurable achievements.")
    # Proba-specific advice
    if placed_proba < 0.5:
        recs.append("Next 6 weeks: 3 projects + 1 certification + 10 mock interviews — then re-check your score.")
    else:
        recs.append("You're on track. Now aim for product-based mock interviews and refine system design basics.")
    return recs


# -------------------------
# UI
# -------------------------
st.title("🎯 Student Career Prediction")
st.caption("Predict placement eligibility, next CGPA, and expected package with actionable recommendations.")

with st.form("student_form"):
    st.subheader("Student Details")
    cols = st.columns(2)
    with cols[0]:
        name = st.text_input("Name", placeholder="e.g., Suraj Kumar")
        roll_no = st.text_input("Roll No")
        branch = st.selectbox("Branch", BRANCHES, index=0)
        current_cgpa = st.number_input("Current CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.2, step=0.1)
        attendance_pct = st.number_input("Attendance % (0–100)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
        backlogs = st.number_input("Backlogs (count)", min_value=0, max_value=20, value=0, step=1)
    with cols[1]:
        projects_count = st.number_input("Projects (count)", min_value=0, max_value=50, value=2, step=1)
        internships_count = st.number_input("Internships (count)", min_value=0, max_value=20, value=0, step=1)
        tech_skill_score = st.slider("Technical Skill Score", min_value=0, max_value=100, value=68)
        soft_skill_score = st.slider("Soft Skill Score", min_value=0, max_value=100, value=65)
        certifications_count = st.number_input("Certifications (count)", min_value=0, max_value=20, value=1, step=1)
        applied_companies = st.number_input("Companies applied (count)", min_value=0, max_value=100, value=5, step=1)

    submitted = st.form_submit_button("Predict")

# Build feature vector
branch_one_hot = {f"branch_{b}": 1 if branch == b else 0 for b in BRANCHES}
input_dict = {
    "current_cgpa": float(current_cgpa),
    "attendance_pct": float(attendance_pct),
    "backlogs": int(backlogs),
    "projects_count": int(projects_count),
    "internships_count": int(internships_count),
    "tech_skill_score": int(tech_skill_score),
    "soft_skill_score": int(soft_skill_score),
    "certifications_count": int(certifications_count),
    "applied_companies": int(applied_companies),
    **branch_one_hot,
}

X = np.array([[input_dict[feat] for feat in FEATURES]], dtype=float)

if submitted:
    # Placement prediction
    if MODELS.placement_clf is not None:
        try:
            proba = float(MODELS.placement_clf.predict_proba(X)[0,1])
        except Exception:
            proba = heuristic_proba_placement(input_dict)
    else:
        proba = heuristic_proba_placement(input_dict)

    placed = proba >= 0.5

    # Next CGPA
    if MODELS.next_cgpa_reg is not None:
        try:
            next_cgpa = float(MODELS.next_cgpa_reg.predict(X)[0])
        except Exception:
            next_cgpa = heuristic_next_cgpa(input_dict)
    else:
        next_cgpa = heuristic_next_cgpa(input_dict)

    # Package
    if MODELS.package_reg is not None:
        try:
            package_lpa = float(MODELS.package_reg.predict(X)[0])
        except Exception:
            package_lpa = heuristic_package_lpa(input_dict, proba)
    else:
        package_lpa = heuristic_package_lpa(input_dict, proba)

    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Placement Eligibility", "Yes" if placed else "No", delta=f"{proba*100:.1f}% chance")
    with c2:
        st.metric("Next Semester CGPA", f"{next_cgpa:.2f}")
    with c3:
        st.metric("Expected Package", f"{package_lpa:.2f} LPA")

    st.progress(proba)

    # Recommendations
    recs = build_recommendations(input_dict, proba)
    st.markdown("### Recommendations")
    for i, r in enumerate(recs, start=1):
        st.write(f"{i}. {r}")

    # Debug / Feature view
    with st.expander("See input features"):
        st.json(input_dict)

# -------------------------
# Training stub (optional): displayed as info only
# -------------------------
st.markdown("---")
st.caption(
    "Tip: Train your models separately with scikit-learn (train/test split, pipelines, scaling/encoding), "
    "save with joblib.dump(model, 'models/placement_clf.pkl'), and ensure FEATURES order matches here."
)
