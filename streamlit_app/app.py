import streamlit as st
import joblib
import pandas as pd
import os
import sys

# ===============================
# PATH HANDLING
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.utils.config import (
    THRESHOLD_SCREENER,
    THRESHOLD_EARLY_WARNING,
    MODEL_PATH_SCREENER,
    MODEL_PATH_EARLY_WARNING,
    NUMERIC_FEATURES_SCREENER,
    NUMERIC_FEATURES_EARLY_WARNING,
    get_risk_bucket,
)
from backend.utils.preprocess import build_feature_row

# ===============================
# LOAD MODELS
# ===============================
screener_model = joblib.load(os.path.join(PROJECT_ROOT, "models", MODEL_PATH_SCREENER))
early_warning_model = joblib.load(os.path.join(PROJECT_ROOT, "models", MODEL_PATH_EARLY_WARNING))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Student Academic Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Academic Risk Prediction</h1>
    <p style="text-align:center; color:gray;">
    ML-based system to identify students at risk of academic failure and support early intervention
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.info(
    "ℹ️ This model predicts risk of **academic failure** (final grade below 10/20), "
    "used as a proxy for dropout risk. The source dataset does not contain literal "
    "withdrawal/dropout records.",
    icon="ℹ️",
)

# ===============================
# MODE SELECTION
# ===============================
mode_label = st.radio(
    "Which stage of the term are you assessing?",
    ["🆕 Day 1 Screener (no grades yet)", "📈 Early-Warning Checkpoint (after 2 grading periods)"],
    help="The Early-Warning model uses the first two period grades and is substantially "
         "more accurate (~0.98 AUC vs ~0.71 AUC for the grade-less screener)."
)
mode = "screener" if mode_label.startswith("🆕") else "early_warning"

st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 15, 22, 17)
    studytime = st.slider("📘 Weekly Study Time (1=<2h ... 4=>10h)", 1, 4, 2)
    failures = st.number_input("❌ Past Class Failures", min_value=0, max_value=4, value=0)
    absences = st.number_input("📅 Absences", min_value=0, max_value=93, value=4)
    goout = st.slider("🎉 Going Out With Friends (1=low, 5=high)", 1, 5, 3)
    famrel = st.slider("👨‍👩‍👧 Family Relationship Quality (1=bad, 5=great)", 1, 5, 4)

with col2:
    schoolsup = st.selectbox("🏫 Extra School Support", ["Yes", "No"])
    famsup = st.selectbox("👨‍👩‍👧 Family Educational Support", ["Yes", "No"], index=0)
    higher = st.selectbox("🎯 Wants Higher Education", ["Yes", "No"], index=0)
    internet = st.selectbox("🌐 Internet Access at Home", ["Yes", "No"], index=0)
    health = st.slider("💪 Health Status (1=bad, 5=great)", 1, 5, 4)
    Dalc = st.slider("🍷 Workday Alcohol Use (1=low, 5=high)", 1, 5, 1)

G1, G2 = None, None
if mode == "early_warning":
    st.subheader("📝 First Two Period Grades (0-20 scale)")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        G1 = st.number_input("Period 1 Grade (G1)", min_value=0, max_value=20, value=11)
    with gcol2:
        G2 = st.number_input("Period 2 Grade (G2)", min_value=0, max_value=20, value=11)

def yn(val):
    return 1 if val == "Yes" else 0

# ===============================
# PREDICTION
# ===============================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Academic Risk", use_container_width=True):

    data = {
        "age": age, "studytime": studytime, "failures": failures,
        "absences": absences, "goout": goout, "famrel": famrel,
        "schoolsup": yn(schoolsup), "famsup": yn(famsup),
        "higher": yn(higher), "internet": yn(internet),
        "health": health, "Dalc": Dalc,
    }

    if mode == "screener":
        model = screener_model
        threshold = THRESHOLD_SCREENER
        numeric_features = NUMERIC_FEATURES_SCREENER
    else:
        data["G1"] = G1
        data["G2"] = G2
        model = early_warning_model
        threshold = THRESHOLD_EARLY_WARNING
        numeric_features = NUMERIC_FEATURES_EARLY_WARNING

    X = build_feature_row(data, numeric_features)
    proba = model.predict_proba(X)[0][1]
    risk = get_risk_bucket(proba)

    st.markdown("### 📊 Prediction Result")

    if proba < threshold:
        st.success(f"✅ Prediction: Not At Risk  (Risk Level: {risk})")
    else:
        st.error(f"🚨 Prediction: At Risk  (Risk Level: {risk})")

    st.write(f"**Dropout/Failure Probability:** {proba:.2f}")
    st.write(f"**Threshold Used ({'Screener' if mode=='screener' else 'Early-Warning'}):** {threshold}")
    st.progress(min(float(proba), 1.0))

    if mode == "screener":
        st.caption(
            "⚠️ The Screener model has moderate accuracy (~0.71 AUC) since no grade "
            "data is available yet -- treat this as a coarse first-pass flag, not a "
            "precise diagnosis. Re-check with the Early-Warning model once the first "
            "two period grades are recorded."
        )

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:14px;">
    End-to-End Machine Learning Project | Streamlit Demo Interface
    </p>
    """,
    unsafe_allow_html=True
)
