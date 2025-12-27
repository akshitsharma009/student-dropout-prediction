import streamlit as st
import joblib
import numpy as np
import os

# ===============================
# PATH HANDLING (ROBUST)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "dropout_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.6

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Dropout Risk Prediction</h1>
    <p style="text-align:center; color:gray;">
    ML-based system to identify students at risk and support early intervention
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===============================
# INPUT SECTION
# ===============================
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("📘 Study Time (1–4)", 1, 4, 2)
    failures = st.number_input("❌ Past Failures", min_value=0, value=0)
    absences = st.number_input("📅 Absences", min_value=0, value=0)

with col2:
    schoolsup = st.selectbox("🏫 School Support", ["Yes", "No"])
    famsup = st.selectbox("👨‍👩‍👧 Family Support", ["Yes", "No"])
    paid = st.selectbox("💰 Paid Classes", ["Yes", "No"])
    higher = st.selectbox("🎯 Higher Education Goal", ["Yes", "No"])
    internet = st.selectbox("🌐 Internet Access", ["Yes", "No"])

def yn(val):
    return 1 if val == "Yes" else 0

# ===============================
# PREDICTION
# ===============================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Dropout Risk", use_container_width=True):

    X = np.array([[
        studytime,
        failures,
        absences,
        yn(schoolsup),
        yn(famsup),
        yn(paid),
        yn(higher),
        yn(internet)
    ]])

    proba = model.predict_proba(X)[0][1]

    st.markdown("### 📊 Prediction Result")

    if proba < 0.4:
        risk = "Low Risk"
        st.success("✅ Prediction: No Dropout")
    elif proba < 0.6:
        risk = "Medium Risk"
        st.warning("⚠️ Prediction: No Dropout (Monitor Closely)")
    else:
        risk = "High Risk"
        st.error("🚨 Prediction: Dropout")

    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Dropout Probability:** {proba:.2f}")
    st.write(f"**Threshold Used:** {THRESHOLD}")

    # Visual indicator
    st.progress(min(float(proba), 1.0))

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
