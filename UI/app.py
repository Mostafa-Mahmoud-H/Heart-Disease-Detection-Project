import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_model.predict import make_prediction
from rule_based_system.expert_system import run_engine, interpret_risk

st.set_page_config(page_title="Heart Disease Detector", layout="wide")

st.title("🏥 Heart Disease Detection System")
st.markdown("Predict heart disease risk using **Machine Learning** or **Expert System Rules**.")

# --- Sidebar: User Input ---
st.sidebar.header("Patient Clinical Data")

def get_user_input():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                              help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 60, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversible Defect")
    
    return {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

patient_data = get_user_input()

# --- Main Interface ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Machine Learning Prediction")
    if st.button("Run ML Model"):
        prediction, probability = make_prediction(patient_data)
        if prediction == 1:
            st.error(f"Prediction: POSITIVE for Heart Disease")
            st.warning(f"Confidence: {probability[1]:.2%}")
        else:
            st.success(f"Prediction: NEGATIVE (Healthy)")
            st.info(f"Confidence: {probability[0]:.2%}")

with col2:
    st.subheader("🧠 Expert System Analysis")
    if st.button("Run Expert System"):
        risk_level = run_engine(patient_data)
        advice = interpret_risk(risk_level)
        
        if risk_level == "high":
            st.error(advice)
        elif risk_level == "moderate":
            st.warning(advice)
        else:
            st.success(advice)

st.divider()
st.info("Note: This tool is for educational purposes. Always consult a real doctor.") 