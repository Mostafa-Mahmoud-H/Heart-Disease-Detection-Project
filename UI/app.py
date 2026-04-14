import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_model.predict import make_prediction
from rule_based_system.expert_system import run_engine, interpret_risk


st.set_page_config(page_title="Heart Disease Detector", layout="wide", page_icon="🏥")

st.title("🏥 Heart Disease Detection & Analytics Dashboard")
st.markdown("---")


@st.cache_data
def load_cleaned_data():

    data_path = "data/CleanData/cleaned_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_cleaned_data()

st.sidebar.header("📋 Clinical Input Data")

def get_user_input():
    age = st.sidebar.slider("Age", 1, 100, 55)
    sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                              help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 60, 200, 130)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope", options=[0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], help="1: Normal, 2: Fixed, 3: Reversible")
    
    return {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

patient_data = get_user_input()


tab_predict, tab_data, tab_model = st.tabs(["🎯 Risk Prediction", "📊 Data Analysis", "📈 Model Insights"])


with tab_predict:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Machine Learning Result")
        if st.button("Run ML Model"):
            res, prob = make_prediction(patient_data)
            if res == 1:
                st.error(f"Prediction: POSITIVE (Heart Disease Detected)")
                st.warning(f"Probability: {prob[1]:.2%}")
            else:
                st.success(f"Prediction: NEGATIVE (Healthy)")
                st.info(f"Probability: {prob[0]:.2%}")

    with col2:
        st.subheader("🧠 Expert System Result")
        if st.button("Run Expert System"):
            risk_level = run_engine(patient_data)
            advice = interpret_risk(risk_level)
            if risk_level == "high":
                st.error(advice)
            elif risk_level == "moderate":
                st.warning(advice)
            else:
                st.success(advice)


with tab_data:
    if df is not None:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        st.subheader("📊 Target Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='target', data=df, palette='viridis', ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("❌ Cleaned data file not found in 'data/CleanData/'. Please run train_model.py first.")


with tab_model:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📉 Confusion Matrix")
        if os.path.exists("ConfusionMatrix.png"):
            st.image("ConfusionMatrix.png", use_container_width=True)
        else:
            st.info("Confusion Matrix image will appear here after training.")

    with col_b:
        st.subheader("🏆 Feature Importance")
        model_path = os.path.join('ml_model', 'saved_models', 'heart_disease_pipeline.pkl')
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
  
            model = pipeline.named_steps['model']
            

            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots()
            pd.Series(importances).sort_values().plot(kind='barh', ax=ax_imp)
            ax_imp.set_title("Top Predictive Features")
            st.pyplot(fig_imp)
        else:
            st.error("Model file (.pkl) not found.")
