import joblib
import pandas as pd
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def make_prediction(patient_data):

    model_path = os.path.join('ml_model', 'saved_models', 'heart_disease_pipeline.pkl')
    
    if not os.path.exists(model_path):
        return 


    pipeline = joblib.load(model_path)


    df_patient = pd.DataFrame([patient_data])


    prediction = pipeline.predict(df_patient)[0]
    probability = pipeline.predict_proba(df_patient)[0]

    return prediction, probability

if __name__ == "__main__":

    sample_patient = {
        'age': 58,
        'sex': 1,
        'cp': 1,        # Atypical Angina
        'trestbps': 120,
        'chol': 284,
        'restecg': 0,
        'thalach': 160,
        'exang': 0,
        'oldpeak': 1.8,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }

    print("Running a sample prediction...")
    res, prob = make_prediction(sample_patient)

    if isinstance(res, str):
        print(res)
    else:
        status = "Heart Disease Detected (1)" if res == 1 else "No Heart Disease (0)"
        confidence = prob[res] * 100
        print(f"\n--- Result ---")
        print(f"Diagnosis: {status}")
        print(f"Confidence: {confidence:.2f}%")