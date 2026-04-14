# 🏥 Heart Disease Detection System (Hybrid AI Approach)

A comprehensive intelligent system for heart disease risk assessment, integrating **Rule-Based Expert Systems** (Inference Engines) and **Machine Learning** methodologies.

## 🚀 Overview
This project demonstrates two distinct paradigms of Artificial Intelligence to solve a clinical diagnostic problem:
1. **Rule-Based Approach:** Uses predefined medical heuristics to provide explainable risk assessments.
2. **Machine Learning Approach:** Uses statistical patterns to predict the probability of heart disease based on historical data.

## 🛠️ Implementation Phases

### 1. Data Processing & EDA
- **Cleaning:** Handled missing values, removed duplicates, and treated outliers.
- **Normalization:** Applied `RobustScaler` to numerical features (Age, Blood Pressure, Cholesterol).
- **Encoding:** Utilized `OneHotEncoder` for categorical variables (Chest Pain type, ECG results).
- **Visualization:** Generated correlation heatmaps and feature importance plots to understand data trends.

### 2. Rule-Based Expert System (Experta)
- Implemented an inference engine using the `experta` library.
- **Rules:** Defined 13+ medical rules (e.g., Age/Cholesterol interaction, Blood Pressure thresholds).
- **Inference:** Uses a forward-chaining mechanism to categorize patients into `Low`, `Moderate`, or `High` risk levels.

### 3. Machine Learning Model (Decision Tree)
- **Algorithm:** Scikit-Learn's `DecisionTreeClassifier`.
- **Pipeline:** Integrated preprocessing and model training into a single `joblib` pipeline for consistency.
- **Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score.

### 4. Integration & UI (Streamlit)
- Developed an interactive web dashboard where users can input patient clinical data and receive dual-diagnosis results (ML vs. Expert System).

## 📂 Project Structure
```text
Heart_Disease_Detection/
│── data/                # Raw and preprocessed CSV datasets
│── notebooks/           # Data analysis and model training notebooks
│── rule_based_system/   # Expert system engine (expert_system.py) and rules (rules.py)
│── ml_model/            # Training scripts, prediction logic, and saved .pkl models
│── reports/             # Accuracy comparison and evaluation scripts
│── ui/                  # Streamlit web application (app.py)
│── utils/               # Helper scripts for data cleaning
│── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```
## 🏁 How to Run
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Dashboard:

Bash
streamlit run ui/app.py
## 👥 Development Team
- * Nour Saudi
- * Omar Mohamed Mostafa
- * Mostafa Mahmoud