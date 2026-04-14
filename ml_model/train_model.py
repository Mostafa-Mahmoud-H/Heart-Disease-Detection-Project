import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder , RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.data_processing import Data_Preprocessor

def train():
    Raw_Data_Path = "data/RawData/heart.csv" 
    if not os.path.exists(Raw_Data_Path):
        print(f"Error: File not found at {Raw_Data_Path}")
        return

    df = pd.read_csv(Raw_Data_Path)
    preprocessor_tool = Data_Preprocessor(df)
    cleaned_data = preprocessor_tool.cleaning(df)
    preprocessor_tool.save_cleaned_data(cleaned_data)
    

    X = cleaned_data.drop('target', axis=1)
    y = cleaned_data['target']
    
    cat_cols = preprocessor_tool.categorical_columns
    num_cols = preprocessor_tool.continuous_columns
    

    preprocessor_transformer = ColumnTransformer([
        ('scale', MinMaxScaler(), num_cols),
        ('encode', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ], remainder='passthrough')

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor_transformer),
        ('model', DecisionTreeClassifier(random_state=42))
    ])

    print("Starting Hyperparameter Tuning...")
    param_grid = {
        'model__max_depth': [3, 5, 7, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__criterion': ['gini', 'entropy']
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    

    best_model = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")


    y_pred = best_model.predict(X_test)
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred))


    print("\nGenerating Feature Importance Plot...")
    encoded_cat_names = best_model.named_steps['preprocessor'].named_transformers_['encode'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_names) + [col for col in X.columns if col not in num_cols + cat_cols]

    importances = best_model.named_steps['model'].feature_importances_
    feat_importances = pd.Series(importances, index=all_feature_names)
    
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Important Features (Decision Tree)')
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


    model_dir = os.path.join('ml_model', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'heart_disease_pipeline.pkl')
    joblib.dump(best_model, save_path)
    print(f"Model saved successfully at: {save_path}")

if __name__ == "__main__":
    train()
