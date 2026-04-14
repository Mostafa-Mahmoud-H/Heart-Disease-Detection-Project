import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.data_processing import Data_Preprocessor
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train():
    Raw_Data_Path=r"D:\College\Semmester_6\IntelligentProgramming\Assignment_2\Heart Disease Detection Project\Data\RawData\heart.csv" 
    df=pd.read_csv(Raw_Data_Path)
    preprocessor=Data_Preprocessor(df)
    cleaned_Data=preprocessor.cleaning(df)
    preprocessor.save_cleaned_data(cleaned_Data)
    
    x=cleaned_Data.drop('target',axis=1)
    y=cleaned_Data['target']
    
    cat_cols = preprocessor.categorical_columns
    num_cols = preprocessor.continuous_columns
    
    Preprocessor_transformer=ColumnTransformer([
        ('scale',RobustScaler() , num_cols),
        ('encode',OneHotEncoder(drop='first', sparse_output=False),cat_cols)
    ] ,remainder='passthrough',
       verbose_feature_names_out=False                                        
    ) 
    
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor',Preprocessor_transformer),
        ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
    model_pipeline.fit(X_train, y_train)

    
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)       
    f1 = f1_score(y_test, y_pred)              
    
    print(f" Accuracy (Overall):  {accuracy*100}) %")
    print(f" Precision:           {precision:.4f}")
    print(f" Recall (Sensitivity): {recall:.4f}")
    print(f" F1-Score:            {f1:.4f}")
    
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("="*30)
    cm=confusion_matrix(y_test , y_pred) 
    plt.figure(figsize=(8,6))
    sns.heatmap(cm ,annot=True ,cmap='Blues' ,fmt='d' , xticklabels=['No Disease', 'Disease'] , yticklabels=['No Disease', 'Disease'])    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Diabetes Prediction')
    os.makedirs("Reports" ,exist_ok=True)
    plt.savefig(r"Reports\ConfusionMatrix.png")
    print("Saved")
    plt.show()

    os.makedirs('ml_model/saved_models', exist_ok=True)
    joblib.dump(model_pipeline, 'ml_model/saved_models/heart_disease_pipeline.pkl')
    
    print("Full Pipeline saved successfully.")
    return train

if __name__ == "__main__":
    train()
    