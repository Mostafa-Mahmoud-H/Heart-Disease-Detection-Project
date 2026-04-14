import pandas as pd
import numpy as np 

class Data_Preprocessor:
    def __init__(self , df):
        self.categorical_columns = ['cp', 'restecg', 'slope', 'thal']
        self.continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.feature_to_drop = 'fbs' 

    
    def cleaning(self , df):
        # Remove duplications:
        df=df.drop_duplicates()
        
        
        # Dealing With Missing Values :
        if 'restecg' in df.columns:
            df['restecg']=df['restecg'].fillna(df['restecg'].mode()[0])
        if 'oldpeak' in df.columns:
            df['oldpeak']= df['oldpeak'].fillna( df['oldpeak'].median())   
        if 'thal' in df.columns:
            df['thal'] = df['thal'].replace(0, df['thal'].mode()[0])   
        
        # Drop Feature with the least corr with target 
        if self.feature_to_drop in df.columns:
            df=df.drop(columns=[self.feature_to_drop] , axis=1)
            print(f" Feature {self.feature_to_drop} IS Dropped")
            
            
        return df
    
    
    def save_cleaned_data(self, df, path='data/CleanData/cleaned_data.csv'):
        df.to_csv(path, index=False)
        print(f"Cleaned data saved successfully to {path}")    