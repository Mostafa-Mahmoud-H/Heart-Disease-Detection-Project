import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rule_based_system.expert_system import run_engine

def evaluate_expert_system():
    
    df = pd.read_csv(r"D:\College\Semmester_6\IntelligentProgramming\Assignment_2\Heart Disease Detection Project\data\CleanData\cleaned_data.csv")
    
    correct = 0
    total = len(df)
    
    print(f"Evaluating Expert System on {total} patients...")

    for index, row in df.iterrows():
        patient_data = row.to_dict()
        

        risk_level = run_engine(patient_data)
        

        prediction = 1 if risk_level in ['high', 'moderate'] else 0
        actual = int(row['target'])
        
        if prediction == actual:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"Expert System Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    evaluate_expert_system()