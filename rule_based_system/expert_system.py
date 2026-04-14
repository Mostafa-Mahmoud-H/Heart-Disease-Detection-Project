import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

from experta import KnowledgeEngine
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rules import CardiacRiskEngine, PatientData, RiskAssessment, RiskFactor


FIELDS = [
    {"key": "age",      "prompt": "Age (years)",                                                                    "type": int,   "valid": lambda v: 1 <= v <= 120,    "hint": "e.g. 55"},
    {"key": "sex",      "prompt": "Sex (1=Male, 0=Female)",                                                         "type": int,   "valid": lambda v: v in (0, 1),      "hint": "0 or 1"},
    {"key": "cp",       "prompt": "Chest pain type (0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic)",  "type": int,   "valid": lambda v: v in (0,1,2,3),   "hint": "0,1,2 or 3"},
    {"key": "trestbps", "prompt": "Resting blood pressure (mm Hg)",                                                 "type": int,   "valid": lambda v: 60 <= v <= 250,   "hint": "e.g. 130"},
    {"key": "chol",     "prompt": "Serum cholesterol (mg/dl)",                                                      "type": int,   "valid": lambda v: 100 <= v <= 600,  "hint": "e.g. 240"},
    {"key": "restecg",  "prompt": "Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)",                  "type": int,   "valid": lambda v: v in (0, 1, 2),   "hint": "0, 1 or 2"},
    {"key": "thalach",  "prompt": "Maximum heart rate achieved (bpm)",                                              "type": int,   "valid": lambda v: 60 <= v <= 250,   "hint": "e.g. 150"},
    {"key": "exang",    "prompt": "Exercise-induced angina (1=Yes, 0=No)",                                          "type": int,   "valid": lambda v: v in (0, 1),      "hint": "0 or 1"},
    {"key": "oldpeak",  "prompt": "ST depression (oldpeak)",                                                        "type": float, "valid": lambda v: 0.0 <= v <= 10.0, "hint": "e.g. 1.5"},
    {"key": "slope",    "prompt": "ST slope (0=Upsloping, 1=Flat, 2=Downsloping)",                                 "type": int,   "valid": lambda v: v in (0, 1, 2),   "hint": "0, 1 or 2"},
    {"key": "ca",       "prompt": "Number of major vessels blocked (0-4)",                                          "type": int,   "valid": lambda v: 0 <= v <= 4,      "hint": "0,1,2,3 or 4"},
    {"key": "thal",     "prompt": "Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)",                   "type": int,   "valid": lambda v: v in (1, 2, 3),   "hint": "1, 2 or 3"},
]


def prompt_patient():
    print("\nEnter patient information:")
    patient = {}
    for field in FIELDS:
        while True:
            raw = input(f"{field['prompt']} [{field['hint']}]: ").strip()
            try:
                value = field["type"](raw)
                if not field["valid"](value):
                    raise ValueError
                patient[field["key"]] = value
                break
            except (ValueError, TypeError):
                print(f"    Invalid input. Expected: {field['hint']}")
    return patient


def run_engine(patient):
    engine = CardiacRiskEngine()
    engine.reset()
    engine.declare(PatientData(**patient))
    engine.run()

    for fact in engine.facts.values():
        if isinstance(fact, RiskAssessment):
            return fact["level"]
    return "low"


def interpret_risk(level):
    advice = {
        "low": (
            "RESULT: LOW RISK\n"
            "Your indicators suggest a low probability of cardiac disease.\n"
            "Maintain a healthy lifestyle, regular checkups, and balanced diet."
        ),
        "moderate": (
            "RESULT: MODERATE RISK\n"
            "Some cardiac risk factors are present.\n"
            "Consult your physician for a cardiovascular evaluation.\n"
            "Consider lifestyle changes: diet, exercise, and stress management."
        ),
        "high": (
            "RESULT: HIGH RISK\n"
            "Multiple significant cardiac risk factors detected.\n"
            "Immediate medical consultation is strongly recommended."
        ),
    }
    return advice.get(level, "Risk level could not be determined.")


def main():
    print("   Cardiac Risk Assessment Expert System")

    while True:
        patient = prompt_patient()
        level   = run_engine(patient)

        
        print(interpret_risk(level))

        again = input("Assess another patient? (y/n): ").strip().lower()
        if again != "y":
            print("\nThank you. Stay healthy!\n")
            break


if __name__ == "__main__":
    main()