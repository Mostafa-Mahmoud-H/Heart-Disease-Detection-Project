Markdown
# 📊 Performance Comparison Report: ML vs. Expert System

This report provides a detailed evaluation of the **Heart Disease Detection System**, comparing the data-driven **Decision Tree Classifier** against the rule-based **Expert System**.

## 1. Quantitative Metrics Summary

| Metric | Machine Learning (Decision Tree) | Rule-Based System (Experta) |
| :--- | :--- | :--- |
| **Accuracy** | **73.77%** | **37.38%** |
| **Precision** | 0.7241 | N/A (Rule-driven) |
| **Recall (Sensitivity)** | 0.7241 | N/A (Rule-driven) |
| **F1-Score** | 0.7241 | N/A (Rule-driven) |

### 🔍 Decision Tree Detailed Performance
* **Healthy (Class 0):** Precision 0.75 | Recall 0.75
* **Disease (Class 1):** Precision 0.72 | Recall 0.72
* **Total Support:** 61 test samples.

### 📉 Confusion Matrix (ML Model)
```text
[[24  8]  -> (TN: 24 | FP: 8)
 [ 8 21]]  -> (FN: 8  | TP: 21)
 ```
2. Qualitative Analysis & Description
🤖 Machine Learning Approach
The Decision Tree Classifier shows a strong performance with ~74% accuracy.

Strengths: It successfully learned complex, non-linear correlations from the data. For instance, it can detect patterns where multiple slight symptoms combine to indicate risk, which a human might miss.

Weakness: It is a "Black Box" to some extent compared to rules; it makes decisions based on statistical probability rather than medical reasoning.


## 🧠 Rule-Based Expert System (Experta)
* - The Expert System achieved a lower accuracy of 37.38%.

Reasoning: This is expected in medical informatics. The Expert System relies on 13 static clinical rules. If a patient's data does not perfectly match these hardcoded thresholds (e.g., if blood pressure is 139 instead of 140), the system defaults to "Low Risk."

Strengths: High Explainability. Every result is tied to a specific clinical rule, making it safer for doctors to understand the "Why" behind a diagnosis.

3. Final Conclusion
The Machine Learning model is significantly more effective for high-accuracy screening, while the Expert System serves as a vital tool for clinical validation and explainable AI. For a real-world application, a hybrid approach using the ML model for prediction and the Expert System for justification would be ideal.

# Prepared by:

* Nour Saudi

* Omar Mohamed Mostafa

* Mostafa Mahmoud