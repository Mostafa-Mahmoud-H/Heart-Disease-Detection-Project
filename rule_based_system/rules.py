from experta import *

# Facts
class PatientData(Fact):
    pass

class RiskFactor(Fact):
    name  = Field(str, mandatory=True)
    level = Field(str, mandatory=True)   # 'low','moderate','high'

class RiskAssessment(Fact):
    level = Field(str, mandatory=True)   # 'low','moderate','high'


# Expert System
class CardiacRiskEngine(KnowledgeEngine):

    # RULE 1 
    # High cholesterol + older age → high risk
    @Rule(PatientData(chol=MATCH.chol, age=MATCH.age),
          TEST(lambda chol, age: chol > 240 and age > 50))
    def rule_high_cholesterol_older(self, chol, age):
        print(f" [Rule 1] Cholesterol {chol} > 240 AND Age {age} > 50 → HIGH risk factor")
        self.declare(RiskFactor(name="high_cholesterol_age", level="high"))

    # RULE 2
    # High resting blood pressure + typical/atypical angina chest pain → high risk
    @Rule(PatientData(trestbps=MATCH.trestbps, cp=MATCH.cp),
          TEST(lambda trestbps, cp: trestbps > 140 and cp in (0, 1)))
    def rule_hypertension_angina(self, trestbps, cp):
        print(f" [Rule 2] BloodPressure {trestbps} > 140 AND ChestPain type {cp} (angina) → HIGH risk factor")
        self.declare(RiskFactor(name="hypertension_angina", level="high"))

    # RULE 3 
    # Exercise-induced angina + significant ST depression → high risk
    @Rule(PatientData(exang=MATCH.exang, oldpeak=MATCH.oldpeak),
          TEST(lambda exang, oldpeak: exang == 1 and oldpeak > 2.0))
    def rule_exercise_angina_st_depression(self, exang, oldpeak):
        print(f"  [Rule 3] ExerciseAngina=Yes AND OldPeak {oldpeak} > 2.0 → HIGH risk factor")
        self.declare(RiskFactor(name="exercise_angina_st", level="high"))

    # RULE 4 
    # Multiple blocked vessels → high risk
    @Rule(PatientData(ca=MATCH.ca),
          TEST(lambda ca: ca >= 3))
    def rule_multiple_vessels(self, ca):
        print(f" [Rule 4] Blocked vessels ca={ca} ≥ 3 → HIGH risk factor")
        self.declare(RiskFactor(name="multiple_vessel_disease", level="high"))

    # RULE 5 
    # Reversible thalassemia defect → high risk
    @Rule(PatientData(thal=MATCH.thal),
          TEST(lambda thal: thal == 3))
    def rule_reversible_thal_defect(self, thal):
        print(f"  [Rule 5] Thalassemia type {thal} = reversible defect → HIGH risk factor")
        self.declare(RiskFactor(name="reversible_thal", level="high"))

    # RULE 6 
    # Abnormal/LVH ECG + high blood pressure → high risk
    @Rule(PatientData(restecg=MATCH.restecg, trestbps=MATCH.trestbps),
          TEST(lambda restecg, trestbps: restecg in (1, 2) and trestbps > 130))
    def rule_ecg_abnormality_bp(self, restecg, trestbps):
        print(f"  [Rule 6] ECG restecg={restecg} abnormal AND BloodPressure {trestbps} > 130 → HIGH risk factor")
        self.declare(RiskFactor(name="ecg_bp_abnormality", level="high"))

    # RULE 7 
    # Flat/upsloping ST slope + exercise angina → moderate risk
    @Rule(PatientData(slope=MATCH.slope, exang=MATCH.exang),
          TEST(lambda slope, exang: slope in (0, 1) and exang == 1))
    def rule_slope_exang(self, slope, exang):
        print(f"  [Rule 7] Slope={slope} (flat/upsloping) AND ExerciseAngina=Yes → MODERATE risk factor")
        self.declare(RiskFactor(name="slope_exercise_angina", level="moderate"))

    # RULE 8
    # Male + age > 55 → moderate risk (sex-based age threshold)
    @Rule(PatientData(sex=MATCH.sex, age=MATCH.age),
          TEST(lambda sex, age: sex == 1 and age > 55))
    def rule_male_age(self, sex, age):
        print(f"  [Rule 8] Male AND Age {age} > 55 → MODERATE risk factor")
        self.declare(RiskFactor(name="male_age_risk", level="moderate"))

    # RULE 9 
    # Low max heart rate achieved (< 120) → moderate risk
    @Rule(PatientData(thalach=MATCH.thalach),
          TEST(lambda thalach: thalach < 120))
    def rule_low_max_heart_rate(self, thalach):
        print(f"  [Rule 9] Max heart rate thalach={thalach} < 120 → MODERATE risk factor")
        self.declare(RiskFactor(name="low_thalach", level="moderate"))

    # RULE 10 
    # Very high cholesterol (> 300) alone → moderate risk
    @Rule(PatientData(chol=MATCH.chol),
          TEST(lambda chol: chol > 300))
    def rule_very_high_cholesterol(self, chol):
        print(f"  [Rule 10] Cholesterol {chol} > 300 → MODERATE risk factor")
        self.declare(RiskFactor(name="very_high_cholesterol", level="moderate"))

    # RULE 11
    # Normal ECG + low blood pressure + good heart rate → low risk
    @Rule(PatientData(restecg=MATCH.restecg, trestbps=MATCH.trestbps, thalach=MATCH.thalach),
          TEST(lambda restecg, trestbps, thalach: restecg == 0 and trestbps < 120 and thalach > 150))
    def rule_healthy_indicators(self, restecg, trestbps, thalach):
        print(f"  [Rule 11] Normal ECG + BP {trestbps} < 120 + Heart rate {thalach} > 150 → LOW risk factor")
        self.declare(RiskFactor(name="healthy_indicators", level="low"))

    # RULE 12 
    # No vessel blockage + normal thal → low risk
    @Rule(PatientData(ca=MATCH.ca, thal=MATCH.thal),
          TEST(lambda ca, thal: ca == 0 and thal == 1))
    def rule_clean_vessels_normal_thal(self, ca, thal):
        print(f"  [Rule 12] No blocked vessels (ca=0) AND Normal thal (thal=1) → LOW risk factor")
        self.declare(RiskFactor(name="clean_vessels_thal", level="low"))

    # RULE 13 
    # No exercise angina + minimal ST depression → low risk
    @Rule(PatientData(exang=MATCH.exang, oldpeak=MATCH.oldpeak),
          TEST(lambda exang, oldpeak: exang == 0 and oldpeak <= 0.5))
    def rule_no_angina_no_st(self, exang, oldpeak):
        print(f" [Rule 13] No ExerciseAngina AND OldPeak {oldpeak} ≤ 0.5 → LOW risk factor")
        self.declare(RiskFactor(name="no_angina_no_st", level="low"))

    # AGGREGATION: Compute final risk level from all declared RiskFactor facts
    @Rule(AS.pd << PatientData(),
          NOT(RiskAssessment()),
          salience=-1)
    def aggregate_risk(self, pd):
        high_count     = 0
        moderate_count = 0
        low_count      = 0

        for f in self.facts.values():
            try:
                level = f["level"]
                name  = f["name"]
                print(f"    DEBUG → name={name}, level={level}")
                if level == "high":
                    high_count += 1
                elif level == "moderate":
                    moderate_count += 1
                elif level == "low":
                    low_count += 1
            except (KeyError, TypeError):
                continue

        print(f" DEBUG COUNTS → high={high_count}, moderate={moderate_count}, low={low_count}")

        if high_count >= 2:
            final = "high"
        elif high_count == 1 or moderate_count >= 2:
            final = "moderate"
        elif moderate_count == 1:
            final = "moderate"
        elif low_count > 0:
            final = "low"
        else:
            final = "low"

        self.declare(RiskAssessment(level=final))
