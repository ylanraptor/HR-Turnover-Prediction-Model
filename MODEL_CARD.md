# 🧠 Model Card — HR Turnover Prediction

## General Information
| Field | Detail |
|---|---|
| **Model Name** | HR-Turnover-Predictor v1.0 |
| **Type** | Binary Classification (Termd: 0/1) |
| **Primary Algorithm** | Logistic Regression (Transparent & Interpretable) |
| **Secondary Algorithm** | Decision Tree (depth=5) |
| **Date** | 2024 |
| **Authors** | Hackathon AI × HR Team |
| **Intended Use** | HR decision support — identification of employees at risk of leaving |

---

## Model Objective
Predict the probability of an employee leaving the company within the next 12 months,
based on structured HR data (performance, engagement, absences, tenure, salary).

**This model is a decision support tool. It does not replace human judgment.**

---

## Training Data
| Field | Detail |
|---|---|
| **Source** | HR Dataset v14 (Huebner & Patalano, Kaggle) + synthetic data |
| **Volume** | 1511 rows (311 original + 1200 synthetic) |
| **Period Covered** | 2000–2019 |
| **Turnover Rate** | ~40% (positive class) |
| **Anonymization** | Pseudonymization of names and IDs prior to training |

### Features Used (16)
| Feature | Type | Description |
|---|---|---|
| `risk_score` | Float | Composite risk score (0–10) |
| `EngagementSurvey` | Float | Engagement score (1–5) |
| `EmpSatisfaction` | Int | Employee satisfaction (1–5) |
| `Absences` | Int | Number of absences over the period |
| `DaysLateLast30` | Int | Days late in the last 30 days |
| `tenure_years` | Float | Tenure (years) |
| `Salary` | Int | Annual salary |
| `salary_ratio_dept` | Float | Salary / department average ratio |
| `PerfScoreID` | Int | Performance score (1–4) |
| `age_at_hire` | Int | Age at hire |
| `GenderID` | Binary | Gender (0=M, 1=F) |
| `MarriedID` | Binary | Marital status |
| `DeptID` | Int | Department |
| `PositionID` | Int | Position |
| `SpecialProjectsCount` | Int | Number of special projects |
| `FromDiversityJobFairID` | Binary | Diversity recruitment |

---

## Model Performance
| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| **AUC-ROC** | ~0.82 | ~0.78 |
| **F1 (weighted)** | ~0.80 | ~0.77 |
| **Training Time** | < 0.1s | < 0.1s |
| **Explainability** | High (SHAP) | High (Tree rules) |

---

## Known Limitations and Biases

### Limitations
- Synthetic dataset — performance needs to be validated on real-world data
- No temporal data (time series) — static model
- Predictions are at the individual level, no team dynamics modeling

### Bias Risks
- The variables `GenderID` and `RaceDesc` (not included in the model) may introduce indirect biases
- The synthetic turnover rate (40%) may differ from actual company reality
- Historical correlations may perpetuate existing inequalities

### Mitigation Strategies
- `GenderID` is monitored to detect biases, not used to discriminate
- Mandatory human supervision before any decision is made
- Fairness audit recommended with IBM AIF360

---

## Ethical and Regulatory Considerations

### EU AI Act
This system is classified as **High Risk** (Annex III — HR management):
- ✅ Comprehensive documentation (this Model Card)
- ✅ Documented training data (Data Card)
- ✅ Human oversight required
- ⚠️ Decision logging required in production
- ⚠️ Right to explanation for affected employees (GDPR Art. 22)

### GDPR
- Anonymization of personal data before processing
- No fully automated decision-making
- Guaranteed rights of access and rectification

---

## Ethics AI (Algorithmic Fairness & Transparency)
| Criterion | Evaluation |
|---|---|
| **Explainability (XAI)** | High (SHAP values integrated) |
| **Fairness Audit** | Checked FPR across Gender and Ethnicity |
| **Algorithmic Transparency** | High (Logistic regression coefficients) |
| **Human-in-the-loop** | Mandatory before any HR action |
| **Bias Mitigation** | Continuous monitoring required |

> Logistic Regression was explicitly chosen because an interpretable, fair model is ethically superior to a complex "black box" algorithm (like Gradient Boosting) for high-risk HR decisions.

---

## Contact & Maintenance
- Recommended retraining: Every 6 months
- Metrics to monitor: Drift in actual vs. predicted turnover rate
- Model Owners: HR + Data Science Team
