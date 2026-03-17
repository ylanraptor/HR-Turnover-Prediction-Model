# Hackathon AI x HR: Frugal & Secure Turnover Prediction

## Context & Objective
An imaginary company is facing a high turnover rate and wants to use AI to better understand the causes of turnover and preserve its talent. Our goal is to provide an AI solution that helps HR manage and retain talent, built on two core pillars of Responsible AI:
- **AI & Cybersecurity:** Securing HR data through strict anonymization and GDPR compliance.
- **Frugal AI:** Optimizing the AI to be highly resource-efficient (energy and data) while remaining effective.

## Personas
- **The Client (HR Management):** "I need a solution to retain my employees while guaranteeing the protection of their sensitive data".
- **Our Team (HR-AI Provider):** "We bring AI solutions for HR that are secure, ethical, frugal, and explainable".

## Project Scope & Architecture
Our technical pipeline is broken down into three main stages, plus an interactive application:
1. **Privacy-by-Design (Cybersecurity):** We process the raw dataset to ensure GDPR compliance. We apply pseudonymization (non-reversible SHA-256 hashes for names and IDs), suppression of geographic quasi-identifiers (Zip, State), and generalization for age and salary brackets.
2. **Prediction & Explainability:** We trained a lightweight Logistic Regression model to predict the resignation variable (`Termd`). We use the XAI method SHAP to provide transparent explanations to HR regarding individual risk factors.
3. **Frugality & Impact:** We benchmarked computational costs ($CO_2$ footprint and CPU time). Our analysis proves that using only 5 features out of 19 maintains 98% of the maximum AUC performance, validating our data frugality approach.
4. **Interactive Demo (Python App):** We built a custom Python dashboard (using Streamlit) where users can input an employee's data and instantly see the probability that they will quit, complete with an automated narrative explanation.

## Execution Instructions
To reproduce our analysis and generate the required deliverables, please run the Jupyter notebooks in the following strict order:

1. **`UC2_Anonymisation_RGPD.ipynb`**: Applies anonymization techniques to the raw dataset and outputs the secure `HRDataset_anonymized.csv`.
2. **`UC1_Modele_Predictif_Frugal.ipynb`**: Ingests the anonymized data, trains the predictive model, generates SHAP explanations, and exports the active employee risk scores to `employee_risk_scores.csv`.
3. **`UC3_Benchmark_Frugalite.ipynb`**: Compares multiple algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) based on performance and resource cost, then generates the final `model_card.csv`.
4. **App Execution**: Run our Python application file to launch the interactive HR dashboard.

## Key Results
- **Validated Security:** The dataset is fully anonymized and audited for k-anonymity to prevent re-identification.
- **Ultra-Frugality:** Our frugal Logistic Regression model achieves an AUC of 88.7%. It is ~47 times faster to train and emits ~47 times less $CO_2$ than heavy models like Gradient Boosting.
- **Data Minimization:** Only 5 features are required to achieve optimal predictive performance, proving that collecting more data is useless and increases GDPR risks