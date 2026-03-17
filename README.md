# Hackathon AI x HR: Ethical & Secure Turnover Prediction

## Context & Objective
An imaginary company is facing a high turnover rate and wants to use AI to better understand the causes of turnover and preserve its talent. Our goal is to provide an AI solution that helps HR manage and retain talent, built on two core pillars of Responsible AI:
- **AI & Cybersecurity:** Securing HR data through strict anonymization and GDPR compliance to prevent data breaches and protect employee privacy.
- **Ethics AI:** Avoiding algorithmic bias and discrimination. We ensure our model treats all employee groups fairly and provides transparent, explainable results to support human-in-the-loop HR decisions.

## Personas
- **The Client (HR Management):** "I need a fair and transparent solution to retain my employees while guaranteeing the absolute protection of their sensitive data."
- **Our Team (HR-AI Provider):** "We bring AI solutions for HR that are secure, highly ethical, equitable, and explainable."

## Project Scope & Architecture
Our technical pipeline is broken down into three main stages, plus an interactive application:
1. **Privacy-by-Design (Cybersecurity):** We process the raw dataset to ensure GDPR compliance before any machine learning occurs. We apply pseudonymization (non-reversible SHA-256 hashes for names and IDs), suppression of geographic quasi-identifiers (Zip, State), and generalization for age and salary brackets.
2. **Prediction & Explainability (Ethics):** We trained a transparent Logistic Regression model to predict the resignation variable (`Termd`). To avoid "black box" decisions, we integrated the XAI method SHAP to provide clear, narrative explanations to HR regarding individual risk factors.
3. **Algorithmic Fairness Audit (Ethics):** We conducted rigorous fairness audits to check the False Positive Rates (FPR) across protected demographic attributes (such as Gender and Ethnicity). This ensures the model does not systematically discriminate against specific minority or demographic groups.
4. **Interactive Demo (Python App):** We built a custom Python dashboard (using Streamlit) where users can input an employee's data and instantly see the probability that they will quit, complete with an automated, unbiased narrative explanation.

## Execution Instructions
To reproduce our analysis and generate the required deliverables, please run the Jupyter notebooks in the following strict order:

1. **`UC2_Anonymisation_RGPD.ipynb`**: Applies cybersecurity and anonymization techniques to the raw dataset and outputs the secure `HRDataset_anonymized.csv`.
2. **`UC1_Modele_Predictif_Frugal.ipynb`**: Ingests the anonymized data, trains the predictive model, conducts the fairness audit (Gender/Ethnicity bias checks), generates SHAP explanations, and exports the active employee risk scores to `employee_risk_scores.csv`.
3. **`UC3_Benchmark_Frugalite.ipynb`**: Compares algorithmic tradeoffs (complexity vs. transparency) and generates the final `model_card.csv` to document the model's ethical and technical boundaries.
4. **App Execution**: Run our Python application file to launch the interactive, secure HR dashboard.

## Key Results
- **Validated Security:** The dataset is fully anonymized and audited for k-anonymity to prevent re-identification, complying with EU AI Act "High Risk" data requirements.
- **Audited Fairness:** The model's predictions were rigorously tested for demographic parity, identifying and mitigating potential historical biases related to gender and ethnicity.
- **Actionable Transparency:** By rejecting complex "black box" algorithms in favor of explainable AI (SHAP), the system guarantees that no automated HR decision is made without clear, human-readable justification.
