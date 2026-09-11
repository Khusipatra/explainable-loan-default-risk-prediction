## 🏦 Loan Default Prediction using Machine Learning

**Author:** Khusi Patra

## 📘 Overview
Predicts whether a loan applicant will default, using the Loan Default Dataset (~149,000 applications, Yasserh / Kaggle). The project covers the full pipeline — cleaning, leakage detection, model comparison, threshold tuning, and explainability — resulting in a realistic, deployment-ready credit-risk model.

## 🧠 Problem Statement
Not every applicant carries the same risk, and lenders need to score that risk **before** approving a loan — using only information genuinely available at application time. The main challenge here wasn't accuracy, it was trust: several raw columns in this dataset only exist *after* a loan is approved or rejected, and a model trained on them looks great but is useless in production. Finding and removing that leakage was the core of this project.

## 🎯 Objectives
- Detect and remove data leakage hiding in the raw features
- Handle class imbalance (~25% default rate) without distorting real probabilities
- Compare multiple classifiers under proper cross-validation
- Tune a decision threshold and evaluate once, honestly, on unseen data
- Explain *why* the model predicts what it predicts

## 🧩 Approach

**1. Data Prep** — Train/test split before any imputation, so no information from the test set leaks into preprocessing. Missing values checked for correlation with the target, not just filled blindly.

**2. Leakage Detection** — An ablation study (removing features one at a time and tracking CV performance) exposed the problem directly: with all columns in, ROC-AUC sat at a suspicious ~1.00. `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, and `property_value` turned out to be almost perfectly predictive *only because they're missing exactly when a loan defaults* — a textbook leakage pattern. Removing them dropped ROC-AUC to a realistic ~0.89, confirming the fix.

**3. Model Comparison** — Logistic Regression, Random Forest, XGBoost, and LightGBM, each tested with three imbalance strategies (class weighting, SMOTE, both combined), scored via stratified 3-fold cross-validation. SMOTE is applied inside each fold only, never before the split.

**4. Threshold Tuning & Final Evaluation** — The winning pipeline (XGBoost + SMOTE) is refit on a validation slice to pick a decision threshold from the precision/recall/F1 trade-off, then evaluated exactly once on the untouched test set.

**5. Explainability** — Feature importance, SHAP summary/dependence/waterfall plots, and risk-band + segment-level default rates, so the model's decisions are interpretable, not a black box.

## ⚙️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 🐍 |
| ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook |

## 📊 Results
| Metric | Score |
|---|---|
| Best model | XGBoost + SMOTE |
| ROC-AUC (test) | 0.896 |
| PR-AUC (test) | 0.845 |
| F1-score (test) | 0.759 |
| Precision / Recall | 0.856 / 0.681 |

**Risk bands:** 🟢 Low (~80% of applicants, ~10% default) · 🟡 Medium (~5%, ~49% default) · 🔴 High (~15%, ~98% default)

## 📂 Getting Started
```bash
pip install -r requirements.txt
```
1. Download the dataset from Kaggle: [Loan Default Dataset (Yasserh)](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
2. Place it at `Data/Loan_Default.csv`
3. Run `Loan_Default_Fixed.ipynb` top to bottom

## 📈 Future Improvements
- Calibrate predicted probabilities for more reliable risk scoring
- Add cost-sensitive evaluation (a missed default costs more than a false alarm)
- Serve the model behind a simple API for real-time scoring
 
