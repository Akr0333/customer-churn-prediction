# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn and converts model output into business-oriented retention insights.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Portfolio-success)

## 🎯 Business Problem

Customer churn can reduce recurring revenue and increase acquisition costs.

The goal is to identify customers with a higher probability of leaving so retention teams can prioritise outreach.

## 🔄 ML Pipeline

```text
Data → EDA → Cleaning → Feature Engineering → Model Training → Evaluation → Churn Risk → Business Insights
```

## 🤖 Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## 📊 Evaluation

Accuracy • Precision • Recall • F1-score • Confusion Matrix • ROC-AUC

## 📁 Structure

```text
customer-churn-prediction/
├── data/
├── notebooks/
├── src/
├── models/
├── visualisations/
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ Run

```bash
pip install -r requirements.txt
python src/train_model.py
```

## 💼 Business Questions

- Which customer segments show higher churn risk?
- How do tenure, contract type and monthly charges relate to churn?
- Which model provides the most useful trade-off between precision and recall?
- Which factors should be monitored for retention?

## 🔮 Improvements

- Hyperparameter optimisation
- SHAP explainability
- Probability-based risk scoring
- Streamlit dashboard
- Model monitoring and drift detection

> Educational portfolio project using a synthetic dataset.

⭐ **Predict risk → Explain the model → Support retention decisions**