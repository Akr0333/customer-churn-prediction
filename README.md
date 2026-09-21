# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn and translates model output into **business-oriented retention insights**.

## 🎯 Business Problem

Customer churn can reduce recurring revenue and increase acquisition costs.

The goal is to identify customers with a higher probability of leaving so retention teams can prioritise outreach.

## 🔄 ML Pipeline

`Data → EDA → Cleaning → Feature Engineering → Model Training → Evaluation → Churn Risk → Business Insights`

## 🤖 Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## 📊 Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## 💼 Business Questions

- Which customer segments show higher churn risk?
- How do tenure, contract type and monthly charges relate to churn?
- Which model provides the most useful precision/recall trade-off?
- Which customer factors should a retention team monitor?

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

## 🔎 Portfolio Notes

This project is intentionally written as a business case rather than only a model-training exercise.

The important output is not just a prediction — it is a workflow for identifying risk, evaluating the model and translating the results into a retention decision.

> Educational portfolio project using a synthetic dataset.

## 🔮 Next Improvements

- Hyperparameter optimisation
- SHAP explainability
- Probability-based risk scoring
- Streamlit dashboard
- Model monitoring and drift detection

⭐ **Predict risk → Explain the model → Support retention decisions**


## 📸 Project Demo

> Add dashboard screenshots, model evaluation charts and a short demo GIF here so recruiters can understand the result before reading the code.

### Suggested visuals
- Problem / dataset overview
- KPI or EDA chart
- Model evaluation (confusion matrix / ROC-AUC where applicable)
- Final dashboard or application

