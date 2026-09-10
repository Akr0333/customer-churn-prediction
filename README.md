# Customer Churn Prediction 📉

End-to-end machine learning project for predicting customer churn and translating model results into actionable retention insights.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-success)

## 🎯 Business Problem

Customer churn reduces recurring revenue and increases acquisition costs. The goal is to identify customers who are most likely to leave so a business can prioritise retention efforts.

## 🔄 Workflow

`Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Business Insights`

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

For churn prediction, recall and ROC-AUC are especially useful because missing a likely churner can be more costly than reviewing an extra customer.

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

For exploratory analysis, open the notebook in `notebooks/`.

## 💼 Business Questions

- Which customer segments have the highest churn risk?
- How do tenure, contract type and monthly charges relate to churn?
- Which model gives the best balance of precision and recall?
- Which features should the business monitor for retention campaigns?

## 🔮 Future Improvements

- Hyperparameter optimisation
- SHAP explainability
- Probability-based retention scoring
- Streamlit prediction dashboard
- Model monitoring and drift checks

## ⚠️ Note

This repository is designed as an educational portfolio project. The included dataset is synthetic so the project is reproducible without exposing customer information.

---

⭐ **Predict risk. Explain the model. Turn predictions into retention action.**
