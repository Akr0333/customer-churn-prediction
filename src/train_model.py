"""Train and evaluate customer churn models on a reproducible synthetic dataset."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def make_dataset(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(70, 22, n).clip(20, 180)
    support = rng.poisson(2, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.27, 0.18])
    internet = rng.choice(["DSL", "Fiber optic", "No internet"], n, p=[0.35, 0.45, 0.20])
    age = rng.integers(18, 75, n)

    logit = (
        -2.2
        + 0.028 * (monthly - 65)
        - 0.028 * tenure
        + 0.32 * support
        + 1.15 * (contract == "Month-to-month")
        + 0.45 * (internet == "Fiber optic")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly.round(2),
        "support_calls": support,
        "contract": contract,
        "internet_service": internet,
        "churn": churn,
    })


def main() -> None:
    df = make_dataset()
    df.to_csv(DATA_DIR / "customer_churn.csv", index=False)

    X = df.drop(columns="churn")
    y = df["churn"]
    numeric = ["age", "tenure_months", "monthly_charges", "support_calls"]
    categorical = ["contract", "internet_service"]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        probability = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
        results.append((name, auc))
        print(f"\n{name} | ROC-AUC={auc:.3f}")
        print(classification_report(y_test, prediction, zero_division=0))

    best_name, _ = max(results, key=lambda item: item[1])
    best_model = models[best_name]
    final_pipeline = Pipeline([("preprocess", preprocess), ("model", best_model)])
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_DIR / "churn_model.pkl")
    print(f"Saved best model: {best_name}")


if __name__ == "__main__":
    main()
