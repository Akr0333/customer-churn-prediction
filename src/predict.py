"""Score a customer with the trained churn model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"


def predict_churn(customer: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python src/train_model.py")
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([customer])
    return float(model.predict_proba(frame)[0, 1])


if __name__ == "__main__":
    sample = {
        "age": 29,
        "tenure_months": 5,
        "monthly_charges": 110,
        "support_calls": 4,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
    }
    probability = predict_churn(sample)
    print(f"Predicted churn probability: {probability:.1%}")
