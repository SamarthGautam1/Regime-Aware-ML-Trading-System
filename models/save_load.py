import os
import joblib
from sklearn.pipeline import Pipeline

ARTIFACTS_DIR = "artifacts"


def save_model(model: Pipeline, symbol: str) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(symbol: str) -> Pipeline:
    path = os.path.join(ARTIFACTS_DIR, f"{symbol}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}")
    return joblib.load(path)
