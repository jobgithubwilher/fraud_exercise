import pickle
from typing import Any

MODEL_PATH = "./app/artifacts/XGBClassifier_final_model.pkl"


def load_model() -> Any:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()


def predict(data):
    return model.predict(data)
