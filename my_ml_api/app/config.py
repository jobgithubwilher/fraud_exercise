import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MODEL_PATH = os.getenv(
    "MODEL_PATH", "./app/artifacts/XGBClassifier_final_model.json"
)
