import pytest
import pandas as pd
import xgboost as xgb
from app.feature_engineering import FraudDetectionDataset
from app.model import predict

# Define the correct feature order expected by the model
FEATURE_COLUMNS = [
    "avg_price_so_far",
    "rolling_sum_3",
    "hour_of_day",
    "rolling_mean_3",
    "rolling_std_3",
    "device_android",
    "total_transactions_so_far",
    "time_since_first",
    "total_price_so_far",
    "period_Afternoon",
    "day_of_week",
    "is_weekend",
    "price",
]


@pytest.fixture
def sample_input():
    return {
        "timestamp": "2025-01-01T12:00:00",
        "session_id": "session_1",
        "device": "android",
        "price": 100.0,
    }


@pytest.fixture
def sample_dataframe():
    data = {
        "timestamp": ["2025-01-01T12:00:00"],
        "session_id": ["session_1"],
        "device": ["android"],
        "price": [100.0],
    }
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def transformed_dataframe(sample_dataframe):
    dataset = FraudDetectionDataset(sample_dataframe)
    dataset.enrich_features()

    # Drop non-numeric columns and reorder according to model's expected order
    feature_df = dataset.get_dataset().drop(
        columns=["timestamp", "session_id", "device"], errors="ignore"
    )
    feature_df = feature_df[FEATURE_COLUMNS]  # Ensure correct order

    return feature_df


@pytest.fixture
def model():
    model = xgb.Booster()
    model.load_model("app/artifacts/XGBClassifier_final_model.json")
    return model


def test_feature_engineering(transformed_dataframe):
    for col in FEATURE_COLUMNS:
        assert col in transformed_dataframe.columns


def test_predict(sample_input):
    # Convert input to DataFrame
    df = pd.DataFrame([sample_input])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Process dataset using FraudDetectionDataset
    dataset = FraudDetectionDataset(df)
    dataset.enrich_features()

    # Drop non-numeric columns and reorder
    transformed_df = dataset.get_dataset().drop(
        columns=["timestamp", "session_id", "device"], errors="ignore"
    )
    transformed_df = transformed_df[FEATURE_COLUMNS]  # Ensure correct order

    # Call predict function with correctly ordered input
    prediction = predict(transformed_df)

    # Convert NumPy types to native Python types
    prediction_value = float(prediction[0])  # Ensure casting to Python float

    # Validate prediction output
    assert isinstance(
        prediction_value, float
    ), "First prediction output should be a float"
