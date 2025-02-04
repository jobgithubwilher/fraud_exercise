import pytest
import sys
import os
import pandas as pd

# Import modules after modifying sys.path (fixes E402)
from app.feature_engineering import FraudDetectionDataset

# Ensure `app/` is in Python's import path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture
def sample_dataframe():
    data = {
        "timestamp": [
            "2024-01-01 12:00:00",
            "2024-01-01 13:30:00",
            "2024-01-01 15:00:00",
        ],
        "session_id": ["A", "A", "B"],
        "price": [100, 200, 150],
        "device": ["android", "ios", "android"],
    }
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def dataset(sample_dataframe):
    return FraudDetectionDataset(sample_dataframe)


def test_calculate_time_features(dataset):
    dataset.calculate_time_features()
    assert "day_of_week" in dataset.df.columns
    assert "is_weekend" in dataset.df.columns
    assert "hour_of_day" in dataset.df.columns
    assert "period_Afternoon" in dataset.df.columns  # Line break to fix E501
