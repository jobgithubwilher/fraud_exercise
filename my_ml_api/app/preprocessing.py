import pandas as pd
from app.feature_engineering import FraudDetectionDataset

# Define the required model features
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


def preprocess_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame to extract only necessary features.

    Args:
        dataframe (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Transformed DataFrame with only required features.
    """
    dataset = FraudDetectionDataset(dataframe)
    dataset.enrich_features()
    transformed_df = dataset.get_dataset()

    # Ensure missing columns are added with default value 0
    for col in FEATURE_COLUMNS:
        if col not in transformed_df.columns:
            transformed_df[col] = 0

    # Select only required features in correct order
    return transformed_df[FEATURE_COLUMNS]
