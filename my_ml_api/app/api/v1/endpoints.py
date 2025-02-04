from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import os
import logging
from functools import lru_cache
from datetime import datetime
from app.feature_engineering import FraudDetectionDataset
from app.config import MODEL_PATH

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the FastAPI router
router = APIRouter()


# Define the request body schema
class PredictionInput(BaseModel):
    # Pydantic will automatically parse ISO 8601 timestamps
    timestamp: datetime
    session_id: str
    device: str
    price: float


# Check if model file exists
if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model file not found at: {MODEL_PATH}")
    raise HTTPException(status_code=500, detail="Model file not found")


# Function to load model efficiently with caching
@lru_cache(maxsize=1)
def get_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
    return model


# Expected feature order for the model
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


@router.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Endpoint to predict fraud detection probability based on input features.
    """
    try:
        # Feature Engineering
        feature_data = FraudDetectionDataset.transform(input_data)

        # Convert to DMatrix format
        dmatrix = xgb.DMatrix(
            pd.DataFrame([feature_data], columns=FEATURE_COLUMNS)
        )

        # Get the model
        model = get_model()

        # Predict fraud probability
        prediction = model.predict(dmatrix)[0]

        logger.info(
            f"🔍 Prediction for session {input_data.session_id}: "
            f"{prediction:.4f}"
        )

        # Convert numpy.float32 to Python float before returning response
        return {
            "session_id": input_data.session_id,
            "fraud_probability": float(prediction),
        }

    except ValueError as e:
        logger.error(f"❌ Data processing error: {str(e)}")
        raise HTTPException(
            status_code=400, detail="Invalid input data format"
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
