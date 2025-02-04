from fastapi import FastAPI
from app.api.v1.endpoints import router as v1_router
from app.logging import logger

app = FastAPI()

# Include routes
app.include_router(v1_router, prefix="/v1", tags=["Prediction"])


# Log startup
@app.on_event("startup")
def startup_event():
    logger.info("Application started successfully.")
