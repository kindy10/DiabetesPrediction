import logging

from fastapi import FastAPI

from src.predict import predict_diabetes
from src.routes.prediction import router as prediction_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "A machine learning API that predicts the likelihood "
        "of diabetes using patient health measurements."
    ),
    version="1.0.0",
    contact={
        "name": "Diabetes Prediction Project"
    }
)

@app.get(
    "/",
    tags=["System"],
    summary="API information"
)
def root():
    return {
        "message": "Diabetes Prediction API",
        "status": "running"
    }

@app.get(
    "/health",
    tags=["System"],
    summary="Check API health"
)
def health():
    return {
        "status": "healthy"
    }

app.include_router(prediction_router)

