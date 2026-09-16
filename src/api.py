import logging

from fastapi import FastAPI

from src.predict import predict_diabetes
from src.routes.health import router as health_router
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


app.include_router(health_router)
app.include_router(prediction_router)

