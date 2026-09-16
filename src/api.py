import logging
from datetime import datetime, timezone
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from src.predict import predict_diabetes

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


class PatientInput(BaseModel):
    Pregnancies:float =Field(...,ge=0,le=20)
    Glucose: float = Field(..., ge=0,le=300)
    BloodPressure: float = Field(..., ge=0,le=200)
    SkinThickness: float = Field(..., ge=0,le=100)
    Insulin: float = Field(..., ge=0,le=1000)
    BMI: float = Field(..., ge=0,le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0,le=3)
    Age: float = Field(..., ge=0,le=120)

class PredictionResponse(BaseModel):
    prediction:int
    result:str
    probability:float
    threshold:float
    predicted_at:str

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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict diabetes",
    description=(
        "Predict whether a patient is likely to have diabetes "
        "based on the provided health measurements."
    )
)
def predict(patient: PatientInput):

    try:
        result = predict_diabetes(
            patient.model_dump()
        )
        logger.info("Diabetes prediction   completed")

        prediction = result["prediction"]

        return {
            "prediction": prediction,
            "result": (
                "Positive"
                if prediction == 1
                else "Negative"
            ),
            "probability": result["probability"],
            "threshold": result["threshold"],
            "predicted_at":datetime.now(timezone.utc).isoformat()
        }
    except FileNotFoundError:
        logger.error("Prediction model is unavailable")

        raise HTTPException(
        status_code=500,
        detail="Prediction model is unavailable."
    )
    except Exception:
        logger.exception(
            "Unexpected error during diabetes prediction"
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        )

