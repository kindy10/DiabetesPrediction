import logging

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
    description="Machine learning API for diabetes prediction",
    version="1.0.0"
)


class PatientInput(BaseModel):
    Pregnancies:float =Field(...,ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    prediction:int
    result:str
    probability:float
    threshold:float

@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
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
            "threshold": result["threshold"]
        }
    except FileNotFoundError:
        logger.error("Prediction model is unavailable")

        raise HTTPException(
        status_code=500,
        detail="Prediction model is unavailable."
    )

