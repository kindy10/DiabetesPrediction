from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from src.predict import predict_diabetes

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


@app.post("/predict")
def predict(patient:PatientInput):

    try:
        result = predict_diabetes(patient.model_dump())
        return result

    except(ValueError,TypeError) as error:
        raise HTTPException(status_code=400,detail=str(error))

