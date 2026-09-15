from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


VALID_PATIENT = {
    "Pregnancies":2,
    "Glucose":120,
    "BloodPressure":70,
    "SkinThickness":30,
    "Insulin":100,
    "BMI":30.5,
    "DiabetesPedigreeFunction":0.5,
    "Age":35
}

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Diabetes Prediction API"
    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=VALID_PATIENT
    )

    assert response.status_code == 200

    data = response.json()

    assert "probability" in data
    assert "threshold" in data
    assert "prediction" in data
    assert "result" in data

    assert 0 <= data["probability"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Positive","Negative"]

def test_prediction_result_matches_prediction():
    response = client.post(
        "/predict",
        json=VALID_PATIENT
    )

    assert response.status_code == 200

    data = response.json()

    if data["prediction"] == 1:
        assert data["result"] == "Positive"
    else:
        assert data["result"] == "Negative"


def test_prediction_rejects_negative_value():
    patient = VALID_PATIENT.copy()

    patient["Glucose"] = -10

    response = client.post(
        "/predict",
        json=patient
    )

    assert response.status_code == 422


def test_prediction_rejects_missing_feature():
    patient = VALID_PATIENT.copy()

    del patient["Glucose"]

    response = client.post(
        "/predict",
        json=patient
    )

    assert response.status_code == 422