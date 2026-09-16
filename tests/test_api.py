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
    assert "predicted_at" in data

    assert 0 <= data["probability"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Positive","Negative"]
    assert isinstance(data["predicted_at"],str)
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


def test_prediction_returns_500_when_model_is_missing(monkeypatch):
    from src.routes import prediction

    def mock_predict(data):
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr(
        prediction.prediction_service,
        "predict",
        mock_predict
    )

    response = client.post(
        "/predict",
        json=VALID_PATIENT
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Prediction model is unavailable."
    )

def test_prediction_rejects_unrealistic_value():
    patient = VALID_PATIENT.copy()
    patient["Glucose"] = 5000

    response = client.post(
        "/predict",
        json=patient
    )

    assert response.status_code == 422

def test_prediction_rejects_unrealistic_age():
    patient = VALID_PATIENT.copy()
    patient["Age"] = 200

    response = client.post(
        "/predict",
        json=patient
    )

    assert response.status_code == 422


def test_prediction_returns_500_on_unexpected_error(monkeypatch):
    from src.routes import prediction

    def mock_predict(data):
        raise RuntimeError("Unexpected prediction failure")

    monkeypatch.setattr(
        prediction.prediction_service,
        "predict",
        mock_predict
    )

    response = client.post(
        "/predict",
        json=VALID_PATIENT
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An unexpected error occurred during prediction."
    )
