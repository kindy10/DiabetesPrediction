import pytest

from src.predict import predict_diabetes


VALID_PATIENT = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 30,
    "Insulin": 100,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
}


def test_prediction_returns_expected_structure():
    """
    A valid patient should produce a prediction result
    with probability, threshold and prediction.
    """

    result = predict_diabetes(VALID_PATIENT)

    assert "probability" in result
    assert "threshold" in result
    assert "prediction" in result


def test_prediction_probability_is_valid():
    """
    Probability must be between 0 and 1.
    """

    result = predict_diabetes(VALID_PATIENT)

    assert 0 <= result["probability"] <= 1


def test_prediction_threshold_is_valid():
    """
    The saved threshold must be between 0 and 1.
    """

    result = predict_diabetes(VALID_PATIENT)

    assert 0 <= result["threshold"] <= 1


def test_prediction_is_binary():
    """
    Prediction must be either 0 or 1.
    """

    result = predict_diabetes(VALID_PATIENT)

    assert result["prediction"] in [0, 1]


def test_zero_values_are_accepted():
    """
    Zero measurements should be accepted because the
    preprocessing pipeline treats them as missing values.
    """

    patient = VALID_PATIENT.copy()

    patient["Glucose"] = 0
    patient["Insulin"] = 0

    result = predict_diabetes(patient)

    assert result["prediction"] in [0, 1]


def test_missing_feature_is_rejected():
    """
    A patient missing a required feature should raise ValueError.
    """

    patient = VALID_PATIENT.copy()

    del patient["Glucose"]

    with pytest.raises(ValueError):
        predict_diabetes(patient)


def test_negative_value_is_rejected():
    """
    Negative measurements should raise ValueError.
    """

    patient = VALID_PATIENT.copy()

    patient["Glucose"] = -10

    with pytest.raises(ValueError):
        predict_diabetes(patient)


def test_non_numeric_value_is_rejected():
    """
    Non-numeric values should raise ValueError.
    """

    patient = VALID_PATIENT.copy()

    patient["Glucose"] = "not-a-number"

    with pytest.raises(ValueError):
        predict_diabetes(patient)

def test_extra_feature_is_rejected():
    """
    Input containing an unexpected feature should raise ValueError.
    """

    patient = VALID_PATIENT.copy()

    patient["UnexpectedFeature"] = 123

    with pytest.raises(ValueError):
        predict_diabetes(patient)


def test_model_is_cached():
    from src.predict import load_model

    load_model.cache_clear()

    first_model = load_model()
    second_model = load_model()

    assert first_model is second_model