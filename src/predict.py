from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Zero represents an invalid/missing measurement
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Saved model
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_knn_final.pkl"


def load_model():
    """
    Load the saved diabetes prediction model package.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def prepare_input(data, expected_features):
    """
    Prepare one patient's data for prediction.
    """

    # Create exactly one row
    df = pd.DataFrame(
        [data],
        columns=expected_features
    )

    # Make sure all expected columns exist
    if df.shape != (1, len(expected_features)):
        raise ValueError(
            f"Expected 1 row and {len(expected_features)} features, "
            f"but received shape {df.shape}"
        )

    # Convert values to numeric
    for column in expected_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace invalid zero measurements with NaN
    for column in ZERO_AS_MISSING:
        if column in df.columns:
            df.loc[
                df[column] == 0,
                column
            ] = np.nan

    return df


def predict_diabetes(data):
    """
    Predict diabetes outcome for one patient.

    Returns:
        dictionary containing probability,
        threshold and prediction.
    """

    model_package = load_model()

    model = model_package["model"]
    threshold = model_package["threshold"]
    expected_features = model_package["features"]

    prepared_data = prepare_input(
        data,
        expected_features
    )

    # Safety check
    if prepared_data.shape != (1, len(expected_features)):
        raise ValueError(
            "Prepared input has an unexpected shape: "
            f"{prepared_data.shape}"
        )

    probability = model.predict_proba(
        prepared_data
    )[0, 1]

    prediction = int(
        probability >= threshold
    )

    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "prediction": prediction
    }


if __name__ == "__main__":

    patient = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 30,
        "Insulin": 100,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35
    }

    result = predict_diabetes(patient)

    print("Diabetes Prediction")
    print("-------------------")

    print(
        f"Probability: "
        f"{result['probability']:.4f}"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    if result["prediction"] == 1:
        print("Result: Positive")
    else:
        print("Result: Negative")