from pathlib import Path

import joblib
import numpy as np
import pandas as pd


#Add validation constants
NON_NEGATIVE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
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


#Validation function
def validate_input(data, expected_features):
    """
    Validate one patient's input before prediction.
    """

    if not isinstance(data, dict):
        raise TypeError(
            "Patient data must be provided as a dictionary."
        )

    # Check for missing features
    missing_features = [
        feature
        for feature in expected_features
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    #Check for unexpected features
    extra_features = [
        feature
        for feature in data
        if feature not in expected_features
    ]

    if extra_features:
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )
    # Check numeric values
    for feature in expected_features:

        value = data[feature]

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be numeric. "
                f"Received: {value}"
            )

        if not np.isfinite(numeric_value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        if feature in NON_NEGATIVE_FEATURES:
            if numeric_value < 0:
                raise ValueError(
                    f"{feature} cannot be negative."
                )
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

    validate_input(data,expected_features)

    prepared_data = prepare_input(data,expected_features)

    # Safety check
    if prepared_data.shape != (1, len(expected_features)):
        raise ValueError(
            "Prepared input has an unexpected shape: "
            f"{prepared_data.shape}"
        )

    probability = model.predict_proba(
        prepared_data
    )[0, 1]

    prediction = int(probability >= threshold)

    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "prediction": prediction
    }

