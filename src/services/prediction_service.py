from src.predict import predict_diabetes


class PredictionService:
    """
    Service responsible for handling diabetes predictions.
    """

    def predict(self, patient_data):
        """
        Generate a diabetes prediction for a patient.
        """
        return predict_diabetes(patient_data)