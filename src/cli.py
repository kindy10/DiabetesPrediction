from predict import predict_diabetes


def main():
    print("Diabetes Prediction")
    print("-------------------")
    print("Enter patient information.")
    print()

    patient = {
        "Pregnancies": float(input("Pregnancies: ")),
        "Glucose": float(input("Glucose: ")),
        "BloodPressure": float(input("BloodPressure: ")),
        "SkinThickness": float(input("SkinThickness: ")),
        "Insulin": float(input("Insulin: ")),
        "BMI": float(input("BMI: ")),
        "DiabetesPedigreeFunction": float(
            input("DiabetesPedigreeFunction: ")
        ),
        "Age": float(input("Age: "))
    }

    try:
        result = predict_diabetes(patient)

        print()
        print("Prediction Result")
        print("-----------------")

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

    except (ValueError, TypeError) as error:
        print()
        print("Input error:", error)


if __name__ == "__main__":
    main()