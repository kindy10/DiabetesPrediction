\# DiabetesPrediction



A machine learning classification project for predicting diabetes outcomes using the Pima Indians Diabetes Dataset.



> This project is an educational machine learning exercise. The model is not a medical diagnostic tool.



\## Project Overview



The goal of this project is to build and compare several binary classification algorithms and select a final model based on F1-score, while also considering precision and recall.



\## Dataset



The project uses the Pima Indians Diabetes Dataset.



The dataset contains 768 observations and 8 input features.



\### Features



\- Pregnancies

\- Glucose

\- BloodPressure

\- SkinThickness

\- Insulin

\- BMI

\- DiabetesPedigreeFunction

\- Age



\### Target



`Outcome`



\- `0` = Negative

\- `1` = Positive



\## Data Preprocessing



The dataset contains zero values that represent missing measurements in:



\- Glucose

\- BloodPressure

\- SkinThickness

\- Insulin

\- BMI



These zero values were converted to `NaN`.



Median imputation was then applied using values learned from the training set.



For KNN, features were standardized using `StandardScaler`.



The preprocessing steps are included in the saved model pipeline.



\## Models Tested



The following classification algorithms were evaluated:



\- Logistic Regression

\- Random Forest

\- K-Nearest Neighbors

\- Decision Tree

\- Support Vector Machine



\## Model Comparison



| Model | Accuracy | Precision | Recall | F1 |

|---|---:|---:|---:|---:|

| KNN | 0.7532 | 0.6600 | 0.6111 | 0.6346 |

| Random Forest | 0.7532 | 0.6739 | 0.5741 | 0.6200 |

| Decision Tree | 0.7273 | 0.6429 | 0.5000 | 0.5625 |

| SVM | 0.7273 | 0.6429 | 0.5000 | 0.5625 |

| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 |



F1-score was used as the primary model selection metric.



\## Final Model



The final model is K-Nearest Neighbors.



Configuration:



\- `n\_neighbors = 5`

\- `weights = uniform`

\- `metric = euclidean`

\- Classification threshold = `0.25`



The threshold was selected using 5-fold stratified cross-validation on the training data.



\## Final Test Performance



| Metric | Score |

|---|---:|

| Accuracy | 0.7143 |

| Precision | 0.5714 |

| Recall | 0.7407 |

| F1-score | 0.6452 |



Confusion matrix:



```text

\[\[70 30]

&#x20;\[14 40]]



DiabetesPrediction/

│

├── data/

│

├── models/

│   └── diabetes\_knn\_final.pkl

│

├── notebooks/

│   └── diabetes\_prediction.ipynb

│

├── src/

│   ├── \_\_init\_\_.py

│   └── predict.py

│

├── tests/

│   └── test\_predict.py

│

├── .gitignore

├── README.md

└── requirements.txt



\## Making a Prediction



&#x20; The saved model can be used through:



&#x20; python src\\predict.py



&#x20; The prediction script:



&#x20; Loads the saved model.

&#x20; Validates the input.

&#x20; Converts invalid zero measurements to missing values.

&#x20; Applies the saved preprocessing pipeline.

&#x20; Calculates the probability.

&#x20; Applies the saved classification threshold.

&#x20; Returns the prediction.

&#x20; Running Tests



&#x20; Run:

&#x20;   python -m pytest

