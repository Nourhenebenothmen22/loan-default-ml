import os
import sys
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


# paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(project_root, "models", "random_forest.pkl")

# load model
model = joblib.load(model_path)

app = FastAPI(title="Loan Default Prediction API")


# input schema
class LoanData(BaseModel):

    data: list


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(input_data: LoanData):

    if len(input_data.data) != model.n_features_in_:
        return {
            "error": f"Model expects {model.n_features_in_} features"
        }

    features = np.array(input_data.data).reshape(1, -1)

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}