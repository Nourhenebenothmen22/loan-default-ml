import os
import joblib
import pandas as pd


def load_model():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "random_forest.pkl")

    model = joblib.load(model_path)
    return model


def predict(data):

    model = load_model()

    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    X_test_path = os.path.join(project_root, "data", "processed", "X_test.csv")

    X_test = pd.read_csv(X_test_path)

    model = load_model()

    preds = model.predict(X_test)

    print("Predictions example:")
    print(preds[:10])