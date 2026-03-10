import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model():

    # -------------------------
    # Paths
    # -------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    X_test_path = os.path.join(project_root, "data", "processed", "X_test.csv")
    y_test_path = os.path.join(project_root, "data", "processed", "y_test.csv")
    model_path = os.path.join(project_root, "models", "random_forest.pkl")
    # -------------------------
    # Load data
    # -------------------------
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # -------------------------
    # Load trained model
    # -------------------------
    model = joblib.load(model_path)

    # -------------------------
    # Predictions
    # -------------------------
    y_pred = model.predict(X_test)

    # -------------------------
    # Evaluation metrics
    # -------------------------
    print("\n===== Model Evaluation =====\n")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()