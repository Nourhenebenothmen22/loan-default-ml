import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model():

    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Load trained model
    model = joblib.load("models/random_forest.pkl")

    # Predictions
    predictions = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Detailed report
    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    evaluate_model()