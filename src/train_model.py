import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():

    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Set MLflow experiment
    mlflow.set_experiment("loan-default-model")

    with mlflow.start_run():

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_train, y_train.values.ravel())

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        joblib.dump(model, "models/random_forest.pkl")

        print("Model trained with accuracy:", accuracy)

if __name__ == "__main__":
    train_model()