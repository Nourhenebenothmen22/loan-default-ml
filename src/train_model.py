import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def train_model():

    # ========================
    # Load data
    # ========================
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # ========================
    # SMOTE (handle imbalance)
    # ========================
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # ========================
    # Model + Hyperparameters
    # ========================
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    # ========================
    # MLflow experiment
    # ========================
    mlflow.set_experiment("loan-default-model")

    with mlflow.start_run():

        # Train
        grid.fit(X_train_resampled, y_train_resampled.values.ravel())

        best_model = grid.best_estimator_

        # Predictions
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        # ========================
        # Metrics
        # ========================
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        # ========================
        # Log parameters
        # ========================
        mlflow.log_param("n_estimators", grid.best_params_["n_estimators"])
        mlflow.log_param("max_depth", grid.best_params_["max_depth"])
        mlflow.log_param("min_samples_split", grid.best_params_["min_samples_split"])
        mlflow.log_param("min_samples_leaf", grid.best_params_["min_samples_leaf"])
        mlflow.log_param("sampling_method", "SMOTE")

        # ========================
        # Log metrics
        # ========================
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # ========================
        # Confusion Matrix
        # ========================
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # ========================
        # Log model
        # ========================
        mlflow.sklearn.log_model(best_model, "model")

        # Save model locally
        joblib.dump(best_model, "models/random_forest.pkl")

        print("\nBest Parameters:", grid.best_params_)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1 Score:", f1)
        print("ROC AUC:", roc_auc)


if __name__ == "__main__":
    train_model()