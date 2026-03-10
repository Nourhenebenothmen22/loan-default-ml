import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    model = RandomForestClassifier()

    model.fit(X_train, y_train.values.ravel())

    joblib.dump(model, "models/random_forest.pkl")

if __name__ == "__main__":
    train_model()