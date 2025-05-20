# src/rul_estimation/models/random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def train(X, y, **params) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", None),
        n_jobs=-1,
        random_state=params.get("seed", 42),
    )
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test) -> float:
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
