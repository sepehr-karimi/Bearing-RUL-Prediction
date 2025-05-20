# src/rul_estimation/evaluation.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def print_metrics(y_true, y_pred) -> None:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
