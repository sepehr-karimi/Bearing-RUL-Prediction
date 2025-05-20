# src/rul_estimation/models/cnn_lstm.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, Input
import numpy as np


def build_model(cfg):
    """
    cfg.params example:
        {"filters": 32, "kernel": 3, "lstm_units": 64, "drop": 0.3}
    """
    filters = cfg.params.get("filters", 32)
    kernel = cfg.params.get("kernel", 3)
    lstm_units = cfg.params.get("lstm_units", 64)
    drop = cfg.params.get("drop", 0.3)

    model = Sequential(
        [
            Input(shape=(None, 1)),
            Conv1D(filters, kernel, activation="relu"),
            Dropout(drop),
            LSTM(lstm_units, return_sequences=False),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
