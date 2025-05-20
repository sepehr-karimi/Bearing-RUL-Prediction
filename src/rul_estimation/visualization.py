# src/rul_estimation/visualization.py
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true, y_pred, title: str = "Test predictions") -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
