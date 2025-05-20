# tests/test_models.py
import numpy as np
from rul_estimation.models.random_forest import train

def test_rf_train():
    X = np.random.rand(50, 4)
    y = np.random.rand(50)
    model = train(X, y, n_estimators=10)
    assert hasattr(model, "predict")
