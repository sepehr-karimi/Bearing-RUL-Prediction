# src/rul_estimation/features/engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def extract_features(
    df: pd.DataFrame,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very simple example: rolling mean + std on all numeric cols
    """
    win = cfg.window
    num_cols = df.select_dtypes("number").columns.tolist()
    feat_df = pd.DataFrame(index=df.index)

    for col in num_cols:
        feat_df[f"{col}_mean{win}"] = df[col].rolling(win, min_periods=1).mean()
        feat_df[f"{col}_std{win}"] = df[col].rolling(win, min_periods=1).std()

    feat_df.fillna(method="bfill", inplace=True)
    y = df["health_index"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.astype(np.float32))

    return X, y
