# src/rul_estimation/data/preprocessing.py
import pandas as pd
import numpy as np


def assign_health_index(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Adds column 'health_index' ∈ [0,1] (1 = new, 0 = failed).
    Simple linear degradation if no domain-specific rule given.
    """
    if target_col and target_col in df.columns:
        return df  # already exists
    n = len(df)
    df["health_index"] = np.linspace(1.0, 0.0, n)
    return df.reset_index(drop=True)


# -------- stage helpers (placeholders) -------- #

def fix_stage_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: implement your notebook's optimized stage reorder."""
    return df
