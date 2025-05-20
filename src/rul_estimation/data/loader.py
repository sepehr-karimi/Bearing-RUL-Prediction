# src/rul_estimation/data/loader.py
from pathlib import Path
import pandas as pd


def load_bearing_data(path: str | Path) -> pd.DataFrame:
    """
    Accepts a CSV / Parquet / folder of CSVs and concatenates them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path.resolve()}")

    if path.is_file():
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    else:  # folder
        files = sorted(path.glob("*.csv"))
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    return df
