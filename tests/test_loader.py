# tests/test_loader.py
from pathlib import Path
import pandas as pd
from rul_estimation.data.loader import load_bearing_data

def test_roundtrip(tmp_path: Path):
    path = tmp_path / "toy.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    df = load_bearing_data(path)
    assert df["a"].sum() == 6
