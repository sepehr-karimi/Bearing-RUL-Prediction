# scripts/predict.py
import argparse
import joblib
from pathlib import Path
import pandas as pd
from rul_estimation.features.engineering import extract_features
from rul_estimation.data.preprocessing import assign_health_index
from rul_estimation.data.loader import load_bearing_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict RUL on new data")
    p.add_argument("--model", type=Path, required=True, help="Checkpoint file")
    p.add_argument("--data", type=Path, required=True, help="CSV/Parquet to predict on")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    df = load_bearing_data(args.data)
    df = assign_health_index(df)
    X, _ = extract_features(df, cfg=None)  # quick hack – cfg not needed for defaults
    preds = model.predict(X)
    print("First 10 predictions:", preds[:10])


if __name__ == "__main__":
    main()
