# scripts/train.py
import argparse
from pathlib import Path
from rul_estimation.config import TrainingConfig
from rul_estimation.pipeline import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RUL model")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
