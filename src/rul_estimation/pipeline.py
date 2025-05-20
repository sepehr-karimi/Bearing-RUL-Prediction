# src/rul_estimation/pipeline.py
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import TrainingConfig
from .utils import set_seed, save_checkpoint
from .data.loader import load_bearing_data
from .data.preprocessing import assign_health_index
from .features.engineering import extract_features
from .models import (
    train_rf,
    build_cnn_lstm,
    build_transformer,
)
from .evaluation import print_metrics

import numpy as np


def run(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)

    # 1. Load & preprocess
    raw = load_bearing_data(cfg.data.path)
    processed = assign_health_index(raw, cfg.data.health)

    # 2. Feature engineering
    X, y = extract_features(processed, cfg.features)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, shuffle=False
    )

    # 4. Model selection
    if cfg.model.name == "random_forest":
        model = train_rf(X_train, y_train, **cfg.model.params)
    elif cfg.model.name == "cnn_lstm":
        model = build_cnn_lstm(cfg)
        model.fit(
            X_train[..., None],  # add channel dim
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=0.1,
            verbose=2,
        )
    elif cfg.model.name == "transformer":
        model = build_transformer(cfg)
        model.fit(
            X_train[:, None, :],  # [samples, seq, feat]
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=0.1,
            verbose=2,
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    # 5. Evaluate
    if cfg.model.name == "random_forest":
        y_pred = model.predict(X_test)
    elif cfg.model.name == "cnn_lstm":
        y_pred = model.predict(X_test[..., None]).squeeze()
    else:
        y_pred = model.predict(X_test[:, None, :]).squeeze()

    print_metrics(y_test, y_pred)

    # 6. Save checkpoint
    save_checkpoint(model, Path(cfg.output_dir))
