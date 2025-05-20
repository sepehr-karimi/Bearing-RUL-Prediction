# src/rul_estimation/utils.py
from pathlib import Path
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_checkpoint(model, out_dir: Path, name: str = "model") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{name}.pkl"
    if hasattr(model, "save"):
        # Keras / tf
        ckpt = out_dir / f"{name}.h5"
        model.save(ckpt)
    else:
        import joblib
        joblib.dump(model, ckpt)
    return ckpt
