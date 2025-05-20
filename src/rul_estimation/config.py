
---

## `src/rul_estimation/config.py`

```python
# src/rul_estimation/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass
class DataConfig:
    path: Path
    health: str | None = None


@dataclass
class FeatureConfig:
    selected: List[str] | None = None
    window: int = 50               # rolling window length


@dataclass
class ModelConfig:
    name: str                      # "random_forest", "cnn_lstm", "transformer"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    # paths
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    output_dir: Path = Path("artifacts/")
    seed: int = 42
    test_size: float = 0.2         # fraction
    epochs: int = 5                # DL models
    batch_size: int = 32

    @classmethod
    def from_yaml(cls, file: str | Path) -> "TrainingConfig":
        with open(file, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)
