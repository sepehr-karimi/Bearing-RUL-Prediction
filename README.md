# RUL-Estimation 📦  
**Bearing Remaining-Useful-Life prediction toolkit**

> A clean, reproducible Python package that grew out of the original Colab notebook  
> `notebooks/RUL_Estimation.ipynb`.  It streamlines data ingestion, feature engineering,
> model training (Random-Forest, CNN-LSTM, Transformer) and evaluation—turning a one-off
> experiment into production-ready code.

---

## Table of Contents
1. [Project Goals](#project-goals)
2. [Folder Layout](#folder-layout)
3. [Quick Install](#quick-install)
4. [Training in 60 seconds](#training-in-60-seconds)
5. [Making Predictions](#making-predictions)
6. [Configuration files](#configuration-files)
7. [Notebooks for Exploration](#notebooks-for-exploration)
8. [Running Tests & CI](#running-tests--ci)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Goals
* **Reproducible research** – deterministic seeds, YAML configs, unit tests.  
* **Modular codebase** – separate data, features, models, evaluation.  
* **CLI workflow** – `rul-train` / `rul-predict` for headless runs or cron jobs.  
* **Extensibility** – drop-in new feature extractors or models with minimal boilerplate.

---

## Folder Layout

```text
rul-estimation-project/
│
├─ pyproject.toml        ← PEP 621 build metadata (editable install)
├─ requirements.txt      ← pinned run-time deps
├─ README.md             ← **← you are here**
│
├─ data/
│   ├─ raw/              ← untouched source CSV / Parquet
│   ├─ interim/          ← intermediate artefacts
│   └─ processed/        ← final model-ready files
│
├─ notebooks/
│   └─ RUL_Estimation.ipynb
│
├─ configs/
│   └─ train.yaml        ← sample hyper-parameters & paths
│
├─ scripts/              ← CLI entry points (`python -m scripts.train …`)
│   ├─ train.py
│   └─ predict.py
│
├─ src/
│   └─ rul_estimation/
│       ├─ data/         ← loaders & preprocessing
│       ├─ features/     ← feature engineering
│       ├─ models/       ← RF, CNN-LSTM, Transformer
│       ├─ evaluation.py
│       ├─ visualization.py
│       ├─ pipeline.py   ← end-to-end orchestration
│       ├─ config.py     ← dataclass config loader
│       └─ utils.py
│
└─ tests/                ← fast pytest suite
