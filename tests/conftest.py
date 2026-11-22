import numpy as np
import pandas as pd
import pytest

from rul.data import ALL_SENSORS, COLUMNS, add_condition, add_rul

SENSORS = ["s2", "s3", "s4"]


def make_engines(n_units: int, conditions: int = 1, seed: int = 0, min_life: int = 40,
                 max_life: int = 70) -> pd.DataFrame:
    """Synthetic run-to-failure engines: sensors drift upward near failure and shift with the condition."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(min_life, max_life))
        for cycle in range(1, life + 1):
            cond = int(rng.integers(conditions))
            wear = np.exp(-(life - cycle) / 15)
            row = {"unit": unit, "cycle": cycle, "setting_1": 10.0 * cond + rng.normal(0, 0.001),
                   "setting_2": 0.25 * cond + rng.normal(0, 0.0005), "setting_3": 100.0}
            row |= {s: 0.0 for s in ALL_SENSORS}
            for i, s in enumerate(SENSORS):
                row[s] = 100 * cond + (i + 1) * wear + rng.normal(0, 0.05)
            rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


@pytest.fixture
def engines():
    return add_rul(add_condition(make_engines(12)), cap=30)


@pytest.fixture
def multi_condition_engines():
    return add_rul(add_condition(make_engines(12, conditions=3)), cap=30)


def make_cfg(tmp_path) -> dict:
    return {
        "seed": 0,
        "paths": {"data": tmp_path / "data", "models": tmp_path / "models", "reports": tmp_path / "reports"},
        "data": {"subsets": ["FD001"]},
        "labels": {"rul_cap": 30},
        "split": {"val_fraction": 0.25},
        "features": {"window": 10, "sensors": SENSORS},
        "lstm": {"hidden": 16, "layers": 1, "dropout": 0.0, "lr": 0.01, "batch_size": 64,
                 "max_epochs": 15, "patience": 4, "seeds": [0]},
        "lightgbm": {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
        "anomaly": {"smooth": 5, "baseline_cycles": 10, "quantile": 0.99, "persist": 3, "near_failure_rul": [10]},
        "serve": {"subset": "FD001"},
        "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"},
    }


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


def write_subset(cfg, name="FD001"):
    """Write synthetic train, test and RUL files in the C-MAPSS layout; return the true test RUL."""
    root = cfg["paths"]["data"]
    root.mkdir(parents=True)
    make_engines(24, seed=1).to_csv(root / f"train_{name}.txt", sep=" ", header=False, index=False)
    test_full = make_engines(8, seed=2)
    life = test_full.groupby("unit")["cycle"].transform("max")
    cut = (life * 0.7).astype(int)
    test_full[test_full["cycle"] <= cut].to_csv(root / f"test_{name}.txt", sep=" ", header=False, index=False)
    rul = (life - cut).groupby(test_full["unit"]).first()
    rul.to_csv(root / f"RUL_{name}.txt", header=False, index=False)
    return rul.to_numpy()
