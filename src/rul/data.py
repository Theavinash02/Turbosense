import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["setting_1", "setting_2", "setting_3"]
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *ALL_SENSORS]


def download(cfg: dict) -> Path:
    """Fetch the C-MAPSS text files into the data folder, unless they are already there."""
    target = cfg["paths"]["data"]
    if not (target / "train_FD001.txt").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", cfg["data"]["mirror"], str(target)], check=True)
    return target


def add_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a `condition` label: one value per operating regime.

    The simulator uses a few fixed regimes plus jitter, so rounding the three settings recovers them.
    """
    out = df.copy()
    out["condition"] = (
        out["setting_1"].round(0).astype(int).astype(str)
        + "_"
        + (out["setting_2"].round(2) + 0.0).astype(str)  # + 0.0 turns -0.0 into 0.0
        + "_"
        + out["setting_3"].round(0).astype(int).astype(str)
    )
    return out


def add_rul(train: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Return train with `rul` (cycles until the engine's last cycle) and `y` (rul clipped at cap)."""
    out = train.copy()
    out["rul"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    out["y"] = out["rul"].clip(upper=cap).astype("float32")
    return out


def load_subset(cfg: dict, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train with labels, test, true RUL per test engine) for one subset such as "FD001"."""
    root = cfg["paths"]["data"]

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(root / name, sep=r"\s+", header=None, names=COLUMNS)

    train = add_rul(add_condition(read(f"train_{subset}.txt")), cfg["labels"]["rul_cap"])
    test = add_condition(read(f"test_{subset}.txt"))
    rul = pd.read_csv(root / f"RUL_{subset}.txt", header=None)[0].to_numpy()
    if len(rul) != test["unit"].nunique():
        raise ValueError(f"{subset}: {len(rul)} RUL labels for {test['unit'].nunique()} test engines")
    return train, test, rul


def split_engines(train: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by engine, never by row: every cycle of an engine lands on the same side."""
    units = np.random.default_rng(seed).permutation(train["unit"].unique())
    val_units = set(units[: max(1, int(round(val_fraction * len(units))))])
    in_val = train["unit"].isin(val_units)
    return train[~in_val], train[in_val]


def check_split(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Raise if any engine appears on both sides."""
    shared = set(train["unit"]) & set(val["unit"])
    if shared:
        raise AssertionError(f"{len(shared)} engines appear in both train and validation")
