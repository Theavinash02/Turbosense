import numpy as np
import pandas as pd


def pad_front(a: np.ndarray, window: int) -> np.ndarray:
    """Pad a short history to `window` rows by repeating its first row."""
    if len(a) < window:
        a = np.vstack([np.repeat(a[:1], window - len(a), axis=0), a])
    return a


def all_windows(df: pd.DataFrame, sensors: list[str], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return every (window, sensors) slice of every engine, with the target at the slice's last cycle."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        a = pad_front(g[sensors].to_numpy(np.float32), window)
        target, offset = g["y"].to_numpy(np.float32), max(0, window - len(g))
        for end in range(window, len(a) + 1):
            X.append(a[end - window : end])
            y.append(target[end - 1 - offset])
    return np.stack(X), np.array(y, np.float32)


def last_windows(df: pd.DataFrame, sensors: list[str], window: int) -> np.ndarray:
    """Return one slice per engine, ending at its last observed cycle (engines in unit order)."""
    return np.stack([pad_front(g[sensors].to_numpy(np.float32), window)[-window:] for _, g in df.groupby("unit")])


def rolling_features(df: pd.DataFrame, sensors: list[str], window: int) -> pd.DataFrame:
    """Return current sensor values plus their rolling mean and std per engine, for the tabular baseline."""
    g = df.groupby("unit")[sensors]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).add_suffix("_mean")
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0).add_suffix("_std")
    return pd.concat([df[sensors], mean, std], axis=1)
