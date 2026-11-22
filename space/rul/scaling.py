import json
from pathlib import Path

import pandas as pd


class ConditionScaler:
    """Standardise sensors, either per operating condition or with one global mean and std.

    Fit on training engines only. A condition that was not seen when fitting falls back to the
    global statistics; `unseen_share_` records what share of the last transformed rows that was.
    """

    def __init__(self, sensors: list[str], mode: str = "condition"):
        if mode not in ("condition", "global"):
            raise ValueError(f"unknown scaling mode: {mode}")
        self.sensors, self.mode = list(sensors), mode
        self.unseen_share_ = 0.0

    def fit(self, df: pd.DataFrame) -> "ConditionScaler":
        self.g_mean = df[self.sensors].mean()
        self.g_std = df[self.sensors].std().fillna(0)
        self.g_std = self.g_std.where(self.g_std > 1e-8, 1.0)
        if self.mode == "condition":
            grouped = df.groupby("condition")[self.sensors]
            self.c_mean = grouped.mean()
            std = grouped.std().fillna(0)
            # A sensor that is constant within a condition carries no signal there: scale by 1 so it becomes 0.
            self.c_std = std.where(std > 1e-8, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        values = (df[self.sensors] - self.g_mean) / self.g_std
        self.unseen_share_ = 0.0
        if self.mode == "condition":
            known = df["condition"].isin(self.c_mean.index)
            if known.any():
                part = df.loc[known]
                mean = self.c_mean.loc[part["condition"]].to_numpy()
                std = self.c_std.loc[part["condition"]].to_numpy()
                values.loc[known] = (part[self.sensors].to_numpy() - mean) / std
            self.unseen_share_ = float(1 - known.mean())
        out[self.sensors] = values.astype("float32")
        return out

    def save(self, path: Path) -> None:
        state = {"sensors": self.sensors, "mode": self.mode,
                 "g_mean": self.g_mean.to_dict(), "g_std": self.g_std.to_dict()}
        if self.mode == "condition":
            state["c_mean"] = self.c_mean.to_dict(orient="index")
            state["c_std"] = self.c_std.to_dict(orient="index")
        Path(path).write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "ConditionScaler":
        state = json.loads(Path(path).read_text())
        scaler = cls(state["sensors"], state["mode"])
        scaler.g_mean, scaler.g_std = pd.Series(state["g_mean"]), pd.Series(state["g_std"])
        if scaler.mode == "condition":
            scaler.c_mean = pd.DataFrame.from_dict(state["c_mean"], orient="index")[scaler.sensors]
            scaler.c_std = pd.DataFrame.from_dict(state["c_std"], orient="index")[scaler.sensors]
        return scaler
