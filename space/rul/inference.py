"""Loading saved models and scoring one engine. Imports only what serving needs (no LightGBM, no MLflow)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rul.anomaly import HealthMonitor
from rul.data import add_condition
from rul.model import RULNet, default_device, predict
from rul.scaling import ConditionScaler
from rul.windows import last_windows


def model_dir(cfg: dict, subset: str, scaling: str = "condition") -> Path:
    d = cfg["paths"]["models"] / f"{subset}_{scaling}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_models(cfg: dict, subset: str, scaling: str = "condition", device: str | None = None):
    """Return (scaler, [models]) saved by `rul train`."""
    p, d = cfg["lstm"], model_dir(cfg, subset, scaling)
    device = device or default_device()
    if not (d / "scaler.json").exists():
        raise FileNotFoundError(f"no saved models in {d}; run `rul train --subset {subset}` first")
    scaler = ConditionScaler.load(d / "scaler.json")
    models = []
    for path in sorted(d.glob("lstm_seed*.pt")):
        model = RULNet(len(scaler.sensors), p["hidden"], p["layers"], p["dropout"]).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.eval())
    if not models:
        raise FileNotFoundError(f"no saved models in {d}; run `rul train --subset {subset}` first")
    return scaler, models


class Predictor:
    """Everything needed to score one engine's history: scaler, LSTM seeds and (if saved) the health monitor."""

    def __init__(self, cfg: dict, subset: str, device: str = "cpu"):
        self.cfg, self.subset = cfg, subset
        self.cap, self.window = cfg["labels"]["rul_cap"], cfg["features"]["window"]
        self.scaler, self.models = load_models(cfg, subset, "condition", device)
        self.sensors = self.scaler.sensors
        monitor_path = model_dir(cfg, subset) / "monitor.json"
        self.monitor = HealthMonitor.load(monitor_path) if monitor_path.exists() else None

    def predict(self, history: pd.DataFrame) -> dict:
        """Score one engine. `history` has cycle, the three settings and the sensors, oldest cycle first."""
        df = add_condition(history.sort_values("cycle").reset_index(drop=True))
        df["unit"] = 1
        scaled = self.scaler.transform(df)
        X = last_windows(scaled, self.sensors, self.window)
        per_seed = np.array([float(predict(m, X, self.cap)[0]) for m in self.models])
        out = {
            "cycles_received": int(len(df)),
            "padded": bool(len(df) < self.window),
            "rul_cycles": float(per_seed.mean()),
            "rul_min": float(per_seed.min()),
            "rul_max": float(per_seed.max()),
            "at_cap": bool(per_seed.mean() >= self.cap - 1),
            "unseen_condition_share": float(self.scaler.unseen_share_),
            "anomaly": self._anomaly(df, scaled),
        }
        return out

    def _anomaly(self, df: pd.DataFrame, scaled: pd.DataFrame) -> dict:
        if self.monitor is None:
            return {"available": False, "reason": "no health monitor saved for this subset"}
        m = self.monitor
        if int(df["cycle"].iloc[0]) != 1 or len(df) <= m.baseline:
            return {"available": False,
                    "reason": f"needs the history from cycle 1 and more than {m.baseline} cycles"}
        score = m.score(scaled)
        return {"available": True, "score": float(score.iloc[-1]), "threshold": float(m.threshold),
                "alarm": bool(m.alarms(scaled, score).iloc[-1])}
