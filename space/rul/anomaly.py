"""Unsupervised health monitor: how far has an engine moved from its own healthy baseline?

It is fitted on early-life cycles only, uses no failure labels, and is separate from the RUL model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class HealthMonitor:
    """Mahalanobis distance of an engine's smoothed sensors from its own early-life baseline.

    - Each engine's baseline is the mean of its first `baseline` cycles (0 = compare with the fleet mean).
    - The distance uses the covariance of healthy training residuals.
    - An alarm fires once the distance has been above the threshold for `persist` cycles in a row.

    Frames must be scaled and sorted by unit and cycle, with each engine's history starting at its first cycle.
    """

    def __init__(self, sensors: list[str], smooth: int = 10, baseline: int = 20, quantile: float = 0.99,
                 persist: int = 5):
        self.sensors, self.smooth, self.baseline = list(sensors), smooth, baseline
        self.quantile, self.persist = quantile, persist

    def residuals(self, df: pd.DataFrame) -> np.ndarray:
        rolled = df.groupby("unit")[self.sensors].rolling(self.smooth, min_periods=1).mean()
        x = rolled.reset_index(level=0, drop=True).loc[df.index].to_numpy(float)
        if self.baseline > 0:
            early = df[df.groupby("unit").cumcount() < self.baseline]
            x = x - early.groupby("unit")[self.sensors].mean().loc[df["unit"]].to_numpy(float)
        return x

    def fit(self, train: pd.DataFrame, healthy_rul: int) -> "HealthMonitor":
        """Fit on training rows with at least `healthy_rul` cycles left, after the baseline period."""
        x = self.residuals(train)
        healthy = ((train["rul"] >= healthy_rul) & (train.groupby("unit").cumcount() >= self.baseline)).to_numpy()
        self.mean = x[healthy].mean(axis=0)
        self.precision = np.linalg.pinv(np.cov(x[healthy], rowvar=False))
        self.threshold = float(np.quantile(self._distance(x[healthy]), self.quantile))
        return self

    def _distance(self, x: np.ndarray) -> np.ndarray:
        d = x - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d).clip(min=0))

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return the distance for every row (higher = further from healthy)."""
        return pd.Series(self._distance(self.residuals(df)), index=df.index, name="anomaly_score")

    def alarms(self, df: pd.DataFrame, score: pd.Series | None = None) -> pd.Series:
        """Return True on every cycle that ends a run of `persist` cycles above the threshold."""
        above = ((self.score(df) if score is None else score) > self.threshold).astype(int)
        streak = above.groupby(df["unit"]).rolling(self.persist, min_periods=self.persist).sum()
        return (streak.reset_index(level=0, drop=True).loc[df.index] >= self.persist).rename("alarm")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "sensors": self.sensors, "smooth": self.smooth, "baseline": self.baseline, "quantile": self.quantile,
            "persist": self.persist, "mean": self.mean.tolist(), "precision": self.precision.tolist(),
            "threshold": self.threshold}))

    @classmethod
    def load(cls, path: Path) -> "HealthMonitor":
        s = json.loads(Path(path).read_text())
        monitor = cls(s["sensors"], s["smooth"], s["baseline"], s["quantile"], s["persist"])
        monitor.mean, monitor.precision = np.array(s["mean"]), np.array(s["precision"])
        monitor.threshold = s["threshold"]
        return monitor


def auroc(labels, scores) -> float:
    """Area under the ROC curve: the chance a positive outranks a negative (ties count half)."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_run_to_failure(monitor: HealthMonitor, df: pd.DataFrame, healthy_rul: int) -> dict:
    """Score a monitor on held-out engines that run to failure (scaled frame with a `rul` column)."""
    score = monitor.score(df)
    alarm = monitor.alarms(df, score)
    first = df.loc[alarm].groupby("unit")["rul"].max()  # cycles left when the alarm first fired
    n = df["unit"].nunique()
    healthy = (df["rul"] >= healthy_rul) & (df.groupby("unit").cumcount() >= monitor.baseline)
    lead = first.to_numpy(float)
    return {
        "engines": int(n),
        "detected_share": float(len(first) / n),
        "median_lead_cycles": float(np.median(lead)) if len(lead) else None,
        "lead_p10_cycles": float(np.quantile(lead, 0.1)) if len(lead) else None,
        "early_alarm_share": float((first >= healthy_rul).sum() / n),  # fired while still in the healthy range
        "healthy_row_exceed_rate": float((score[healthy] > monitor.threshold).mean()),
    }


def run_anomaly(cfg: dict, subset: str) -> dict:
    """Fit the monitor on one subset, evaluate it, save it next to the models and write a JSON report."""
    from rul.inference import model_dir
    from rul.train import prepare

    a, cap, sensors = cfg["anomaly"], cfg["labels"]["rul_cap"], cfg["features"]["sensors"]
    prepared = prepare(cfg, subset, "condition")
    last = prepared["test"].groupby("unit")["cycle"].idxmax()
    rul = prepared["rul"]

    def evaluate(baseline: int) -> tuple[HealthMonitor, dict]:
        monitor = HealthMonitor(sensors, a["smooth"], baseline, a["quantile"], a["persist"])
        monitor.fit(prepared["train"], cap)
        test_score = monitor.score(prepared["test"]).loc[last].to_numpy()
        return monitor, {
            "validation": evaluate_run_to_failure(monitor, prepared["val"], cap),
            "test_auroc": {f"rul_le_{k}": auroc(rul <= k, test_score) for k in a["near_failure_rul"]},
        }

    monitor, own = evaluate(a["baseline_cycles"])
    _, fleet = evaluate(0)
    monitor.save(model_dir(cfg, subset, "condition") / "monitor.json")
    result = {"subset": subset, "settings": dict(a), "threshold": monitor.threshold,
              "own_baseline": own, "fleet_reference": fleet}
    reports = cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"anomaly_{subset}.json").write_text(json.dumps(result, indent=2))
    v = own["validation"]
    print(f"{subset} monitor: detected {v['detected_share']:.0%} of validation engines, "
          f"median lead {v['median_lead_cycles']} cycles, early alarms {v['early_alarm_share']:.0%}")
    return result
