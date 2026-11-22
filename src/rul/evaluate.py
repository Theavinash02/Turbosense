import numpy as np


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def nasa_score(y, pred) -> float:
    """Return the PHM08 score (Saxena et al., 2008): late predictions cost more than early ones. Lower is better."""
    d = np.asarray(pred, float) - np.asarray(y, float)  # positive = predicted too late
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def test_metrics(rul: np.ndarray, pred: np.ndarray, cap: int) -> dict:
    """Return the test metrics for one prediction per test engine."""
    return {
        "rmse_capped": rmse(np.minimum(rul, cap), pred),
        "rmse_true": rmse(rul, pred),
        "nasa": nasa_score(rul, pred),
        "nasa_per_engine": nasa_score(rul, pred) / len(rul),
    }


test_metrics.__test__ = False  # not a pytest test, despite the name
