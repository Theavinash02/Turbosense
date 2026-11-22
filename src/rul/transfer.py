import json

import numpy as np

from rul import data
from rul.evaluate import rmse
from rul.inference import load_models
from rul.model import predict
from rul.windows import last_windows


def transfer_matrix(cfg: dict, subsets: list[str] | None = None) -> dict:
    """Apply each subset's saved models to every subset's test engines, with no retraining.

    Target sensors are scaled with the source subset's statistics, as they would be in deployment.
    """
    subsets = subsets or cfg["data"]["subsets"]
    sensors, window, cap = cfg["features"]["sensors"], cfg["features"]["window"], cfg["labels"]["rul_cap"]
    tests = {s: data.load_subset(cfg, s)[1:] for s in subsets}
    out = {"subsets": subsets, "rmse_capped": {}, "unseen_condition_share": {}}
    for src in subsets:
        scaler, models = load_models(cfg, src, "condition")
        out["rmse_capped"][src], out["unseen_condition_share"][src] = {}, {}
        for tgt in subsets:
            test, rul = tests[tgt]
            X = last_windows(scaler.transform(test), sensors, window)
            truth = np.minimum(rul, cap)
            out["rmse_capped"][src][tgt] = float(np.mean([rmse(truth, predict(m, X, cap)) for m in models]))
            out["unseen_condition_share"][src][tgt] = scaler.unseen_share_
    reports = cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "transfer.json").write_text(json.dumps(out, indent=2))
    return out
