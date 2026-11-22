"""Train end to end on synthetic engines: no download needed."""

import json

import numpy as np

from rul import inference, report, train, transfer
from rul.evaluate import rmse

from .conftest import write_subset


def test_train_transfer_and_report(cfg):
    rul = write_subset(cfg)
    result = train.train_subset(cfg, "FD001", "condition", track=True)

    run = result["lstm"][0]
    assert run["last_train_mse"] < run["first_train_mse"]          # the loss actually went down
    constant = rmse(np.minimum(rul, 30), np.full(len(rul), 15.0))
    assert run["rmse_capped"] < constant                           # and the model beats a constant guess
    assert (cfg["paths"]["models"] / "FD001_condition" / "lstm_seed0.pt").exists()

    scaler, models = inference.load_models(cfg, "FD001")
    assert len(models) == 1 and scaler.mode == "condition"

    matrix = transfer.transfer_matrix(cfg)
    assert np.isclose(matrix["rmse_capped"]["FD001"]["FD001"], run["rmse_capped"], atol=1e-4)

    text = report.build_report(cfg)
    assert "| FD001 |" in text
    saved = json.loads((cfg["paths"]["reports"] / "train_FD001_condition.json").read_text())
    assert saved["lightgbm"]["rmse_capped"] > 0


def test_mlflow_run_is_recorded(cfg):
    import mlflow

    write_subset(cfg)
    train.train_subset(cfg, "FD001", "condition", track=True)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == 2                                           # one LightGBM run, one LSTM run
    assert "metrics.test_rmse_capped" in runs.columns
    assert runs["metrics.epoch_val_rmse"].notna().any()
