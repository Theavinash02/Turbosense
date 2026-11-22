from pathlib import Path

import mlflow


def _flat(params: dict) -> dict:
    return {k: (",".join(map(str, v)) if isinstance(v, list) else v) for k, v in params.items()}


def log_run(cfg: dict, name: str, params: dict, metrics: dict,
            history: list[dict] | None = None, artifacts: list[Path] | None = None) -> None:
    """Record one training run in MLflow: parameters, final metrics, per-epoch curves and saved files."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=name):
        mlflow.log_params(_flat(params))
        mlflow.log_metrics({f"test_{k}" if k != "val_rmse" else k: float(v) for k, v in metrics.items()})
        for row in history or []:
            mlflow.log_metric("epoch_train_mse", row["train_mse"], step=row["epoch"])
            mlflow.log_metric("epoch_val_rmse", row["val_rmse"], step=row["epoch"])
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
