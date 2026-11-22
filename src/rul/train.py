import json
import time

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

from rul import data
from rul.evaluate import rmse, test_metrics
from rul.inference import model_dir
from rul.model import RULNet, default_device, predict
from rul.scaling import ConditionScaler
from rul.windows import all_windows, last_windows, rolling_features


def prepare(cfg: dict, subset: str, scaling: str) -> dict:
    """Load one subset, split it by engine and scale it with statistics from the training engines."""
    train, test, rul = data.load_subset(cfg, subset)
    tr, va = data.split_engines(train, cfg["split"]["val_fraction"], cfg["seed"])
    data.check_split(tr, va)
    scaler = ConditionScaler(cfg["features"]["sensors"], scaling).fit(tr)
    return {"subset": subset, "scaling": scaling, "scaler": scaler, "rul": rul,
            "train": scaler.transform(tr), "val": scaler.transform(va), "test": scaler.transform(test)}


def fit_lstm(cfg: dict, Xtr, ytr, Xva, yva, seed: int, device: str | None = None):
    """Train one LSTM with early stopping on validation RMSE. Returns (model, best val RMSE, history)."""
    p, cap = cfg["lstm"], cfg["labels"]["rul_cap"]
    device = device or default_device()
    torch.manual_seed(seed)
    model = RULNet(Xtr.shape[2], p["hidden"], p["layers"], p["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = nn.MSELoss()
    X, y = torch.as_tensor(Xtr).to(device), (torch.as_tensor(ytr) / cap).to(device)
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(p["max_epochs"]):
        model.train()
        order, total = torch.randperm(len(X), device=device), 0.0
        for i in range(0, len(X), p["batch_size"]):
            idx = order[i : i + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val_rmse = rmse(yva, predict(model, Xva, cap))
        history.append({"epoch": epoch + 1, "train_mse": total / len(X), "val_rmse": val_rmse})
        if val_rmse < best - 1e-3:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    return model, float(best), history


def fit_lightgbm(cfg: dict, prepared: dict) -> dict:
    """Train the tabular baseline on rolling features and return its validation and test metrics."""
    sensors, window, cap = cfg["features"]["sensors"], cfg["features"]["window"], cfg["labels"]["rul_cap"]
    model = lgb.LGBMRegressor(**cfg["lightgbm"], random_state=cfg["seed"], verbose=-1)
    model.fit(rolling_features(prepared["train"], sensors, window), prepared["train"]["y"])
    last = prepared["test"].groupby("unit")["cycle"].idxmax().to_numpy()
    pred = np.clip(model.predict(rolling_features(prepared["test"], sensors, window).loc[last]), 0, cap)
    val_pred = np.clip(model.predict(rolling_features(prepared["val"], sensors, window)), 0, cap)
    return {"val_rmse": rmse(prepared["val"]["y"], val_pred), **test_metrics(prepared["rul"], pred, cap)}


def train_subset(cfg: dict, subset: str, scaling: str = "condition", track: bool = True) -> dict:
    """Train the baseline and one LSTM per seed on a subset; save models, a report and MLflow runs."""
    sensors, window, cap = cfg["features"]["sensors"], cfg["features"]["window"], cfg["labels"]["rul_cap"]
    if track:
        from rul import tracking  # imported here so serving does not need MLflow
    prepared = prepare(cfg, subset, scaling)
    out_dir = model_dir(cfg, subset, scaling)
    prepared["scaler"].save(out_dir / "scaler.json")

    result = {"subset": subset, "scaling": scaling, "test_engines": int(len(prepared["rul"])),
              "train_engines": int(prepared["train"]["unit"].nunique()),
              "val_engines": int(prepared["val"]["unit"].nunique()),
              "conditions": int(prepared["train"]["condition"].nunique()),
              "lightgbm": fit_lightgbm(cfg, prepared), "lstm": []}
    if track:
        tracking.log_run(cfg, f"{subset}-{scaling}-lightgbm",
                         {"subset": subset, "scaling": scaling, "model": "lightgbm", **cfg["lightgbm"]},
                         result["lightgbm"])

    Xtr, ytr = all_windows(prepared["train"], sensors, window)
    Xva, yva = all_windows(prepared["val"], sensors, window)
    Xte = last_windows(prepared["test"], sensors, window)
    for seed in cfg["lstm"]["seeds"]:
        t0 = time.time()
        model, best_val, history = fit_lstm(cfg, Xtr, ytr, Xva, yva, seed)
        metrics = {"val_rmse": best_val, **test_metrics(prepared["rul"], predict(model, Xte, cap), cap)}
        torch.save(model.state_dict(), out_dir / f"lstm_seed{seed}.pt")
        result["lstm"].append({"seed": seed, "epochs": len(history), "seconds": round(time.time() - t0, 1),
                               "first_train_mse": history[0]["train_mse"],
                               "last_train_mse": history[-1]["train_mse"], **metrics})
        print(f"{subset} [{scaling}] seed {seed}: {len(history)} epochs, "
              f"train MSE {history[0]['train_mse']:.4f} -> {history[-1]['train_mse']:.4f}, "
              f"val RMSE {best_val:.2f}, test RMSE {metrics['rmse_capped']:.2f} (capped)")
        if track:
            tracking.log_run(cfg, f"{subset}-{scaling}-lstm-seed{seed}",
                             {"subset": subset, "scaling": scaling, "model": "lstm", "seed": seed,
                              "window": window, "rul_cap": cap, "n_sensors": len(sensors), **cfg["lstm"]},
                             metrics, history=history,
                             artifacts=[out_dir / f"lstm_seed{seed}.pt", out_dir / "scaler.json"])

    reports = cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"train_{subset}_{scaling}.json").write_text(json.dumps(result, indent=2))
    return result
