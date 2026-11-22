"""HTTP service: send an engine's sensor history, get its remaining useful life and health status.

Run with `rul serve`, or `uvicorn rul.api:create_app --factory`.
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rul.config import load_config
from rul.data import SETTINGS
from rul.inference import Predictor


class Cycle(BaseModel):
    cycle: int = Field(ge=1, description="Flight cycle number, counted from the engine's first cycle")
    setting_1: float
    setting_2: float
    setting_3: float
    sensors: dict[str, float] = Field(description="Sensor readings by name, e.g. {'s2': 642.1, ...}")


class PredictRequest(BaseModel):
    engine_id: str | None = None
    cycles: list[Cycle] = Field(min_length=1, description="The engine's history, oldest cycle first")


class Anomaly(BaseModel):
    available: bool
    reason: str | None = None
    score: float | None = None
    threshold: float | None = None
    alarm: bool | None = None


class PredictResponse(BaseModel):
    engine_id: str | None
    model_subset: str
    cycles_received: int
    padded: bool = Field(description="True if the history was shorter than the model's window")
    rul_cycles: float = Field(description="Predicted remaining useful life, mean over the model seeds")
    rul_min: float = Field(description="Lowest prediction among the seeds")
    rul_max: float = Field(description="Highest prediction among the seeds")
    at_cap: bool = Field(description="True if the prediction is at the model's ceiling: read it as 'at least this'")
    unseen_condition_share: float = Field(description="Share of cycles in operating conditions the model never saw")
    anomaly: Anomaly


def create_app(cfg: dict | None = None, subset: str | None = None) -> FastAPI:
    cfg = cfg or load_config()
    subset = subset or os.environ.get("RUL_SUBSET") or cfg["serve"]["subset"]
    predictor = Predictor(cfg, subset)
    app = FastAPI(title="Turbofan remaining useful life", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_subset": subset, "models": len(predictor.models),
                "rul_cap": predictor.cap, "window": predictor.window, "sensors": predictor.sensors,
                "health_monitor": predictor.monitor is not None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> dict:
        numbers = [c.cycle for c in request.cycles]
        if any(b <= a for a, b in zip(numbers, numbers[1:], strict=False)):
            raise HTTPException(422, "cycles must be in strictly increasing order")
        missing = sorted({s for c in request.cycles for s in predictor.sensors if s not in c.sensors})
        if missing:
            raise HTTPException(422, f"every cycle needs these sensors; missing: {missing}")
        rows = [{"cycle": c.cycle, **{k: getattr(c, k) for k in SETTINGS},
                 **{s: c.sensors[s] for s in predictor.sensors}} for c in request.cycles]
        return {"engine_id": request.engine_id, "model_subset": subset, **predictor.predict(pd.DataFrame(rows))}

    return app
