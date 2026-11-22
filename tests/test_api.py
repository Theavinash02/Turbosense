import pytest
from fastapi.testclient import TestClient

from rul import anomaly, train
from rul.api import create_app

from .conftest import SENSORS, make_cfg, make_engines, write_subset


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = make_cfg(tmp_path_factory.mktemp("api"))
    write_subset(cfg)
    train.train_subset(cfg, "FD001", "condition", track=False)
    anomaly.run_anomaly(cfg, "FD001")
    return TestClient(create_app(cfg, "FD001"))


def history(first=1, last=45, drop=None):
    engine = make_engines(1, seed=7, min_life=60, max_life=61)
    rows = engine[(engine.cycle >= first) & (engine.cycle <= last)]
    return [{"cycle": int(r.cycle), "setting_1": r.setting_1, "setting_2": r.setting_2, "setting_3": r.setting_3,
             "sensors": {s: float(getattr(r, s)) for s in SENSORS if s != drop}} for r in rows.itertuples()]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models"] == 1 and body["health_monitor"] is True
    assert body["sensors"] == SENSORS


def test_predict_returns_rul_and_anomaly(client):
    response = client.post("/predict", json={"engine_id": "E7", "cycles": history()})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine_id"] == "E7" and body["cycles_received"] == 45 and body["padded"] is False
    assert 0 <= body["rul_min"] <= body["rul_cycles"] <= body["rul_max"] <= 30
    assert body["unseen_condition_share"] == 0
    assert body["anomaly"]["available"] is True and body["anomaly"]["score"] >= 0


def test_prediction_falls_as_the_engine_ages(client):
    early = client.post("/predict", json={"cycles": history(last=25)}).json()["rul_cycles"]
    late = client.post("/predict", json={"cycles": history(last=58)}).json()["rul_cycles"]
    assert late < early


def test_short_history_is_padded_and_anomaly_is_unavailable(client):
    body = client.post("/predict", json={"cycles": history(first=1, last=4)}).json()
    assert body["padded"] is True
    assert body["anomaly"]["available"] is False

    body = client.post("/predict", json={"cycles": history(first=20, last=50)}).json()
    assert body["anomaly"]["available"] is False       # history does not start at the engine's first cycle


def test_bad_requests_are_rejected(client):
    assert client.post("/predict", json={"cycles": []}).status_code == 422
    missing = client.post("/predict", json={"cycles": history(drop="s3")})
    assert missing.status_code == 422 and "s3" in missing.text
    out_of_order = history()[::-1]
    assert client.post("/predict", json={"cycles": out_of_order}).status_code == 422
