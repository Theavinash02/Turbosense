import numpy as np

from rul.anomaly import HealthMonitor, auroc, evaluate_run_to_failure
from rul.scaling import ConditionScaler

from .conftest import SENSORS


def fitted(engines):
    scaled = ConditionScaler(SENSORS, "condition").fit(engines).transform(engines)
    return HealthMonitor(SENSORS, smooth=5, baseline=10, persist=3).fit(scaled, healthy_rul=30), scaled


def test_auroc_known_values():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_score_rises_towards_failure_and_every_engine_alarms(engines):
    monitor, scaled = fitted(engines)
    score = monitor.score(scaled)
    assert score[scaled.rul <= 3].mean() > 3 * score[scaled.rul >= 30].mean()
    result = evaluate_run_to_failure(monitor, scaled, healthy_rul=30)
    assert result["detected_share"] == 1.0
    assert result["median_lead_cycles"] > 0


def test_alarm_needs_consecutive_cycles(engines):
    monitor, scaled = fitted(engines)
    one = scaled[scaled.unit == 1]
    spike = one.copy()
    spike.loc[spike.index[25], SENSORS] += 50                 # one wild reading in mid-life
    assert monitor.score(spike).iloc[25] > monitor.threshold
    assert not monitor.alarms(spike).iloc[: 25 + 1].any()     # a single spike does not raise an alarm


def test_baseline_removes_an_engine_specific_offset(engines):
    monitor, scaled = fitted(engines)
    shifted = scaled.copy()
    shifted[SENSORS] += 3.0                                    # an engine that simply reads higher on every sensor
    np.testing.assert_allclose(monitor.score(shifted), monitor.score(scaled), rtol=1e-4)  # float32 inputs


def test_save_and_load_give_the_same_scores(engines, tmp_path):
    monitor, scaled = fitted(engines)
    monitor.save(tmp_path / "monitor.json")
    again = HealthMonitor.load(tmp_path / "monitor.json")
    np.testing.assert_allclose(monitor.score(scaled), again.score(scaled))
    assert again.threshold == monitor.threshold
