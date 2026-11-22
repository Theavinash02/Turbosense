import numpy as np

from rul.scaling import ConditionScaler

from .conftest import SENSORS


def test_condition_scaling_removes_the_regime_offset(multi_condition_engines):
    df = multi_condition_engines
    per_condition = ConditionScaler(SENSORS, "condition").fit(df).transform(df)
    global_ = ConditionScaler(SENSORS, "global").fit(df).transform(df)
    # After per-condition scaling every regime is centred on zero; with one global scaler it is not.
    assert per_condition.groupby("condition")["s2"].mean().abs().max() < 1e-4
    assert global_.groupby("condition")["s2"].mean().abs().max() > 0.5


def test_unseen_condition_falls_back_to_global_statistics(multi_condition_engines):
    df = multi_condition_engines
    seen = df[df.condition != df.condition.iloc[0]]
    scaler = ConditionScaler(SENSORS, "condition").fit(seen)
    out = scaler.transform(df)
    assert 0 < scaler.unseen_share_ < 1
    assert np.isfinite(out[SENSORS].to_numpy()).all()


def test_constant_sensor_becomes_zero_not_nan(engines):
    df = engines.assign(s4=5.0)
    out = ConditionScaler(SENSORS, "condition").fit(df).transform(df)
    assert (out["s4"] == 0).all()


def test_save_and_load_give_the_same_transform(multi_condition_engines, tmp_path):
    df = multi_condition_engines
    scaler = ConditionScaler(SENSORS, "condition").fit(df)
    scaler.save(tmp_path / "scaler.json")
    again = ConditionScaler.load(tmp_path / "scaler.json")
    np.testing.assert_allclose(scaler.transform(df)[SENSORS], again.transform(df)[SENSORS], rtol=1e-5)
