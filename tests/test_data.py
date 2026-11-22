import pandas as pd
import pytest

from rul.data import add_condition, check_split, split_engines


def test_rul_counts_down_to_zero_and_is_capped(engines):
    one = engines[engines.unit == 1]
    assert one["rul"].iloc[-1] == 0
    assert (one["rul"].diff().dropna() == -1).all()
    assert engines["y"].max() == 30
    assert (engines["y"] <= engines["rul"]).all()


def test_split_keeps_every_engine_on_one_side(engines):
    train, val = split_engines(engines, 0.25, seed=0)
    assert not set(train.unit) & set(val.unit)
    assert val.unit.nunique() == 3
    assert len(train) + len(val) == len(engines)
    check_split(train, val)


def test_check_split_catches_a_shared_engine(engines):
    train, val = split_engines(engines, 0.25, seed=0)
    leaked = pd.concat([val, train.iloc[:1]])
    with pytest.raises(AssertionError, match="both train and validation"):
        check_split(train, leaked)


def test_condition_ignores_jitter_and_negative_zero():
    df = pd.DataFrame({"setting_1": [0.0004, -0.0003, 10.002], "setting_2": [0.0004, -0.0004, 0.2503],
                       "setting_3": [100.0, 100.0, 100.0]})
    labels = add_condition(df)["condition"].tolist()
    assert labels[0] == labels[1] == "0_0.0_100"
    assert labels[2] == "10_0.25_100"
