import numpy as np

from rul.windows import all_windows, last_windows, pad_front, rolling_features

from .conftest import SENSORS


def test_window_count_and_targets(engines):
    X, y = all_windows(engines, SENSORS, window=10)
    expected = sum(len(g) - 10 + 1 for _, g in engines.groupby("unit"))
    assert X.shape == (expected, 10, len(SENSORS))
    first = engines[engines.unit == 1]
    assert y[0] == first["y"].iloc[9]            # target is taken at the window's last cycle
    np.testing.assert_allclose(X[0], first[SENSORS].to_numpy(np.float32)[:10])


def test_last_windows_end_at_the_last_cycle(engines):
    X = last_windows(engines, SENSORS, window=10)
    assert X.shape == (engines.unit.nunique(), 10, len(SENSORS))
    last = engines[engines.unit == 1][SENSORS].to_numpy(np.float32)[-1]
    np.testing.assert_allclose(X[0, -1], last)


def test_short_history_is_padded_with_its_first_row():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    padded = pad_front(a, 5)
    assert padded.shape == (5, 2)
    np.testing.assert_array_equal(padded[:3], np.repeat(a[:1], 3, axis=0))
    np.testing.assert_array_equal(padded[-3:], a)


def test_rolling_features_never_look_across_engines(engines):
    feats = rolling_features(engines, SENSORS, window=10)
    first_rows = engines.groupby("unit").head(1).index
    # On an engine's first cycle the rolling mean is just that cycle's value.
    np.testing.assert_allclose(feats.loc[first_rows, "s2_mean"], engines.loc[first_rows, "s2"])
