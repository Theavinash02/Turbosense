import numpy as np

from rul.evaluate import nasa_score, rmse, test_metrics


def test_rmse():
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)


def test_nasa_score_is_zero_when_exact_and_punishes_late_more_than_early():
    assert nasa_score([50, 60], [50, 60]) == 0
    late, early = nasa_score([50], [60]), nasa_score([50], [40])
    assert late > early > 0
    assert np.isclose(late, np.exp(1) - 1) and np.isclose(early, np.exp(10 / 13) - 1)


def test_capped_and_true_rmse_differ_only_above_the_cap():
    rul, pred = np.array([10, 200]), np.array([10, 125])
    m = test_metrics(rul, pred, cap=125)
    assert m["rmse_capped"] == 0
    assert m["rmse_true"] > 0
