"""Smoke tests for the models and metrics."""
import numpy as np
import pytest

from dengue import data, evaluate, features
from dengue.models import NGBoostGrowth, Persistence


@pytest.fixture(scope="module")
def sup():
    p = data.build_panels(data.load_long())
    return features.build_supervised(p, data.clean_weekly(p))


def test_metrics_perfect_prediction():
    y = np.array([1.0, 10.0, 100.0])
    m = evaluate.point_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2_log"] == 1


def test_persistence_returns_last_value(sup):
    assert np.array_equal(Persistence().predict(sup), sup["cases_t"].to_numpy())


def test_ngboost_fits_and_gives_ordered_intervals(sup):
    train = sup[sup["target_week_end"] < sup["target_week_end"].max()]
    test = sup[sup["target_week_end"] == sup["target_week_end"].max()]
    m = NGBoostGrowth(params={"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2}).fit(train)
    yhat = m.predict(test)
    lo, hi = m.predict_interval(test, 0.9)
    assert yhat.shape == (26,) and np.all(yhat >= 0)
    assert np.all(lo <= yhat + 1e-9) and np.all(yhat <= hi + 1e-9)
