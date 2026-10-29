"""Forecasting models behind one small, uniform interface.

Learned models (except NGBoost-Poisson) predict the *log growth* of cases,
    r = log(1 + y_{t+1}) - log(1 + y_t),
which makes 26 divisions with very different scales (1 to >1,600 cases/week)
comparable, and turns back into a count forecast as
    y_hat = exp(log(1 + y_t) + r_hat) - 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config as C


def _log_growth_target(df: pd.DataFrame) -> np.ndarray:
    return (np.log1p(df["y_next"]) - np.log1p(df["cases_t"])).to_numpy()


def _to_counts(df: pd.DataFrame, r_hat: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.log1p(df["cases_t"].to_numpy()) + r_hat), 0, None)


# --------------------------------------------------------------------------- #
# Baselines without learning
# --------------------------------------------------------------------------- #
class Persistence:
    """y_hat(t+1) = y(t). The standard benchmark any forecaster must beat."""
    name = "Naive persistence"
    probabilistic = False

    def fit(self, df):
        return self

    def predict(self, df):
        return df["cases_t"].to_numpy(dtype=float)


class Drift:
    """Continue last week's growth: y_hat = y_t * (1+y_t)/(1+y_{t-1})."""
    name = "Naive drift"
    probabilistic = False

    def fit(self, df):
        return self

    def predict(self, df):
        g = df["growth_t"].fillna(0.0).to_numpy()
        return _to_counts(df, g)


# --------------------------------------------------------------------------- #
# Learned models
# --------------------------------------------------------------------------- #
@dataclass
class _Learned:
    params: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(C.FEATURES))

    def _X(self, df, fit=False):
        X = df[self.features].to_numpy(dtype=float)
        if fit:
            self.imputer_ = SimpleImputer(strategy="median").fit(X)
        return self.imputer_.transform(X)


class RidgeGrowth(_Learned):
    name = "Ridge regression"
    probabilistic = False
    grid = [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)]

    def fit(self, df):
        self.model_ = make_pipeline(StandardScaler(), Ridge(**self.params))
        self.model_.fit(self._X(df, fit=True), _log_growth_target(df))
        return self

    def predict(self, df):
        return _to_counts(df, self.model_.predict(self._X(df)))


class RFGrowth(_Learned):
    name = "Random forest"
    probabilistic = False
    grid = [{"max_depth": d, "min_samples_leaf": m}
            for d in (2, 3, None) for m in (3, 5, 10)]

    def fit(self, df):
        self.model_ = RandomForestRegressor(n_estimators=500, random_state=C.RANDOM_STATE,
                                            n_jobs=-1, **self.params)
        self.model_.fit(self._X(df, fit=True), _log_growth_target(df))
        return self

    def predict(self, df):
        return _to_counts(df, self.model_.predict(self._X(df)))


def _ngb_grid():
    return [{"n_estimators": n, "learning_rate": lr, "max_depth": d}
            for n in (100, 300) for lr in (0.01, 0.03, 0.1) for d in (2, 3)]


class NGBoostGrowth(_Learned):
    """NGBoost with a Normal distribution over log growth (proposed model)."""
    name = "NGBoost (Normal, log-growth)"
    probabilistic = True
    grid = _ngb_grid()

    def _make(self):
        p = dict(self.params)
        base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=p.pop("max_depth", 3),
                                     random_state=C.RANDOM_STATE)
        return NGBRegressor(Dist=Normal, Base=base, verbose=False,
                            random_state=C.RANDOM_STATE, **p)

    def fit(self, df):
        self.model_ = self._make()
        self.model_.fit(self._X(df, fit=True), _log_growth_target(df))
        return self

    def dist(self, df):
        return self.model_.pred_dist(self._X(df))

    def predict(self, df):
        # median of the predictive distribution on the count scale
        return _to_counts(df, self.dist(df).params["loc"])

    def predict_interval(self, df, level=0.9):
        d = self.dist(df).params
        z = stats.norm.ppf(0.5 + level / 2)
        return (_to_counts(df, d["loc"] - z * d["scale"]),
                _to_counts(df, d["loc"] + z * d["scale"]))

    def predict_log_growth(self, X: np.ndarray) -> np.ndarray:
        """Raw model output used by explainability tools (expects imputed X)."""
        return self.model_.predict(X)


class NGBoostPoisson(_Learned):
    """NGBoost with a Poisson distribution over next-week counts (variant)."""
    name = "NGBoost (Poisson, counts)"
    probabilistic = True
    grid = _ngb_grid()

    def _make(self):
        p = dict(self.params)
        base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=p.pop("max_depth", 3),
                                     random_state=C.RANDOM_STATE)
        return NGBRegressor(Dist=Poisson, Base=base, verbose=False,
                            random_state=C.RANDOM_STATE, **p)

    def fit(self, df):
        self.model_ = self._make()
        self.model_.fit(self._X(df, fit=True), df["y_next"].to_numpy().astype(int))
        return self

    def predict(self, df):
        return np.asarray(self.model_.pred_dist(self._X(df)).params["mu"], dtype=float)

    def predict_interval(self, df, level=0.9):
        mu = self.predict(df)
        a = (1 - level) / 2
        return stats.poisson.ppf(a, mu), stats.poisson.ppf(1 - a, mu)


MODEL_CLASSES = [Persistence, Drift, RidgeGrowth, RFGrowth, NGBoostPoisson, NGBoostGrowth]
