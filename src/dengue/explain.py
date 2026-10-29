"""Explainability: SHAP, permutation importance and partial dependence / ICE.

All explanations refer to the NGBoost location parameter, i.e. the predicted
log growth of cases from week t to week t+1 (positive = cases expected to rise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config as C
from .evaluate import point_metrics
from .models import NGBoostGrowth, RFGrowth


def shap_values(model: NGBoostGrowth, df: pd.DataFrame) -> shap.Explanation:
    X = model._X(df)
    explainer = shap.TreeExplainer(model.model_, model_output=0)   # 0 = 'loc' parameter
    sv = explainer(X)
    sv.feature_names = [C.FEATURE_LABELS[f] for f in model.features]
    return sv


def shap_importance(sv: shap.Explanation, features: list[str]) -> pd.DataFrame:
    imp = np.abs(sv.values).mean(axis=0)
    return (pd.DataFrame({"feature": features,
                          "label": [C.FEATURE_LABELS[f] for f in features],
                          "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def rf_shap_importance(rf: RFGrowth, df: pd.DataFrame) -> pd.DataFrame:
    """Same analysis for the random-forest baseline, to check robustness."""
    sv = shap.TreeExplainer(rf.model_)(rf._X(df))
    return shap_importance(sv, rf.features)


def permutation_importance(model, test: pd.DataFrame, n_repeats: int = 30,
                           seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Increase in test MAE (log scale) when one feature is shuffled."""
    rng = np.random.default_rng(seed)
    base = point_metrics(test["y_next"], model.predict(test))["MAE_log"]
    rows = []
    for f in model.features:
        drops = []
        for _ in range(n_repeats):
            shuffled = test.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            drops.append(point_metrics(test["y_next"], model.predict(shuffled))["MAE_log"] - base)
        rows.append({"feature": f, "label": C.FEATURE_LABELS[f],
                     "mean_increase_MAE_log": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("mean_increase_MAE_log", ascending=False).reset_index(drop=True)


def partial_dependence(model: NGBoostGrowth, df: pd.DataFrame, feature: str,
                       n_grid: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid, ICE curves [n_rows x n_grid], PDP = mean of ICE).

    Curves are the predicted log growth with `feature` fixed at each grid value.
    """
    X = model._X(df)
    j = model.features.index(feature)
    grid = np.linspace(np.nanpercentile(X[:, j], 2), np.nanpercentile(X[:, j], 98), n_grid)
    ice = np.empty((X.shape[0], n_grid))
    for k, v in enumerate(grid):
        Xv = X.copy()
        Xv[:, j] = v
        ice[:, k] = model.predict_log_growth(Xv)
    return grid, ice, ice.mean(axis=0)
