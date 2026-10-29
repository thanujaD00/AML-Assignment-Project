"""Turn the division x week panel into a supervised one-week-ahead dataset.

For every division d and week t, the features describe what is known at the end
of week t; the target is the number of cases notified in week t+1.
All features are computed from the dataset itself - no external data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import DenguePanels


def _national_clean(panels: DenguePanels, week26_policy: str) -> pd.Series:
    nat = panels.national.astype(float).copy()
    if week26_policy == "drop":
        nat.loc[pd.Timestamp(C.DUPLICATE_WEEK)] = np.nan
    return nat


def build_supervised(panels: DenguePanels, weekly_clean: pd.DataFrame,
                     week26_policy: str = C.WEEK26_POLICY,
                     include_future: bool = False) -> pd.DataFrame:
    """Return one row per (division, week t) with features and target y_{t+1}.

    include_future=True also returns rows for the last observed week whose target
    (the following week) is not in the dataset - used for genuine forecasts.
    """
    nat = _national_clean(panels, week26_policy)
    hist_annual = panels.annual.mean(axis=0)               # mean of 2023, 2024, 2025
    weeks = list(weekly_clean.index)
    rows = []
    for div in weekly_clean.columns:
        s = weekly_clean[div]
        for i, t in enumerate(weeks):
            c_t = s.iloc[i]
            if np.isnan(c_t):
                continue
            c_t1 = s.iloc[i - 1] if i > 0 else np.nan
            if i + 1 < len(weeks):
                target_week, y_next = weeks[i + 1], s.iloc[i + 1]
                if np.isnan(y_next):
                    continue
            elif include_future:
                target_week, y_next = t + pd.Timedelta(days=7), np.nan
            else:
                continue
            n_t = nat.iloc[i]
            n_t1 = nat.iloc[i - 1] if i > 0 else np.nan
            hist = hist_annual[div]
            rows.append({
                "division": div,
                "week_end": t,
                "target_week_end": target_week,
                "cases_t": c_t,
                "cases_t1": c_t1,
                "y_next": y_next,
                "log_cases_t": np.log1p(c_t),
                "log_cases_t1": np.log1p(c_t1),
                "growth_t": np.log1p(c_t) - np.log1p(c_t1),
                "log_hist_annual": np.log1p(hist),
                "season_ratio_t": np.log1p(c_t) - np.log1p(hist / 52.0),
                "log_national_t": np.log1p(n_t),
                "national_growth_t": np.log1p(n_t) - np.log1p(n_t1),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["target_week_end", "division"]).reset_index(drop=True)


def make_feature_row(division: str, cases_t: float, cases_t1: float | None,
                     national_t: float, national_t1: float | None,
                     hist_annual_mean: float) -> pd.DataFrame:
    """Build one feature row from user inputs (same formulas as build_supervised)."""
    c_t1 = np.nan if cases_t1 is None else float(cases_t1)
    n_t1 = np.nan if national_t1 is None else float(national_t1)
    return pd.DataFrame([{
        "division": division,
        "cases_t": float(cases_t),
        "cases_t1": c_t1,
        "y_next": np.nan,
        "log_cases_t": np.log1p(cases_t),
        "log_cases_t1": np.log1p(c_t1),
        "growth_t": np.log1p(cases_t) - np.log1p(c_t1),
        "log_hist_annual": np.log1p(hist_annual_mean),
        "season_ratio_t": np.log1p(cases_t) - np.log1p(hist_annual_mean / 52.0),
        "log_national_t": np.log1p(national_t),
        "national_growth_t": np.log1p(national_t) - np.log1p(n_t1),
    }])
