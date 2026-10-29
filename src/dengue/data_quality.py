"""Automated data-quality audit. Every check is computed from the raw file."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .data import DenguePanels, sha256


def audit(panels: DenguePanels) -> dict[str, pd.DataFrame]:
    w, ytd = panels.weekly, panels.ytd
    checks = []

    # 1. File integrity against the hash published by the compiler
    digest = sha256(C.RAW_XLSX)
    checks.append(("raw file SHA-256 matches published hash", digest == C.RAW_XLSX_SHA256, digest))

    # 2. Divisions sum to the national total every week
    diff = (w.sum(axis=1) - panels.national).abs()
    checks.append(("division counts sum to national total (all weeks)", bool((diff == 0).all()),
                   f"max abs diff = {diff.max():.0f}"))

    # 3. Year-to-date(t) - year-to-date(t-1) == weekly(t)
    ytd_gap = (ytd.diff() - w).iloc[1:].abs().to_numpy().max()
    checks.append(("YTD increments equal weekly counts (weeks 25-33)", bool(ytd_gap == 0),
                   f"max abs diff = {ytd_gap:.0f}"))

    # 4. Missing / negative values
    checks.append(("no missing weekly values", bool(w.notna().all().all()), f"{int(w.isna().sum().sum())} missing"))
    checks.append(("no negative counts", bool((w >= 0).all().all()), f"min = {w.min().min():.0f}"))

    # 5. Exact duplicate consecutive weeks
    dup = [str(w.index[i].date()) for i in range(1, len(w)) if w.iloc[i].equals(w.iloc[i - 1])]
    checks.append(("no week identical to the previous week", len(dup) == 0, ", ".join(dup) or "none"))

    checks_df = pd.DataFrame(checks, columns=["check", "passed", "detail"])

    # 6. Cross-sectional consistency: rank correlation of division counts
    hist = panels.annual.mean(axis=0)
    rows = []
    for i, d in enumerate(w.index):
        prev = spearmanr(w.iloc[i], w.iloc[i - 1])[0] if i > 0 else np.nan
        nxt = spearmanr(w.iloc[i], w.iloc[i + 1])[0] if i < len(w) - 1 else np.nan
        rows.append({
            "week_end": d.date(),
            "national_cases": int(panels.national.loc[d]),
            "spearman_vs_prev_week": round(prev, 3),
            "spearman_vs_next_week": round(nxt, 3),
            "spearman_vs_hist_annual_mean": round(spearmanr(w.iloc[i], hist)[0], 3),
        })
    rank_df = pd.DataFrame(rows)
    return {"checks": checks_df, "rank_consistency": rank_df}


def describe_dataset(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("measure").agg(
        rows=("value", "size"),
        divisions=("division", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        distinct_dates=("date", "nunique"),
    )
    g["first_date"] = g["first_date"].dt.date
    g["last_date"] = g["last_date"].dt.date
    return g.reset_index()
