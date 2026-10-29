"""Loading and tidying the Epidemiology Unit disease-surveillance workbook."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_long(path: Path = C.RAW_XLSX) -> pd.DataFrame:
    """Read the 'Data' sheet and split the indicator into measure + division."""
    df = pd.read_excel(path, sheet_name="Data")
    df["date"] = pd.to_datetime(df["date"])
    parts = df["indicator"].str.split(": ", n=1, expand=True)
    df["measure"] = parts[0]
    df["division"] = parts[1]
    df["disease"] = df["measure"].str.split(",").str[0]
    return df


def load_about(path: Path = C.RAW_XLSX) -> pd.DataFrame:
    about = pd.read_excel(path, sheet_name="About", header=None)
    about.columns = ["field", "value"]
    return about


@dataclass
class DenguePanels:
    weekly: pd.DataFrame      # index=week_end, columns=26 divisions (as printed)
    national: pd.Series       # national weekly cases (as printed)
    ytd: pd.DataFrame         # year-to-date cases, same shape as weekly
    annual: pd.DataFrame      # index=year-end date (2023-2025), columns=divisions


def build_panels(long: pd.DataFrame) -> DenguePanels:
    dengue = long[long["disease"] == "Dengue"]

    def pivot(measure: str) -> pd.DataFrame:
        p = dengue[dengue["measure"] == measure].pivot(
            index="date", columns="division", values="value"
        )
        return p.sort_index().astype(float)

    weekly_all = pivot(C.WEEKLY_INDICATOR)
    ytd_all = pivot(C.YTD_INDICATOR)
    annual_all = pivot(C.ANNUAL_INDICATOR)
    return DenguePanels(
        weekly=weekly_all.drop(columns=C.NATIONAL),
        national=weekly_all[C.NATIONAL],
        ytd=ytd_all.drop(columns=C.NATIONAL),
        annual=annual_all.drop(columns=C.NATIONAL),
    )


def epi_week(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Epidemiological week number (2026) from the week-ending date."""
    d = pd.to_datetime(pd.Series(dates))
    return (C.FIRST_EPI_WEEK + (d - pd.Timestamp(C.FIRST_WEEK_END)).dt.days // 7).to_numpy()


def clean_weekly(panels: DenguePanels, week26_policy: str = C.WEEK26_POLICY) -> pd.DataFrame:
    """Return division x week counts with unreliable weeks set to NaN.

    No value is imputed or invented: unreliable cells become missing.
    """
    w = panels.weekly.copy()
    w.loc[pd.Timestamp(C.MISALIGNED_WEEK), :] = np.nan
    if week26_policy == "drop":
        w.loc[pd.Timestamp(C.DUPLICATE_WEEK), :] = np.nan
    elif week26_policy != "as_printed":
        raise ValueError(f"unknown week26_policy: {week26_policy}")
    return w


def tidy_weekly_csv(panels: DenguePanels) -> pd.DataFrame:
    """Long, analysis-friendly table of the dengue series with quality flags."""
    rows = panels.weekly.stack().rename("cases").reset_index()
    rows.columns = ["week_end", "division", "cases"]
    ytd = panels.ytd.stack().rename("ytd_cases").reset_index()
    ytd.columns = ["week_end", "division", "ytd_cases"]
    out = rows.merge(ytd, on=["week_end", "division"], how="left")
    out["epi_week_2026"] = epi_week(out["week_end"])
    out["quality_flag"] = "ok"
    out.loc[out["week_end"] == pd.Timestamp(C.MISALIGNED_WEEK), "quality_flag"] = (
        "suspected_misaligned_division_counts"
    )
    out.loc[out["week_end"] == pd.Timestamp(C.DUPLICATE_WEEK), "quality_flag"] = (
        "source_reprinted_previous_week"
    )
    out["cases"] = out["cases"].astype(int)
    out["ytd_cases"] = out["ytd_cases"].astype(int)
    return out.sort_values(["division", "week_end"]).reset_index(drop=True)
