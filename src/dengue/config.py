"""Central configuration: paths, constants and experiment settings."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = ROOT / "data" / "raw" / "disease-surveillance-2026-10-10.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
MODEL_DIR = ROOT / "models"

# SHA-256 of the Excel file as published on
# https://research.dulajmendis.com/data/disease-surveillance/ (version 2026-10-10)
RAW_XLSX_SHA256 = "9f075df094ad55691bb07d0d1a594f047d719ee1dc637f8acc244e739a3f70bb"

# --------------------------------------------------------------------------- #
# Dataset semantics
# --------------------------------------------------------------------------- #
WEEKLY_INDICATOR = "Dengue, cases notified in the week"
YTD_INDICATOR = "Dengue, cases notified this year to date"
ANNUAL_INDICATOR = "Dengue, cases notified in weeks 1 to 52"
NATIONAL = "Sri Lanka"

# Week-ending date -> epidemiological week of 2026 (from the source notes:
# Vol. 53 Nos. 25-34 = weeks 24-33, ending 2026-06-14 ... 2026-08-16).
FIRST_WEEK_END = "2026-06-14"
FIRST_EPI_WEEK = 24

# Data-quality decisions (see data_quality.py and reports/tables/data_quality_*.csv)
#  * Week 26 (2026-06-28): the source states Vol. 53 No. 27 re-printed the week-25
#    dengue counts in all divisions -> the true week-26 counts are unknown.
#  * Week 24 (2026-06-14): division-level weekly counts are inconsistent with every
#    other week (rank correlation with neighbouring weeks ~0.08 vs >=0.95 elsewhere),
#    although they still sum to the national total. Treated as unreliable.
DUPLICATE_WEEK = "2026-06-28"
MISALIGNED_WEEK = "2026-06-14"

# "drop"       -> set week-26 division counts to missing (main experiment)
# "as_printed" -> keep week-26 counts exactly as printed (sensitivity analysis)
WEEK26_POLICY = "drop"

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

FEATURES = [
    "log_cases_t",          # log(1 + cases notified in week t)
    "log_cases_t1",         # log(1 + cases notified in week t-1)  (may be missing)
    "growth_t",             # log_cases_t - log_cases_t1           (may be missing)
    "log_hist_annual",      # log(1 + mean annual cases 2023-2025)
    "season_ratio_t",       # log_cases_t - log(mean weekly cases 2023-2025)
    "log_national_t",       # log(1 + national cases in week t)
    "national_growth_t",    # national log growth from t-1 to t
]

FEATURE_LABELS = {
    "log_cases_t": "Cases this week (log)",
    "log_cases_t1": "Cases last week (log)",
    "growth_t": "Division growth (log ratio)",
    "log_hist_annual": "Historical annual burden (log)",
    "season_ratio_t": "This week vs historical weekly mean (log ratio)",
    "log_national_t": "National cases this week (log)",
    "national_growth_t": "National growth (log ratio)",
}

# Rolling-origin design (target week-ending dates).
# Hyper-parameter tuning uses the earlier target weeks only; the last two
# target weeks are a hold-out test that is never used for any decision.
TEST_TARGET_WEEKS = ["2026-08-09", "2026-08-16"]

PI_LEVELS = (0.8, 0.9)
