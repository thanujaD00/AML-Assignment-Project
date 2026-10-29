"""Tests for data loading, quality decisions and feature construction."""
import numpy as np
import pandas as pd
import pytest

from dengue import config as C
from dengue import data, data_quality, features


@pytest.fixture(scope="module")
def panels():
    return data.build_panels(data.load_long())


def test_raw_file_hash_matches_published():
    assert data.sha256(C.RAW_XLSX) == C.RAW_XLSX_SHA256


def test_panel_shape(panels):
    assert panels.weekly.shape == (10, 26)
    assert panels.annual.shape == (3, 26)
    assert C.NATIONAL not in panels.weekly.columns


def test_divisions_sum_to_national(panels):
    assert (panels.weekly.sum(axis=1) == panels.national).all()


def test_audit_flags_duplicate_week(panels):
    checks = data_quality.audit(panels)["checks"].set_index("check")
    assert not checks.loc["no week identical to the previous week", "passed"]
    assert C.DUPLICATE_WEEK in checks.loc["no week identical to the previous week", "detail"]


def test_clean_weekly_never_imputes(panels):
    w = data.clean_weekly(panels, "drop")
    assert w.loc[pd.Timestamp(C.MISALIGNED_WEEK)].isna().all()
    assert w.loc[pd.Timestamp(C.DUPLICATE_WEEK)].isna().all()
    ok = w.dropna()
    assert (ok == panels.weekly.loc[ok.index]).all().all()   # kept values unchanged


def test_supervised_has_no_target_leakage(panels):
    w = data.clean_weekly(panels, "drop")
    sup = features.build_supervised(panels, w)
    assert len(sup) == 156
    assert (sup["target_week_end"] - sup["week_end"] == pd.Timedelta(days=7)).all()
    # the feature 'cases_t' is the observed count of week t, never of week t+1
    for _, r in sup.sample(20, random_state=0).iterrows():
        assert r["cases_t"] == panels.weekly.loc[r["week_end"], r["division"]]
        assert r["y_next"] == panels.weekly.loc[r["target_week_end"], r["division"]]


def test_feature_row_matches_pipeline(panels):
    w = data.clean_weekly(panels, "drop")
    sup = features.build_supervised(panels, w)
    r = sup.iloc[-1]
    hist = panels.annual.mean(axis=0)[r["division"]]
    row = features.make_feature_row(r["division"], r["cases_t"], r["cases_t1"],
                                    np.expm1(r["log_national_t"]),
                                    np.expm1(r["log_national_t"] - r["national_growth_t"]), hist)
    np.testing.assert_allclose(row[C.FEATURES].to_numpy()[0], r[C.FEATURES].to_numpy(dtype=float), rtol=1e-9)
