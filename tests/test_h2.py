"""H2 sort-variable tests, centred on the detrending fix (STATUS.md section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.h2_volatility import DETREND_MODES, detrend_gex


def _trending_panel(n: int = 6 * 24 * 800) -> pd.DataFrame:
    """GEX with a hard downward trend and pure-noise fluctuations around it.

    Construction is the point: there is NO relationship between the level of
    GEX and anything else here -- only a trend plus noise. Any sort that finds
    structure in the level is finding the trend.

    The default span is 800 days deliberately: it crosses three calendar years,
    so `within_year` has more than one group to work with. Inside a single year
    that mode is a no-op and cannot detrend anything.
    """
    idx = pd.date_range("2022-01-01", periods=n, freq="10min", tz="UTC")
    rng = np.random.default_rng(0)
    trend = -np.linspace(0.5e6, 13e6, n)
    return pd.DataFrame({"gex_flow": trend + rng.normal(0, 1e6, n)}, index=idx)


class TestDetrendGex:
    def test_all_modes_run(self):
        p = _trending_panel()
        for m in DETREND_MODES:
            out = detrend_gex(p, m)
            assert len(out) == len(p)
            assert out.notna().sum() > 0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown detrend mode"):
            detrend_gex(_trending_panel(), "gaussianify")

    def test_raw_is_untouched(self):
        p = _trending_panel()
        assert detrend_gex(p, "raw").equals(p["gex_flow"])

    @staticmethod
    def _calendar_separation(p: pd.DataFrame, mode: str) -> float:
        """How far apart in TIME the top and bottom buckets sit, as a share of
        the sample span. 1.0 = the sort is purely a calendar sort."""
        z = detrend_gex(p, mode)
        ok = z.notna()
        q = pd.qcut(z[ok], 5, labels=False)
        idx = p.index[ok]
        gap = (idx[q == 0].mean() - idx[q == 4].mean()).total_seconds()
        return abs(gap) / (p.index[-1] - p.index[0]).total_seconds()

    def test_raw_quintiles_are_a_calendar_sort(self):
        """The bug. On trend-plus-noise the raw sort separates by DATE."""
        assert self._calendar_separation(_trending_panel(), "raw") > 0.5

    @pytest.mark.parametrize("mode", ["rolling_z", "within_month", "within_year"])
    def test_detrending_cuts_the_calendar_sort(self, mode):
        p = _trending_panel()
        raw = self._calendar_separation(p, "raw")
        assert self._calendar_separation(p, mode) < 0.5 * raw

    @pytest.mark.parametrize("mode", ["rolling_z", "within_month"])
    def test_local_detrending_removes_it_almost_entirely(self, mode):
        """rolling_z and within_month rescale against a LOCAL window, so a
        slow trend is gone. within_year is deliberately excluded: it removes
        only the between-year level, leaving the trend within each year, which
        is exactly why it is a robustness mode and not the baseline."""
        assert self._calendar_separation(_trending_panel(), mode) < 0.15

    def test_rolling_z_uses_no_future_information(self):
        """Truncating the sample must not change any value already computed."""
        p = _trending_panel()
        cut = len(p) // 2
        full = detrend_gex(p, "rolling_z").iloc[:cut]
        trunc = detrend_gex(p.iloc[:cut], "rolling_z")
        pd.testing.assert_series_equal(full, trunc)

    def test_within_month_does_peek(self):
        """Stated as robustness-only in config; this documents why."""
        p = _trending_panel()
        cut = len(p) // 2
        full = detrend_gex(p, "within_month").iloc[:cut]
        trunc = detrend_gex(p.iloc[:cut], "within_month")
        assert not np.allclose(full.to_numpy(), trunc.to_numpy(), equal_nan=True)
