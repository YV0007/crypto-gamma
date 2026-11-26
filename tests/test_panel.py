"""Panel construction tests, centred on the no-lookahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.build.panel import build_calendar, _asof

GRID = pd.date_range("2024-03-01", "2024-03-03", freq="10min", tz="UTC")


class TestAsOf:
    def test_takes_last_value_at_or_before(self):
        s = pd.Series([1.0, 2.0],
                      index=pd.DatetimeIndex(["2024-03-01 00:05", "2024-03-01 00:25"], tz="UTC"))
        out = _asof(GRID[:4], s, "x")
        assert np.isnan(out.iloc[0])          # 00:00, before any observation
        assert out.iloc[1] == 1.0             # 00:10 -> the 00:05 value
        assert out.iloc[2] == 1.0             # 00:20 -> still 00:05, NOT 00:25
        assert out.iloc[3] == 2.0             # 00:30 -> 00:25

    def test_never_reaches_forward(self):
        s = pd.Series([9.0], index=pd.DatetimeIndex(["2024-03-01 23:00"], tz="UTC"))
        out = _asof(GRID[:6], s, "x")
        assert out.isna().all()


class TestCalendar:
    def test_time_to_expiry_is_non_negative_and_decreasing(self):
        cal = build_calendar(GRID)
        for col in ("tte_major_hours", "tte_daily_hours"):
            assert (cal[col] >= 0).all()
            # within a stretch before one expiry, tte must fall monotonically
            seg = cal[col].iloc[:20]
            assert (seg.diff().dropna() < 0).all()

    def test_seasonality_columns(self):
        cal = build_calendar(GRID)
        assert set(cal["hour_of_day"]) <= set(range(24))
        assert set(cal["day_of_week"]) <= set(range(7))

    def test_major_expiry_horizon_exceeds_one_day(self):
        """The regression guard for the bug in STATUS.md 4.8.

        Deribit expires an option every single calendar day, so a
        time-to-expiry built off ALL expiries can never exceed 24 hours and is
        really a clock, not an expiry horizon. The major-expiry column must
        reach out weeks, or it is measuring the same nothing again.
        """
        cal = build_calendar(GRID)
        assert cal["tte_major_hours"].max() > 24 * 7
        assert cal["tte_daily_hours"].max() <= 24 + 1e-9

    def test_major_expiry_is_not_a_time_of_day_dummy(self):
        """The old is_expiry_day reduced exactly to `hour_of_day < 8`.

        Needs a grid spanning several monthly cycles: over a two-day window
        the only thing time-to-expiry *can* do is tick down with the clock, so
        hour-of-day would explain most of it however well the column is built.
        """
        long_grid = pd.date_range("2024-01-01", "2024-06-30", freq="10min", tz="UTC")
        cal = build_calendar(long_grid)
        assert not (cal["is_major_expiry_day"] == (cal["hour_of_day"] < 8)).all()
        within = cal.groupby("hour_of_day")["tte_major_days"].transform(
            lambda s: s - s.mean())
        r2 = 1 - within.var() / cal["tte_major_days"].var()
        assert r2 < 0.10, f"tte_major_days is {r2:.2%} explained by hour of day"

    def test_days_since_major_is_never_negative(self):
        cal = build_calendar(GRID)
        d = cal["days_since_major"].dropna()
        assert (d >= 0).all()


@pytest.mark.skipif(
    not (config.path("paths.processed") /
         f"gex_flow_{config.get('sample.start')}_{config.get('sample.end')}.parquet").exists(),
    reason="full-period GEX not built yet",
)
class TestNoLookahead:
    """The decisive test: rebuilding on truncated data must not change history.

    If any feature peeked forward, values computed with the full sample would
    differ from the same values computed with the future removed.
    """

    def test_truncated_rebuild_is_identical(self):
        from src.build.panel import build_panel
        start, end = config.get("sample.start"), config.get("sample.end")
        cut = pd.Timestamp("2024-06-30 23:50", tz="UTC")

        full = build_panel(start, end)
        trunc = build_panel(start, end, truncate_at=cut)

        common = full.index[full.index <= cut]
        assert len(common) > 1000
        a = full.loc[common]
        b = trunc.reindex(common)

        for col in b.columns:
            if not np.issubdtype(b[col].dtype, np.number):
                continue
            # last row of a trailing window legitimately depends on the final
            # bar, so compare everything strictly before the cut
            m = common < cut - pd.Timedelta("1h")
            pd.testing.assert_series_equal(
                a.loc[m, col], b.loc[m, col], check_names=False,
                obj=f"lookahead detected in column {col!r}")
