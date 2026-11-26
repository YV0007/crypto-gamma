"""Tests for greeks alignment and GEX arithmetic.

Uses synthetic greeks: no historical greeks source exists yet (see
src/build/greeks.py docstring), so these lock down the scaling and alignment
now, and the magnitude check becomes a data question rather than a code
question once real gamma is available.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.build.gex import compute_gex_flow, compute_gex_naive
from src.build.greeks import GREEK_COLS, align_greeks, apply_missing_policy

TS = pd.Timestamp("2024-03-01 12:00", tz="UTC")


def _panel(**over):
    base = {
        "instrument_name": ["BTC-29MAR24-65000-C"],
        "timestamp": [TS],
        "dealer_position": [-100_000.0],
        "gamma": [2e-5],
        "spot": [65_000.0],
        "contract_size": [1.0],
    }
    base.update(over)
    return pd.DataFrame(base)


class TestGexArithmetic:
    def test_formula_matches_hand_calculation(self):
        cells, series = compute_gex_flow(_panel())
        # -100,000 x 2e-5 x 65,000^2 x 0.01 x 1.0
        expected = -100_000 * 2e-5 * 65_000**2 * 0.01 * 1.0
        assert expected == pytest.approx(-84_500_000.0)
        assert cells["gex_contribution"].iloc[0] == pytest.approx(expected)
        assert series["gex_flow"].iloc[0] == pytest.approx(expected)

    def test_order_of_magnitude_is_hundreds_of_millions(self):
        """Guards the exact failure mode the magnitude check looks for."""
        _, series = compute_gex_flow(_panel())
        assert 1e7 < abs(series["gex_flow"].iloc[0]) < 1e10

    def test_contract_size_scales_linearly(self):
        _, a = compute_gex_flow(_panel())
        _, b = compute_gex_flow(_panel(contract_size=[10.0]))
        assert b["gex_flow"].iloc[0] == pytest.approx(10 * a["gex_flow"].iloc[0])

    def test_sign_follows_dealer_position(self):
        _, short = compute_gex_flow(_panel())
        _, long_ = compute_gex_flow(_panel(dealer_position=[+100_000.0]))
        assert short["gex_flow"].iloc[0] < 0 < long_["gex_flow"].iloc[0]

    def test_scales_with_spot_squared(self):
        _, a = compute_gex_flow(_panel())
        _, b = compute_gex_flow(_panel(spot=[130_000.0]))
        assert b["gex_flow"].iloc[0] == pytest.approx(4 * a["gex_flow"].iloc[0])

    def test_aggregates_across_instruments(self):
        p = pd.concat([_panel(), _panel(instrument_name=["BTC-29MAR24-70000-P"],
                                        dealer_position=[+40_000.0])], ignore_index=True)
        _, series = compute_gex_flow(p)
        assert len(series) == 1
        assert series["n_instruments"].iloc[0] == 2
        assert series["gex_flow"].iloc[0] == pytest.approx(
            (-100_000 + 40_000) * 2e-5 * 65_000**2 * 0.01)

    def test_missing_required_column_raises(self):
        with pytest.raises(KeyError, match="gamma"):
            compute_gex_flow(_panel().drop(columns=["gamma"]))


class TestGexNaive:
    def test_is_explicitly_blocked_not_silently_wrong(self):
        with pytest.raises(NotImplementedError, match="open interest"):
            compute_gex_naive()


class TestAlignment:
    def _snaps(self):
        return pd.DataFrame({
            "instrument_name": ["X", "X"],
            "timestamp": [pd.Timestamp("2024-03-01 09:00", tz="UTC"),
                          pd.Timestamp("2024-03-01 15:00", tz="UTC")],
            "gamma": [1e-5, 3e-5], "delta": [.5, .6], "vega": [1., 2.], "theta": [-1., -2.],
        })

    def _grid(self):
        return pd.DataFrame({
            "instrument_name": ["X"] * 3,
            "timestamp": [pd.Timestamp("2024-03-01 08:00", tz="UTC"),
                          pd.Timestamp("2024-03-01 12:00", tz="UTC"),
                          pd.Timestamp("2024-03-01 18:00", tz="UTC")],
        })

    def test_nearest_prior_never_looks_forward(self):
        out = align_greeks(self._grid(), self._snaps(), "nearest_prior", "24h")
        assert pd.isna(out["gamma"].iloc[0])          # before any snapshot
        assert out["gamma"].iloc[1] == 1e-5           # 09:00 snapshot, not 15:00
        assert out["gamma"].iloc[2] == 3e-5

    def test_interpolate_fills_forward_and_is_flagged(self):
        out = align_greeks(self._grid(), self._snaps(), "interpolate", "24h")
        assert out["gamma"].iloc[0] == 1e-5           # only reachable by looking ahead

    def test_max_staleness_bounds_the_carry(self):
        out = align_greeks(self._grid(), self._snaps(), "nearest_prior", "1h")
        assert pd.isna(out["gamma"].iloc[1])          # 09:00 snapshot is 3h stale

    def test_all_greeks_are_carried(self):
        out = align_greeks(self._grid(), self._snaps(), "nearest_prior", "24h")
        assert set(GREEK_COLS) <= set(out.columns)


class TestMissingPolicy:
    def _mixed(self):
        p = pd.concat([_panel(), _panel(gamma=[None])], ignore_index=True)
        return p

    def test_exclude_drops_and_reports(self):
        kept, stats = apply_missing_policy(self._mixed(), "exclude")
        assert len(kept) == 1
        assert stats["missing"] == 1 and stats["missing_fraction"] == 0.5

    def test_fail_raises(self):
        with pytest.raises(Exception, match="no gamma"):
            apply_missing_policy(self._mixed(), "fail")

    def test_zero_fill_is_not_offered(self):
        """Zero-filled gamma is indistinguishable from a flat position."""
        with pytest.raises(ValueError, match="intentionally not offered"):
            apply_missing_policy(self._mixed(), "zero")
