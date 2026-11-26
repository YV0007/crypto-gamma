"""Unit tests for the Stage-1 fetch layer. No network access."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import config
from src.fetch.instruments import parse_instrument_name
from src.fetch.runner import day_range


class TestInstrumentNames:
    def test_call(self):
        got = parse_instrument_name("BTC-27DEC24-100000-C")
        assert got["underlying"] == "BTC"
        assert got["strike"] == 100000.0
        assert got["option_type"] == "call"
        assert got["expiry_name"] == datetime(2024, 12, 27, 8, tzinfo=timezone.utc)

    def test_put_single_digit_day(self):
        got = parse_instrument_name("BTC-5APR24-62000-P")
        assert got["option_type"] == "put"
        assert got["expiry_name"] == datetime(2024, 4, 5, 8, tzinfo=timezone.utc)

    def test_fractional_strike(self):
        assert parse_instrument_name("BTC-1MAR24-61000.5-C")["strike"] == 61000.5

    @pytest.mark.parametrize("name", ["BTC-PERPETUAL", "BTC-27DEC24", "garbage"])
    def test_non_options_return_none(self, name):
        assert parse_instrument_name(name) is None


class TestDayRange:
    def test_splits_on_utc_boundaries(self):
        start = config.to_ms("2024-03-01")
        end = config.to_ms("2024-03-04")
        days = day_range(start, end)
        assert [d[0] for d in days] == ["2024-03-01", "2024-03-02", "2024-03-03"]
        assert days[0][1] == start
        assert days[-1][2] == end

    def test_windows_are_contiguous_and_non_overlapping(self):
        days = day_range(config.to_ms("2022-01-01"), config.to_ms("2022-02-01"))
        assert len(days) == 31
        for a, b in zip(days, days[1:]):
            assert a[2] == b[1]


class TestConfig:
    def test_sample_bounds_are_end_inclusive(self):
        start, end = config.sample_bounds_ms()
        assert config.from_ms(start).date().isoformat() == config.get("sample.start")
        # end is exclusive, so the last covered day is sample.end
        assert config.from_ms(end - 1).date().isoformat() == config.get("sample.end")

    def test_dvol_resolution_is_not_a_silent_no_data_value(self):
        # Deribit returns ZERO points, without erroring, for "1" and "60".
        assert config.get("fetch.market_data.dvol_resolution") not in ("1", "60")

    def test_perp_instrument_is_pinned(self):
        assert config.get("fetch.perp_trades.instrument") == "BTC-PERPETUAL"

    def test_contract_size_constant_is_marked_fallback_only(self):
        # A bare `contract_size` key invites code to use it instead of joining
        # the per-instrument value from instruments.parquet.
        assert "contract_size" not in config.get("construction")
        assert config.get("construction.contract_size_fallback") == 1.0
