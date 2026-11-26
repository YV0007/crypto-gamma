"""Tests for dealer inventory reconstruction.

The first test is the one that matters: a taker BUY must DECREASE the dealer's
position. If that invariant ever flips, every downstream result reverses sign
while still looking perfectly plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build.inventory import (
    accumulate,
    apply_block_trade_policy,
    compute_dealer_flow,
    to_interval_grid,
)

CREATION = pd.Timestamp("2024-03-01 00:00", tz="UTC")
EXPIRY = pd.Timestamp("2024-03-03 08:00", tz="UTC")


@pytest.fixture
def instruments() -> pd.DataFrame:
    return pd.DataFrame([{
        "instrument_name": "BTC-3MAR24-60000-C",
        "kind": "option",
        "contract_size": 1.0,
        "creation_ts": CREATION,
        "expiry_ts": EXPIRY,
    }])


def _trade(ts: str, direction: str, amount: float, seq: int, block: str | None = None) -> dict:
    t = pd.Timestamp(ts, tz="UTC")
    return {
        "instrument_name": "BTC-3MAR24-60000-C",
        "dt": t,
        "timestamp": int(t.value // 1_000_000),
        "direction": direction,
        "amount": amount,
        "trade_seq": seq,
        "block_trade_id": block,
    }


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame([
        _trade("2024-03-01 10:00", "buy", 10.0, 1),
        _trade("2024-03-01 12:00", "sell", 4.0, 2),
        _trade("2024-03-02 09:00", "buy", 1.0, 3, block="BLOCK-1"),
    ])


def _run(trades, instruments, mode):
    kept, stats = apply_block_trade_policy(trades, mode)
    flow, _ = compute_dealer_flow(kept, instruments)
    return accumulate(flow), stats


class TestSignConvention:
    def test_taker_buy_DECREASES_dealer_position(self, trades, instruments):
        """THE critical invariant: customer buys -> dealer is short."""
        out, _ = _run(trades.iloc[[0]], instruments, "exclude")
        assert out["dealer_flow"].iloc[0] == -10.0
        assert out["dealer_position"].iloc[0] == -10.0

    def test_taker_sell_INCREASES_dealer_position(self, trades, instruments):
        out, _ = _run(trades.iloc[[1]], instruments, "exclude")
        assert out["dealer_flow"].iloc[0] == +4.0
        assert out["dealer_position"].iloc[0] == +4.0

    def test_accumulation_is_ordered_and_additive(self, trades, instruments):
        out, _ = _run(trades, instruments, "exclude")
        # buy 10 -> -10, then sell 4 -> -6
        assert list(out["dealer_position"]) == [-10.0, -6.0]

    def test_position_is_in_contracts_not_notional(self, trades, instruments):
        inst = instruments.copy()
        inst["contract_size"] = 10.0          # e.g. a perpetual-style size
        out, _ = _run(trades.iloc[[0]], inst, "exclude")
        assert out["dealer_flow"].iloc[0] == -1.0   # 10 amount / 10 size


class TestBlockTrades:
    def test_excluded_under_baseline(self, trades, instruments):
        out, stats = _run(trades, instruments, "exclude")
        assert len(out) == 2
        assert stats["n_block"] == 1
        assert out["dealer_position"].iloc[-1] == -6.0

    def test_included_under_include(self, trades, instruments):
        out, stats = _run(trades, instruments, "include")
        assert len(out) == 3
        assert out["dealer_position"].iloc[-1] == -7.0   # -6 then buy 1

    def test_half_weight(self, trades, instruments):
        out, _ = _run(trades, instruments, "half_weight")
        assert out["dealer_flow"].iloc[-1] == -0.5
        assert out["dealer_position"].iloc[-1] == -6.5


class TestLifecycle:
    def test_zero_before_creation_and_after_expiry(self, trades, instruments):
        out, _ = _run(trades, instruments, "exclude")
        grid = to_interval_grid(out, instruments, "2024-02-28", "2024-03-04", interval="1h")
        before = grid[grid["timestamp"] < CREATION]
        after = grid[grid["timestamp"] >= EXPIRY]
        assert len(before) > 0 and (before["dealer_position"] == 0).all()
        assert len(after) > 0 and (after["dealer_position"] == 0).all()

    def test_forward_fill_persists_position_between_trades(self, trades, instruments):
        out, _ = _run(trades, instruments, "exclude")
        grid = to_interval_grid(out, instruments, "2024-03-01", "2024-03-02", interval="1h")
        g = grid.set_index("timestamp")["dealer_position"]
        assert g[pd.Timestamp("2024-03-01 11:00", tz="UTC")] == -10.0   # between trades
        assert g[pd.Timestamp("2024-03-01 13:00", tz="UTC")] == -6.0
        assert g[pd.Timestamp("2024-03-01 09:00", tz="UTC")] == 0.0     # before first trade

    def test_grid_has_no_gaps(self, trades, instruments):
        out, _ = _run(trades, instruments, "exclude")
        grid = to_interval_grid(out, instruments, "2024-03-01", "2024-03-02", interval="1h")
        ts = grid["timestamp"].sort_values()
        assert ts.is_unique
        assert set(np.diff(ts.to_numpy()).astype("timedelta64[h]").astype(int)) == {1}
        assert grid["dealer_position"].notna().all()


class TestBadDirection:
    def test_null_and_unexpected_directions_are_dropped(self, trades, instruments):
        bad = pd.concat([trades.iloc[[0]], pd.DataFrame([
            _trade("2024-03-01 11:00", None, 5.0, 9),
            _trade("2024-03-01 11:30", "unknown", 5.0, 10),
        ])], ignore_index=True)
        kept, _ = apply_block_trade_policy(bad, "exclude")
        flow, stats = compute_dealer_flow(kept, instruments)
        assert stats["n_bad_direction"] == 2
        assert len(flow) == 1


class TestContractSizeSafety:
    def test_unmapped_instrument_raises(self, trades, instruments):
        orphan = trades.copy()
        orphan["instrument_name"] = "BTC-9MAR24-1-C"
        kept, _ = apply_block_trade_policy(orphan, "exclude")
        with pytest.raises(RuntimeError, match="absent from instruments.parquet"):
            compute_dealer_flow(kept, instruments)


class TestBlockTradeShareIsReportedByVolume:
    """Regression guard for STATUS.md 4.7.

    The `exclude` baseline was chosen because blocks are ~1.3% of TRADES. They
    are ~31% of volume and ~67% of net dealer flow, which is what GEX is built
    from. If `apply_block_trade_policy` ever goes back to reporting the count
    alone, the same mistake becomes available again.
    """

    @staticmethod
    def _tape():
        # 100 small customer buys, plus one huge block sell that dominates the
        # net flow while being 1% of the trade count.
        n = 100
        return pd.DataFrame({
            "instrument_name": ["BTC-29MAR24-70000-C"] * (n + 1),
            "direction": ["buy"] * n + ["sell"],
            "amount": [1.0] * n + [400.0],
            "block_trade_id": [None] * n + ["blk-1"],
        })

    def test_stats_carry_volume_and_net_flow_shares(self):
        _, stats = apply_block_trade_policy(self._tape(), "include")
        assert {"block_fraction", "block_fraction_volume",
                "block_fraction_net_flow"} <= set(stats)

    def test_count_share_understates_the_others(self):
        _, stats = apply_block_trade_policy(self._tape(), "include")
        assert stats["block_fraction"] == pytest.approx(1 / 101, rel=1e-6)
        assert stats["block_fraction_volume"] == pytest.approx(400 / 500, rel=1e-6)
        # net flow: 100 customer buys => dealer -100; block sell => dealer +400
        assert stats["block_fraction_net_flow"] == pytest.approx(400 / 300, rel=1e-6)
        assert stats["block_fraction"] < stats["block_fraction_volume"]

    def test_excluding_blocks_flips_the_net_dealer_sign_here(self):
        keep_in, _ = apply_block_trade_policy(self._tape(), "include")
        keep_ex, _ = apply_block_trade_policy(self._tape(), "exclude")
        sign = lambda d: np.sign(
            (np.where(d["direction"] == "buy", -1.0, 1.0) * d["amount"] * d["weight"]).sum())
        assert sign(keep_in) == 1.0 and sign(keep_ex) == -1.0
