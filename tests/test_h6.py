"""H6 backtest mechanics.

The tests that matter here are not "does it return a number" but "is the number
a backtest rather than a fantasy". Three failure modes are pinned:

  * the forward-return shift (off by one = look at tomorrow's close today)
  * cost accounting (a reversal must pay twice, not once)
  * the holding period (silently rebalancing every bar makes costs vanish)

Everything else is a sanity check on the sign conventions of the rule itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import h6_strategy as h6


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A minimal panel with the columns H6 touches, on the real 10-min grid."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC")
    ret = rng.normal(0, 0.002, n)
    close = 40000 * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {
            "close": close,
            "ret": pd.Series(np.log(close), index=idx).diff(),
            "gex_flow": rng.normal(0, 1e6, n).cumsum(),
        },
        index=idx,
    ).rename_axis("timestamp")


class TestNoLookahead:
    def test_forward_return_is_next_period(self):
        """forward_return[t] must equal ret[t+1], never ret[t]."""
        p = _panel(50)
        fwd = h6.forward_return(p)
        assert np.isclose(fwd.iloc[10], p["ret"].iloc[11])
        assert np.isnan(fwd.iloc[-1]), "last row cannot have a forward return"

    def test_position_uses_only_past_closes(self):
        """Truncating the panel must not change any earlier position."""
        p = _panel(300)
        kw = dict(threshold=0.5, lookback="10min", detrend="raw", trade_middle=False)
        full = h6.raw_position(p, **kw)
        trunc = h6.raw_position(p.iloc[:200], **kw)
        pd.testing.assert_series_equal(full.iloc[:200], trunc, check_freq=False)

    def test_backtest_drops_unearnable_final_row(self):
        p = _panel(100)
        pos = pd.Series(1.0, index=p.index)
        assert len(h6.backtest(p, pos)) == len(p) - 1


class TestCosts:
    def test_reversal_pays_twice(self):
        """-1 -> +1 is two units of turnover, not one."""
        p = _panel(10)
        pos = pd.Series([0, 1, 1, -1, -1, 0, 0, 0, 0, 0], index=p.index, dtype=float)
        bt = h6.backtest(p, pos)
        c = h6.cost_bps() / 1e4
        assert np.isclose(bt["turnover"].iloc[3], 2.0)
        assert np.isclose(bt["cost"].iloc[3], 2.0 * c)

    def test_holding_position_costs_nothing(self):
        p = _panel(20)
        pos = pd.Series(1.0, index=p.index)
        bt = h6.backtest(p, pos)
        assert bt["cost"].iloc[1:].sum() == 0.0, "an unchanged position must not be charged"

    def test_net_is_gross_minus_cost(self):
        p = _panel(50)
        pos = h6.raw_position(p, threshold=0.5, lookback="10min",
                              detrend="raw", trade_middle=False)
        bt = h6.backtest(p, pos)
        assert np.allclose(bt["net"], bt["gross"] - bt["cost"])

    def test_cost_bps_is_taker_plus_slippage(self):
        from src import config
        f = config.get("h6_strategy.fees")
        assert h6.cost_bps() == f["perp_taker_bps"] + f["slippage_bps"]

    def test_breakeven_cost_recovers_actual_cost(self):
        """At the breakeven cost the net total must be ~0 -- a self-check on the metric."""
        p = _panel(500, seed=3)
        pos = h6.apply_hold(
            h6.raw_position(p, threshold=0.3, lookback="10min",
                            detrend="raw", trade_middle=False), "1h")
        bt = h6.backtest(p, pos)
        res = h6.evaluate(bt, "t")
        be = res["breakeven_cost_bps"] / 1e4
        assert np.isclose((bt["gross"] - bt["turnover"] * be).sum(), 0.0, atol=1e-9)


class TestHold:
    def test_hold_reduces_turnover(self):
        p = _panel(600, seed=1)
        raw = h6.raw_position(p, threshold=0.2, lookback="10min",
                              detrend="raw", trade_middle=False)
        fast = h6.backtest(p, raw)["turnover"].sum()
        slow = h6.backtest(p, h6.apply_hold(raw, "4h"))["turnover"].sum()
        assert slow < fast

    def test_hold_of_one_interval_is_identity(self):
        p = _panel(100)
        raw = h6.raw_position(p, threshold=0.2, lookback="10min",
                              detrend="raw", trade_middle=False)
        pd.testing.assert_series_equal(h6.apply_hold(raw, "10min"), raw)

    def test_position_constant_within_hold_window(self):
        p = _panel(120)
        raw = h6.raw_position(p, threshold=0.0, lookback="10min",
                              detrend="raw", trade_middle=True)
        held = h6.apply_hold(raw, "1h")  # 6 intervals
        for i in range(0, 60, 6):
            assert held.iloc[i:i + 6].nunique() == 1


class TestRuleDirection:
    def test_high_gamma_mean_reverts_low_gamma_follows(self):
        """The sign convention IS the hypothesis; a flip here would invert H6."""
        idx = pd.date_range("2024-01-01", periods=4, freq="10min", tz="UTC")
        p = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0, 103.0],
             "ret": [np.nan, 0.01, 0.01, 0.01],
             "gex_flow": [0.0, 0.0, 0.0, 0.0]},
            index=idx,
        ).rename_axis("timestamp")

        # price is rising, so momentum = +1 throughout
        hi = h6.raw_position(p.assign(gex_flow=[5.0, 5.0, 5.0, 5.0]),
                             threshold=1.0, lookback="10min",
                             detrend="raw", trade_middle=False)
        lo = h6.raw_position(p.assign(gex_flow=[-5.0, -5.0, -5.0, -5.0]),
                             threshold=1.0, lookback="10min",
                             detrend="raw", trade_middle=False)
        assert (hi.iloc[1:] == -1.0).all(), "high gamma must fade the move"
        assert (lo.iloc[1:] == +1.0).all(), "low gamma must follow the move"

    def test_middle_regime_is_flat_by_default(self):
        p = _panel(200, seed=2)
        pos = h6.raw_position(p, threshold=1e9, lookback="10min",
                              detrend="raw", trade_middle=False)
        assert (pos == 0.0).all(), "an unreachable threshold must never trade"

    def test_warmup_nans_do_not_trade(self):
        """rolling_z is NaN during warm-up; that must mean flat, not a bet."""
        p = _panel(50)
        pos = h6.raw_position(p, threshold=1.0, lookback="10min",
                              detrend="rolling_z", trade_middle=False)
        assert pos.notna().all()
        assert (pos.iloc[:5] == 0.0).all()


class TestBenchmarks:
    def test_buy_and_hold_is_always_long(self):
        p = _panel(50)
        assert (h6.benchmark_position(p, "buy_and_hold", "10min", "1h") == 1.0).all()

    def test_mean_revert_is_negation_of_momentum(self):
        p = _panel(200, seed=4)
        mom = h6.benchmark_position(p, "always_momentum", "10min", "1h")
        mr = h6.benchmark_position(p, "always_mean_revert", "10min", "1h")
        assert np.allclose(mr, -mom)

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            h6.benchmark_position(_panel(10), "always_win", "10min", "1h")


class TestDrawdown:
    def test_monotone_gains_have_no_drawdown(self):
        assert h6.max_drawdown(pd.Series([0.01] * 10)) == 0.0

    def test_drawdown_is_peak_to_trough(self):
        # cumsum path: 1, 0, 1 -> worst dip below the running peak is -1
        assert np.isclose(h6.max_drawdown(pd.Series([1.0, -1.0, 1.0])), -1.0)
