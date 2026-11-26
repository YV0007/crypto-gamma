"""H6 -- can the gamma signal be traded, net of Deribit fees and slippage?

THE RULE, stated before it was run:

    dealer gamma HIGH  -> dealers dampen moves      -> MEAN-REVERT
    dealer gamma LOW   -> dealers amplify moves     -> FOLLOW MOMENTUM
    in between                                      -> FLAT

Position at t is -sign(r_lookback) in the high-gamma regime and +sign(r_lookback)
in the low-gamma regime, held for `hold`, and it earns ret[t+1..t+hold]. Costs
are charged on |position change| from the first run, never bolted on afterwards.

THIS HYPOTHESIS IS PERMITTED TO FAIL. "Mechanism real, alpha absent" is a
legitimate and expected finding: H3 tests whether dealers actually hedge, which
is the scientific claim; H6 asks the separate and much harder question of
whether what they do is exploitable by an outsider paying retail fees. A null
here does not weaken H3. What would be dishonest is searching until something
clears, so every specification tried is written to the specification log and
the Deflated Sharpe bar rises to match the search actually performed.

WHY GROSS IS REPORTED ALONGSIDE NET
-----------------------------------
Net is the answer to "is this tradable". Gross is the answer to "is there any
predictive content at all", which is the scientifically interesting half and is
invisible once 7 bps of costs are subtracted at 10-minute frequency. Reporting
only net would hide a real (if unprofitable) effect; reporting only gross would
be the usual backtest lie. Both, always.

NO LOOKAHEAD -- the three places it would creep in
--------------------------------------------------
1. The regime variable is the rolling-60d z-score of GEX_flow, the same
   strictly-trailing quantity H2 uses as its baseline. Sorting the RAW GEX
   level would sort calendar time (see h2_volatility) and the "strategy" would
   be a bet on the year.
2. The regime threshold is a FIXED number in z units, not a sample quantile.
   An unconditional quantile of z computed over the whole sample would leak the
   future into every early observation.
3. Returns are shifted explicitly: panel `ret[t]` is the return INTO t, so a
   position formed at t must earn `ret` shifted by -1. `forward_return` does
   that shift in one place and `tests/test_h6.py` pins it.

References
    Bailey & Lopez de Prado (2014) for the deflation; see src/utils/stats.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.analysis.h2_volatility import detrend_gex
from src.utils import speclog
from src.utils.stats import (
    annualise,
    block_bootstrap_ci,
    deflate_from_log,
    nw_mean,
    sharpe,
)

log = logging.getLogger("analysis.h6")

BENCHMARKS = ("always_mean_revert", "always_momentum", "buy_and_hold")


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

def cost_bps() -> float:
    """Round-trip cost per UNIT of position change, in basis points.

    `execution: taker` is not a detail. The rule fires on a return that has
    already happened, so it lifts the offer; pricing it at the maker fee would
    be assuming the one fill a latency-disadvantaged outsider cannot count on.
    Slippage is added on top rather than folded into the fee, so the two can be
    stressed separately in the sensitivity sweep.
    """
    fees = config.get("h6_strategy.fees")
    mode = config.get("h6_strategy.execution")
    fee = fees["perp_taker_bps"] if mode == "taker" else fees["perp_maker_bps"]
    return float(fee) + float(fees["slippage_bps"])


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------

def forward_return(panel: pd.DataFrame) -> pd.Series:
    """The return a position formed at t actually earns.

    panel `ret[t]` is log(close_t / close_{t-1}) -- the return INTO t, already
    realised when the signal is formed. Shifting by -1 is the entire difference
    between a backtest and a fantasy, so it happens here and nowhere else.
    """
    return panel["ret"].shift(-1)


def regime(panel: pd.DataFrame, threshold: float, detrend: str) -> pd.Series:
    """+1 where dealer gamma is HIGH, -1 where LOW, 0 in between.

    NaN during the rolling-window warm-up propagates to a flat position, which
    is correct: no signal is not the same as a signal of zero, but both mean
    "do not trade", and conflating them here would only ever cost fees.
    """
    z = detrend_gex(panel, detrend)
    out = pd.Series(0, index=panel.index, dtype=float)
    out[z >= threshold] = 1.0
    out[z <= -threshold] = -1.0
    out[z.isna()] = 0.0
    return out


def raw_position(
    panel: pd.DataFrame,
    *,
    threshold: float,
    lookback: str,
    detrend: str,
    trade_middle: bool,
) -> pd.Series:
    """The desired position before the holding period is imposed.

    High gamma flips the sign of the recent move (mean-revert), low gamma keeps
    it (momentum). `trade_middle=True` treats the middle of the z distribution
    as a mild momentum zone; the baseline is False, because the hypothesis is a
    claim about the TAILS and trading the middle just pays fees on noise.
    """
    step = pd.Timedelta(config.get("construction.interval"))
    k = max(int(pd.Timedelta(lookback) / step), 1)
    # Return over the lookback, ending at t: strictly backward-looking.
    mom = np.sign(np.log(panel["close"]).diff(k))

    g = regime(panel, threshold, detrend)
    pos = pd.Series(0.0, index=panel.index)
    pos[g == 1.0] = -mom[g == 1.0]      # high gamma -> mean-revert
    pos[g == -1.0] = mom[g == -1.0]     # low gamma  -> momentum
    if trade_middle:
        pos[g == 0.0] = mom[g == 0.0]
    return pos.fillna(0.0)


def apply_hold(pos: pd.Series, hold: str) -> pd.Series:
    """Allow the position to change only at `hold` boundaries.

    Rebalancing every 10 minutes at 7 bps a turn is a fee-generation machine,
    so the holding period is a declared parameter rather than an afterthought.
    Between boundaries the previous position is carried forward, which is what
    "holding" means and also what keeps measured turnover honest.
    """
    step = pd.Timedelta(config.get("construction.interval"))
    k = max(int(pd.Timedelta(hold) / step), 1)
    if k == 1:
        return pos
    gate = np.zeros(len(pos), dtype=bool)
    gate[::k] = True
    held = pos.where(gate).ffill()
    return held.fillna(0.0)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def backtest(panel: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Per-interval gross return, cost and net return for a position series.

    The position is assumed to be taken at the CLOSE of t and to earn the
    return into t+1. Costs are charged in the interval the trade happens, on
    |pos_t - pos_{t-1}|, so a full reversal pays twice -- which it does.
    """
    fwd = forward_return(panel)
    c = cost_bps() / 1e4
    turnover = pos.diff().abs().fillna(pos.abs())
    gross = pos * fwd
    cost = turnover * c
    out = pd.DataFrame({
        "position": pos, "forward_ret": fwd,
        "gross": gross, "cost": cost, "net": gross - cost,
        "turnover": turnover,
    })
    # Drop the final row: its forward return does not exist yet.
    return out.dropna(subset=["forward_ret"])


def max_drawdown(net: pd.Series) -> float:
    """Worst peak-to-trough on the cumulative log-return path."""
    eq = net.fillna(0.0).cumsum()
    return float((eq - eq.cummax()).min())


def evaluate(bt: pd.DataFrame, label: str, *, bootstrap: bool = True) -> dict:
    """Everything reported about one strategy, gross and net side by side.

    `bootstrap=False` skips the block bootstrap interval, which costs ~39 s on
    the full panel (1000 reps x ~1460 daily blocks). It is worth that for the
    baseline and the benchmarks; across a 48-cell sweep it is half an hour to
    re-learn what the HAC t-stat on each cell already says. The headline number
    always carries the interval.
    """
    net, gross = bt["net"].dropna(), bt["gross"].dropna()
    if len(net) < 100:
        return {"label": label, "n_obs": int(len(net)), "insufficient": True}

    m = nw_mean(net)
    days = (bt.index[-1] - bt.index[0]).total_seconds() / 86400 or np.nan
    if bootstrap:
        blocks = bt.index.normalize()
        lo, hi = block_bootstrap_ci(net.to_numpy(), blocks[: len(net)].to_numpy())
    else:
        lo = hi = float("nan")

    # Hit rate must condition on actually HOLDING something. Counting flat
    # intervals as losses drags it toward zero and makes a rule that is flat
    # 63% of the time look far worse than it is.
    live = bt[bt["position"] != 0]
    turn = float(bt["turnover"].sum())
    gross_total = float(gross.sum())

    return {
        "label": label,
        "n_obs": int(len(net)),
        "sharpe_net": sharpe(net),
        "sharpe_net_annual": annualise(sharpe(net)),
        "sharpe_gross": sharpe(gross),
        "sharpe_gross_annual": annualise(sharpe(gross)),
        "mean_net": float(net.mean()),
        "mean_gross": float(gross.mean()),
        "hac_t_stat": float(m.t_stat[0]),
        "hac_p_value": float(m.p_value[0]),
        "hac_lags": int(m.lags),
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "total_net_log_return": float(net.sum()),
        "total_gross_log_return": gross_total,
        "total_cost": float(bt["cost"].sum()),
        "turnover_per_year": turn / days * 365 if days else np.nan,
        "n_rebalances": int((bt["turnover"] > 0).sum()),
        "time_in_market": float((bt["position"] != 0).mean()),
        "hit_rate_when_in_market": float((live["net"] > 0).mean()) if len(live) else np.nan,
        "max_drawdown": max_drawdown(net),
        # The single most useful number for the deck: the cost, in bps per unit
        # of turnover, at which this rule would exactly break even. Compare it
        # with cost_bps() to see how far from tradable the signal is, in units
        # a reader can argue with -- rather than a Sharpe they cannot.
        "breakeven_cost_bps": (gross_total / turn * 1e4) if turn > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

def benchmark_position(panel: pd.DataFrame, name: str, lookback: str, hold: str) -> pd.Series:
    """The three pre-declared benchmarks.

    They exist to answer "compared with what?". A gamma rule that merely
    reproduces always-momentum has found nothing about gamma, and buy-and-hold
    is the only one of the three that pays costs once instead of continuously.
    """
    step = pd.Timedelta(config.get("construction.interval"))
    k = max(int(pd.Timedelta(lookback) / step), 1)
    mom = np.sign(np.log(panel["close"]).diff(k)).fillna(0.0)
    if name == "always_mean_revert":
        return apply_hold(-mom, hold)
    if name == "always_momentum":
        return apply_hold(mom, hold)
    if name == "buy_and_hold":
        return pd.Series(1.0, index=panel.index)
    raise ValueError(f"unknown benchmark {name!r}")


# ---------------------------------------------------------------------------
# one specification
# ---------------------------------------------------------------------------

def run_spec(
    panel: pd.DataFrame,
    *,
    threshold: float,
    lookback: str,
    hold: str,
    detrend: str,
    trade_middle: bool,
    name: str,
    record: bool = True,
    bootstrap: bool = True,
) -> dict:
    """Backtest one parameterisation and (by default) log it as a trial.

    `record=True` on EVERY call, winners and losers alike. A specification log
    containing only the good runs deflates nothing while looking like rigour.
    """
    pos = apply_hold(
        raw_position(panel, threshold=threshold, lookback=lookback,
                     detrend=detrend, trade_middle=trade_middle),
        hold,
    )
    res = evaluate(backtest(panel, pos), name, bootstrap=bootstrap)
    res.update({"threshold": threshold, "lookback": lookback, "hold": hold,
                "detrend": detrend, "trade_middle": trade_middle})

    if record and not res.get("insufficient"):
        speclog.record(
            "h6", name,
            params={"threshold": threshold, "lookback": lookback, "hold": hold,
                    "detrend": detrend, "trade_middle": trade_middle,
                    "cost_bps": cost_bps()},
            n_obs=res["n_obs"], statistic="net_return_per_interval",
            estimate=res["mean_net"], t_stat=res["hac_t_stat"],
            p_value=res["hac_p_value"], sharpe=res["sharpe_net"],
            notes="H6 gamma-conditional timing rule, net of taker fees + slippage",
        )
    return res


def sweep(panel: pd.DataFrame, sig: dict) -> list[dict]:
    """The full alternatives grid. Every cell is logged as a distinct trial.

    This is deliberately run BEFORE the deflation is computed, so that the
    Deflated Sharpe the baseline has to clear already accounts for the whole
    search -- not just the specification that happened to be written down.
    """
    alt = config.get("h6_strategy.alternatives")
    grid = list(itertools.product(
        sorted({sig["gamma_z_threshold"], *alt["gamma_z_threshold"]}),
        list(dict.fromkeys([sig["lookback"], *alt["lookback"]])),
        list(dict.fromkeys([sig["hold"], *alt["hold"]])),
    ))
    log.info("sweeping %d specifications", len(grid))
    out = []
    for thr, lb, hd in grid:
        r = run_spec(panel, threshold=thr, lookback=lb, hold=hd,
                     detrend=sig["detrend"], trade_middle=sig["trade_middle"],
                     name=f"gamma_timing_z{thr}_lb{lb}_h{hd}", bootstrap=False)
        if not r.get("insufficient"):
            out.append(r)
            log.info("  z>=%.1f lb=%-5s hold=%-4s  SR_net=%+.5f  SR_gross=%+.5f  net=%+.4f",
                     thr, lb, hd, r["sharpe_net"], r["sharpe_gross"],
                     r["total_net_log_return"])
    return out


# ---------------------------------------------------------------------------
# what the gross edge actually is
# ---------------------------------------------------------------------------

def microstructure_diagnostic(panel: pd.DataFrame) -> dict:
    """Is the gross edge a gamma effect, or is it bid-ask bounce?

    This is the diagnostic that decides how to read a positive GROSS Sharpe,
    and leaving it out would let a microstructure artefact be presented as a
    finding about dealer hedging.

    Bid-ask bounce puts a NEGATIVE first-order autocorrelation into measured
    returns -- trades alternate between hitting the bid and lifting the offer,
    so the recorded price zig-zags inside a spread that never moved. Its
    signature is sharp: rho(1) < 0 and rho(k) ~ 0 for every k > 1. A genuine
    dealer-hedging effect would decay over the horizon dealers actually hedge
    on, not die after one interval.

    It also explains any sub-1bp breakeven cost. The bounce is bounded by the
    spread, so a rule that harvests it is trying to earn the spread while
    paying the spread, which nets to a loss for anyone who is not the
    market maker collecting it.
    """
    r = panel["ret"].dropna()
    rho = {int(k): float(r.autocorr(k)) for k in (1, 2, 3, 6, 12)}
    bounce = rho[1] < 0 and all(abs(rho[k]) < abs(rho[1]) / 2 for k in (2, 3, 6))
    return {
        "autocorr": rho,
        "n_obs": int(len(r)),
        "return_sd": float(r.std()),
        "bounce_signature": bool(bounce),
        "interpretation": (
            "rho(1) < 0 with rho(k>1) ~ 0: the classic bid-ask bounce signature. "
            "Any gross edge from a sign-fade rule is the spread, not a gamma "
            "effect, and cannot be earned net of paying that same spread."
            if bounce else
            "No clean bounce signature; a gross edge here needs another explanation."
        ),
    }


def plot_equity(panel: pd.DataFrame, sig: dict, dest) -> None:
    """Cumulative log return: baseline gross vs net, against the benchmarks.

    The gap between the two baseline lines IS the result -- it is the fee bill,
    drawn to scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = apply_hold(
        raw_position(panel, threshold=sig["gamma_z_threshold"],
                     lookback=sig["lookback"], detrend=sig["detrend"],
                     trade_middle=sig["trade_middle"]),
        sig["hold"])
    bt = backtest(panel, pos)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(bt.index, bt["gross"].cumsum(), lw=1.4, color="tab:blue",
            label=f"H6 gamma rule, GROSS (SR {annualise(sharpe(bt['gross'])):+.2f}/yr)")
    net_lab = (f"H6 gamma rule, NET of {cost_bps():.0f}bps "
               f"(SR {annualise(sharpe(bt['net'])):+.2f}/yr)")
    ax.plot(bt.index, bt["net"].cumsum(), lw=1.4, color="tab:red", label=net_lab)
    for name, colour in [("always_mean_revert", "tab:orange"), ("buy_and_hold", "tab:green")]:
        b = backtest(panel, benchmark_position(panel, name, sig["lookback"], sig["hold"]))
        lab = f"{name}, net (SR {annualise(sharpe(b['net'])):+.2f}/yr)"
        ax.plot(b.index, b["net"].cumsum(), lw=1.0, ls="--", color=colour, label=lab)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative log return")
    ax.set_title("H6 -- gamma-conditional timing rule. "
                 "The gap between blue and red is the fee bill.")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    log.info("wrote %s", dest)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run(start: str, end: str) -> dict:
    sig = config.get("h6_strategy.signal")
    panel = pd.read_parquet(config.path("paths.processed") / f"panel_{start}_{end}.parquet")
    log.info("panel %s -> %s, %d rows; cost %.1f bps per unit turnover",
             start, end, len(panel), cost_bps())

    # Baseline FIRST, so it is on the record before any sweep result is seen.
    baseline = run_spec(
        panel, threshold=sig["gamma_z_threshold"], lookback=sig["lookback"],
        hold=sig["hold"], detrend=sig["detrend"],
        trade_middle=sig["trade_middle"], name="baseline",
    )
    log.info("baseline SR_net=%.5f (annual %.3f)  SR_gross=%.5f",
             baseline["sharpe_net"], baseline["sharpe_net_annual"],
             baseline["sharpe_gross"])

    bench = {}
    for b in config.get("h6_strategy.benchmarks"):
        pos = benchmark_position(panel, b, sig["lookback"], sig["hold"])
        bench[b] = evaluate(backtest(panel, pos), b)

    grid = sweep(panel, sig)

    # Deflation is computed AFTER the sweep so the trial count includes it.
    pos = apply_hold(
        raw_position(panel, threshold=sig["gamma_z_threshold"],
                     lookback=sig["lookback"], detrend=sig["detrend"],
                     trade_middle=sig["trade_middle"]),
        sig["hold"],
    )
    net = backtest(panel, pos)["net"].dropna()
    dsr = deflate_from_log(net.to_numpy(), hypothesis=None)
    log.info("\n%s", dsr)

    micro = microstructure_diagnostic(panel)
    log.info("microstructure: %s", micro["interpretation"])

    fig_dest = config.path("paths.figures") / "h6_equity_curve.png"
    fig_dest.parent.mkdir(parents=True, exist_ok=True)
    plot_equity(panel, sig, fig_dest)

    best = max(grid, key=lambda r: r["sharpe_net"]) if grid else None
    return {
        "microstructure_diagnostic": micro,
        "sample": {"start": start, "end": end},
        "cost_bps_per_unit_turnover": cost_bps(),
        "execution": config.get("h6_strategy.execution"),
        "signal": sig,
        "baseline": baseline,
        "benchmarks": bench,
        "sweep": grid,
        "best_in_sweep": best,
        "deflated_sharpe_of_baseline": {
            k: (float(v) if isinstance(v, int | float | np.floating) else v)
            for k, v in dsr.__dict__.items()
        },
        "verdict": verdict(baseline, bench, dsr),
    }


def verdict(baseline: dict, bench: dict, dsr) -> dict:
    """The pre-committed decision rule, applied without discretion.

    Tradable requires ALL of: positive net Sharpe, a HAC t-stat clearing 2, and
    survival of the deflation at the configured threshold. Anything less is
    reported as "mechanism may be real, alpha absent", which the plan permits.
    """
    thr = config.get("stats.deflated_sharpe.survival_threshold")
    passes = {
        "net_sharpe_positive": bool(baseline.get("sharpe_net", float("nan")) > 0),
        "hac_t_above_2": bool(abs(baseline.get("hac_t_stat", 0.0)) > 2.0),
        "survives_deflation": bool(np.isfinite(dsr.dsr) and dsr.dsr >= thr),
        "beats_best_benchmark": bool(
            baseline.get("sharpe_net", float("-inf"))
            > max((b.get("sharpe_net", float("-inf")) for b in bench.values()),
                  default=float("-inf"))
        ),
    }
    return {
        "criteria": passes,
        "tradable": all(passes.values()),
        "statement": ("tradable net of fees" if all(passes.values())
                      else "NOT tradable net of fees -- mechanism may be real, alpha absent"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="H6: is the gamma signal tradable?")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")
    res = run(start, end)
    dest = config.path("paths.tables") / "h6_results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, default=str))
    log.info("wrote %s", dest)
    print(json.dumps(res["verdict"], indent=2))


if __name__ == "__main__":
    main()
