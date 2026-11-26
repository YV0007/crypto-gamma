"""Reconstruct cumulative dealer inventory from taker-flagged option flow.

SIGN CONVENTION -- the entire hypothesis rests on this
======================================================
Deribit flags the *taker* (aggressor) on every trade. Market makers sit in the
book quoting; customers cross the spread to get filled. Therefore:

    direction == "buy"   ->  customer BOUGHT   ->  dealer SOLD    ->  dealer_flow = -amount
    direction == "sell"  ->  customer SOLD     ->  dealer BOUGHT  ->  dealer_flow = +amount

    dealer_position[i, t] = dealer_position[i, t-1] + dealer_flow[i, t]

`direction` describes the TAKER, and the dealer is the counterparty, so the
dealer's flow is the negation of the taker's. A sign error here does not weaken
the study's results -- it reverses them. This is why `tests/test_inventory.py`
asserts the invariant directly and why STEP 3(a) prints a hand-checkable trade
ledger.

Position is initialised at 0 on the instrument's creation_timestamp and forced
hard to 0 at expiration_timestamp (config `construction.expiry_handling`).
Between trades the position persists: it is forward-filled onto the interval
grid, because a position exists whether or not anyone trades.

UNITS
-----
Position is carried in CONTRACTS, not BTC notional: `amount / contract_size`,
with contract_size joined per instrument from instruments.parquet (options
1.0, BTC-PERPETUAL 10.0 -- never a single global constant). contract_size is
carried through to the output so the GEX step can apply it explicitly:

    GEX = dealer_position x gamma x S^2 x 0.01 x contract_size

NO GREEKS ARE APPLIED HERE. Position only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("build.inventory")

VALID_DIRECTIONS = {"buy", "sell"}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_trades(start: str, end: str) -> pd.DataFrame:
    """Load option trades for [start, end] inclusive from data/interim/."""
    src = config.path("paths.interim") / "option_trades"
    if not src.exists():
        raise SystemExit(f"{src} not found -- run scripts/to_parquet.py first")

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    months = pd.period_range(lo, hi - pd.Timedelta(seconds=1), freq="M")
    filters = [[("year", "==", p.year), ("month", "==", p.month)] for p in months]

    df = pd.read_parquet(src, filters=filters)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["dt"] >= lo) & (df["dt"] < hi)].copy()
    log.info("loaded %d trades %s .. %s", len(df), df["dt"].min(), df["dt"].max())
    return df


def load_instruments() -> pd.DataFrame:
    p = config.path("paths.interim") / "instruments.parquet"
    if not p.exists():
        raise SystemExit(f"{p} not found -- run python3 -m src.fetch.instruments")
    inst = pd.read_parquet(p)
    return inst[inst["kind"] == "option"].copy()


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def apply_block_trade_policy(df: pd.DataFrame, mode: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply config `construction.block_trades`.

    include      -- keep them at full weight (BASELINE)
    exclude      -- drop block trades entirely (robustness)
    half_weight  -- keep them at 0.5 weight (robustness alternative)

    Block trades are negotiated off-book. Whether the taker flag carries the
    same meaning for them is exactly the kind of choice Glassnode does not
    disclose, so it is a config parameter with a documented baseline.

    The baseline was `exclude` until 2026-09-01, on the grounds that blocks are
    only 1.3% of trades. That is the wrong denominator: they are 31% of volume
    and 67% of net dealer flow, and excluding them halves the measured gamma
    exposure and flips the sign of 15% of intervals. See DIAGNOSTIC_H2.md §2.
    """
    mode = mode or config.get("construction.block_trades.baseline")
    is_block = df["block_trade_id"].notna() if "block_trade_id" in df.columns else pd.Series(False, index=df.index)
    n_block = int(is_block.sum())

    if mode == "exclude":
        out = df.loc[~is_block].copy()
        out["weight"] = 1.0
    elif mode == "include":
        out = df.copy()
        out["weight"] = 1.0
    elif mode == "half_weight":
        out = df.copy()
        out["weight"] = np.where(is_block, 0.5, 1.0)
    else:
        raise ValueError(f"unknown block_trades mode {mode!r} (exclude|include|half_weight)")

    # Report the block share three ways. By COUNT alone blocks look negligible
    # (~1.3%) and that number is what motivated the old `exclude` baseline. By
    # VOLUME they are ~31%, and because they are large and directional they
    # carry ~2/3 of the NET dealer flow -- the only quantity GEX actually uses.
    # Never judge this policy on the count again.
    vol_all = float(df["amount"].sum()) if "amount" in df.columns else 0.0
    vol_blk = float(df.loc[is_block, "amount"].sum()) if vol_all else 0.0
    net_all = net_blk = 0.0
    if {"amount", "direction"} <= set(df.columns):
        signed = np.where(df["direction"].to_numpy() == "buy", -1.0, 1.0) * df["amount"].to_numpy()
        net_all = float(signed.sum())
        net_blk = float(signed[is_block.to_numpy()].sum())

    stats = {
        "mode": mode,
        "n_trades_in": len(df),
        "n_block": n_block,
        "block_fraction": n_block / len(df) if len(df) else 0.0,
        "block_fraction_volume": vol_blk / vol_all if vol_all else 0.0,
        "block_fraction_net_flow": abs(net_blk) / abs(net_all) if net_all else 0.0,
        "n_trades_out": len(out),
    }
    log.info("block trade policy %s: blocks are %.2f%% of trades but %.2f%% of "
             "VOLUME and %.2f%% of NET dealer flow (%d/%d trades, %+.0f of %+.0f BTC net) "
             "-> %d trades kept",
             mode, 100 * stats["block_fraction"], 100 * stats["block_fraction_volume"],
             100 * stats["block_fraction_net_flow"], n_block, len(df),
             net_blk, net_all, len(out))
    return out, stats


def compute_dealer_flow(df: pd.DataFrame, inst: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Signed dealer flow per trade, in contracts."""
    cs = inst.set_index("instrument_name")["contract_size"]
    df = df.copy()
    df["contract_size"] = df["instrument_name"].map(cs)

    n_unmapped = int(df["contract_size"].isna().sum())
    if n_unmapped:
        missing = sorted(df.loc[df["contract_size"].isna(), "instrument_name"].unique())[:5]
        raise RuntimeError(
            f"{n_unmapped} trades on instruments absent from instruments.parquet, "
            f"e.g. {missing}. contract_size MUST come per-instrument."
        )

    bad_dir = ~df["direction"].isin(VALID_DIRECTIONS)
    n_bad = int(bad_dir.sum())
    if n_bad:
        log.error("%d trades with null/unexpected direction -- DROPPED: %s",
                  n_bad, df.loc[bad_dir, "direction"].value_counts(dropna=False).to_dict())
        df = df.loc[~bad_dir].copy()

    # THE sign convention. taker buy -> dealer short.
    taker_sign = np.where(df["direction"].to_numpy() == "buy", -1.0, +1.0)
    df["dealer_flow"] = taker_sign * (df["amount"] / df["contract_size"]) * df["weight"]

    stats = {"n_bad_direction": n_bad, "bad_direction_fraction": n_bad / max(len(df) + n_bad, 1)}
    return df, stats


def accumulate(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative dealer position per instrument, in strict timestamp order."""
    df = df.sort_values(["instrument_name", "timestamp", "trade_seq"], kind="mergesort")
    df["dealer_position"] = df.groupby("instrument_name", sort=False)["dealer_flow"].cumsum()
    return df


def to_interval_grid(
    trades: pd.DataFrame,
    inst: pd.DataFrame,
    start: str,
    end: str,
    interval: str | None = None,
) -> pd.DataFrame:
    """Forward-fill positions onto a regular interval grid.

    Grid spans the requested window for every instrument that traded in it.
    Position is 0 before creation_timestamp, the last observed cumulative
    position between trades, and hard 0 at/after expiration_timestamp.
    """
    interval = interval or config.get("construction.interval")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    grid = pd.date_range(lo, hi, freq=interval, inclusive="left")

    names = sorted(trades["instrument_name"].unique())
    log.info("grid: %d instruments x %d intervals (%s) = %d rows",
             len(names), len(grid), interval, len(names) * len(grid))

    panel = pd.DataFrame({
        "instrument_name": np.repeat(names, len(grid)),
        "timestamp": np.tile(grid, len(names)),
    })

    last = (trades[["instrument_name", "dt", "dealer_position"]]
            .sort_values("dt", kind="mergesort")
            .rename(columns={"dt": "timestamp"}))

    panel = pd.merge_asof(
        panel.sort_values("timestamp", kind="mergesort"),
        last,
        on="timestamp",
        by="instrument_name",
        direction="backward",      # last position at or before the grid point
    )
    # before the first trade in the window there is no observation -> 0
    panel["dealer_position"] = panel["dealer_position"].fillna(0.0)

    meta = inst.set_index("instrument_name")
    panel["creation_ts"] = panel["instrument_name"].map(meta["creation_ts"])
    panel["expiry_ts"] = panel["instrument_name"].map(meta["expiry_ts"])
    panel["contract_size"] = panel["instrument_name"].map(meta["contract_size"])

    # 0 before listing, and hard 0 at expiry
    before = panel["timestamp"] < panel["creation_ts"]
    panel.loc[before, "dealer_position"] = 0.0

    if config.get("construction.expiry_handling") == "hard_zero":
        expired = panel["timestamp"] >= panel["expiry_ts"]
        panel.loc[expired, "dealer_position"] = 0.0
        log.info("hard-zeroed %d grid points at/after expiry", int(expired.sum()))

    return (panel[["instrument_name", "timestamp", "dealer_position",
                   "contract_size", "creation_ts", "expiry_ts"]]
            .sort_values(["instrument_name", "timestamp"], kind="mergesort")
            .reset_index(drop=True))


def build_inventory(start: str, end: str, block_mode: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full pipeline. Returns (grid_panel, trade_level, stats)."""
    inst = load_instruments()
    raw = load_trades(start, end)
    kept, block_stats = apply_block_trade_policy(raw, block_mode)
    flow, dir_stats = compute_dealer_flow(kept, inst)
    trades = accumulate(flow)
    panel = to_interval_grid(trades, inst, start, end)
    return panel, trades, {**block_stats, **dir_stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dealer inventory from taker flow")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--block-mode", default=None, help="override construction.block_trades.baseline")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s")

    panel, trades, stats = build_inventory(args.start, args.end, args.block_mode)
    out_dir = config.path("paths.processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"inventory_{args.start}_{args.end}.parquet"
    panel.set_index(["instrument_name", "timestamp"]).to_parquet(dest)
    log.info("wrote %s (%d rows) stats=%s", dest, len(panel), stats)


if __name__ == "__main__":
    main()
