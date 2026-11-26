"""Full-period inventory -> greeks -> GEX, computed in bounded memory.

WHY THIS EXISTS SEPARATELY FROM inventory.py
--------------------------------------------
The March validation built an explicit (instrument x interval) grid. Over the
full sample that is 76,028 instruments x 210,384 ten-minute intervals ~ 1.6e10
cells: impossible to materialise.

The decomposition that makes it tractable:

  1. Dealer position is a STEP FUNCTION that only changes at trade times, so
     the cumulative position from listing is one groupby-cumsum over all
     16.9M trades -- exact, and cheap.
  2. Gamma must be evaluated per (instrument, interval) because it depends on
     spot and time-to-expiry, so that part is irreducible. It is streamed in
     time chunks and aggregated to GEX[t] immediately, never stored.

Positions therefore initialise at listing exactly as CONTEXT.md §11 requires
(position is 0 until an instrument's first trade, which is at or after its
creation_timestamp), removing the one-month truncation that qualified every
number in the March slice.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.build.black_scholes import clean_iv, greeks as bs_greeks, time_to_expiry_years
from src.build.inventory import apply_block_trade_policy

log = logging.getLogger("build.pipeline")

TRADE_COLS = ["instrument_name", "timestamp", "direction", "amount",
              "block_trade_id", "iv", "index_price", "trade_seq"]


def load_all_trades(start: str, end: str) -> pd.DataFrame:
    """All option trades in the window, memory-conscious dtypes."""
    src = config.path("paths.interim") / "option_trades"
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    months = pd.period_range(lo, hi - pd.Timedelta(seconds=1), freq="M")
    filters = [[("year", "==", p.year), ("month", "==", p.month)] for p in months]
    t0 = time.time()
    df = pd.read_parquet(src, filters=filters, columns=TRADE_COLS)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["dt"] >= lo) & (df["dt"] < hi)]
    df["instrument_name"] = df["instrument_name"].astype("category")
    df["direction"] = df["direction"].astype("category")
    log.info("loaded %s trades in %.1fs (%.2f GB) %s .. %s",
             f"{len(df):,}", time.time() - t0,
             df.memory_usage(deep=True).sum() / 1e9, df["dt"].min(), df["dt"].max())
    return df


def cumulative_positions(trades: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Signed dealer position at every trade time, accumulated FROM LISTING."""
    cs = inst.set_index("instrument_name")["contract_size"]
    df = trades.copy()
    df["contract_size"] = df["instrument_name"].map(cs).astype(float)
    if df["contract_size"].isna().any():
        n = int(df["contract_size"].isna().sum())
        raise RuntimeError(f"{n} trades on instruments missing from instruments.parquet")

    bad = ~df["direction"].isin(["buy", "sell"])
    if bad.any():
        log.error("dropping %d trades with null/unexpected direction", int(bad.sum()))
        df = df[~bad]

    # taker buy -> customer bought -> dealer sold -> negative
    sign = np.where(df["direction"].to_numpy() == "buy", -1.0, 1.0)
    df["dealer_flow"] = sign * (df["amount"] / df["contract_size"]) * df["weight"]

    df = df.sort_values(["instrument_name", "timestamp", "trade_seq"], kind="mergesort")
    df["dealer_position"] = df.groupby("instrument_name", observed=True)["dealer_flow"].cumsum()
    log.info("accumulated positions over %s trades, %s instruments",
             f"{len(df):,}", f"{df['instrument_name'].nunique():,}")
    return df


def _chunk_gex(
    chunk_lo: pd.Timestamp,
    chunk_hi: pd.Timestamp,
    trades: pd.DataFrame,
    meta: pd.DataFrame,
    interval: str,
    staleness: pd.Timedelta,
    exclude_tte_days: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GEX for one time chunk. Returns (per-interval series, diagnostics)."""
    grid = pd.date_range(chunk_lo, chunk_hi, freq=interval, inclusive="left")
    if len(grid) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # instruments that could carry a position in this chunk: already traded at
    # least once before the chunk ends, and not yet expired at its start
    alive = meta[(meta["creation_ts"] < chunk_hi) & (meta["expiry_ts"] > chunk_lo)]
    traded_by = trades[trades["dt"] < chunk_hi]["instrument_name"].unique()
    names = alive.index.intersection(pd.Index(traded_by))
    if len(names) == 0:
        return pd.DataFrame(), pd.DataFrame()

    panel = pd.DataFrame({
        "instrument_name": np.repeat(names.to_numpy(), len(grid)),
        "timestamp": np.tile(grid, len(names)),
    })

    sub = trades[trades["instrument_name"].isin(names) & (trades["dt"] < chunk_hi)]

    # position: last cumulative value at or before t, persists indefinitely
    pos = (sub[["instrument_name", "dt", "dealer_position"]]
           .rename(columns={"dt": "timestamp"}).sort_values("timestamp", kind="mergesort"))
    pos["instrument_name"] = pos["instrument_name"].astype(str)
    panel["instrument_name"] = panel["instrument_name"].astype(str)
    panel = panel.sort_values("timestamp", kind="mergesort")
    panel = pd.merge_asof(panel, pos, on="timestamp", by="instrument_name", direction="backward")
    panel["dealer_position"] = panel["dealer_position"].fillna(0.0)

    # IV: last published mark IV at or before t, bounded by staleness
    iv = (sub[["instrument_name", "dt", "iv_pct"]].dropna(subset=["iv_pct"])
          .rename(columns={"dt": "timestamp"}).sort_values("timestamp", kind="mergesort"))
    iv["instrument_name"] = iv["instrument_name"].astype(str)
    panel = pd.merge_asof(panel, iv, on="timestamp", by="instrument_name",
                          direction="backward", tolerance=staleness)

    # spot: last index price at or before t (from the trade tape itself)
    spot = (trades[trades["dt"] < chunk_hi][["dt", "index_price"]]
            .rename(columns={"dt": "timestamp", "index_price": "spot"})
            .sort_values("timestamp", kind="mergesort"))
    gsp = pd.merge_asof(pd.DataFrame({"timestamp": grid}), spot,
                        on="timestamp", direction="backward")
    panel = panel.merge(gsp, on="timestamp", how="left")

    panel["strike"] = panel["instrument_name"].map(meta["strike"])
    panel["option_type"] = panel["instrument_name"].map(meta["option_type"])
    panel["expiry_ts"] = panel["instrument_name"].map(meta["expiry_ts"])
    panel["contract_size"] = panel["instrument_name"].map(meta["contract_size"])

    # hard zero at expiry, zero before listing
    panel.loc[panel["timestamp"] >= panel["expiry_ts"], "dealer_position"] = 0.0
    panel.loc[panel["timestamp"] < panel["instrument_name"].map(meta["creation_ts"]),
              "dealer_position"] = 0.0

    tte = time_to_expiry_years(panel["timestamp"], panel["expiry_ts"])
    g = bs_greeks(panel["spot"].to_numpy(), panel["strike"].to_numpy(),
                  tte.to_numpy(), (panel["iv_pct"] / 100.0).to_numpy(),
                  panel["option_type"].to_numpy())
    panel["gamma"] = g["gamma"].to_numpy()

    panel["contrib"] = (panel["dealer_position"] * panel["gamma"]
                        * panel["spot"] ** 2 * 0.01 * panel["contract_size"])

    # Optional: drop the cells closest to expiry. Black-Scholes gamma diverges
    # as T->0, so the largest numbers in the series come from options in their
    # final hours and their ceiling is set by `min_tte_days`, an arbitrary
    # constant. On March 2024 the last day before expiry is 8.5% of cells but
    # 22% of gross gamma dollars, and removing it changes the sign of 13.7% of
    # intervals. Deribit expires daily, so this is a pervasive component of
    # every interval rather than a tail. A headline that survives this is much
    # harder to argue with.
    if exclude_tte_days > 0:
        near = tte.to_numpy() < (exclude_tte_days / 365.25)
        panel.loc[near, "contrib"] = 0.0

    live = panel["timestamp"] < panel["expiry_ts"]
    has_g = panel["gamma"].notna()
    absp = panel["dealer_position"].abs()

    series = panel.groupby("timestamp").agg(
        gex_flow=("contrib", "sum"),
        spot=("spot", "first"),
        n_priced=("gamma", "count"),
    )
    series["n_alive"] = panel[live].groupby("timestamp").size()
    series["n_alive_pos"] = panel[live & (absp > 0)].groupby("timestamp").size()

    diag = pd.DataFrame({
        "cells": [len(panel)],
        "cells_live": [int(live.sum())],
        "cells_with_gamma": [int(has_g.sum())],
        "pos_weight_total": [float(absp[live].sum())],
        "pos_weight_with_gamma": [float(absp[live & has_g].sum())],
        # is the missing mass disproportionately HIGH gamma? compare the gamma
        # actually observed against gamma for the same cells with staleness off
        "gamma_sum_observed": [float(panel.loc[has_g, "gamma"].sum())],
        "abs_contrib": [float(panel["contrib"].abs().sum())],
    })
    return series, diag


def run(start: str, end: str, staleness: str | None = None, tag: str = "",
        block_mode: str | None = None, exclude_tte_days: float = 0.0) -> dict:
    t0 = time.time()
    interval = config.get("construction.interval")
    stale = pd.Timedelta(staleness or config.get("construction.greeks.max_staleness"))
    chunk_days = config.get("construction.pipeline_chunk_days", 5)

    inst_all = pd.read_parquet(config.path("paths.interim") / "instruments.parquet")
    inst = inst_all[inst_all["kind"] == "option"].copy()
    meta = inst.set_index("instrument_name")

    trades = load_all_trades(start, end)
    trades, bstats = apply_block_trade_policy(trades, block_mode)
    trades["iv_pct"] = clean_iv(trades["iv"])
    trades = cumulative_positions(trades, inst)

    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    bounds = pd.date_range(lo, hi, freq=f"{chunk_days}D")
    if bounds[-1] < hi:
        bounds = bounds.append(pd.DatetimeIndex([hi]))

    parts, diags = [], []
    for i in range(len(bounds) - 1):
        s, d = _chunk_gex(bounds[i], bounds[i + 1], trades, meta, interval, stale,
                          exclude_tte_days)
        if len(s):
            parts.append(s); diags.append(d)
        if i % 20 == 0:
            log.info("chunk %d/%d  %s  elapsed %.1f min",
                     i + 1, len(bounds) - 1, bounds[i].date(), (time.time() - t0) / 60)

    series = pd.concat(parts).sort_index()
    diag = pd.concat(diags, ignore_index=True).sum().to_dict()

    proc = config.path("paths.processed")
    proc.mkdir(parents=True, exist_ok=True)
    suffix = f"{start}_{end}{tag}"
    series.to_parquet(proc / f"gex_flow_{suffix}.parquet")

    final = (trades.groupby("instrument_name", observed=True)["dealer_position"].last()
             .rename("final_position"))
    final.to_frame().to_parquet(proc / f"final_positions_{suffix}.parquet")

    elapsed = time.time() - t0
    log.info("DONE in %.1f min: %s intervals, %s trades, %s instruments",
             elapsed / 60, f"{len(series):,}", f"{len(trades):,}", f"{final.size:,}")
    return {"elapsed_min": elapsed / 60, "intervals": len(series),
            "instruments": int(final.size), **bstats, **diag}


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-period GEX pipeline")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--staleness", default=None, help="override greeks max_staleness")
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude-tte-days", type=float, default=0.0,
                    help="drop cells within this many days of expiry from the GEX sum")
    ap.add_argument("--block-mode", default=None,
                    help="override construction.block_trades.baseline (exclude|include|half_weight)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s")
    stats = run(args.start or config.get("sample.start"),
                args.end or config.get("sample.end"), args.staleness, args.tag,
                args.block_mode, args.exclude_tte_days)
    # Persist the diagnostics. They used to live only in the log, which meant
    # downstream reports hardcoded them and silently went stale the moment the
    # pipeline was re-run under a different specification.
    import json
    dest = config.path("paths.tables") / f"pipeline_stats{args.tag or ''}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats, indent=2, default=str))
    log.info("stats=%s -> %s", stats, dest)


if __name__ == "__main__":
    main()
