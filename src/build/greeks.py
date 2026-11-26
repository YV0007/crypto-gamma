"""Attach Deribit-published greeks to the inventory panel.

*** BLOCKER, VERIFIED 2026-08-31 ***
CONTEXT.md §6 says "Greeks: do not recompute. Deribit publishes gamma, delta,
vega, theta per instrument. Pull them." That is true only for instruments that
are STILL ACTIVE. It is false for any backdated sample:

    ticker(BTC-29MAR24-72000-C)                  -> "instrument is not active"
    get_book_summary_by_instrument(same)         -> "instrument is not open"
    get_book_summary_by_instrument(LIVE option)  -> no greeks columns at all
    historical trades                            -> `iv` + `index_price`, no greeks

Greeks are reachable only through `ticker`, and only while the instrument is
live. There is no historical greeks endpoint on www or history. Deribit's own
`get_book_summary_*` does not carry them even for live instruments.

Consequences: a historical gamma series must come from one of

  deribit_ticker_snapshot  live instruments only -- forward-collectable from
                           today, worthless for March 2024
  tardis_options_chain     genuinely Deribit-published greeks, but the free
                           tier covers only the FIRST DAY of each month
  black_scholes_from_iv    closed-form gamma from Deribit's own published mark
                           `iv`, which IS carried on every historical trade

The third is deliberately NOT implemented: it recomputes what CONTEXT.md says
to pull, which is a methodological decision, not an implementation detail.

ALIGNMENT
---------
Greeks are point-in-time snapshots; positions are continuous. `nearest_prior`
(the default) takes the last snapshot at or before each grid point via a
backward merge_asof, bounded by `max_staleness` -- strictly backward-looking,
so no lookahead bias. `interpolate` is offered for robustness only and must
never produce a headline number: it reads greeks from the future.

MISSING GREEKS ARE NEVER ZERO-FILLED. A zeroed gamma silently shrinks GEX
toward zero and is indistinguishable from a genuinely flat position. Cells
without gamma are excluded and their coverage is reported.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.build.black_scholes import clean_iv, greeks as bs_greeks, time_to_expiry_years

log = logging.getLogger("build.greeks")

GREEK_COLS = ["gamma", "delta", "vega", "theta"]


class GreeksUnavailable(RuntimeError):
    """Raised when no historical greeks source can satisfy the requested window."""


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def load_greeks_snapshots(source: str | None = None,
                          window: tuple[str, str] | None = None) -> pd.DataFrame:
    """Return long greeks snapshots: instrument_name, timestamp, gamma, ...

    Every source returns the same schema so `align_greeks` is source-agnostic.
    """
    source = source or config.get("construction.greeks.source")

    if source == "deribit_ticker_snapshot":
        snap_dir = config.path("paths.oi_snapshots")
        files = sorted(snap_dir.glob("book_summary_*.parquet")) if snap_dir.exists() else []
        if not files:
            raise GreeksUnavailable(
                "no ticker snapshots under data/raw/oi_snapshots/. Run "
                "`python3 -m src.fetch.book_summary` to start collecting. NOTE: "
                "snapshots only ever cover instruments live at collection time, "
                "so they cannot supply greeks for a backdated sample."
            )
        frames = []
        for f in files:
            d = pd.read_parquet(f)
            keep = ["instrument_name", "snapshot_ts", *[c for c in GREEK_COLS if c in d.columns]]
            frames.append(d[keep].rename(columns={"snapshot_ts": "timestamp"}))
        out = pd.concat(frames, ignore_index=True)
        log.info("loaded %d ticker-snapshot rows from %d files", len(out), len(files))
        return out

    if source == "tardis_options_chain":
        src = config.path("paths.interim") / "tardis_options_chain"
        files = sorted(src.glob("*.parquet")) if src.exists() else []
        if not files:
            raise GreeksUnavailable(
                "no Tardis options_chain files under data/interim/tardis_options_chain/. "
                "Free tier covers only the FIRST DAY of each month: "
                "https://datasets.tardis.dev/v1/deribit/options_chain/YYYY/MM/01/OPTIONS.csv.gz "
                "(this environment receives HTTP 403 from their WAF -- the file must be "
                "supplied out of band). Columns needed: symbol, timestamp, gamma, delta, "
                "vega, theta, open_interest, underlying_price."
            )
        out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = out.rename(columns={"symbol": "instrument_name"})
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="us", utc=True)
        return out[["instrument_name", "timestamp", *GREEK_COLS]]

    if source == "black_scholes_from_iv":
        # The "snapshot" here is Deribit's own published mark IV, carried on
        # every historical trade. Greeks are NOT joined -- they are computed
        # per grid cell by `compute_bs_greeks` after this IV series has been
        # aligned, because gamma depends on the grid's spot and time-to-expiry,
        # not only on the instrument.
        if window is None:
            raise ValueError("black_scholes_from_iv needs a (start, end) window")
        start, end = window
        src = config.path("paths.interim") / "option_trades"
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        months = pd.period_range(lo, hi - pd.Timedelta(seconds=1), freq="M")
        filters = [[("year", "==", p.year), ("month", "==", p.month)] for p in months]
        df = pd.read_parquet(src, filters=filters,
                             columns=["instrument_name", "timestamp", "iv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]
        n_raw = len(df)
        df["iv_pct"] = clean_iv(df["iv"])
        n_bad = int(df["iv_pct"].isna().sum())
        df = df.dropna(subset=["iv_pct"])
        log.info("IV source: %d trade observations, %d dropped as out-of-bounds "
                 "(%.4f%%), %d usable across %d instruments",
                 n_raw, n_bad, 100 * n_bad / max(n_raw, 1), len(df),
                 df["instrument_name"].nunique())
        return df[["instrument_name", "timestamp", "iv_pct"]]

    raise ValueError(f"unknown greeks source {source!r}")


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def align_greeks(
    panel: pd.DataFrame,
    snaps: pd.DataFrame,
    method: str | None = None,
    max_staleness: str | None = None,
) -> pd.DataFrame:
    """Attach greeks to the (instrument_name, timestamp) inventory grid."""
    method = method or config.get("construction.greeks.alignment")
    tol = pd.Timedelta(max_staleness or config.get("construction.greeks.max_staleness"))

    left = panel.sort_values("timestamp", kind="mergesort")
    right = (snaps.dropna(subset=["timestamp"])
                  .sort_values("timestamp", kind="mergesort")
                  .drop_duplicates(subset=["instrument_name", "timestamp"], keep="last"))

    if method == "nearest_prior":
        out = pd.merge_asof(left, right, on="timestamp", by="instrument_name",
                            direction="backward", tolerance=tol)
    elif method == "interpolate":
        log.warning("alignment=interpolate USES FUTURE INFORMATION -- robustness only")
        back = pd.merge_asof(left, right, on="timestamp", by="instrument_name",
                             direction="backward", tolerance=tol)
        fwd = pd.merge_asof(left, right, on="timestamp", by="instrument_name",
                            direction="forward", tolerance=tol)
        out = back.copy()
        for c in [*GREEK_COLS, "iv_pct"]:
            if c in back.columns:
                out[c] = back[c].combine_first(fwd[c])
    else:
        raise ValueError(f"unknown alignment {method!r} (nearest_prior|interpolate)")

    payload = "gamma" if "gamma" in out.columns else "iv_pct"
    log.info("aligned %s by %s (tolerance %s): %d/%d cells populated",
             payload, method, tol,
             int(out[payload].notna().sum()) if payload in out else 0, len(out))
    return out


def compute_bs_greeks(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill gamma/delta/vega/theta per cell from aligned IV, spot and TTE.

    Used only when greeks.source == black_scholes_from_iv. `panel` must already
    carry `iv_pct` (aligned) and `spot`.
    """
    inst = pd.read_parquet(config.path("paths.interim") / "instruments.parquet")
    meta = inst.set_index("instrument_name")
    out = panel.copy()
    if "strike" not in out.columns:
        out["strike"] = out["instrument_name"].map(meta["strike"])
    if "option_type" not in out.columns:
        out["option_type"] = out["instrument_name"].map(meta["option_type"])
    if "expiry_ts" not in out.columns:
        out["expiry_ts"] = out["instrument_name"].map(meta["expiry_ts"])

    tte = time_to_expiry_years(out["timestamp"], out["expiry_ts"])
    g = bs_greeks(out["spot"].to_numpy(), out["strike"].to_numpy(),
                  tte.to_numpy(), (out["iv_pct"] / 100.0).to_numpy(),
                  out["option_type"].to_numpy())
    for c in GREEK_COLS:
        out[c] = g[c].to_numpy()
    out["tte_years"] = tte.to_numpy()
    log.info("computed BS greeks for %d/%d cells (%.2f%% have gamma)",
             int(out["gamma"].notna().sum()), len(out),
             100 * out["gamma"].notna().mean())
    return out


# --------------------------------------------------------------------------
# coverage diagnostics
# --------------------------------------------------------------------------
def coverage_report(panel: pd.DataFrame, spot: pd.Series | None = None) -> dict:
    """Missing-gamma coverage, broken down by moneyness and time to expiry.

    Missing greeks concentrated in deep OTM or near expiry would bias GEX
    systematically rather than just adding noise, so the breakdown matters more
    than the headline fraction.
    """
    df = panel.copy()
    has = df["gamma"].notna() if "gamma" in df.columns else pd.Series(False, index=df.index)
    overall = float(1 - has.mean()) if len(df) else 1.0

    inst = pd.read_parquet(config.path("paths.interim") / "instruments.parquet")
    strike = inst.set_index("instrument_name")["strike"]
    df["strike"] = df["instrument_name"].map(strike)
    if spot is not None:
        df["spot"] = df["timestamp"].map(spot)
    if "spot" not in df.columns:
        df["spot"] = np.nan
    df["moneyness"] = df["strike"] / df["spot"]
    df["tte_days"] = (df["expiry_ts"] - df["timestamp"]).dt.total_seconds() / 86400

    m_bins = [0, .8, .95, 1.05, 1.2, np.inf]
    m_lab = ["deep ITM-put/OTM-call <0.8", "0.8-0.95", "ATM 0.95-1.05", "1.05-1.2", "deep >1.2"]
    t_bins = [-np.inf, 0, 1, 7, 30, 90, np.inf]
    t_lab = ["expired", "<1d", "1-7d", "7-30d", "30-90d", ">90d"]

    df["m_bucket"] = pd.cut(df["moneyness"], m_bins, labels=m_lab)
    df["t_bucket"] = pd.cut(df["tte_days"], t_bins, labels=t_lab)
    df["_has"] = has.astype(float)

    by_m = df.groupby("m_bucket", observed=False)["_has"].agg(["size", "mean"])
    by_t = df.groupby("t_bucket", observed=False)["_has"].agg(["size", "mean"])
    by_m["missing_frac"] = 1 - by_m["mean"]
    by_t["missing_frac"] = 1 - by_t["mean"]

    return {
        "cells": len(df),
        "missing_fraction_overall": overall,
        "by_moneyness": by_m[["size", "missing_frac"]],
        "by_time_to_expiry": by_t[["size", "missing_frac"]],
    }


def apply_missing_policy(panel: pd.DataFrame, policy: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Handle cells without gamma. Never zero-fills."""
    policy = policy or config.get("construction.greeks.missing_policy")
    missing = panel["gamma"].isna() if "gamma" in panel.columns else pd.Series(True, index=panel.index)
    stats = {"policy": policy, "cells": len(panel), "missing": int(missing.sum()),
             "missing_fraction": float(missing.mean()) if len(panel) else 1.0}
    if policy == "fail":
        if missing.any():
            raise GreeksUnavailable(
                f"{stats['missing']:,}/{len(panel):,} cells "
                f"({100*stats['missing_fraction']:.2f}%) have no gamma")
        return panel, stats
    if policy == "exclude":
        return panel.loc[~missing].copy(), stats
    raise ValueError(f"unknown missing_policy {policy!r} (exclude|fail). "
                     "'zero' is intentionally not offered.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach greeks to the inventory panel")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s")

    from src.build.gex import attach_spot, load_spot   # local: avoids a cycle

    source = args.source or config.get("construction.greeks.source")
    inv = config.path("paths.processed") / f"inventory_{args.start}_{args.end}.parquet"
    panel = pd.read_parquet(inv).reset_index()

    snaps = load_greeks_snapshots(source, window=(args.start, args.end))
    aligned = align_greeks(panel, snaps)

    if source == "black_scholes_from_iv":
        # gamma depends on the grid's spot and time-to-expiry, so spot must be
        # attached before greeks can be computed.
        aligned = attach_spot(aligned, load_spot(args.start, args.end))
        aligned = compute_bs_greeks(aligned)

    rep = coverage_report(aligned, spot=None)
    log.info("missing gamma overall: %.4f%%", 100 * rep["missing_fraction_overall"])
    print("\n-- missing gamma by moneyness --");      print(rep["by_moneyness"])
    print("\n-- missing gamma by time to expiry --"); print(rep["by_time_to_expiry"])

    kept, stats = apply_missing_policy(aligned)
    dest = Path(str(inv).replace("inventory_", "inventory_greeks_"))
    kept.to_parquet(dest, index=False)
    log.info("wrote %s (%d rows) %s", dest, len(kept), stats)


if __name__ == "__main__":
    main()
