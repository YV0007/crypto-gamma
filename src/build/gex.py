"""Signed gamma exposure from reconstructed dealer inventory.

    GEX_flow[t] = SUM_i  dealer_position[i,t] * gamma[i,t] * S[t]^2 * 0.01 * contract_size[i]

Units: notional USD per 1% move in spot.

    dealer_position  contracts (signed; negative = dealer short gamma)
    gamma            dTdelta/dS per contract, as published by Deribit
    S^2 * 0.01       converts a per-unit-move gamma into exposure per 1% move
    contract_size    PER INSTRUMENT from instruments.parquet -- options 1.0,
                     BTC-PERPETUAL 10.0. Never one global constant: a wrong
                     contract size rescales GEX by a silent constant factor
                     that nothing downstream would catch.

Sign reading: positive GEX means dealers are long gamma and hedge AGAINST the
move (selling into rallies, buying dips) -- volatility dampening. Negative GEX
means dealers are short gamma and hedge WITH the move -- amplifying.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import config

log = logging.getLogger("build.gex")


def load_spot(start: str, end: str, source: str | None = None) -> pd.Series:
    """Spot series for the S^2 term, indexed by timestamp (UTC)."""
    source = source or config.get("construction.spot_source")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    if source == "ohlcv":
        p = config.path("paths.interim") / "ohlcv_1m.parquet"
        if not p.exists():
            raise SystemExit(f"{p} missing -- run `python3 -m src.fetch.market_data`")
        df = pd.read_parquet(p)
        s = df.set_index("dt")["close"]
    elif source == "index_price":
        # index_price rides on every trade, so this needs no extra fetch and is
        # available even while the market_data stage is still running.
        src = config.path("paths.interim") / "option_trades"
        months = pd.period_range(lo, hi - pd.Timedelta(seconds=1), freq="M")
        filters = [[("year", "==", p.year), ("month", "==", p.month)] for p in months]
        df = pd.read_parquet(src, filters=filters, columns=["timestamp", "index_price"])
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        s = df.set_index("dt")["index_price"].sort_index()
    else:
        raise ValueError(f"unknown spot_source {source!r} (ohlcv|index_price)")

    s = s[(s.index >= lo) & (s.index < hi)].sort_index()
    log.info("spot from %s: %d points %s .. %s", source, len(s), s.index.min(), s.index.max())
    return s


def attach_spot(panel: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """Nearest-prior spot for each grid point (backward-looking only).

    Idempotent: the greeks step already attaches spot when the BS source is in
    use, and re-merging would silently produce spot_x/spot_y.
    """
    if "spot" in panel.columns:
        log.info("spot already attached (%d/%d non-null) -- reusing",
                 int(panel["spot"].notna().sum()), len(panel))
        return panel
    grid = pd.DataFrame({"timestamp": sorted(panel["timestamp"].unique())})
    sp = spot.reset_index()
    sp.columns = ["timestamp", "spot"]
    aligned = pd.merge_asof(grid, sp.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    out = panel.merge(aligned, on="timestamp", how="left")
    n_missing = int(out["spot"].isna().sum())
    if n_missing:
        log.warning("%d grid cells have no prior spot observation", n_missing)
    return out


def compute_gex_flow(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cell contribution and the aggregated GEX_flow series.

    `panel` must carry dealer_position, gamma, spot and contract_size.
    """
    for col in ("dealer_position", "gamma", "spot", "contract_size"):
        if col not in panel.columns:
            raise KeyError(f"panel is missing required column {col!r}")

    out = panel.copy()
    out["gex_contribution"] = (
        out["dealer_position"] * out["gamma"] * out["spot"] ** 2 * 0.01 * out["contract_size"]
    )
    series = (out.groupby("timestamp")["gex_contribution"].sum()
                 .rename("gex_flow").to_frame())
    series["n_instruments"] = out.groupby("timestamp")["instrument_name"].nunique()
    series["spot"] = out.groupby("timestamp")["spot"].first()
    log.info("GEX_flow over %d intervals: median %.3e USD per 1%% move",
             len(series), series["gex_flow"].median())
    return out, series


def compute_gex_naive(*_args, **_kwargs):
    """OI-based benchmark GEX. NOT IMPLEMENTED -- blocked on historical OI.

        GEX_naive[t] = SUM_i sign_i * OI[i,t] * gamma[i,t] * S[t]^2 * 0.01 * contract_size[i]
        sign_i = -1 for calls, +1 for puts

    Blocked because Deribit publishes no historical open interest at all
    (CONTEXT.md §7, re-verified 2026-08-31: `get_book_summary_by_instrument`
    errors with "instrument is not open" on any expired instrument). OI must
    come from Tardis.dev monthly samples or a forward-collected snapshot
    series.

    This is contribution #4 (the horse race), not the core result. H2, H3, H4
    and H6 need no OI, so this is deliberately deferred rather than blocking
    the pipeline.
    """
    raise NotImplementedError(
        "GEX_naive is blocked on historical open interest -- see CONTEXT.md §7. "
        "Not on the critical path: H2/H3/H4/H6 need only GEX_flow."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute flow-based GEX")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s")

    proc = config.path("paths.processed")
    src = proc / f"inventory_greeks_{args.start}_{args.end}.parquet"
    if not src.exists():
        raise SystemExit(f"{src} missing -- run `python3 -m src.build.greeks` first")

    panel = pd.read_parquet(src)
    panel = attach_spot(panel, load_spot(args.start, args.end))
    cells, series = compute_gex_flow(panel)

    dest = proc / f"gex_flow_{args.start}_{args.end}.parquet"
    series.to_parquet(dest)
    cells.to_parquet(proc / f"gex_cells_{args.start}_{args.end}.parquet", index=False)
    log.info("wrote %s (%d intervals)", dest, len(series))


if __name__ == "__main__":
    main()
