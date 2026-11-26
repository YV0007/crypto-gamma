"""Merge everything into one analysis-ready panel at config `construction.interval`.

NO LOOKAHEAD -- the design rule
-------------------------------
Every column at time t uses only information observable at or before t.
Concretely:

  * returns          log(close_t / close_{t-1})           -- closes at or before t
  * realized_vol     trailing window ENDING at t          -- never centred, never forward
  * dvol / funding   last published value AT OR BEFORE t  -- backward merge_asof
  * gex_flow         positions and IV at or before t      -- built that way upstream
  * tte_nearest      next expiry >= t, from the LISTING calendar

`tte_nearest` deserves a note: it references a future timestamp (the next
expiry) but is not lookahead, because the expiry calendar is published when an
instrument is listed and is therefore known at t. What would be lookahead is
using anything that happens AT that expiry.

Forward-looking targets (e.g. RV[t+1] for H2) are NOT built here. The analysis
modules construct them explicitly by shifting, so the shift is always visible
at the point of use rather than buried in the panel.

`tests/test_panel.py::TestNoLookahead` enforces this by rebuilding the panel on
data truncated at T and asserting every row at t <= T is bit-identical.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("build.panel")


def _interval() -> str:
    return config.get("construction.interval")


def load_gex(start: str, end: str) -> pd.DataFrame:
    p = config.path("paths.processed") / f"gex_flow_{start}_{end}.parquet"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run `python3 -m src.build.pipeline` first")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def build_returns(grid: pd.DatetimeIndex, truncate_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Interval returns and TRAILING realized volatility from 1-minute bars."""
    p = config.path("paths.interim") / "ohlcv_1m.parquet"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run `python3 -m src.fetch.market_data`")
    ohlcv = pd.read_parquet(p).set_index("dt").sort_index()
    if truncate_at is not None:
        ohlcv = ohlcv[ohlcv.index <= truncate_at]

    m1 = ohlcv["close"]
    r1 = np.log(m1).diff()

    # interval close = last 1m close AT OR BEFORE the grid point
    close = m1.reindex(m1.index.union(grid)).ffill().reindex(grid)
    out = pd.DataFrame(index=grid)
    out["close"] = close
    out["ret"] = np.log(out["close"]).diff()

    # trailing realized vol: sum of squared 1m returns over the window ENDING at t
    win = config.get("h2_volatility.rv_window")
    rv = (r1.pow(2).rolling(win, closed="right").sum().pow(0.5))
    out["realized_vol"] = rv.reindex(rv.index.union(grid)).ffill().reindex(grid)

    # 5-minute returns, for the autocorrelation diagnostics in H2
    r5 = np.log(m1.resample("5min").last()).diff()
    out["ret_5m"] = r5.reindex(r5.index.union(grid)).ffill().reindex(grid)
    out["abs_ret"] = out["ret"].abs()
    return out


def _asof(grid: pd.DatetimeIndex, s: pd.Series, name: str) -> pd.Series:
    """Last observation at or before each grid point."""
    left = pd.DataFrame({"timestamp": grid})
    right = s.rename(name).reset_index()
    right.columns = ["timestamp", name]
    right = right.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged.set_index("timestamp")[name]


def build_controls(grid: pd.DatetimeIndex, truncate_at: pd.Timestamp | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    interim = config.path("paths.interim")

    p = interim / "dvol.parquet"
    if p.exists():
        d = pd.read_parquet(p).set_index("dt")["close"].sort_index()
        if truncate_at is not None:
            d = d[d.index <= truncate_at]
        out["dvol"] = _asof(grid, d, "dvol")
    else:
        log.warning("dvol.parquet missing -- column omitted")

    p = interim / "funding.parquet"
    if p.exists():
        f = pd.read_parquet(p).set_index("dt")
        col = "interest_8h" if "interest_8h" in f.columns else f.columns[0]
        s = f[col].sort_index()
        if truncate_at is not None:
            s = s[s.index <= truncate_at]
        out["funding"] = _asof(grid, s, "funding")
    else:
        log.warning("funding.parquet missing -- column omitted")
    return out


def _next_expiry(exp: np.ndarray, grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First expiry at or after each grid point."""
    idx = np.clip(np.searchsorted(exp, grid.to_numpy(), side="left"), 0, len(exp) - 1)
    return pd.DatetimeIndex(exp[idx])


def build_calendar(grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Expiry-calendar and seasonality features.

    The expiry calendar is known at listing, so time-to-nearest-expiry is
    observable at t. Nothing here uses an outcome.

    DAILY vs MAJOR -- read this before using any column here
    --------------------------------------------------------
    Deribit lists a BTC option expiry on EVERY calendar day: 1,461 distinct
    expiry dates over the 1,461 days of the sample. So a "time to the next
    expiry" built off ALL expiries can never exceed 24 hours, and a
    "is it an expiry day" flag built the same way is true every day.

    That is what the previous version of this function did, and the two
    columns it produced were degenerate: `tte_nearest_hours` was 93% explained
    by hour-of-day dummies alone, and `is_expiry_day` reduced exactly to
    `hour_of_day < 8` (Deribit settles at 08:00 UTC). Neither carried any
    information about expiries. See DIAGNOSTIC_H2.md section 3.

    The gamma-pinning literature is about MAJOR expiries -- the monthly and
    quarterly dates that carry the open interest -- so those get their own
    columns and are what any expiry-conditioned analysis should use:

      tte_major_hours / days   hours & days to the next monthly-or-quarterly
                               expiry. This is the one with real variation.
      is_major_expiry_day      the calendar day of a major expiry.
      days_since_major         days since the last major expiry.
      tte_daily_hours          hours to the next expiry of ANY kind. Kept
                               under an honest name; it is a clock variable.
      is_daily_expiry_window   pre-settlement window of the daily expiry. This
                               is the old `is_expiry_day` under a name that
                               says what it actually is.

    A "major" expiry is the last Friday of a month (Deribit's monthly), which
    subsumes the quarterlies. Identified from the listing calendar, not from
    open interest, so it stays observable at t.
    """
    inst = pd.read_parquet(config.path("paths.interim") / "instruments.parquet")
    exp_all = np.sort(inst.loc[inst["kind"] == "option", "expiry_ts"].dropna().unique())

    ex = pd.DatetimeIndex(exp_all)
    # monthly expiry = the last Friday-settlement of each calendar month
    is_fri = ex.dayofweek == 4
    fri = ex[is_fri]
    if len(fri):
        per = fri.tz_convert(None).to_period("M")   # tz dropped for grouping only
        last_of_month = pd.Series(fri, index=per).groupby(level=0).max().to_numpy()
        exp_major = np.sort(last_of_month)
    else:                                   # defensive: never expected
        log.warning("no Friday expiries found -- falling back to all expiries")
        exp_major = exp_all

    nxt_all = _next_expiry(exp_all, grid)
    nxt_maj = _next_expiry(exp_major, grid)
    # `days_since_major` is undefined before the first major expiry in the
    # sample -- clipping to index 0 there would silently measure the distance
    # to a FUTURE expiry and report it as a negative age.
    prev_raw = np.searchsorted(exp_major, grid.to_numpy(), side="right") - 1
    prv_maj = pd.DatetimeIndex(exp_major[np.clip(prev_raw, 0, None)])

    out = pd.DataFrame(index=grid)
    out["tte_major_hours"] = (nxt_maj - grid).total_seconds() / 3600
    out["tte_major_days"] = out["tte_major_hours"] / 24
    out["is_major_expiry_day"] = (nxt_maj.normalize() == grid.normalize()).astype(int)
    out["days_since_major"] = np.where(
        prev_raw >= 0, (grid - prv_maj).total_seconds() / 86400, np.nan)

    out["tte_daily_hours"] = (nxt_all - grid).total_seconds() / 3600
    out["is_daily_expiry_window"] = (nxt_all.normalize() == grid.normalize()).astype(int)

    out["hour_of_day"] = grid.hour
    out["day_of_week"] = grid.dayofweek

    log.info("calendar: %d expiries of which %d major; tte_major_days spans "
             "%.2f .. %.2f (daily tte spans %.2f .. %.2f hours)",
             len(exp_all), len(exp_major),
             out["tte_major_days"].min(), out["tte_major_days"].max(),
             out["tte_daily_hours"].min(), out["tte_daily_hours"].max())
    return out


def build_panel(start: str, end: str, truncate_at: pd.Timestamp | None = None) -> pd.DataFrame:
    gex = load_gex(start, end)
    grid = pd.DatetimeIndex(gex.index)
    if truncate_at is not None:
        grid = grid[grid <= truncate_at]
        gex = gex.loc[grid]

    panel = pd.concat([
        gex,
        build_returns(grid, truncate_at),
        build_controls(grid, truncate_at),
        build_calendar(grid),
    ], axis=1)
    panel.index.name = "timestamp"
    log.info("panel: %d rows x %d cols, %s .. %s",
             len(panel), panel.shape[1], panel.index.min(), panel.index.max())
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the analysis panel")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")
    panel = build_panel(start, end)
    dest = config.path("paths.processed") / f"panel_{start}_{end}.parquet"
    panel.to_parquet(dest)
    log.info("wrote %s", dest)
    print(panel.describe().T.to_string())


if __name__ == "__main__":
    main()
