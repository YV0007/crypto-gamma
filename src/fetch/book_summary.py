"""Open interest, mark IV and greeks for all LIVE instruments.

*** SNAPSHOT ONLY. ***

Deribit publishes NO historical open interest. `get_book_summary_by_currency`
and `ticker` return the CURRENT state and nothing else; there is no endpoint,
on either host, that returns an OI time series. Verified 2026-08-31.

Historical OI therefore comes from Tardis.dev free monthly samples (first day
of each month, `ticker` channel, includes open_interest and greeks) --
CONTEXT.md section 7. This module does not and cannot backfill it.

NOT ON THE CRITICAL PATH. H2, H3, H4 and H6 need no OI at all; only the naive
GEX benchmark, H1 reconciliation and H5 pinning do. Running this daily from
today forward accumulates forward-looking reconciliation points, which is
worth doing because every uncollected day is permanently lost -- but it does
not gate any part of the core pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.fetch.client import DeribitClient, setup_logging

log = logging.getLogger("fetch.book_summary")


async def snapshot(currency: str = "BTC", light: bool = False) -> pd.DataFrame:
    """Current OI / mark IV / greeks for every live option, plus the perp.

    light=True stops after the book summary: ONE request, ~1 s, and it already
    carries open_interest, mark_iv and underlying_price for every instrument.
    The greeks cost 979 requests and ~68 s on top of that, and we compute our
    own greeks from mark_iv anyway (STATUS.md 4.3, corr 0.9976) -- Deribit's are
    a reconciliation check, not an input. That asymmetry is the whole reason OI
    can be sampled hourly while greeks stay daily.
    """
    async with DeribitClient(host="www") as client:
        book = await client.get(
            "get_book_summary_by_currency", currency=currency, kind="option"
        )
        df = pd.DataFrame(book)
        # Stamp the moment the OI was OBSERVED, not the moment the job happens
        # to finish. If the Mac sleeps during the 979-request greeks loop the
        # process is suspended and can resume hours later: on 2026-09-01 a run
        # begun at 20:34 finished at 23:34 and was written with a 23:34 stamp
        # while every OI value in it was from 20:34. Deribit's own
        # creation_timestamp caught it. OI stamped 3 h late is worse than no OI.
        book_ts = datetime.now(tz=timezone.utc)
        df["snapshot_ts"] = book_ts
        log.info("book summary: %d live options", len(df))

        if light:
            log.info("light mode: skipping greeks (requests=%d)", client.n_requests)
            return df

        # get_book_summary_by_currency carries OI and mark_iv but NOT greeks;
        # greeks require a per-instrument ticker call.
        names = df["instrument_name"].tolist()
        sem = asyncio.Semaphore(config.get("fetch.workers", 8))

        async def one(name: str) -> dict:
            async with sem:
                t = await client.get("ticker", instrument_name=name)
            g = t.get("greeks", {}) or {}
            return {
                "instrument_name": name,
                "open_interest": t.get("open_interest"),
                "mark_iv": t.get("mark_iv"),
                "underlying_price": t.get("underlying_price"),
                "index_price": t.get("index_price"),
                "delta": g.get("delta"), "gamma": g.get("gamma"),
                "vega": g.get("vega"), "theta": g.get("theta"), "rho": g.get("rho"),
            }

        greeks = pd.DataFrame(await asyncio.gather(*(one(n) for n in names)))
        log.info("ticker greeks: %d instruments, requests=%d", len(greeks), client.n_requests)

    merged = df.merge(greeks, on="instrument_name", how="left", suffixes=("", "_ticker"))
    # snapshot_ts came across from df and refers to the OI. The greeks were read
    # later, so they get their own honest timestamp rather than borrowing it.
    greeks_ts = datetime.now(tz=timezone.utc)
    merged["greeks_ts"] = greeks_ts
    lag = (greeks_ts - book_ts).total_seconds()
    if lag > 600:
        log.warning(
            "greeks lag OI by %.0f s -- the machine almost certainly slept "
            "mid-run; OI is stamped %s, greeks %s. Treat the greeks in this "
            "file as a separate observation.", lag, book_ts, greeks_ts)
    return merged


async def main() -> None:
    setup_logging("book_summary")
    light = "--light" in sys.argv
    df = await snapshot(config.get("sample.currency"), light=light)
    # Light files lack the greek columns, so they go to their own directory.
    # Mixing them into paths.oi_snapshots would hand src/build/greeks.py a
    # directory where only some files have a `gamma` column.
    out_dir = config.path("paths.oi_light" if light else "paths.oi_snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename follows the OI observation time, so the name never lies about
    # when the data is from, even if the run itself was suspended and resumed.
    stamp = df["snapshot_ts"].iloc[0].strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / f"book_summary_{stamp}.parquet"
    df.to_parquet(dest, index=False)
    log.info("wrote %s [%s] (%d rows, total OI %.1f)",
             dest, "light" if light else "full", len(df),
             df["open_interest"].fillna(0).sum())


if __name__ == "__main__":
    asyncio.run(main())
