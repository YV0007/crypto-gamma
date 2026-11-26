"""Full backfill of BTC option trades, taker-flagged.

Endpoint choice (deviation from the RiveChen design -- documented)
------------------------------------------------------------------
RiveChen page per-instrument on `trade_seq`. That requires enumerating every
historical instrument (120,929 BTC options) and issuing at least one request
each. `get_last_trades_by_currency_and_time` returns EVERY option trade in a
time window with `instrument_name` embedded, so the same data costs ~5k
requests instead of ~120k. Measured 2026-08-31 on 2024-03-05: 43,201 rows in
5 requests / 38.8 s.

Their dedup rule still applies and is still needed: that same day returned
43,201 rows for 43,196 unique trade_ids -- 5 boundary duplicates. Removed at
parquet export on (instrument_name, trade_seq).

`direction` is the taker (aggressor) side and is the entire point:
    direction == "buy"   -> customer bought  -> dealer sold   -> position -1
    direction == "sell"  -> customer sold    -> dealer bought -> position +1

Block trades are retained in the RAW layer and filtered downstream according
to config `construction.block_trades` -- raw data is never pre-filtered.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src import config
from src.fetch.client import DeribitClient, setup_logging
from src.fetch.runner import paginate_window, run_daily_backfill

log = logging.getLogger("fetch.option_trades")


async def fetch_option_day(client: DeribitClient, start_ms: int, end_ms: int):
    cfg = config.get("fetch.option_trades")
    return await paginate_window(
        client,
        method=cfg["method"],
        start_ms=start_ms,
        end_ms=end_ms,
        page_size=cfg["page_size"],
        extra={"currency": config.get("sample.currency"), "kind": cfg["kind"]},
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill BTC option trades")
    ap.add_argument("--start", help="YYYY-MM-DD override (default: config sample.start)")
    ap.add_argument("--end", help="YYYY-MM-DD override, inclusive (default: config sample.end)")
    ap.add_argument("--stage", default="option_trades", help="checkpoint stage name")
    args = ap.parse_args()

    setup_logging(args.stage)
    if args.start or args.end:
        start = config.to_ms(args.start or config.get("sample.start"))
        end = config.to_ms(args.end or config.get("sample.end")) + 86_400_000
        log.info("window override: %s .. %s", args.start, args.end)
    else:
        start, end = config.sample_bounds_ms()

    out_dir = config.path("fetch.storage.option_dir")
    summary = await run_daily_backfill(
        stage=args.stage, page_fn=fetch_option_day, out_dir=out_dir,
        start_ms=start, end_ms=end,
    )
    log.info("summary: %s", summary)


if __name__ == "__main__":
    asyncio.run(main())
