"""Full backfill of BTC-PERPETUAL trades -- the H3 hedging-flow venue.

ONLY BTC-PERPETUAL. Fetching all BTC futures pulls ~90 GB of expiring-contract
history that this study never uses; the instrument name comes from config
(`fetch.perp_trades.instrument`) and is never widened to a currency-level
futures sweep.

`direction` is again the taker side. For H3 this is the dependent variable:
signed perpetual taker volume in the minutes after a large spot move.

NOTE ON UNITS: BTC-PERPETUAL has contract_size 10.0 (USD per contract), unlike
options at 1.0. The `amount` field is USD notional. Do not assume BTC units.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src import config
from src.fetch.client import DeribitClient, setup_logging
from src.fetch.runner import paginate_window, run_daily_backfill

log = logging.getLogger("fetch.perp_trades")


async def fetch_perp_day(client: DeribitClient, start_ms: int, end_ms: int):
    cfg = config.get("fetch.perp_trades")
    return await paginate_window(
        client,
        method=cfg["method"],
        start_ms=start_ms,
        end_ms=end_ms,
        page_size=cfg["page_size"],
        extra={"instrument_name": cfg["instrument"]},
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill BTC-PERPETUAL trades")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--stage", default="perp_trades")
    args = ap.parse_args()

    setup_logging(args.stage)
    inst = config.get("fetch.perp_trades.instrument")
    if inst != "BTC-PERPETUAL":
        raise SystemExit(f"refusing to run: fetch.perp_trades.instrument={inst!r}, expected BTC-PERPETUAL")

    if args.start or args.end:
        start = config.to_ms(args.start or config.get("sample.start"))
        end = config.to_ms(args.end or config.get("sample.end")) + 86_400_000
    else:
        start, end = config.sample_bounds_ms()

    out_dir = config.path("fetch.storage.perp_dir")
    summary = await run_daily_backfill(
        stage=args.stage, page_fn=fetch_perp_day, out_dir=out_dir,
        start_ms=start, end_ms=end,
    )
    log.info("summary: %s", summary)


if __name__ == "__main__":
    asyncio.run(main())
