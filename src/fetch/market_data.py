"""Spot/perp OHLCV and the DVOL volatility index.

RECON FINDING (2026-08-31): `get_volatility_index_data` SILENTLY returns zero
data points for resolution "1" and "60" -- no error, just an empty array. Only
3600 / 43200 / 1D return data. DVOL history begins in 2022 (2021 requests come
back empty), which the sample period already respects. The resolution lives in
config so this trap cannot be reintroduced by editing code.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import pandas as pd

from src import config
from src.fetch.client import DeribitClient, setup_logging

log = logging.getLogger("fetch.market_data")
DAY_MS = 86_400_000


async def fetch_ohlcv(client: DeribitClient, start_ms: int, end_ms: int) -> pd.DataFrame:
    cfg = config.get("fetch.market_data")
    inst = cfg["ohlcv_instrument"]
    res = cfg["ohlcv_resolution"]
    chunk = cfg["ohlcv_chunk_days"] * DAY_MS
    frames = []
    n_bars = [0, 0]
    cur = start_ms
    while cur < end_ms:
        hi = min(cur + chunk, end_ms)
        out = await client.get(
            "get_tradingview_chart_data",
            instrument_name=inst, start_timestamp=cur, end_timestamp=hi, resolution=res,
        )
        if out.get("status") == "ok" and out.get("ticks"):
            n = len(out["ticks"])
            # The API caps responses at 5,001 bars and truncates SILENTLY,
            # still reporting status "ok". Detect it rather than trust it.
            expected = (hi - cur) // 60_000
            if n >= 5000 and n < expected * 0.95:
                raise RuntimeError(
                    f"ohlcv silently truncated: {n} bars for a window expecting "
                    f"~{expected}. Lower fetch.market_data.ohlcv_chunk_days.")
            frames.append(pd.DataFrame({
                "timestamp": out["ticks"], "open": out["open"], "high": out["high"],
                "low": out["low"], "close": out["close"], "volume": out["volume"],
            }))
            n_bars[0] += n
            if n_bars[0] // 100_000 > n_bars[1]:
                n_bars[1] = n_bars[0] // 100_000
                log.info("ohlcv progress: %d bars, at %s", n_bars[0],
                         pd.to_datetime(cur, unit="ms", utc=True).date())
        else:
            log.warning("ohlcv empty for %s..%s status=%s", cur, hi, out.get("status"))
        cur = hi
    if not frames:
        raise RuntimeError("no OHLCV returned for the whole window")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["instrument_name"] = inst
    return df.reset_index(drop=True)


async def fetch_dvol(client: DeribitClient, start_ms: int, end_ms: int) -> pd.DataFrame:
    cfg = config.get("fetch.market_data")
    res = cfg["dvol_resolution"]
    if res in ("1", "60"):
        raise ValueError(
            f"dvol_resolution={res!r} returns ZERO points from Deribit without erroring. "
            "Use 3600, 43200 or 1D."
        )
    chunk = cfg["dvol_chunk_days"] * DAY_MS
    rows = []
    cur = start_ms
    while cur < end_ms:
        hi = min(cur + chunk, end_ms)
        out = await client.get(
            "get_volatility_index_data",
            currency=config.get("sample.currency"),
            start_timestamp=cur, end_timestamp=hi, resolution=res,
        )
        data = out.get("data", [])
        if not data:
            log.warning("dvol empty for %s..%s", cur, hi)
        rows.extend(data)
        cur = hi
    if not rows:
        raise RuntimeError("no DVOL returned for the whole window")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)


async def fetch_funding(client: DeribitClient, start_ms: int, end_ms: int) -> pd.DataFrame:
    """BTC-PERPETUAL funding rate history -- an H2 control.

    get_funding_rate_history caps the window, so it is chunked. Funding is an
    8h-settled rate reported continuously; we keep the raw series and let the
    panel resample it.
    """
    rows = []
    cur = start_ms
    chunk = 30 * DAY_MS
    while cur < end_ms:
        hi = min(cur + chunk, end_ms)
        out = await client.get(
            "get_funding_rate_history",
            instrument_name=config.get("fetch.perp_trades.instrument"),
            start_timestamp=cur, end_timestamp=hi,
        )
        if out:
            rows.extend(out)
        else:
            log.warning("funding empty for %s..%s", cur, hi)
        cur = hi
    if not rows:
        raise RuntimeError("no funding data returned for the whole window")
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch OHLCV and DVOL")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    setup_logging("market_data")

    if args.start or args.end:
        start = config.to_ms(args.start or config.get("sample.start"))
        end = config.to_ms(args.end or config.get("sample.end")) + DAY_MS
    else:
        start, end = config.sample_bounds_ms()

    interim = config.path("paths.interim")
    interim.mkdir(parents=True, exist_ok=True)

    async with DeribitClient(host="www") as client:
        ohlcv = await fetch_ohlcv(client, start, end)
        ohlcv.to_parquet(interim / "ohlcv_1m.parquet", index=False)
        log.info("ohlcv_1m.parquet: %d bars %s .. %s",
                 len(ohlcv), ohlcv["dt"].min(), ohlcv["dt"].max())

        dvol = await fetch_dvol(client, start, end)
        dvol.to_parquet(interim / "dvol.parquet", index=False)
        log.info("dvol.parquet: %d points %s .. %s",
                 len(dvol), dvol["dt"].min(), dvol["dt"].max())
        funding = await fetch_funding(client, start, end)
        funding.to_parquet(interim / "funding.parquet", index=False)
        log.info("funding.parquet: %d points %s .. %s",
                 len(funding), funding["dt"].min(), funding["dt"].max())

        log.info("requests=%d retries=%d", client.n_requests, client.n_retries)


if __name__ == "__main__":
    asyncio.run(main())
