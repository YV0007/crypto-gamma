"""Enumerate the BTC instrument universe.

RECON FINDING (2026-08-31, scripts/probe_api.py)
------------------------------------------------
`get_instruments(expired=true)` behaves differently per host:

    www.deribit.com      ->     56 instruments, earliest expiry 2026-08-30
    history.deribit.com  -> 120,929 instruments, earliest expiry 2016-07-15

Only the history host can enumerate the historical universe. Using www here
would silently truncate the sample to the current expiry batch.

contract_size is carried per instrument. Every BTC option observed reports
1.0, but BTC-PERPETUAL reports 10.0, so downstream GEX code must join this
column rather than trust a config constant.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.fetch.client import DeribitClient, setup_logging

log = logging.getLogger("fetch.instruments")

# BTC-27DEC24-100000-C  ->  underlying, expiry, strike, option type
_NAME_RE = re.compile(r"^(?P<under>[A-Z0-9]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def parse_instrument_name(name: str) -> dict | None:
    """Parse a Deribit option name into its components.

    Returned expiry is the NAME-implied date at 08:00 UTC; the authoritative
    value is `expiration_timestamp` from the API, which we keep alongside it
    and use for the cross-check in `_validate`.
    """
    m = _NAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    day = int(g["exp"][:-5])
    mon = _MONTHS[g["exp"][-5:-2]]
    year = 2000 + int(g["exp"][-2:])
    return {
        "underlying": g["under"],
        "expiry_name": datetime(year, mon, day, 8, 0, tzinfo=timezone.utc),
        "strike": float(g["strike"]),
        "option_type": "call" if g["cp"] == "C" else "put",
    }


async def fetch_instruments(currency: str = "BTC") -> pd.DataFrame:
    """Fetch live + expired instruments from the history host, merged and deduped."""
    host = config.get("fetch.instrument_host", "history")
    frames: list[dict] = []
    async with DeribitClient(host=host) as client:
        for kind in ("option", "future"):
            for expired in ("false", "true"):
                res = await client.get(
                    "get_instruments", currency=currency, kind=kind, expired=expired
                )
                log.info("kind=%s expired=%s -> %d instruments", kind, expired, len(res))
                frames.extend(res)
        log.info("requests=%d retries=%d", client.n_requests, client.n_retries)

    df = pd.DataFrame(frames)
    before = len(df)
    df = df.drop_duplicates(subset=["instrument_name"], keep="first").reset_index(drop=True)
    log.info("merged %d rows -> %d unique instruments", before, len(df))
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    # get_instruments already returns `strike` and `option_type`. We parse the
    # name independently and keep BOTH, suffixed -- disagreement means our
    # parser is wrong, and _validate turns that into a hard failure rather
    # than letting a mis-parsed strike propagate into GEX.
    parsed = [parse_instrument_name(n) or {} for n in df["instrument_name"]]
    pdf = pd.DataFrame(parsed, index=df.index).add_suffix("_parsed")
    out = pd.concat([df, pdf], axis=1)
    # Perpetuals carry a year-3000 sentinel expiry (32503708800000 ms) which
    # overflows pandas' nanosecond datetime range. Flag them, don't coerce
    # silently -- BTC-PERPETUAL must survive the in-sample filter below.
    NS_MAX_MS = 9_214_646_400_000  # ~2262-01-01, the ns bound
    out["is_perpetual"] = out["expiration_timestamp"] >= NS_MAX_MS
    out["expiry_ts"] = pd.to_datetime(
        out["expiration_timestamp"].where(~out["is_perpetual"]), unit="ms", utc=True
    )
    out["creation_ts"] = pd.to_datetime(out["creation_timestamp"], unit="ms", utc=True)
    if "contract_size" not in out.columns:
        raise RuntimeError("contract_size absent from get_instruments response")
    return out


def _validate(df: pd.DataFrame) -> None:
    """Fail loudly on the things that would silently corrupt GEX."""
    opts = df[df["kind"] == "option"]
    n_unparsed = int(opts["strike_parsed"].isna().sum())
    if n_unparsed:
        bad = opts.loc[opts["strike_parsed"].isna(), "instrument_name"].head(5).tolist()
        raise RuntimeError(f"{n_unparsed} option names failed to parse, e.g. {bad}")
    if bool(opts["contract_size"].isna().any()):
        raise RuntimeError("null contract_size on some options")

    # Parser vs API cross-checks. A wrong strike or option_type would corrupt
    # every downstream gamma weight, so these are errors, not warnings.
    bad_strike = int((opts["strike_parsed"] != opts["strike"]).sum())
    if bad_strike:
        raise RuntimeError(f"parsed strike disagrees with API on {bad_strike} instruments")
    bad_type = int((opts["option_type_parsed"] != opts["option_type"]).sum())
    if bad_type:
        raise RuntimeError(f"parsed option_type disagrees with API on {bad_type} instruments")

    # expiration_timestamp is authoritative; the name only implies a date.
    disagree = int((opts["expiry_name_parsed"] != opts["expiry_ts"]).sum())
    if disagree:
        log.warning("%d options where name-implied expiry != expiration_timestamp "
                    "(expiration_timestamp is authoritative)", disagree)

    sizes = sorted(opts["contract_size"].unique().tolist())
    log.info("option contract_size values: %s", sizes)
    if len(sizes) > 1:
        log.warning("contract_size VARIES across options %s -- the config constant "
                    "MUST NOT be used; join per-instrument from instruments.parquet", sizes)
    fut = df[df["kind"] != "option"]
    if len(fut):
        log.info("futures contract_size values: %s (n=%d instruments)",
                 sorted(fut["contract_size"].unique().tolist()), len(fut))
        perp = fut[fut["instrument_name"] == "BTC-PERPETUAL"]
        if len(perp):
            log.info("BTC-PERPETUAL contract_size=%s (USD per contract, NOT 1.0)",
                     perp["contract_size"].iloc[0])


async def main() -> None:
    setup_logging("instruments")
    df = await fetch_instruments()
    df = _enrich(df)
    _validate(df)

    start, end = config.sample_bounds_ms()
    start_ts = pd.Timestamp(start, unit="ms", tz="UTC")
    end_ts = pd.Timestamp(end, unit="ms", tz="UTC")
    # Keep anything that could have been tradeable during the sample: it must
    # have expired at/after the sample start and been created before its end.
    tradeable = (df["expiry_ts"] >= start_ts) | df["is_perpetual"]
    in_sample = df[tradeable & (df["creation_ts"] < end_ts)].copy()

    dest = config.path("paths.interim") / "instruments.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    in_sample.to_parquet(dest, index=False)

    full = config.path("paths.interim") / "instruments_all.parquet"
    df.to_parquet(full, index=False)

    log.info("universe=%d  in-sample=%d  -> %s", len(df), len(in_sample), dest)
    opts = in_sample[in_sample["kind"] == "option"]
    log.info("in-sample options=%d  expiry %s .. %s",
             len(opts), opts["expiry_ts"].min(), opts["expiry_ts"].max())


if __name__ == "__main__":
    asyncio.run(main())
