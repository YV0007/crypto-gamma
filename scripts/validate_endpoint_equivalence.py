"""Prove the currency-level endpoint loses nothing vs per-instrument paging.

This project fetches option trades with get_last_trades_by_currency_and_time
instead of the per-instrument trade_seq paging used by the RiveChen fetcher.
That is a deliberate deviation (~5k requests instead of ~120k), so it has to
be shown to be lossless rather than assumed: for a sample of instruments we
page the FULL per-instrument history on trade_seq, restrict it to the fetched
month, and require an exact trade_id set match.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.fetch.client import DeribitClient  # noqa: E402

MONTH = "2024-03"
N_INSTRUMENTS = 12
PAGE = 10000


async def full_history(client: DeribitClient, name: str) -> list[dict]:
    """Page an instrument's entire history on trade_seq, per the RiveChen docs."""
    out, seq = [], 1
    for _ in range(500):
        res = await client.get("get_last_trades_by_instrument",
                               instrument_name=name, start_seq=seq,
                               end_seq=seq + PAGE - 1, count=PAGE)
        page = res.get("trades", [])
        if not page:
            break
        out.extend(page)
        # docs: response order is not guaranteed -- compute the max explicitly
        seq = max(t["trade_seq"] for t in page) + 1
        if len(page) < PAGE:
            break
    return out


async def main() -> int:
    raw_dir = config.path("fetch.storage.option_dir")
    rows = []
    for f in sorted(raw_dir.glob(f"{MONTH}*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            rows.extend(json.loads(ln) for ln in fh)
    df = pd.DataFrame(rows).drop_duplicates(subset=["instrument_name", "trade_seq"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    lo, hi = pd.Timestamp(f"{MONTH}-01", tz="UTC"), pd.Timestamp(f"{MONTH}-01", tz="UTC") + pd.offsets.MonthBegin(1)

    counts = df["instrument_name"].value_counts()
    random.seed(config.get("seed", 42))
    # mix of heavily and lightly traded instruments
    pool = counts.index.tolist()
    picks = pool[:4] + random.sample(pool[4:], min(N_INSTRUMENTS - 4, len(pool) - 4))

    print(f"Cross-checking {len(picks)} instruments for {MONTH}\n")
    print(f"{'instrument':<28} {'currency_ep':>11} {'per_instr':>10} {'missing':>8} {'extra':>6}")
    print("-" * 68)

    bad = 0
    async with DeribitClient(host="history") as client:
        for name in picks:
            hist = await full_history(client, name)
            h = pd.DataFrame(hist)
            h["dt"] = pd.to_datetime(h["timestamp"], unit="ms", utc=True)
            h = h[(h["dt"] >= lo) & (h["dt"] < hi)].drop_duplicates(subset=["trade_seq"])
            ours = set(df.loc[df["instrument_name"] == name, "trade_id"])
            theirs = set(h["trade_id"])
            missing, extra = theirs - ours, ours - theirs
            flag = "" if not (missing or extra) else "  <-- MISMATCH"
            if missing or extra:
                bad += 1
            print(f"{name:<28} {len(ours):>11,} {len(theirs):>10,} "
                  f"{len(missing):>8} {len(extra):>6}{flag}")

    print("-" * 68)
    if bad:
        print(f"\nFAIL: {bad}/{len(picks)} instruments disagree -- the currency-level "
              "endpoint is NOT lossless. Do not launch.")
        return 1
    print(f"\nPASS: all {len(picks)} instruments match exactly. "
          "Currency-level fetching is lossless.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
