"""Signed BTC-PERPETUAL taker flow, aggregated to fixed buckets.

This is the DEPENDENT VARIABLE for H3: the hedging trade itself.

    direction == "buy"   ->  taker lifted the offer  ->  +amount (USD bought)
    direction == "sell"  ->  taker hit the bid       ->  -amount (USD sold)

    net_usd = taker_buy_usd - taker_sell_usd

Note the sign convention is the OPPOSITE of the options side and deliberately
so. On the options book we invert the taker flag because the DEALER is the
passive counterparty. Here we want the flow of whoever is crossing the spread
on the perpetual, because a delta-hedging dealer is the aggressor when it
hedges -- it must trade now, at the market, not sit and quote.

Streams the raw daily files rather than materialising a ~20 GB interim table;
139M perpetual trades only ever need to become ~420k five-minute buckets.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("build.perp_flow")


def _one_day(args) -> pd.DataFrame | None:
    path, freq = args
    ts, sd, am = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line:
                continue
            r = json.loads(line)
            ts.append(r["timestamp"])
            sd.append(1.0 if r["direction"] == "buy" else -1.0)
            am.append(r["amount"])
    if not ts:
        return None
    df = pd.DataFrame({
        "dt": pd.to_datetime(np.asarray(ts), unit="ms", utc=True),
        "signed": np.asarray(sd) * np.asarray(am),
        "abs_usd": np.asarray(am),
        "is_buy": np.asarray(sd) > 0,
    })
    g = df.groupby(pd.Grouper(key="dt", freq=freq))
    out = pd.DataFrame({
        "net_usd": g["signed"].sum(),
        "volume_usd": g["abs_usd"].sum(),
        "buy_usd": g.apply(lambda x: x.loc[x["is_buy"], "abs_usd"].sum(), include_groups=False),
        "n_trades": g.size(),
    })
    return out[out["n_trades"] > 0]


def build(freq: str, workers: int = 8) -> pd.DataFrame:
    raw = config.path("fetch.storage.perp_dir")
    files = sorted(raw.glob("*.jsonl.gz")) + sorted(raw.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no perp raw files under {raw}")
    log.info("streaming %d day-files at %s resolution", len(files), freq)

    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_one_day, [(f, freq) for f in files], chunksize=4)):
            if out is not None and len(out):
                parts.append(out)
            if i % 200 == 0:
                log.info("  %d/%d days", i, len(files))

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["sell_usd"] = df["volume_usd"] - df["buy_usd"]
    # order-flow imbalance, scale-free -- lets 2022 and 2025 be compared despite
    # perpetual volume growing several-fold over the sample
    df["ofi"] = df["net_usd"] / df["volume_usd"].replace(0, np.nan)
    log.info("built %d buckets %s .. %s", len(df), df.index.min(), df.index.max())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate signed perpetual taker flow")
    ap.add_argument("--freq", default=None, help="default: h3_perpflow.return_interval")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s")
    freq = args.freq or config.get("h3_perpflow.return_interval")
    df = build(freq, args.workers)
    dest = config.path("paths.interim") / f"perp_flow_{freq}.parquet"
    df.to_parquet(dest)
    log.info("wrote %s", dest)
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
