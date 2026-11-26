"""Raw JSONL -> partitioned Parquet in data/interim/.

Dedupes on (instrument_name, trade_seq) exactly as the RiveChen docs
prescribe: Deribit returns overlapping rows at page boundaries, and the
time-windowed cursor used here advances to max(timestamp) rather than max+1
(so no trade sharing a millisecond is skipped), which makes a handful of
duplicates per day expected rather than exceptional.

data/raw/ is read-only. This script never writes there.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.fetch.client import setup_logging  # noqa: E402

log = logging.getLogger("to_parquet")

KEY = ["instrument_name", "trade_seq"]


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def convert(raw_dir: Path, out_dir: Path, stage: str,
            start: str | None = None, end: str | None = None) -> dict:
    """Convert raw day-files to parquet. `start`/`end` are inclusive YYYY-MM-DD
    day keys matched against the filename, so a slice can be materialised
    without reading the whole raw tree."""
    files = sorted(list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz")))
    if start or end:
        lo, hi = start or "0000-00-00", end or "9999-99-99"
        files = [f for f in files if lo <= f.name[:10] <= hi]
    if not files:
        raise SystemExit(f"no raw files under {raw_dir}")
    log.info("%s: %d raw day-files", stage, len(files))

    # Convert month-by-month: the full option history is ~17M rows and does not
    # fit comfortably in a single frame alongside the parquet writer's buffers.
    by_month: dict[str, list[Path]] = {}
    for f in files:
        by_month.setdefault(f.name[:7], []).append(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_raw = tot = dupes = 0
    first = last = None
    instruments: set[str] = set()
    for month in sorted(by_month):
        rows = []
        for f in by_month[month]:
            rows.extend(read_jsonl(f))
        n_raw += len(rows)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        before = len(df)
        df = df.drop_duplicates(subset=KEY, keep="first")
        dupes += before - len(df)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["year"] = df["dt"].dt.year
        df["month"] = df["dt"].dt.month
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(out_dir, partition_cols=["year", "month"], index=False)
        tot += len(df)
        instruments |= set(df["instrument_name"].unique())
        first = df["dt"].min() if first is None else min(first, df["dt"].min())
        last = df["dt"].max() if last is None else max(last, df["dt"].max())
        log.info("  %s: %d rows", month, len(df))

    stats = {
        "stage": stage, "raw_rows": n_raw, "rows": tot, "duplicates_removed": dupes,
        "first": str(first), "last": str(last), "instruments": len(instruments),
    }
    log.info("%s: %s", stage, stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw JSONL to partitioned parquet")
    ap.add_argument("--stage", choices=["option_trades", "perp_trades", "all"], default="all")
    ap.add_argument("--start", help="inclusive YYYY-MM-DD")
    ap.add_argument("--end", help="inclusive YYYY-MM-DD")
    args = ap.parse_args()
    setup_logging("to_parquet")

    interim = config.path("paths.interim")
    targets = []
    if args.stage in ("option_trades", "all"):
        targets.append(("option_trades", config.path("fetch.storage.option_dir")))
    if args.stage in ("perp_trades", "all"):
        targets.append(("perp_trades", config.path("fetch.storage.perp_dir")))

    for stage, raw_dir in targets:
        if not raw_dir.exists():
            log.warning("%s: %s does not exist, skipping", stage, raw_dir)
            continue
        convert(raw_dir, interim / stage, stage, args.start, args.end)


if __name__ == "__main__":
    main()
