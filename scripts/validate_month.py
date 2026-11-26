"""Stage-1 acceptance checks on a single fetched month.

Every check is a hard pass/fail. If any fail, the full backfill must not be
launched -- a systematic defect costs hours of refetching and, worse, may not
be visible in aggregate statistics later.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def read_all(raw_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(f for f in raw_dir.iterdir()
                   if f.name.startswith(prefix) and ".jsonl" in f.name)
    rows = []
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fh:
            rows.extend(json.loads(ln) for ln in fh if ln.strip())
    return pd.DataFrame(rows), files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2024-03")
    args = ap.parse_args()

    raw_dir = config.path("fetch.storage.option_dir")
    df, files = read_all(raw_dir, args.month)
    print(f"\nValidating {args.month}: {len(files)} day-files, {len(df):,} raw rows\n")

    days_expected = pd.Period(args.month).days_in_month
    check("day files present", len(files) == days_expected,
          f"{len(files)}/{days_expected} days")

    check("trade count non-zero and plausible", 1_000 < len(df) < 50_000_000,
          f"{len(df):,} rows, mean {len(df)/max(len(files),1):,.0f}/day")

    # --- direction (the taker flag -- the entire hypothesis) ---
    have_dir = "direction" in df.columns
    n_missing = int(df["direction"].isna().sum()) if have_dir else len(df)
    dist = Counter(df["direction"]) if have_dir else {}
    check("`direction` populated on every row", have_dir and n_missing == 0,
          f"{n_missing} missing; distribution {dict(dist)}")
    check("`direction` only buy/sell", have_dir and set(dist) <= {"buy", "sell"},
          f"values {sorted(set(dist))}")

    # --- block trades ---
    has_col = "block_trade_id" in df.columns
    n_block = int(df["block_trade_id"].notna().sum()) if has_col else 0
    check("block_trade_id present where expected", has_col and n_block > 0,
          f"{n_block:,} block rows ({100*n_block/max(len(df),1):.2f}%)"
          if has_col else "column absent entirely")

    # --- timestamps ---
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    lo = pd.Timestamp(f"{args.month}-01", tz="UTC")
    hi = lo + pd.offsets.MonthBegin(1)
    inside = bool((dt >= lo).all() and (dt < hi).all())
    check("timestamps parse and fall inside window", inside,
          f"{dt.min()} .. {dt.max()} (window {lo.date()} .. {hi.date()})")

    # --- duplicates ---
    dupes = int(df.duplicated(subset=["instrument_name", "trade_seq"]).sum())
    uniq_ids = int(df["trade_id"].nunique())
    check("duplicates are boundary-only and removable", dupes < len(df) * 0.001,
          f"{dupes} dup (instrument_name,trade_seq) rows; {uniq_ids:,} unique trade_id "
          f"vs {len(df):,} raw")
    dedup = df.drop_duplicates(subset=["instrument_name", "trade_seq"])
    check("no duplicates after dedupe",
          int(dedup.duplicated(subset=["instrument_name", "trade_seq"]).sum()) == 0,
          f"{len(dedup):,} rows survive")
    check("trade_id agrees with (instrument_name,trade_seq)",
          dedup["trade_id"].nunique() == len(dedup),
          f"{dedup['trade_id'].nunique():,} unique ids / {len(dedup):,} rows")

    # --- required fields for the construction ---
    need = {"instrument_name", "direction", "amount", "price", "timestamp",
            "trade_seq", "trade_id", "index_price", "iv"}
    missing = need - set(df.columns)
    check("all construction fields present", not missing,
          f"missing {missing}" if missing else f"{len(need)} fields present")

    # --- instruments resolve against instruments.parquet ---
    inst_path = config.path("paths.interim") / "instruments.parquet"
    if inst_path.exists():
        inst = pd.read_parquet(inst_path)
        known = set(inst["instrument_name"])
        seen = set(df["instrument_name"])
        unknown = seen - known
        check("every traded instrument resolves to the universe", not unknown,
              f"{len(seen)} traded, {len(unknown)} unresolved"
              + (f" e.g. {sorted(unknown)[:3]}" if unknown else ""))
        cs = inst.set_index("instrument_name")["contract_size"]
        check("contract_size joinable for all traded instruments",
              not unknown and cs.reindex(sorted(seen)).notna().all(),
              f"contract_size values {sorted(cs.reindex(sorted(seen)).dropna().unique())}")

    # --- parquet round trip ---
    tmp = config.path("paths.interim") / "_roundtrip_check"
    dedup2 = dedup.copy()
    dedup2["dt"] = dt.loc[dedup2.index]
    dedup2["year"], dedup2["month"] = dedup2["dt"].dt.year, dedup2["dt"].dt.month
    dedup2.to_parquet(tmp, partition_cols=["year", "month"], index=False)
    back = pd.read_parquet(tmp)
    ok = len(back) == len(dedup2) and set(back["trade_id"]) == set(dedup2["trade_id"])
    check("parquet round-trip", ok, f"wrote {len(dedup2):,}, read {len(back):,}")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*64}\n{len(RESULTS)-n_fail}/{len(RESULTS)} checks passed\n{'='*64}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
