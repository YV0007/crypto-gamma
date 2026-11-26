"""One-glance status of the Stage-1 backfill. Safe to run while it is running."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

STAGES = [("option_trades", "fetch.storage.option_dir"),
          ("perp_trades", "fetch.storage.perp_dir")]


def main() -> int:
    db = config.path("fetch.storage.checkpoint_db")
    if not db.exists():
        print("no checkpoint database yet -- backfill has not started")
        return 0
    start, end = config.sample_bounds_ms()
    total_days = (end - start) // 86_400_000

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    print(f"\ncrypto-gamma backfill status  {datetime.now(tz=timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    print(f"sample {config.get('sample.start')} .. {config.get('sample.end')}  ({total_days} days)\n")
    print(f"{'stage':<22}{'done':>7}{'fail':>6}{'pct':>7}{'rows':>14}{'reqs':>8}{'disk':>10}")
    print("-" * 74)

    for stage, path_key in STAGES:
        rows = con.execute(
            """SELECT status, COUNT(*), COALESCE(SUM(rows),0), COALESCE(SUM(requests),0),
                      COALESCE(SUM(bytes),0)
               FROM checkpoints WHERE stage=? GROUP BY status""", (stage,)).fetchall()
        agg = {s: (n, r, q, b) for s, n, r, q, b in rows}
        done, nrows, reqs, nbytes = agg.get("done", (0, 0, 0, 0))
        failed = agg.get("failed", (0, 0, 0, 0))[0]
        pct = 100 * done / total_days if total_days else 0
        print(f"{stage:<22}{done:>7}{failed:>6}{pct:>6.1f}%{nrows:>14,}{reqs:>8,}"
              f"{nbytes/1e9:>9.2f}G")
        if failed:
            for (k, e) in con.execute(
                "SELECT key, error FROM checkpoints WHERE stage=? AND status='failed' LIMIT 5",
                    (stage,)).fetchall():
                print(f"    FAILED {k}: {str(e)[:80]}")

    print("-" * 74)
    for name, key in [("ohlcv_1m.parquet", "paths.interim"), ("dvol.parquet", "paths.interim"),
                      ("instruments.parquet", "paths.interim")]:
        p = config.path(key) / name
        print(f"  {name:<24} {'present' if p.exists() else 'MISSING':>9}"
              f"{(f'  {p.stat().st_size/1e6:.1f} MB' if p.exists() else '')}")

    snaps = config.path("paths.oi_snapshots")
    n_snap = len(list(snaps.glob("*.parquet"))) if snaps.exists() else 0
    print(f"  {'OI snapshots':<24} {n_snap:>9} files")

    running = con.execute(
        "SELECT stage, MAX(updated_at) FROM checkpoints GROUP BY stage").fetchall()
    print("\nlast checkpoint write per stage:")
    for s, t in running:
        print(f"  {s:<22} {t}")
    con.close()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
