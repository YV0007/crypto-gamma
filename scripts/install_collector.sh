#!/usr/bin/env bash
# Install / refresh the standalone OI collectors.
#
# WHY THE COLLECTOR CANNOT LIVE IN THE REPO
# The repo sits in ~/Desktop, which macOS protects with TCC. A LaunchAgent has
# no Full Disk Access, so launchd cannot even read a script there:
#     /bin/bash: .../scripts/collect_oi.sh: Operation not permitted
# Granting Full Disk Access to /bin/bash would fix it, but that hands every
# shell script on the machine unrestricted access to every file. Instead the
# collectors run from a copy outside the protected area, created by this script.
#
# TWO JOBS, because the two API calls cost wildly different amounts:
#   full  (com.crypto-gamma.oi-collector)  979 requests, ~68 s  -> daily
#   light (com.crypto-gamma.oi-light)        1 request,  ~1 s   -> hourly
# get_book_summary_by_currency already carries open_interest and mark_iv for
# every instrument; only the greeks need the per-instrument ticker loop, and we
# compute our own greeks anyway (STATUS.md 4.3).
#
# RE-RUN THIS after changing src/fetch/*, src/config.py or config.yaml.
# Nothing else syncs the copy.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL="$HOME/crypto-gamma-collector"

mkdir -p "$INSTALL/oi_snapshots" "$INSTALL/oi_light" "$HOME/Library/Logs"

# --- sync the code ---------------------------------------------------------
# The whole src/ tree is copied (484 KB) rather than a hand-picked file list, so
# a new import in book_summary.py can never silently break the scheduled job.
rsync -a --delete --exclude '__pycache__' "$REPO/src/" "$INSTALL/src/"
cp "$REPO/config.yaml" "$INSTALL/config.yaml"
printf 'installed %s from %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REPO" > "$INSTALL/INSTALLED_FROM.txt"

# --- shared runner ---------------------------------------------------------
# One script, two modes. $1 is "" (full) or "--light".
cat > "$INSTALL/collect_oi.sh" <<'RUNNER'
#!/usr/bin/env bash
# Open-interest snapshot. Deribit publishes NO historical open interest and NO
# historical greeks; both vanish permanently when a contract expires. Every
# snapshot taken here is a data point that cannot be recovered later at any
# price. This is the only irreplaceable, time-decaying task in the project.
set -u
cd "$(dirname "$0")"

MODE="${1:-}"
TAG=$([ "$MODE" = "--light" ] && echo light || echo full)
LOG="$HOME/Library/Logs/crypto-gamma-oi.log"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Interpreter is PINNED by absolute path. launchd does not inherit the
# interactive shell's PATH, so a bare `python3` resolves to the Command Line
# Tools build (3.9, no pandas) and the job dies with ModuleNotFoundError --
# exactly how the 2026-08-31T23:20Z run was lost.
PY="/opt/anaconda3/bin/python3"
if ! "$PY" -c "import pandas, httpx, pyarrow" 2>/dev/null; then
    echo "=== $STAMP [$TAG] FATAL: $PY cannot import pandas/httpx/pyarrow ===" >> "$LOG"
    exit 1
fi

if "$PY" -m src.fetch.book_summary $MODE >> "$LOG" 2>&1; then
    # Only the daily full run pays for the sqlite rebuild; the hourly job stays
    # at ~1 s. A missed mirror is harmless -- parquet is the source of truth and
    # oi_sqlite.py backfills whatever it has not seen.
    if [ "$TAG" = "full" ]; then
        "$PY" oi_sqlite.py >> "$LOG" 2>&1 || echo "warn: sqlite mirror failed" >> "$LOG"
    fi
    echo "=== $STAMP [$TAG] OK (full: $(ls oi_snapshots/*.parquet 2>/dev/null | wc -l | tr -d ' '), light: $(ls oi_light/*.parquet 2>/dev/null | wc -l | tr -d ' ')) ===" >> "$LOG"
else
    echo "=== $STAMP [$TAG] FAILED ===" >> "$LOG"
fi

if [ "$(wc -l < "$LOG")" -gt 20000 ]; then
    tail -5000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
RUNNER
chmod +x "$INSTALL/collect_oi.sh"

# thin wrapper so the plist needs no arguments
printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/collect_oi.sh" --light\n' > "$INSTALL/collect_oi_light.sh"
chmod +x "$INSTALL/collect_oi_light.sh"

# --- sqlite mirror ---------------------------------------------------------
cat > "$INSTALL/oi_sqlite.py" <<'SQLITE'
"""Mirror both snapshot directories into one queryable SQLite file.

Parquet stays the source of truth -- src/build/greeks.py reads it directly.
This is a convenience layer so the whole history can be inspected with plain
SQL without opening every file:

    sqlite3 ~/crypto-gamma-collector/oi_history.sqlite \
      "select snapshot_ts, kind, round(sum(open_interest),1)
         from oi group by 1,2 order by 1;"

`kind` is 'full' (daily, greeks fetched alongside) or 'light' (hourly, OI+IV
only). The greek columns are deliberately NOT mirrored: we compute our own from
mark_iv, and storing Deribit's would imply they are an input rather than a
reconciliation check.

Idempotent: re-running never duplicates a snapshot already loaded.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE / "oi_history.sqlite"
SOURCES = {"full": HERE / "oi_snapshots", "light": HERE / "oi_light"}

COLS = [
    "instrument_name", "open_interest", "mark_price", "mark_iv",
    "underlying_price", "underlying_index", "base_currency",
    "volume", "volume_usd", "bid_price", "ask_price", "mid_price",
    "last", "interest_rate", "estimated_delivery_price",
]


def main() -> None:
    con = sqlite3.connect(DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS oi ("
        "snapshot_ts TEXT NOT NULL, kind TEXT NOT NULL, "
        "instrument_name TEXT NOT NULL, "
        "open_interest REAL, mark_price REAL, mark_iv REAL, "
        "underlying_price REAL, underlying_index TEXT, base_currency TEXT, "
        "volume REAL, volume_usd REAL, bid_price REAL, ask_price REAL, "
        "mid_price REAL, last REAL, interest_rate REAL, "
        "estimated_delivery_price REAL, "
        # `kind` MUST be in the key. A light and a full run can start in the
        # same second (both RunAtLoad jobs did on 2026-09-01T23:38:37Z), and a
        # key of (snapshot_ts, instrument_name) silently drops one of them.
        "PRIMARY KEY (snapshot_ts, kind, instrument_name))"
    )
    # Migration: an oi table created before `kind` existed is NOT rebuilt by
    # CREATE TABLE IF NOT EXISTS, so the column has to be added explicitly.
    # Everything written before `kind` existed came from oi_snapshots = full.
    have = {r[1] for r in con.execute("PRAGMA table_info(oi)")}
    if "kind" not in have:
        con.execute("ALTER TABLE oi ADD COLUMN kind TEXT")
        con.execute("UPDATE oi SET kind = 'full' WHERE kind IS NULL")
        con.commit()

    # A table built with the old 2-column key cannot be widened by ALTER TABLE.
    # This DB is a pure mirror of the parquet files, so dropping and refilling
    # it is safe and costs a few seconds -- never do this to the parquet.
    ddl = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='oi'"
    ).fetchone()
    if ddl and "PRIMARY KEY (snapshot_ts, kind" not in ddl[0]:
        print("migrating: rebuilding oi with (snapshot_ts, kind, instrument_name) key")
        con.execute("DROP TABLE oi")
        con.commit()
        return main()

    con.execute("CREATE INDEX IF NOT EXISTS oi_ts ON oi(snapshot_ts)")
    con.execute("CREATE INDEX IF NOT EXISTS oi_kind ON oi(kind)")
    done = {(k, t) for k, t in
            con.execute("SELECT DISTINCT kind, snapshot_ts FROM oi")}

    added = 0
    for kind, d in SOURCES.items():
        if not d.is_dir():
            continue
        for f in sorted(d.glob("book_summary_*.parquet")):
            ts = f.stem.replace("book_summary_", "")
            if (kind, ts) in done:
                continue
            df = pd.read_parquet(f)
            out = df[[c for c in COLS if c in df.columns]].copy()
            out.insert(0, "kind", kind)
            out.insert(0, "snapshot_ts", ts)
            out.to_sql("oi", con, if_exists="append", index=False)
            done.add((kind, ts))
            added += 1
    con.commit()
    n_rows = con.execute("SELECT COUNT(*) FROM oi").fetchone()[0]
    by_kind = dict(con.execute(
        "SELECT kind, COUNT(DISTINCT snapshot_ts) FROM oi GROUP BY kind"))
    con.close()
    print(f"sqlite mirror: +{added} new, {by_kind}, {n_rows} rows -> {DB}")


if __name__ == "__main__":
    main()
SQLITE

# --- LaunchAgents ----------------------------------------------------------
for LABEL in com.crypto-gamma.oi-collector com.crypto-gamma.oi-light; do
    AGENT="$HOME/Library/LaunchAgents/$LABEL.plist"
    cp "$REPO/deploy/$LABEL.plist" "$AGENT"
    plutil -lint "$AGENT" >/dev/null
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$AGENT"
    echo "loaded $LABEL"
done

echo
echo "installed to $INSTALL"
echo "log: ~/Library/Logs/crypto-gamma-oi.log"
