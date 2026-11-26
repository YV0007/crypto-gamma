#!/usr/bin/env bash
# SUPERSEDED -- do not schedule this file.
#
# This script cannot be run by launchd from here. The repo is in ~/Desktop,
# which macOS protects with TCC; a LaunchAgent has no Full Disk Access and gets
# "Operation not permitted" trying to read this very file. The scheduled job now
# runs ~/crypto-gamma-collector/collect_oi.sh, installed by
# scripts/install_collector.sh.
#
# Running this by hand from an interactive shell still works (your Terminal does
# have Desktop access), which is exactly why the breakage was invisible for days:
# manual runs succeeded while every scheduled run failed silently.
#
# Daily open-interest + greeks snapshot.
#
# WHY THIS RUNS ON A SCHEDULE
# Deribit publishes NO historical open interest and NO historical greeks. Both
# vanish permanently when a contract expires. Every snapshot this takes is a
# data point that cannot be recovered later at any price. This is the only
# irreplaceable, time-decaying task in the project.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== $STAMP collector starting ===" >> logs/oi_collector.log

# Interpreter is PINNED by absolute path. launchd does not inherit the
# interactive shell's PATH, so a bare `python3` resolves to the Command Line
# Tools build (3.9, no pandas) and the job fails with ModuleNotFoundError.
PY="/opt/anaconda3/bin/python3"
[ -x .venv/bin/python3 ] && PY=.venv/bin/python3
if ! "$PY" -c "import pandas, httpx" 2>/dev/null; then
    echo "FATAL: $PY cannot import pandas/httpx" >> logs/oi_collector.log
    exit 1
fi

if $PY -m src.fetch.book_summary >> logs/oi_collector.log 2>&1; then
    N=$(ls data/raw/oi_snapshots/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo "=== $STAMP OK (total snapshots: $N) ===" >> logs/oi_collector.log
else
    echo "=== $STAMP FAILED (exit $?) ===" >> logs/oi_collector.log
fi

# keep the log from growing without bound
if [ "$(wc -l < logs/oi_collector.log)" -gt 20000 ]; then
    tail -5000 logs/oi_collector.log > logs/oi_collector.log.tmp
    mv logs/oi_collector.log.tmp logs/oi_collector.log
fi
