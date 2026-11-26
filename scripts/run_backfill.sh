#!/usr/bin/env bash
# Full Stage-1 backfill, run sequentially so the three stages never contend
# for the same rate-limit budget. Every stage is independently resumable:
# rerunning this script skips whatever is already checkpointed `done`.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== backfill started $(date -u +%FT%TZ) ===" | tee -a logs/backfill_driver.log

run_stage () {
  local name="$1"; shift
  echo "--- $name starting $(date -u +%FT%TZ) ---" | tee -a logs/backfill_driver.log
  if python3 -m "$@" >> "logs/fetch_${name}.stdout.log" 2>&1; then
    echo "--- $name OK $(date -u +%FT%TZ) ---" | tee -a logs/backfill_driver.log
  else
    echo "--- $name FAILED (exit $?) $(date -u +%FT%TZ) ---" | tee -a logs/backfill_driver.log
  fi
}

run_stage option_trades src.fetch.option_trades
run_stage perp_trades   src.fetch.perp_trades
run_stage market_data   src.fetch.market_data

echo "=== backfill finished $(date -u +%FT%TZ) ===" | tee -a logs/backfill_driver.log
python3 scripts/status.py | tee -a logs/backfill_driver.log
