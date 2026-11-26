#!/bin/sh
# Raw data -> every figure and table in outputs/, in dependency order.
#
# Assumes data/interim/ is already populated (the fetch stage is separate: it
# takes ~40 minutes and hits Deribit 2,399 times, so it is not re-run casually).
# Run from the repo root.  Total runtime ~40 minutes, dominated by the pipeline.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

S=$(python3 -c "from src import config;print(config.get('sample.start'))")
E=$(python3 -c "from src import config;print(config.get('sample.end'))")
echo "sample: $S .. $E"

echo "[1/9] full-period GEX pipeline    (~25 min)"
python3 -m src.build.pipeline

echo "[2/9] analysis panel"
python3 -m src.build.panel

echo "[3/9] perpetual flow"
python3 -m src.build.perp_flow

echo "[4/9] H2  volatility"
python3 -m src.analysis.h2_volatility

echo "[5/9] H3  hedging channel + identification ladder"
cp -f "data/processed/gex_flow_${S}_${E}.parquet" data/processed/gex_flow_h3snapshot.parquet
python3 -m src.analysis.h3_perpflow
python3 -m src.analysis.h3_identification

echo "[6/9] H4  structural break"
python3 -m src.analysis.h4_structural

echo "[7/9] H6 + information coefficients"
python3 -m src.analysis.h6_strategy
python3 -m src.analysis.signal_ic

echo "[8/9] cross-study corrections and validation"
python3 scripts/multiple_testing.py
python3 scripts/validate_gamma_aggregate.py

echo "[9/9] reports and deck outputs"
python3 scripts/report_full.py
python3 scripts/build_deck_outputs.py

echo
echo "done. figures in outputs/figures/, tables in outputs/tables/"
