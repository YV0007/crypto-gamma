#!/bin/sh
# Waits for the blocks-included full-period pipeline, then rebuilds everything
# downstream of it: the panel (with the corrected expiry calendar) and H2 (with
# the detrended sort). Launched 2026-09-01 alongside the pipeline itself.
cd "$(dirname "$0")/.." || exit 1
while [ ! -f logs/pipeline_blockincl_done.txt ]; do sleep 60; done
echo "pipeline done $(date)" > logs/downstream.log
python3 -m src.build.panel        >> logs/downstream.log 2>&1 || exit 1
echo "panel done $(date)"         >> logs/downstream.log
python3 -m src.analysis.h2_volatility >> logs/downstream.log 2>&1 || exit 1
echo "h2 done $(date)"            >> logs/downstream.log
echo DONE > logs/downstream_done.txt
