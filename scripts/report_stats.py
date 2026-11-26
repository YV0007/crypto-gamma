#!/usr/bin/env python
"""What the study has tried so far, and the bar a result now has to clear.

Run this before putting any Sharpe ratio on a slide:

    python scripts/report_stats.py

It prints the specification log -- every analysis this project has ever run --
the resulting trial count per hypothesis, and the selection bar SR0 that the
Deflated Sharpe Ratio would impose today. The bar rises on its own as the
study grows, so the number printed here is only valid for the log as it stands
at the moment of printing. That is the point: it cannot go stale silently.

    --hypothesis h6   restrict the trial family to one hypothesis
    --returns PATH    a parquet/csv of strategy returns to actually deflate
    --column NAME     which column of that file holds the returns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.utils import speclog, stats  # noqa: E402


def load_returns(path: Path, column: str) -> np.ndarray:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if column not in df.columns:
        raise SystemExit(f"column {column!r} not in {path.name}; have: {list(df.columns)}")
    return df[column].to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", default=None, help="restrict to one hypothesis, e.g. h6")
    ap.add_argument("--returns", default=None, help="parquet/csv of strategy returns")
    ap.add_argument("--column", default="ret", help="returns column name")
    ap.add_argument("--full", action="store_true", help="print every logged row, not just trials")
    args = ap.parse_args()

    path = speclog.log_path()
    print(f"specification log: {path}")
    rows = speclog.load()
    if rows.empty:
        print("\n  EMPTY -- nothing has been logged yet.")
        print("  Analyses call src.utils.speclog.record(); until one runs there is")
        print("  no trial count, and any Deflated Sharpe Ratio would be a guess.")
        return

    print(f"  {len(rows)} rows, {rows['run_id'].nunique()} runs, "
          f"{rows['spec_id'].nunique()} distinct specifications")
    print(f"  first {rows['run_ts'].min()}   last {rows['run_ts'].max()}")

    print("\n--- per hypothesis " + "-" * 60)
    print(speclog.summary().to_string(index=False))

    drift = speclog.config_drift()
    if not drift.empty:
        print("\n!! CONFIG DRIFT: these specs were run under more than one config.yaml.")
        print("   Their logged numbers are not comparable. Rerun them, or split the trial.")
        print(drift.to_string(index=False))

    show = speclog.load() if args.full else speclog.trials(hypothesis=args.hypothesis)
    cols = ["run_ts", "hypothesis", "name", "n_obs", "statistic",
            "estimate", "se", "t_stat", "p_value", "sharpe"]
    print(f"\n--- {'all rows' if args.full else 'distinct trials'} " + "-" * 58)
    print(show[cols].to_string(index=False))

    trials = speclog.trials(hypothesis=args.hypothesis)
    n_trials = max(len(trials), 1)
    sr_logged = pd.to_numeric(trials["sharpe"], errors="coerce").dropna()
    sr_var = float(sr_logged.var(ddof=1)) if len(sr_logged) > 1 else float("nan")

    print("\n--- selection bar as of now " + "-" * 51)
    scope = args.hypothesis or "ALL hypotheses"
    print(f"  scope                {scope}")
    print(f"  distinct trials      {n_trials}")
    min_sr = config.get("stats.deflated_sharpe.min_logged_sharpes")
    if len(sr_logged) >= min_sr:
        sr0 = stats.expected_max_sharpe(n_trials, sr_var)
        print(f"  variance of trial Sharpes  {sr_var:.6g}")
        print(f"  SR0 (per obs)        {sr0:.6f}   annualised {stats.annualise(sr0):.3f}")
        print("  Any Sharpe below that line is what this many tries would produce")
        print("  from noise alone.")
    else:
        print(f"  fewer than {min_sr} logged Sharpe ratios: no spread to estimate SR0 from.")
        print("  Log a Sharpe on each backtest specification (speclog.record(sharpe=...))")
        print("  before quoting a deflated number.")

    if args.returns:
        r = load_returns(Path(args.returns), args.column)
        print("\n--- deflated Sharpe of " + args.returns + " " + "-" * 20)
        print(stats.deflate_from_log(r, hypothesis=args.hypothesis))
    else:
        print("\n  (pass --returns PATH --column NAME to deflate an actual return series)")


if __name__ == "__main__":
    main()
