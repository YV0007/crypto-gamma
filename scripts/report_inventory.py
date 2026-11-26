"""STEP 3 report on the reconstructed March-2024 inventory. Reports only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.build.inventory import (  # noqa: E402
    accumulate, apply_block_trade_policy, compute_dealer_flow,
    load_instruments, load_trades,
)

pd.set_option("display.width", 200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2024-03-31")
    args = ap.parse_args()

    inst = load_instruments()
    raw = load_trades(args.start, args.end)
    meta = inst.set_index("instrument_name")

    # ---- (c) block trades, (d) direction quality -- measured on RAW ----
    n_raw = len(raw)
    n_block = int(raw["block_trade_id"].notna().sum()) if "block_trade_id" in raw else 0
    dir_counts = raw["direction"].value_counts(dropna=False)
    n_bad = int(n_raw - raw["direction"].isin(["buy", "sell"]).sum())

    kept, bstats = apply_block_trade_policy(raw)
    flow, dstats = compute_dealer_flow(kept, inst)
    trades = accumulate(flow)

    print("\n" + "=" * 78)
    print("(c) BLOCK TRADES");  print("=" * 78)
    print(f"  raw trades                : {n_raw:,}")
    print(f"  flagged block_trade_id    : {n_block:,}  ({100*n_block/n_raw:.4f}%)")
    print(f"  policy (config baseline)  : {bstats['mode']}")
    print(f"  trades used               : {bstats['n_trades_out']:,}")

    print("\n" + "=" * 78)
    print("(d) DIRECTION FIELD QUALITY");  print("=" * 78)
    print(f"  value counts (incl. NaN)  : {dir_counts.to_dict()}")
    print(f"  null or unexpected        : {n_bad:,}  ({100*n_bad/n_raw:.6f}%)")
    tb = int((raw['direction'] == 'buy').sum()); ts_ = int((raw['direction'] == 'sell').sum())
    print(f"  taker buy / taker sell    : {tb:,} / {ts_:,}  (ratio {tb/ts_:.4f})")

    # ---- (a) hand-check ledgers ----
    print("\n" + "=" * 78)
    print("(a) HAND-CHECK LEDGERS -- verify the sign by hand");  print("=" * 78)
    print("    taker buy  -> dealer_flow NEGATIVE (dealer sold to the customer)")
    print("    taker sell -> dealer_flow POSITIVE (dealer bought from the customer)")
    counts = trades["instrument_name"].value_counts()
    cands = counts[(counts >= 10) & (counts <= 30)].index.tolist()
    rng = np.random.default_rng(config.get("seed", 42))
    picks = list(rng.choice(cands, size=min(3, len(cands)), replace=False))
    for name in picks:
        sub = trades[trades["instrument_name"] == name].sort_values(["timestamp", "trade_seq"])
        raw_sub = raw[raw["instrument_name"] == name]
        print(f"\n  --- {name}  ({len(sub)} trades used, {len(raw_sub)} raw) ---")
        print(f"      created {meta.loc[name,'creation_ts']}   expires {meta.loc[name,'expiry_ts']}"
              f"   contract_size {meta.loc[name,'contract_size']}")
        print(f"      {'timestamp':<26}{'direction':>10}{'amount':>10}{'block':>10}"
              f"{'flow':>10}{'position':>11}")
        for _, r in sub.iterrows():
            blk = str(r.get("block_trade_id"))[:8] if pd.notna(r.get("block_trade_id")) else "-"
            print(f"      {str(r['dt'])[:23]:<26}{r['direction']:>10}{r['amount']:>10.1f}"
                  f"{blk:>10}{r['dealer_flow']:>10.1f}{r['dealer_position']:>11.1f}")

    # ---- (b) final position distribution ----
    final = trades.groupby("instrument_name")["dealer_position"].last()
    print("\n" + "=" * 78)
    print("(b) FINAL DEALER POSITION ACROSS INSTRUMENTS");  print("=" * 78)
    npos, nneg = int((final > 0).sum()), int((final < 0).sum())
    nzero = int((final == 0).sum())
    print(f"  instruments               : {len(final):,}")
    print(f"  positive (dealer long)    : {npos:,}  ({100*npos/len(final):.1f}%)")
    print(f"  negative (dealer short)   : {nneg:,}  ({100*nneg/len(final):.1f}%)")
    print(f"  exactly zero              : {nzero:,}  ({100*nzero/len(final):.1f}%)")
    q = final.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0])
    print("\n  quantiles (contracts):")
    for k, v in q.items():
        print(f"    {k:>5.0%} {v:>14,.2f}")
    print(f"\n  sum across instruments    : {final.sum():,.2f}")
    print(f"  mean / median             : {final.mean():,.2f} / {final.median():,.2f}")
    if npos == 0 or nneg == 0:
        print("\n  *** WARNING: positions are ALL ONE SIGN -- this indicates a bug. ***")
    else:
        print("\n  Mixed signs present, as expected.")

    # ---- (f) plausibility and grid integrity ----
    print("\n" + "=" * 78)
    print("(f) PLAUSIBILITY AND GRID INTEGRITY");  print("=" * 78)
    gross = trades.groupby("instrument_name")["amount"].sum()
    ratio = (final.abs() / gross).replace([np.inf, -np.inf], np.nan)
    viol = int((final.abs() > gross + 1e-6).sum())
    print(f"  |final position| > gross traded volume : {viol}  (must be 0 by construction)")
    print(f"  |position|/gross volume  median {ratio.median():.3f}  p95 {ratio.quantile(.95):.3f}"
          f"  max {ratio.max():.3f}")
    print("\n  largest |final position| (contracts):")
    for name, v in final.reindex(final.abs().sort_values(ascending=False).index).head(5).items():
        print(f"    {name:<28}{v:>12,.1f}   gross traded {gross[name]:>12,.1f}")

    panel_path = config.path("paths.processed") / f"inventory_{args.start}_{args.end}.parquet"
    panel = pd.read_parquet(panel_path).reset_index()
    exp_intervals = len(pd.date_range(pd.Timestamp(args.start, tz="UTC"),
                                      pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1),
                                      freq=config.get("construction.interval"), inclusive="left"))
    per = panel.groupby("instrument_name").size()
    print(f"\n  grid rows                 : {len(panel):,}")
    print(f"  instruments on grid       : {per.size:,}")
    print(f"  expected intervals each   : {exp_intervals:,}")
    print(f"  instruments with wrong count: {int((per != exp_intervals).sum())}")
    print(f"  null dealer_position      : {int(panel['dealer_position'].isna().sum())}")
    dupe = int(panel.duplicated(subset=['instrument_name', 'timestamp']).sum())
    print(f"  duplicate (instrument,ts) : {dupe}")

    # ---- window-truncation caveat ----
    lo = pd.Timestamp(args.start, tz="UTC")
    traded = final.index
    created_before = int((meta.loc[traded, "creation_ts"] < lo).sum())
    print("\n" + "=" * 78)
    print("WINDOW TRUNCATION (contradicts CONTEXT.md §11 'initialise at listing')")
    print("=" * 78)
    print(f"  instruments traded in window        : {len(traded):,}")
    print(f"  ... listed BEFORE the window opened : {created_before:,} "
          f"({100*created_before/len(traded):.1f}%)")
    print("  Their positions omit all pre-window flow and are therefore PARTIAL.")
    print("  Only instruments listed inside the window carry a true from-listing position.")

    # ---- (e) lifecycle plots ----
    hi = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    full_life = [n for n in traded
                 if meta.loc[n, "creation_ts"] >= lo and meta.loc[n, "expiry_ts"] < hi]
    life_counts = counts.reindex(full_life).dropna().sort_values(ascending=False)
    plot_names = life_counts.head(3).index.tolist()

    fig, axes = plt.subplots(len(plot_names), 1, figsize=(11, 3 * len(plot_names)), sharex=False)
    if len(plot_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, plot_names):
        g = panel[panel["instrument_name"] == name].sort_values("timestamp")
        ax.plot(g["timestamp"], g["dealer_position"], lw=1.1, color="#1f4e79")
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.axvline(meta.loc[name, "creation_ts"], color="green", lw=1, ls=":", label="listing")
        ax.axvline(meta.loc[name, "expiry_ts"], color="red", lw=1, ls=":", label="expiry")
        ax.set_title(f"{name}   ({int(counts[name])} trades)", fontsize=10)
        ax.set_ylabel("dealer position\n(contracts)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)
    fig.suptitle("Dealer inventory lifecycle -- full life inside March 2024\n"
                 "positive = dealer long gamma exposure, negative = dealer short",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = config.path("paths.figures") / "inventory_lifecycle_2024-03.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("\n" + "=" * 78)
    print("(e) LIFECYCLE PLOT");  print("=" * 78)
    print(f"  instruments with FULL life inside the window: {len(full_life):,}")
    print(f"  plotted: {plot_names}")
    for name in plot_names:
        g = panel[panel["instrument_name"] == name].sort_values("timestamp")["dealer_position"]
        print(f"    {name:<28} start {g.iloc[0]:>9.1f}  min {g.min():>9.1f} "
              f" max {g.max():>9.1f}  end {g.iloc[-1]:>6.1f}")
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
