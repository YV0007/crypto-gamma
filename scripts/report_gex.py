"""STEP 3 report on GEX_flow. Reports only."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2024-03-01")
ap.add_argument("--end", default="2024-03-31"); a = ap.parse_args()
proc = config.path("paths.processed")
series = pd.read_parquet(proc / f"gex_flow_{a.start}_{a.end}.parquet")
cells = pd.read_parquet(proc / f"gex_cells_{a.start}_{a.end}.parquet")
inv = pd.read_parquet(proc / f"inventory_{a.start}_{a.end}.parquet").reset_index()
g = series["gex_flow"]

print("=" * 78); print("(a) MAGNITUDE CHECK -- GEX_flow, USD per 1% spot move"); print("=" * 78)
for k, v in {"min": g.min(), "1%": g.quantile(.01), "5%": g.quantile(.05),
             "25%": g.quantile(.25), "median": g.median(), "mean": g.mean(),
             "75%": g.quantile(.75), "95%": g.quantile(.95), "99%": g.quantile(.99),
             "max": g.max()}.items():
    print(f"  {k:>7} {v:>18,.0f}   ({v/1e6:>9,.2f} M)")
print(f"\n  |GEX| median {g.abs().median()/1e6:,.2f} M   |GEX| max {g.abs().max()/1e6:,.2f} M")
om = np.log10(g.abs().median())
print(f"  order of magnitude of |median| : 1e{om:.1f}")
if g.abs().median() < 1e7:
    print("  *** BELOW the hundreds-of-millions expectation -- see diagnosis below ***")

print("\n  --- arithmetic on ONE interval ---")
t = g.abs().idxmax()
sub = cells[cells["timestamp"] == t].copy()
sub["abs"] = sub["gex_contribution"].abs()
top = sub.nlargest(5, "abs")
print(f"  timestamp {t}   spot {sub['spot'].iloc[0]:,.2f}")
print(f"  {'instrument':<26}{'position':>11}{'gamma':>12}{'contrib USD':>16}")
for _, r in top.iterrows():
    print(f"  {r['instrument_name']:<26}{r['dealer_position']:>11,.1f}{r['gamma']:>12.3e}"
          f"{r['gex_contribution']:>16,.0f}")
chk = top.iloc[0]
manual = chk["dealer_position"] * chk["gamma"] * chk["spot"]**2 * 0.01 * chk["contract_size"]
print(f"\n  hand-check row 1: {chk['dealer_position']:,.1f} x {chk['gamma']:.6e} x "
      f"{chk['spot']:,.2f}^2 x 0.01 x {chk['contract_size']} = {manual:,.2f}")
print(f"  top 5 sum      {top['gex_contribution'].sum():>18,.0f}")
print(f"  all {len(sub):>5} rows {sub['gex_contribution'].sum():>15,.0f}")
print(f"  reported total {g.loc[t]:>18,.0f}   match={np.isclose(sub['gex_contribution'].sum(), g.loc[t])}")

print("\n" + "=" * 78); print("(c) SIGN DISTRIBUTION"); print("=" * 78)
print(f"  intervals            : {len(g):,}")
print(f"  GEX > 0 (dealer long): {int((g>0).sum()):,}  ({100*(g>0).mean():.1f}%)")
print(f"  GEX < 0 (dealer short): {int((g<0).sum()):,}  ({100*(g<0).mean():.1f}%)")
print(f"  sign changes         : {int((np.sign(g).diff()!=0).sum()-1):,}")

print("\n" + "=" * 78); print("(d) CONCENTRATION"); print("=" * 78)
byi = cells.groupby("instrument_name")["gex_contribution"].apply(lambda x: x.abs().sum())
tot = byi.sum()
print(f"  top 1  instrument : {100*byi.nlargest(1).sum()/tot:5.1f}% of total |GEX|")
print(f"  top 10 instruments: {100*byi.nlargest(10).sum()/tot:5.1f}%")
print(f"  top 50            : {100*byi.nlargest(50).sum()/tot:5.1f}%")
print(f"  instruments contributing: {len(byi):,}")
print("\n  largest contributors:")
for n, v in byi.nlargest(10).items():
    print(f"    {n:<28}{100*v/tot:>6.2f}%")

print("\n" + "=" * 78); print("(e) MISSING-GREEKS IMPACT"); print("=" * 78)
allc = inv.merge(cells[["instrument_name", "timestamp", "gamma"]],
                 on=["instrument_name", "timestamp"], how="left")
allc["has"] = allc["gamma"].notna()
live = allc[allc["timestamp"] < allc["expiry_ts"]]
w = live["dealer_position"].abs()
print(f"  cells (all)                    : {len(allc):,}  gamma present {100*allc['has'].mean():.2f}%")
print(f"  cells (pre-expiry only)        : {len(live):,}  gamma present {100*live['has'].mean():.2f}%")
print(f"  POSITION-WEIGHTED coverage     : {100*(w*live['has']).sum()/w.sum():.2f}%")
print("    (post-expiry cells are hard-zeroed, so they cannot affect GEX)")
per = cells.groupby("timestamp").size()
full = per[per == per.max()].index
print(f"\n  intervals with max instrument coverage: {len(full):,}/{len(per):,}")
comp = g.loc[g.index.isin(full)] if len(full) else g.iloc[:0]
if len(comp):
    print(f"  headline median {g.median()/1e6:,.2f} M  vs complete-coverage median {comp.median()/1e6:,.2f} M")
q = cells.groupby("timestamp")["instrument_name"].nunique()
hi_q = q.quantile(.75)
sub2 = g.loc[q[q >= hi_q].index]
print(f"  median GEX on the best-covered quartile of intervals: {sub2.median()/1e6:,.2f} M")
print(f"  correlation(instruments priced, GEX) : {q.corr(g):.4f}")

print("\n" + "=" * 78); print("(f) CAVEATS"); print("=" * 78)
lo = pd.Timestamp(a.start, tz="UTC")
inst = pd.read_parquet(config.path("paths.interim") / "instruments.parquet").set_index("instrument_name")
traded = inv["instrument_name"].unique()
pre = int((inst.loc[traded, "creation_ts"] < lo).sum())
print(f"  instruments with PARTIAL positions (listed pre-window): {pre:,}/{len(traded):,} "
      f"({100*pre/len(traded):.1f}%)")
print("  GEX is provisional: one month of flow is not inventory from listing.")
print(f"  gamma is RECOMPUTED from Deribit's published mark IV (no historical greeks exist).")

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(series.index, g / 1e6, lw=.8, color="#1f4e79")
ax[0].axhline(0, color="red", lw=.8, ls="--")
ax[0].fill_between(series.index, g / 1e6, 0, where=(g > 0), color="#1f4e79", alpha=.25)
ax[0].fill_between(series.index, g / 1e6, 0, where=(g < 0), color="#c00000", alpha=.25)
ax[0].set_ylabel("GEX_flow\n(USD m per 1% move)")
ax[0].set_title("Flow-based dealer gamma exposure, BTC options, March 2024 (10-min)\n"
                "PROVISIONAL: one-month flow window; gamma recomputed from Deribit published IV",
                fontsize=10)
ax[1].plot(series.index, series["spot"], lw=.8, color="grey")
ax[1].set_ylabel("BTC index"); ax[1].set_xlabel("UTC")
fig.tight_layout()
out = config.path("paths.figures") / "gex_flow_2024-03.png"
fig.savefig(out, dpi=150)
print(f"\n(b) plot -> {out}")
