"""TASK 1 report -- full-period rerun. Reports only, no interpretation."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.build.panel import build_calendar  # noqa: E402

start, end = config.get("sample.start"), config.get("sample.end")
proc = config.path("paths.processed")
series = pd.read_parquet(proc / f"gex_flow_{start}_{end}.parquet")
series.index = pd.to_datetime(series.index, utc=True)
final = pd.read_parquet(proc / f"final_positions_{start}_{end}.parquet")["final_position"]
g = series["gex_flow"]

print("=" * 80); print("(a) FINAL DEALER POSITION -- FULL PERIOD vs MARCH-ONLY"); print("=" * 80)
npos, nneg = int((final > 0).sum()), int((final < 0).sum()); nz = int((final == 0).sum())
n = len(final)
print(f"  instruments                : {n:,}   (March slice: 2,706)")
print(f"  positive (dealer long)     : {npos:>7,}  ({100*npos/n:5.1f}%)   March: 53.7%")
print(f"  negative (dealer short)    : {nneg:>7,}  ({100*nneg/n:5.1f}%)   March: 45.3%")
print(f"  exactly zero               : {nz:>7,}  ({100*nz/n:5.1f}%)   March:  1.0%")
print("\n  quantiles of final position (contracts):")
for q in [0, .01, .05, .25, .50, .75, .95, .99, 1.0]:
    print(f"    {q:>6.0%} {final.quantile(q):>14,.2f}")
print(f"\n  sum {final.sum():,.1f}   mean {final.mean():,.2f}   median {final.median():,.2f}")

print("\n" + "=" * 80); print("(b) GEX_flow -- FULL PERIOD (USD per 1% move)"); print("=" * 80)
for k, v in {"min": g.min(), "1%": g.quantile(.01), "5%": g.quantile(.05), "25%": g.quantile(.25),
             "median": g.median(), "mean": g.mean(), "75%": g.quantile(.75),
             "95%": g.quantile(.95), "99%": g.quantile(.99), "max": g.max()}.items():
    print(f"  {k:>7} {v:>20,.0f}   ({v/1e6:>10,.2f} M)")
print(f"\n  intervals {len(g):,}   |GEX| median {g.abs().median()/1e6:,.2f} M")

print("\n  --- by year ---")
print(f"  {'year':<6}{'N':>8}{'median M':>12}{'mean M':>12}{'min M':>12}{'max M':>12}{'% neg':>8}")
for y, s in g.groupby(g.index.year):
    print(f"  {y:<6}{len(s):>8,}{s.median()/1e6:>12,.2f}{s.mean()/1e6:>12,.2f}"
          f"{s.min()/1e6:>12,.1f}{s.max()/1e6:>12,.1f}{100*(s<0).mean():>7.1f}%")

print("\n" + "=" * 80); print("(c) THE 87.8% QUESTION -- persistence of negative GEX"); print("=" * 80)
neg = (g < 0)
sgn = np.sign(g); changes = int((sgn.diff() != 0).sum() - 1)
print(f"  FULL PERIOD negative       : {100*neg.mean():.1f}%   (March slice: 87.8%)")
print(f"  sign changes               : {changes:,} over {len(g):,} intervals")
print(f"  mean run length            : {len(g)/max(changes,1):,.0f} intervals "
      f"({len(g)/max(changes,1)*10/60/24:.1f} days)")
runs = (sgn != sgn.shift()).cumsum()
rl = g.groupby(runs).size()
print(f"  longest single-sign run    : {rl.max():,} intervals ({rl.max()*10/60/24:.1f} days)")

print("\n  --- by year ---")
for y, s in g.groupby(g.index.year):
    sc = int((np.sign(s).diff() != 0).sum() - 1)
    print(f"    {y}: {100*(s<0).mean():5.1f}% negative, {sc:>5,} sign changes")

cal = build_calendar(pd.DatetimeIndex(g.index))
# Deribit expires an option EVERY day, so a "nearest expiry" horizon is a clock,
# not an expiry measure (STATUS.md 4.8). Use the monthly/quarterly calendar.
tte = cal["tte_major_days"]
bins = [0, 1, 3, 7, 14, np.inf]
labs = ["<1d", "1-3d", "3-7d", "7-14d", ">14d"]
b = pd.cut(tte, bins, labels=labs)
print("\n  --- by time to next MAJOR (monthly/quarterly) expiry ---")
print(f"    {'bucket':<8}{'N':>9}{'% neg':>9}{'median M':>12}{'mean M':>12}")
for lab in labs:
    m = (b == lab).to_numpy()
    if m.sum() == 0: continue
    s = g[m]
    print(f"    {lab:<8}{m.sum():>9,}{100*(s<0).mean():>8.1f}%{s.median()/1e6:>12,.2f}{s.mean()/1e6:>12,.2f}")

print("\n" + "=" * 80); print("(d) GAMMA COVERAGE RECONCILIATION"); print("=" * 80)
import json as _json
_sp = config.path("paths.tables") / "pipeline_stats.json"
if not _sp.exists():
    raise SystemExit(f"{_sp} missing -- run `python3 -m src.build.pipeline` first. "
                     "These numbers must never be hardcoded again.")
d = _json.loads(_sp.read_text())
print(f"  cells (all)                  : {d['cells']:>15,}")
print(f"  cells live (pre-expiry)      : {d['cells_live']:>15,}")
print(f"  cells with gamma             : {d['cells_with_gamma']:>15,}  "
      f"({100*d['cells_with_gamma']/d['cells_live']:.2f}% of live)")
print(f"  POSITION-WEIGHTED coverage   : {100*d['pos_weight_with_gamma']/d['pos_weight_total']:.2f}%"
      f"   (March: 88.06%)")
print("\n  The earlier '~20% downward bias' was NOT a like-for-like measurement:")
print("  it compared the median GEX of well-covered INTERVALS against the median")
print("  of all intervals -- different market conditions, not the same cells with")
print("  and without gamma. The staleness experiment (below) is the correct test.")

print("\n" + "=" * 80); print("(e) PLOT"); print("=" * 80)
fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
d1 = g.resample("1D").mean()
ax[0].plot(d1.index, d1 / 1e6, lw=.7, color="#1f4e79")
ax[0].fill_between(d1.index, d1 / 1e6, 0, where=(d1 > 0), color="#1f4e79", alpha=.3)
ax[0].fill_between(d1.index, d1 / 1e6, 0, where=(d1 < 0), color="#c00000", alpha=.3)
ax[0].axhline(0, color="black", lw=.8)
etf = pd.Timestamp(config.get("sample.etf_launch"), tz="UTC")
for a in ax:
    a.axvline(etf, color="green", lw=1.4, ls="--")
ax[0].text(etf, ax[0].get_ylim()[1]*.85, "  spot BTC ETF launch\n  2024-01-11",
           color="green", fontsize=9)
ax[0].set_ylabel("GEX_flow (USD m per 1% move)\ndaily mean of 10-min series")
ax[0].set_title("Flow-based dealer gamma exposure, BTC options, 2022-2025\n"
                "positions accumulated from listing; gamma from Deribit published IV",
                fontsize=11)
ax[1].plot(series.index, series["spot"], lw=.6, color="grey")
ax[1].set_ylabel("BTC index"); ax[1].set_xlabel("UTC")
fig.tight_layout()
p = config.path("paths.figures") / "gex_flow_full_period.png"
fig.savefig(p, dpi=150)
print(f"  saved -> {p}")
