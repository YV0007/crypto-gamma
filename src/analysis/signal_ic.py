"""Does GEX_flow actually PREDICT anything? Information coefficients.

WHY THIS IS A SEPARATE TEST FROM H2 AND H3
------------------------------------------
Neither existing result is predictive.

  H2 regresses realized volatility on GEX at the same instant, and its
  volatility measure is a TRAILING window ending at t. It is a contemporaneous
  association: "when dealers are short gamma, the market is currently rough."
  That can be true with no forecasting content whatsoever.

  H3 measures a RESPONSE to an event that has already happened. It is a
  mechanism test, not a forecast.

The information coefficient asks the different question a model would need
answered before anyone built one: given only what is knowable at t, does the
signal rank what happens AFTER t?

THE TRAP THIS MODULE IS BUILT TO AVOID
--------------------------------------
Volatility is extremely persistent. Any signal correlated with today's
volatility will appear to "predict" tomorrow's, because tomorrow's volatility
is mostly today's. A raw IC against forward volatility is therefore almost
meaningless on its own and will look impressive.

So every IC is reported twice:

  raw          Spearman rank correlation of the signal with the forward
               outcome. Included because it is what people quote, not because
               it answers the question.
  incremental  the same correlation after BOTH sides have been residualised on
               what a forecaster already knows for free -- trailing realized
               volatility and DVOL, the exchange's own implied-vol index. This
               is the only number that says whether GEX adds anything.

INFERENCE
---------
Observations are 10 minutes apart and the forward windows overlap heavily, so
ordinary correlation standard errors are far too small. Two defences:

  * a moving block bootstrap over whole UTC days, and
  * a daily-IC series -- the IC computed within each day, then averaged. Its
    dispersion across days gives the IC information ratio, which is the
    practitioner's standard and does not assume independence within a day.

A non-overlapping subsample is also reported as a sanity check.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config
from src.analysis.h2_volatility import detrend_gex

log = logging.getLogger("analysis.ic")

# Horizons in 10-minute intervals: 1h, 4h, 24h.
HORIZONS = {"1h": 6, "4h": 24, "24h": 144}
CONTROLS = ["realized_vol", "dvol"]


def _rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, nan_policy="omit")


def _resid_on(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y after linear projection on X (with intercept)."""
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def build(start: str, end: str) -> pd.DataFrame:
    """Panel with the signal and every forward outcome, all backward-safe."""
    p = pd.read_parquet(config.path("paths.processed") / f"panel_{start}_{end}.parquet")
    d = p.dropna(subset=["gex_flow", "realized_vol"]).copy()
    d["signal"] = detrend_gex(d, "rolling_z")
    d["abs_signal"] = d["signal"].abs()
    d["day"] = d.index.normalize()

    for lab, k in HORIZONS.items():
        # forward realized volatility: the trailing-window measure observed k
        # steps ahead, so it covers (t, t+k] and uses nothing known at t
        d[f"fwd_rv_{lab}"] = d["realized_vol"].shift(-k)
        # forward absolute return over the window, and forward signed return
        fwd = np.log(d["close"].shift(-k) / d["close"])
        d[f"fwd_ret_{lab}"] = fwd
        d[f"fwd_absret_{lab}"] = fwd.abs()
    return d


def ic_block_bootstrap(sig: np.ndarray, out: np.ndarray, days: np.ndarray,
                       reps: int = 500, seed: int = 42) -> tuple[float, float, float]:
    """Spearman IC with a moving block bootstrap over whole days.

    Ranks are taken once on the full sample and Pearson is applied to the
    resampled ranks. Re-ranking inside every draw would measure a slightly
    different statistic and costs far more for no gain in honesty here.
    """
    rs, ro = _rank(sig), _rank(out)
    rs = (rs - rs.mean()) / rs.std()
    ro = (ro - ro.mean()) / ro.std()
    point = float(np.mean(rs * ro))

    codes, uniq = pd.factorize(days)
    n = len(uniq)
    prod = rs * ro
    sums = np.bincount(codes, weights=prod, minlength=n)
    cnts = np.bincount(codes, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n, size=(reps, n))
    draws = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def daily_ic(d: pd.DataFrame, sig_col: str, out_col: str, min_obs: int = 30) -> dict:
    """IC computed within each day, then summarised across days.

    mean_ic  average daily IC
    ic_ir    mean / std of the daily IC series -- the information ratio of the
             signal itself, per day. Independent of the overlap problem.
    t_stat   mean / (std / sqrt(n_days))
    hit      share of days with the expected sign
    """
    sub = d[[sig_col, out_col, "day"]].dropna()
    ics = []
    for _, g in sub.groupby("day"):
        if len(g) < min_obs:
            continue
        a, b = g[sig_col].to_numpy(), g[out_col].to_numpy()
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ra, rb = _rank(a), _rank(b)
        ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 10:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "t_stat": np.nan,
                "n_days": int(len(ics)), "hit_rate": np.nan}
    sd = ics.std(ddof=1)
    return {"mean_ic": float(ics.mean()), "ic_ir": float(ics.mean() / sd) if sd > 0 else np.nan,
            "t_stat": float(ics.mean() / (sd / np.sqrt(len(ics)))) if sd > 0 else np.nan,
            "n_days": int(len(ics)), "hit_rate": float((np.sign(ics) == np.sign(ics.mean())).mean())}


def evaluate(d: pd.DataFrame, sig_col: str, out_col: str) -> dict:
    """Raw and incremental IC for one signal / outcome pair."""
    cols = [sig_col, out_col, "day"] + CONTROLS
    sub = d[cols].dropna()
    if len(sub) < 1000:
        return {}
    sig, out = sub[sig_col].to_numpy(), sub[out_col].to_numpy()
    days = sub["day"].to_numpy()

    raw, lo, hi = ic_block_bootstrap(sig, out, days)

    # incremental: strip what a forecaster already knows from BOTH sides,
    # on ranks, so the result stays a rank correlation
    W = np.column_stack([_rank(sub[c].to_numpy()) for c in CONTROLS])
    rs = _resid_on(_rank(sig), W)
    ro = _resid_on(_rank(out), W)
    inc, ilo, ihi = ic_block_bootstrap(rs, ro, days)

    # non-overlapping sanity check: one observation per forward window
    step = max(HORIZONS.get(out_col.split("_")[-1], 6), 1)
    nov = sub.iloc[::step]
    nov_ic = float(np.corrcoef(_rank(nov[sig_col].to_numpy()),
                               _rank(nov[out_col].to_numpy()))[0, 1]) if len(nov) > 200 else np.nan

    return {"n_obs": int(len(sub)),
            "ic_raw": raw, "ic_raw_lo": lo, "ic_raw_hi": hi,
            "ic_incremental": inc, "ic_inc_lo": ilo, "ic_inc_hi": ihi,
            "ic_nonoverlapping": nov_ic,
            "daily": daily_ic(sub.assign(**{sig_col: sig, out_col: out}), sig_col, out_col)}


def run(start: str, end: str) -> dict:
    d = build(start, end)
    out: dict = {}
    pairs = [("signal", f"fwd_rv_{h}") for h in HORIZONS]
    pairs += [("abs_signal", f"fwd_rv_{h}") for h in HORIZONS]
    pairs += [("signal", f"fwd_ret_{h}") for h in HORIZONS]
    pairs += [("signal", f"fwd_absret_{h}") for h in HORIZONS]
    for sig, tgt in pairs:
        r = evaluate(d, sig, tgt)
        if r:
            out[f"{sig} -> {tgt}"] = r
            log.info("%-28s raw %+.4f  incremental %+.4f  [%.4f,%.4f]  n=%d",
                     f"{sig}->{tgt}", r["ic_raw"], r["ic_incremental"],
                     r["ic_inc_lo"], r["ic_inc_hi"], r["n_obs"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Information coefficients for GEX_flow")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")

    res = run(start, end)
    tab = config.path("paths.tables")
    (tab / "signal_ic.json").write_text(json.dumps(res, indent=2, default=str))
    rows = [{"pair": k, "n_obs": v["n_obs"], "ic_raw": v["ic_raw"],
             "ic_incremental": v["ic_incremental"], "inc_lo": v["ic_inc_lo"],
             "inc_hi": v["ic_inc_hi"], "ic_nonoverlap": v["ic_nonoverlapping"],
             "daily_mean_ic": v["daily"]["mean_ic"], "daily_ic_ir": v["daily"]["ic_ir"],
             "daily_t": v["daily"]["t_stat"]} for k, v in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv(tab / "signal_ic.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
