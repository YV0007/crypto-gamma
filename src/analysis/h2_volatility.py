"""H2, stylized fact only: does GEX sort realized volatility and autocorrelation?

RAW SORT ONLY. No regression, no controls, no tuning. The prediction under
test, stated before looking:

    highest GEX quintile -> LOWEST realized vol, MOST NEGATIVE autocorrelation
    lowest  GEX quintile -> HIGHEST realized vol, POSITIVE autocorrelation
    and the pattern should be MONOTONIC across quintiles

Autocorrelation is computed on PAIRS: for every interval t classified into
quintile q, pair r5[t] with r5[t-lag] and take the Pearson correlation across
those pairs. Restricting to a quintile breaks time contiguity, so a
series-wise acf would be meaningless.

Uncertainty uses a MOVING BLOCK BOOTSTRAP with one-day blocks, because
10-minute observations are strongly autocorrelated and an iid bootstrap would
understate the error bars.

WHAT GETS SORTED -- the choice that decides the answer
-----------------------------------------------------
GEX_flow is a trending dollar level, not a stationary state variable. Its
median runs -$0.6M (2022) -> -$13.1M (2025): partly mechanical, since
GEX = pos x gamma x S^2 and gamma ~ 1/S, so GEX ~ S and spot rose 3.6x over
the sample; partly the option book growing. 50.1% of its variance is
between-month.

Sorting the RAW level therefore sorts calendar time. Empirically quintile 1 is
95% drawn from 2024-25 and quintile 4 is 55% from 2022, and only 0.1% of 2022
ever reaches quintile 1. Because realized vol ALSO trends over the sample, the
sort manufactures a relationship out of two trends:

    corr(gex_flow, year)      = -0.361
    corr(realized_vol, year)  = -0.135
    product                   = +0.049
    observed corr(gex, rv)    = +0.032      <- essentially all of it

That is the whole of the "wrong sign, and it isn't noise" result that STATUS.md
section 6 reported. Detrended, the sign flips to the PREDICTED direction and
the interval covers zero:

    raw level              Q5-Q1 = +0.000443   CI [+0.000133, +0.000743]
    rolling-60d z          Q5-Q1 = -0.000113   CI [-0.000407, +0.000168]
    within-month z         Q5-Q1 = -0.000207   CI [-0.000494, +0.000094]
    within-year z          Q5-Q1 = -0.000017   CI [-0.000376, +0.000332]

So the honest finding is a NULL, not a reversal.

The baseline is `rolling_z` (config `h2_volatility.detrend`), a trailing-window
z-score: it is the only detrended option that is strictly backward-looking and
so the only one admissible as a headline. `raw` is still computed and printed,
labelled as confounded, because showing the bias is part of the result. See
DIAGNOSTIC_H2.md section 1.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.utils import speclog  # noqa: E402
from src.utils.stats import (  # noqa: E402
    block_bootstrap_ci,
    block_bootstrap_contrast,
    nw_diff,
    nw_mean,
)

log = logging.getLogger("analysis.h2")


def paired_autocorr(df: pd.DataFrame, mask: np.ndarray, lag: int) -> float:
    """Pearson corr between r5[t] and r5[t-lag] over intervals selected by mask."""
    r = df["ret_5m"]
    lagged = r.shift(lag)
    ok = mask & r.notna().to_numpy() & lagged.notna().to_numpy()
    if ok.sum() < 30:
        return np.nan
    return float(np.corrcoef(r[ok], lagged[ok])[0, 1])


DETREND_MODES = ("raw", "rolling_z", "within_month", "within_year")


def detrend_gex(panel: pd.DataFrame, mode: str | None = None) -> pd.Series:
    """The variable actually sorted on, under one of `DETREND_MODES`.

    Returns a Series aligned to `panel.index`; NaN where the mode cannot be
    evaluated (the warm-up of the rolling window).

      raw           GEX_flow untouched. CONFOUNDED with calendar time -- see
                    the module docstring. Kept to display the bias, never as a
                    headline.
      rolling_z     (x - trailing mean) / trailing sd over `rolling_window`.
                    Strictly backward-looking, so it is the only detrended
                    option with no lookahead. BASELINE.
      within_month  z-score within calendar month. Uses the whole month, so it
                    peeks at the future within that month; robustness only.
      within_year   z-score within calendar year. Peeks harder; robustness.
    """
    mode = mode or config.get("h2_volatility.detrend.baseline")
    if mode not in DETREND_MODES:
        raise ValueError(f"unknown detrend mode {mode!r} (one of {DETREND_MODES})")

    x = panel["gex_flow"]
    if mode == "raw":
        return x.copy()

    if mode == "rolling_z":
        win = config.get("h2_volatility.detrend.rolling_window")
        frac = config.get("h2_volatility.detrend.min_periods_frac")
        step = pd.Timedelta(config.get("construction.interval"))
        n_win = max(int(pd.Timedelta(win) / step), 2)
        mp = max(int(frac * n_win), 2)
        mu = x.rolling(win, min_periods=mp).mean()
        sd = x.rolling(win, min_periods=mp).std()
        z = (x - mu) / sd.where(sd > 0)
        log.info("detrend=rolling_z window=%s (%d intervals, min_periods=%d): "
                 "%d/%d usable after warm-up", win, n_win, mp, int(z.notna().sum()), len(z))
        return z

    key = (x.index.tz_localize(None).to_period("M")   # tz dropped for grouping only
           if mode == "within_month" else x.index.year)
    g = x.groupby(key)
    sd = g.transform("std")
    return (x - g.transform("mean")) / sd.where(sd > 0)


def quintile_sort(panel: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Quintile sort on the detrended GEX chosen by `mode`.

    `gex_mean_musd` and friends stay in RAW dollars whatever the sort variable
    is, so the buckets remain readable; only the assignment changes.
    """
    mode = mode or config.get("h2_volatility.detrend.baseline")
    n_q = config.get("h2_volatility.n_quantiles")
    lags = config.get("h2_volatility.autocorr_lags")

    df = panel.dropna(subset=["gex_flow", "realized_vol"]).copy()
    df["gex_sort"] = detrend_gex(df, mode)
    df = df.dropna(subset=["gex_sort"])
    df["q"] = pd.qcut(df["gex_sort"], n_q, labels=False, duplicates="drop") + 1
    df["day"] = df.index.normalize()

    rows = []
    for q in sorted(df["q"].dropna().unique()):
        m = (df["q"] == q).to_numpy()
        sub = df[m]
        rv = sub["realized_vol"].to_numpy()
        lo, hi = block_bootstrap_ci(rv, sub["day"].to_numpy())
        # HAC standard error reported ALONGSIDE the bootstrap, not instead of
        # it. Selecting a quintile pulls scattered intervals out of the time
        # series, so "lag 1" inside `sub` is not one interval later in
        # calendar time and the HAC estimate misses most of the dependence.
        # It comes out roughly 4x tighter than the block bootstrap on this
        # panel. The bootstrap is the headline; `rv_nw_se` is a lower bound,
        # kept because the gap between the two is itself diagnostic.
        hac = nw_mean(rv)
        row = {
            "detrend": mode,
            "quintile": int(q),
            "N": int(m.sum()),
            "gex_mean_musd": sub["gex_flow"].mean() / 1e6,
            "gex_min_musd": sub["gex_flow"].min() / 1e6,
            "gex_max_musd": sub["gex_flow"].max() / 1e6,
            "realized_vol": float(np.nanmean(rv)),
            "rv_ci_lo": lo, "rv_ci_hi": hi,
            "rv_nw_se": float(hac.se[0]),
            "nw_lags": int(hac.lags),
            "mean_abs_ret": float(sub["abs_ret"].mean()),
        }
        for lag in lags:
            row[f"AC{lag}"] = paired_autocorr(df, m, lag)
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_contrast(panel: pd.DataFrame, mode: str | None = None) -> dict:
    """The top-minus-bottom quintile difference in realized volatility.

    This is the number the hypothesis actually rides on, so it gets an error
    bar rather than two overlapping dots a reader has to eyeball. The
    prediction is a NEGATIVE difference: the most long-gamma quintile should
    be the calmest. STATUS.md section 6 reports the opposite sign.

    The block bootstrap is the headline interval, drawing whole days and
    applying each draw to both quintiles at once. The HAC figures are
    reported next to it and are known to be too tight for the reason given in
    `quintile_sort` -- quoting them alone would overstate the result.
    """
    mode = mode or config.get("h2_volatility.detrend.baseline")
    n_q = config.get("h2_volatility.n_quantiles")
    df = panel.dropna(subset=["gex_flow", "realized_vol"]).copy()
    df["gex_sort"] = detrend_gex(df, mode)
    df = df.dropna(subset=["gex_sort"])
    df["q"] = pd.qcut(df["gex_sort"], n_q, labels=False, duplicates="drop") + 1
    df["day"] = df.index.normalize()

    top = df[df["q"] == df["q"].max()]
    bot = df[df["q"] == df["q"].min()]
    hi, lo = top["realized_vol"].to_numpy(), bot["realized_vol"].to_numpy()

    res = nw_diff(hi, lo)
    point, boot_lo, boot_hi = block_bootstrap_contrast(
        hi, top["day"].to_numpy(), lo, bot["day"].to_numpy())

    out = res.get("diff")
    out["detrend"] = mode
    out["n_obs"] = res.n_obs
    out["lags"] = res.lags
    out["boot_lo"] = boot_lo
    out["boot_hi"] = boot_hi
    # The bootstrap interval is the one that decides the call.
    out["boot_excludes_zero"] = bool(np.isfinite(boot_lo) and (boot_lo > 0 or boot_hi < 0))
    out["point"] = point
    return out


def plots(tab: pd.DataFrame, out_dir, mode: str = "") -> list:
    paths = []
    lags = config.get("h2_volatility.autocorr_lags")
    tag = f"_{mode}" if mode else ""
    note = {"raw": "RAW LEVEL -- CONFOUNDED with calendar time",
            "rolling_z": "trailing-60d z-score (backward-looking)",
            "within_month": "within-month z-score",
            "within_year": "within-year z-score"}.get(mode, mode)

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.vstack([tab["realized_vol"] - tab["rv_ci_lo"],
                      tab["rv_ci_hi"] - tab["realized_vol"]])
    ax.errorbar(tab["quintile"], tab["realized_vol"], yerr=yerr, fmt="o-",
                capsize=5, color="#1f4e79", lw=1.5)
    ax.set_xlabel("GEX_flow quintile  (1 = most short gamma, 5 = most long)")
    ax.set_ylabel("realized volatility (trailing, 1h of 1m returns)")
    ax.set_title(f"H2 sort: realized volatility by dealer-gamma quintile\n"
                 f"sorted on {note}  |  prediction: DECREASING left to right",
                 fontsize=10)
    ax.set_xticks(tab["quintile"])
    ax.grid(alpha=.3)
    fig.tight_layout()
    p = out_dir / f"h2_quintile_realized_vol{tag}.png"
    fig.savefig(p, dpi=150); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lag, c in zip(lags, ["#1f4e79", "#c00000"]):
        ax.plot(tab["quintile"], tab[f"AC{lag}"], "o-", color=c, label=f"AC({lag})", lw=1.5)
    ax.axhline(0, color="grey", lw=.8, ls="--")
    ax.set_xlabel("GEX_flow quintile  (1 = most short gamma, 5 = most long)")
    ax.set_ylabel("autocorrelation of 5-minute returns")
    ax.set_title(f"H2 sort: return autocorrelation by dealer-gamma quintile\n"
                 f"sorted on {note}  |  prediction: positive at low GEX, negative at high",
                 fontsize=10)
    ax.set_xticks(tab["quintile"]); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = out_dir / f"h2_quintile_autocorr{tag}.png"
    fig.savefig(p, dpi=150); paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="H2 stylized fact: quintile sort")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--detrend", default=None,
                    help=f"override h2_volatility.detrend.baseline {DETREND_MODES}")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")
    panel = pd.read_parquet(config.path("paths.processed") / f"panel_{start}_{end}.parquet")

    baseline = args.detrend or config.get("h2_volatility.detrend.baseline")
    modes = [baseline] + [m for m in DETREND_MODES if m != baseline]
    out = config.path("paths.tables"); out.mkdir(parents=True, exist_ok=True)

    tabs, contrasts = [], []
    for mode in modes:
        tab = quintile_sort(panel, mode)
        con = extreme_contrast(panel, mode)
        tabs.append(tab); contrasts.append(con)

        head = "HEADLINE" if mode == baseline else "robustness"
        print(f"\n=== detrend = {mode}   [{head}] ===")
        if mode == "raw":
            print("    NOTE: sorts calendar time as much as gamma -- see the module docstring.")
        print(tab.drop(columns=["detrend"]).to_string(index=False))
        for p in plots(tab, config.path("paths.figures"), mode):
            print(f"  plot -> {p}")

    pd.concat(tabs, ignore_index=True).to_csv(out / "h2_quintile_sort.csv", index=False)

    cf = pd.DataFrame(contrasts)
    cols = [c for c in ("detrend", "point", "boot_lo", "boot_hi",
                        "boot_excludes_zero", "se", "n_obs") if c in cf.columns]
    cf = cf[cols]
    cf.to_csv(out / "h2_extreme_contrast.csv", index=False)

    print("\n=== top-minus-bottom quintile realized vol (prediction: NEGATIVE) ===")
    print(cf.to_string(index=False))
    b = cf[cf["detrend"] == baseline].iloc[0]
    verdict = ("REJECTS the null" if b["boot_excludes_zero"] else
               "cannot reject the null -- no detectable relation")
    sign = "predicted (long gamma calmer)" if b["point"] < 0 else "OPPOSITE to prediction"
    print(f"\nHEADLINE ({baseline}): {b['point']:+.6f} "
          f"[{b['boot_lo']:+.6f}, {b['boot_hi']:+.6f}] -> {verdict}; sign is {sign}.")


if __name__ == "__main__":
    main()
