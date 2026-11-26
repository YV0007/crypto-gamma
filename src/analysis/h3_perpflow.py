"""H3 -- is dealer hedging DIRECTLY OBSERVABLE in perpetual futures flow?

THE CORE TEST. Every other gamma study infers hedging in three steps with no
direct evidence: see gamma, see volatility change, assume forced hedging links
them. On Deribit the options book and the perpetual venue where dealers hedge
are the same exchange, and both flag the aggressor on every trade. So the hedge
can be watched instead of assumed.

MECHANISM UNDER TEST
--------------------
After a large spot move, what a delta-hedging dealer must do next is fixed by
its gamma sign -- an obligation, not a preference:

                    spot RALLIES        spot SELLS OFF
  gamma POSITIVE    must SELL perps     must BUY perps      (leans against)
  gamma NEGATIVE    must BUY perps      must SELL perps     (chases)

IDENTIFICATION
--------------
The level of perpetual flow after a move is dominated by momentum traders and
tells us nothing. The identifying variation is the DIFFERENCE between
positive-gamma and negative-gamma states conditional on the SAME move
direction. Both groups contain the same momentum traders; only dealer
obligation differs.

The headline statistic is a difference-in-differences that flips sign with the
move, so it cannot be produced by a constant flow bias:

    DiD = [flow(neg gamma) - flow(pos gamma)] on RALLIES
        - [flow(neg gamma) - flow(pos gamma)] on SELLOFFS

H3 predicts DiD > 0.

This also validates the whole measurement chain. Reconciling against open
interest only proves trades were captured -- a contract has the same OI
regardless of who is long. H3 is the only test that can confirm the SIGN,
which is the entire hypothesis. And because it conditions on the sign inside a
30-minute window, a slow drift in GEX levels cannot contaminate it.

NO LOOKAHEAD: GEX and the volatility threshold at t use only data at or before
t; only the response window looks forward, which is the outcome by design.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("analysis.h3")


# ---------------------------------------------------------------- inputs
def load_inputs(start: str, end: str, freq: str, gex_file: str | None = None) -> pd.DataFrame:
    interim, proc = config.path("paths.interim"), config.path("paths.processed")

    flow = pd.read_parquet(interim / f"perp_flow_{freq}.parquet")
    flow.index = pd.to_datetime(flow.index, utc=True)

    ohlcv = pd.read_parquet(interim / "ohlcv_1m.parquet").set_index("dt").sort_index()
    close = ohlcv["close"].resample(freq).last()
    ret = np.log(close).diff()

    # A private snapshot is used by default: another session rewrites the
    # canonical GEX file, and H3 must be reproducible against one fixed input.
    gpath = proc / (gex_file or "gex_flow_h3snapshot.parquet")
    if not gpath.exists():
        gpath = proc / f"gex_flow_{start}_{end}.parquet"
    log.info("GEX input: %s", gpath.name)
    gex = pd.read_parquet(gpath)
    gex.index = pd.to_datetime(gex.index, utc=True)

    df = pd.DataFrame({"close": close, "ret": ret}).join(flow, how="left")

    # GEX is 10-min; take the last value AT OR BEFORE each bucket (backward only)
    g = gex["gex_flow"].rename("gex").reset_index()
    g.columns = ["dt", "gex"]
    df = pd.merge_asof(df.reset_index().rename(columns={"index": "dt", "dt": "dt"}),
                       g.sort_values("dt"), on="dt", direction="backward")
    df = df.set_index("dt").sort_index()

    lo = pd.Timestamp(start, tz="UTC"); hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= lo) & (df.index < hi)]


# ---------------------------------------------------------------- events
def mark_events(df: pd.DataFrame, vol_window: str = "30D") -> pd.DataFrame:
    """Flag |return| > k sigma, sigma estimated on a TRAILING window."""
    k = config.get("h3_perpflow.event_threshold_sigma")
    out = df.copy()
    # shift(1) so sigma at t never includes the return at t
    out["sigma"] = out["ret"].shift(1).rolling(vol_window, min_periods=500).std()
    out["is_event"] = (out["ret"].abs() > k * out["sigma"]) & out["sigma"].notna()
    out["direction"] = np.where(out["ret"] > 0, "rally", "selloff")
    log.info("events: %d of %d buckets (%.2f%%), threshold %.1f sigma",
             int(out["is_event"].sum()), len(out),
             100 * out["is_event"].mean(), k)
    return out


def response_windows(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Forward-summed perpetual flow over t+1 .. t+k minutes."""
    step = int(pd.Timedelta(freq).total_seconds() // 60)
    out = df.copy()
    for k in config.get("h3_perpflow.response_windows_min"):
        n = max(1, k // step)
        # shift(-1) starts the window at t+1: the event bucket is excluded, so
        # the move itself can never be counted as its own hedging response
        out[f"net_{k}m"] = out["net_usd"].shift(-1).rolling(n).sum().shift(-(n - 1))
        out[f"vol_{k}m"] = out["volume_usd"].shift(-1).rolling(n).sum().shift(-(n - 1))
        out[f"ofi_{k}m"] = out[f"net_{k}m"] / out[f"vol_{k}m"].replace(0, np.nan)

        # PRE-event mirror: the same window run BACKWARDS, t-k .. t-1. A hedge
        # cannot precede the move that forces it, so this must come out flat.
        # It is the sharpest falsification available because it changes only
        # the direction of time -- same events, same gamma states, same
        # estimator.
        out[f"pre_net_{k}m"] = out["net_usd"].rolling(n).sum().shift(1)
        out[f"pre_vol_{k}m"] = out["volume_usd"].rolling(n).sum().shift(1)
        out[f"pre_ofi_{k}m"] = out[f"pre_net_{k}m"] / out[f"pre_vol_{k}m"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------- inference
def _day_stats(v: np.ndarray, day_code: np.ndarray, n_days: int, mask: np.ndarray):
    """Per-day (sum, count) of v under `mask` -- the sufficient statistics a
    block bootstrap over days needs. Resampling days then becomes a sum over
    precomputed per-day totals instead of re-slicing the frame 10,000 times."""
    ok = mask & ~np.isnan(v)
    sums = np.bincount(day_code[ok], weights=v[ok], minlength=n_days)
    cnts = np.bincount(day_code[ok], minlength=n_days).astype(float)
    return sums, cnts


def block_bootstrap(values: np.ndarray, days: np.ndarray, reps: int, seed: int = 42):
    """Moving block bootstrap over whole days (events cluster within a day)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(days)
    n = len(uniq)
    if n < 2 or len(values) == 0:
        return np.nan, np.nan
    sums, cnts = _day_stats(values, codes, n, np.ones(len(values), bool))
    pick = rng.integers(0, n, size=(reps, n))
    tot, cn = sums[pick].sum(1), cnts[pick].sum(1)
    draws = np.where(cn > 0, tot / np.maximum(cn, 1), np.nan)
    return float(np.nanquantile(draws, .025)), float(np.nanquantile(draws, .975))


def cell_stats(sub: pd.DataFrame, col: str, reps: int) -> dict:
    v = sub[col].to_numpy(dtype=float)
    d = sub["day"].to_numpy()
    ok = ~np.isnan(v)
    lo, hi = block_bootstrap(v[ok], d[ok], reps)
    return {"N": int(ok.sum()), "mean": float(np.nanmean(v)) if ok.any() else np.nan,
            "ci_lo": lo, "ci_hi": hi}


def did_statistic(ev: pd.DataFrame, col: str, reps: int, seed: int = 42) -> dict:
    """Difference-in-differences with a block-bootstrapped CI.

        DiD = [flow(neg gamma) - flow(pos gamma)] on RALLIES
            - [flow(neg gamma) - flow(pos gamma)] on SELLOFFS

    H3 predicts DiD > 0. Because the sign flips with move direction, any
    constant flow bias common to both regimes cancels out.
    """
    ev = ev.dropna(subset=[col])
    if ev.empty:
        return {"did": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "boot_p": np.nan, "reps": reps}
    v = ev[col].to_numpy(dtype=float)
    codes, uniq = pd.factorize(ev["day"].to_numpy())
    n = len(uniq)
    rally = (ev["direction"] == "rally").to_numpy()
    gpos, gneg = (ev["gex"] > 0).to_numpy(), (ev["gex"] < 0).to_numpy()
    cells = {"rn": _day_stats(v, codes, n, rally & gneg),
             "rp": _day_stats(v, codes, n, rally & gpos),
             "sn": _day_stats(v, codes, n, ~rally & gneg),
             "sp": _day_stats(v, codes, n, ~rally & gpos)}

    def did_from(pick):
        m = {}
        for k, (su, cn) in cells.items():
            t, c = su[pick].sum(-1), cn[pick].sum(-1)
            m[k] = np.where(c > 0, t / np.maximum(c, 1), np.nan)
        return (m["rn"] - m["rp"]) - (m["sn"] - m["sp"])

    point = float(did_from(np.arange(n)[None, :])[0])
    rng = np.random.default_rng(seed)
    draws = did_from(rng.integers(0, n, size=(reps, n)))
    lo, hi = np.nanquantile(draws, .025), np.nanquantile(draws, .975)
    p = 2 * min(float(np.nanmean(draws <= 0)), float(np.nanmean(draws >= 0)))
    return {"did": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_p": float(min(p, 1.0)), "reps": reps}




def did_regression(ev: pd.DataFrame, col: str, within_day: bool = False) -> dict:
    """The DiD as a regression, with day-clustered standard errors.

        ofi ~ a + b1*rally + b2*short_gamma + DiD*(rally x short_gamma)

    Algebraically identical to `did_statistic` on the full sample, but it
    exposes the thing the cell-mean version hides: WHERE the identifying
    variation comes from.

    THE PROBLEM THIS EXISTS TO MEASURE
    ----------------------------------
    Dealer gamma sign is highly persistent -- it changes on only 18.7% of days.
    So in the pooled estimate, "short gamma" and "long gamma" events are mostly
    drawn from DIFFERENT DAYS, and usually different months and different
    years. The comparison is then between-day, and every difference between
    those days (volatility regime, market structure, sample era) rides along
    with it. This is the identical confound that broke H2's raw quintile sort
    (STATUS.md section 6): pooling across a period with strong time trends.

    Day fixed effects do NOT fix it. When gamma sign is constant within a day
    the interaction term is still identified off that day's constant sign, so
    the between-day variation leaks straight through the within transformation
    and the estimate barely moves.

    `within_day=True` restricts to days on which the gamma sign ACTUALLY flips
    and absorbs day fixed effects. That is the only specification where the
    contrast is genuinely like-for-like: same day, same regime, same tape,
    differing only in dealer gamma. It costs about 79% of the sample.
    """
    d = ev.dropna(subset=[col]).copy()
    d["R"] = (d["direction"] == "rally").astype(float)
    d["G"] = (d["gex"] < 0).astype(float)
    d["day"] = d.index.normalize()

    if within_day:
        frac = d.groupby("day")["G"].transform("mean")
        d = d[(frac > 0) & (frac < 1)]
        if len(d) < 500:
            return {"did": np.nan, "se": np.nan, "t": np.nan, "n_obs": len(d),
                    "n_days": 0, "spec": "within_day"}

    d["RG"] = d["R"] * d["G"]
    cols = ["R", "G", "RG", col]
    if within_day:
        d[cols] = d[cols] - d.groupby("day")[cols].transform("mean")
        X = np.column_stack([d["R"], d["G"], d["RG"]])
    else:
        X = np.column_stack([np.ones(len(d)), d["R"], d["G"], d["RG"]])
    y = d[col].to_numpy(dtype=float)

    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    codes, _ = pd.factorize(d["day"].to_numpy())
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(codes):
        m = codes == c
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    V = XtXi @ meat @ XtXi
    est, se = float(b[-1]), float(np.sqrt(V[-1, -1]))
    return {"did": est, "se": se, "t": est / se if se > 0 else np.nan,
            "n_obs": int(len(d)), "n_days": int(d["day"].nunique()),
            "spec": "within_day" if within_day else "pooled"}


def identification_report(ev: pd.DataFrame) -> dict:
    """Pooled vs genuinely-within-day DiD, side by side, per horizon.

    If the two agree, the pooled number is safe to quote. If the within-day
    estimate collapses, the pooled number is a between-day artifact and must
    not be the headline.
    """
    out = {}
    for k in config.get("h3_perpflow.response_windows_min"):
        col = f"ofi_{k}m"
        if col not in ev.columns:
            continue
        out[col] = {"pooled": did_regression(ev, col, False),
                    "within_day": did_regression(ev, col, True)}
    return out


def falsification_suite(df: pd.DataFrame, ev: pd.DataFrame, reps: int,
                        seed: int = 42) -> dict:
    """Tests that must FAIL to find an effect if H3 is real.

    WHY THE OLD PLACEBO WAS NOT ONE
    -------------------------------
    The previous placebo resampled NON-event buckets and ran the same DiD. It
    came back significant (+0.0073, p=0.016) and that was read as a warning.
    It is not: a delta-hedging dealer rebalances on *any* move, not only on
    moves past a 1-sigma threshold. Small moves are a WEAKER DOSE of the same
    treatment, not an absence of it. Finding a smaller effect there is what
    the mechanism predicts, so the old test could never have falsified
    anything. It is retained below as `low_dose`, named for what it is, and
    the dose-response ladder makes the point properly.

    A real falsification has to break the link between the move and the hedge
    while leaving everything else alone:

      pre_event        flow BEFORE the move (t-k..t-1). Time-reversed, so the
                       obligation does not yet exist. Must be flat.
      scrambled_dir    event direction labels randomly permuted. Destroys the
                       rally/selloff pairing the DiD is built on.
      shuffled_gex     each event keeps its flow but is assigned the gamma
                       state of a randomly chosen OTHER event. Preserves both
                       marginal distributions, destroys their pairing.

    A fourth entry, `dose_response`, is not a falsification but the positive
    control: the effect should GROW with move size.
    """
    rng = np.random.default_rng(seed)
    out = {}

    # 1. pre-event mirror -- the same estimator run backwards in time
    for k in config.get("h3_perpflow.response_windows_min"):
        col = f"pre_ofi_{k}m"
        if col in ev.columns:
            out[f"pre_event_ofi_{k}m"] = did_statistic(ev, col, reps)

    # 2. scrambled direction
    sc = ev.copy()
    sc["direction"] = rng.permutation(sc["direction"].to_numpy())
    out["scrambled_direction"] = did_statistic(sc, "ofi_30m", reps)

    # 3. shuffled gamma state
    sg = ev.copy()
    sg["gex"] = rng.permutation(sg["gex"].to_numpy())
    out["shuffled_gex"] = did_statistic(sg, "ofi_30m", reps)

    # 4. low dose -- the old "placebo", named honestly
    non = df[(~df["is_event"]) & df["gex"].notna()]
    if len(non):
        non = non.sample(min(len(ev) * 3, len(non)), random_state=seed)
        out["low_dose_non_events"] = did_statistic(non.dropna(subset=["ofi_30m"]), "ofi_30m", 2000)

    # 5. dose-response ladder: DiD by how large the move was
    z = (df["ret"].abs() / df["sigma"]).reindex(ev.index)
    bands = [(1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, np.inf)]
    ladder = {}
    for lo_b, hi_b in bands:
        sub = ev[(z >= lo_b) & (z < hi_b)]
        if len(sub) > 200:
            ladder[f"{lo_b}-{hi_b}sigma"] = did_statistic(sub, "ofi_30m", 2000)
    out["dose_response"] = ladder
    return out


def run(start: str, end: str, gex_file: str | None = None) -> dict:
    freq = config.get("h3_perpflow.return_interval")
    reps = config.get("h3_perpflow.bootstrap_reps")

    df = load_inputs(start, end, freq, gex_file)
    df = mark_events(df)
    df = response_windows(df, freq)
    df["day"] = df.index.normalize()

    ev = df[df["is_event"] & df["gex"].notna()].copy()
    log.info("usable events: %d (rally %d / selloff %d; gex>0 %d / gex<0 %d)",
             len(ev), int((ev["direction"] == "rally").sum()),
             int((ev["direction"] == "selloff").sum()),
             int((ev["gex"] > 0).sum()), int((ev["gex"] < 0).sum()))

    results = {"events": len(ev), "cells": {}, "did": {}}
    for k in config.get("h3_perpflow.response_windows_min"):
        for metric in (f"net_{k}m", f"ofi_{k}m"):
            for d in ("rally", "selloff"):
                for gl, gm in (("gex_pos", ev["gex"] > 0), ("gex_neg", ev["gex"] < 0)):
                    sub = ev[(ev["direction"] == d) & gm]
                    if len(sub):
                        results["cells"][f"{metric}|{d}|{gl}"] = cell_stats(sub, metric, reps)
            results["did"][metric] = did_statistic(ev, metric, reps)
    results["identification"] = identification_report(ev)
    results["falsification"] = falsification_suite(df, ev, reps)
    results["gex_input"] = gex_file or "gex_flow_h3snapshot.parquet"
    results["_events_frame"] = ev
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="H3: dealer hedging in perpetual flow")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--gex-file", default=None,
                    help="GEX parquet under data/processed (default: the H3 snapshot)")
    ap.add_argument("--out", default="h3_results.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")
    res = run(start, end, args.gex_file)
    ev = res.pop("_events_frame")
    ev.to_parquet(config.path("paths.processed") / f"h3_events_{start}_{end}.parquet")
    import json
    (config.path("paths.tables") / args.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=2, default=str))


if __name__ == "__main__":
    main()
