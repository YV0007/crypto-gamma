"""H4 -- did the gamma effect strengthen as the market institutionalised?

The spot ETFs launched 2024-01-11. If dealer hedging drives the mechanism, the
effect should be stronger afterwards: the option book grew, dealer inventories
grew, and the hedging flow they force grew with them.

WHAT IS TESTED, AND ON WHICH RESULT
-----------------------------------
CONTEXT.md frames H4 as a structural break in H2's beta. That is run below, but
it is the weaker of the two available targets: the corrected H2 effect is small
and rests on a single year (STATUS.md section 6), so a break test on it has
little to detect either way.

The sharper target is H3, which is the result that actually holds. Its
difference-in-differences is estimated separately either side of the launch, so
the question becomes: is the DIRECTLY OBSERVED hedging response larger after
the ETFs than before? That is a real test of the same hypothesis on a statistic
that has power.

Both are reported. Neither uses open interest.

NO LOOKAHEAD: the breakpoint is a published calendar date, and every input is
the same backward-looking series the other hypotheses use.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.analysis.h2_volatility import detrend_gex
from src.analysis.h3_perpflow import did_statistic, load_inputs, mark_events, response_windows
from src.utils.stats import hac_ols

log = logging.getLogger("analysis.h4")


def _beta(panel: pd.DataFrame, mode: str) -> dict:
    """HAC slope of realized vol on detrended GEX. H2's beta, one number."""
    d = panel.dropna(subset=["gex_flow", "realized_vol"]).copy()
    d["z"] = detrend_gex(d, mode)
    d = d.dropna(subset=["z"])
    if len(d) < 500:
        return {"coef": np.nan, "se": np.nan, "p_value": np.nan, "n_obs": len(d)}
    r = hac_ols(d["realized_vol"].to_numpy(), d[["z"]].to_numpy(), names=["z"])
    out = r.get("z")
    out["n_obs"] = r.n_obs
    return out


def h2_break(panel: pd.DataFrame, brk: pd.Timestamp, mode: str) -> dict:
    """H2's beta before and after the breakpoint, plus a rolling path."""
    pre, post = panel[panel.index < brk], panel[panel.index >= brk]
    res = {"pre": _beta(pre, mode), "post": _beta(post, mode)}

    # Difference of two independent HAC estimates
    a, b = res["pre"], res["post"]
    if np.isfinite(a["se"]) and np.isfinite(b["se"]):
        diff = b["coef"] - a["coef"]
        se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
        z = diff / se if se > 0 else np.nan
        from scipy.stats import norm
        res["difference"] = {"coef": float(diff), "se": se, "z": float(z),
                             "p_value": float(2 * (1 - norm.cdf(abs(z))))}

    win = config.get("h4_structural.rolling_window_days")
    path = {}
    for (per, sub) in panel.groupby(panel.index.to_period("Q")):
        r = _beta(sub, mode)
        if np.isfinite(r["coef"]):
            path[str(per)] = {"coef": r["coef"], "se": r["se"], "n_obs": r["n_obs"]}
    res["by_quarter"] = path
    res["rolling_window_days"] = win
    return res


def h3_break(start: str, end: str, brk: pd.Timestamp) -> dict:
    """The H3 difference-in-differences, estimated either side of the break."""
    freq = config.get("h3_perpflow.return_interval")
    reps = config.get("h3_perpflow.bootstrap_reps")
    df = response_windows(mark_events(load_inputs(start, end, freq)), freq)
    df["day"] = df.index.normalize()
    ev = df[df["is_event"] & df["gex"].notna()].copy()

    out = {}
    for metric in ("ofi_5m", "ofi_30m"):
        pre = ev[ev.index < brk]
        post = ev[ev.index >= brk]
        out[metric] = {
            "pre_etf": did_statistic(pre, metric, reps),
            "post_etf": did_statistic(post, metric, reps),
        }
        log.info("%s  pre n=%d  post n=%d", metric, len(pre), len(post))
    return out


def run(start: str, end: str) -> dict:
    brk = pd.Timestamp(config.get("h4_structural.chow_breakpoint"), tz="UTC")
    mode = config.get("h2_volatility.detrend.baseline")
    panel = pd.read_parquet(config.path("paths.processed") / f"panel_{start}_{end}.parquet")
    log.info("breakpoint %s (spot ETF launch); H2 detrend=%s", brk.date(), mode)
    return {
        "breakpoint": str(brk.date()),
        "h2_detrend": mode,
        "h2_beta_break": h2_break(panel, brk, mode),
        "h3_did_break": h3_break(start, end, brk),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="H4: structural break at the ETF launch")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")
    res = run(start, end)
    dest = config.path("paths.tables") / "h4_results.json"
    dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
