"""Family-wise correction across every headline test the study reports.

WHY THIS RUNS LAST
------------------
Six hypotheses, several horizons each, and a handful of construction choices
tried along the way. Any one of them at p<0.05 in isolation means much less
than it looks. Benjamini-Hochberg controls the share of false positives among
the rejections, which is the right target for an exploratory study.

Only PRIMARY tests enter the family -- the specification each hypothesis
pre-committed to. Robustness variants of the same test are not independent
hypotheses and would dilute the correction if counted (`stats.spec_log` keeps
the full trial count for the Deflated Sharpe Ratio separately).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.utils.stats import benjamini_hochberg

TAB = config.path("paths.tables")


def _p_from_ci(point: float, lo: float, hi: float) -> float:
    """Two-sided normal p implied by a bootstrap interval (se from the width)."""
    if not all(np.isfinite([point, lo, hi])) or hi <= lo:
        return np.nan
    from scipy.stats import norm
    se = (hi - lo) / (2 * 1.959963985)
    return float(2 * (1 - norm.cdf(abs(point / se)))) if se > 0 else np.nan


def collect() -> pd.DataFrame:
    rows = []

    h2 = pd.read_csv(TAB / "h2_extreme_contrast.csv")
    base = config.get("h2_volatility.detrend.baseline")
    r = h2[h2["detrend"] == base].iloc[0]
    rows.append({"hypothesis": "H2", "test": f"top-minus-bottom quintile RV ({base})",
                 "estimate": r["point"],
                 "p_value": _p_from_ci(r["point"], r["boot_lo"], r["boot_hi"])})

    # H2 also has a more powerful within-day regression form. It is reported
    # alongside the pre-committed quintile contrast rather than instead of it:
    # swapping in the stronger statistic after seeing the result is exactly
    # what this correction exists to penalise.
    wd = TAB / "h2_within_day.json"
    if wd.exists():
        w = json.loads(wd.read_text())
        from scipy.stats import norm
        for k, v in w.items():
            rows.append({"hypothesis": "H2", "test": f"within-day beta ({k})",
                         "estimate": v["beta"],
                         "p_value": float(2 * (1 - norm.cdf(abs(v["t"]))))})

    h3 = json.loads((TAB / "h3_results.json").read_text())
    # The PRIMARY H3 test is the within-day identified one. The pooled figure
    # is a between-day comparison (STATUS.md 6b) and is logged as secondary.
    for h, v in h3.get("identification", {}).items():
        w = v["within_day"]
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(w["t"])))) if np.isfinite(w["t"]) else np.nan
        rows.append({"hypothesis": "H3", "test": f"DiD {h} (within-day)",
                     "estimate": w["did"], "p_value": p})

    h4p = TAB / "h4_results.json"
    if h4p.exists():
        h4 = json.loads(h4p.read_text())
        d = h4.get("h2_beta_break", {}).get("difference")
        if d:
            rows.append({"hypothesis": "H4", "test": "H2 beta pre-vs-post ETF",
                         "estimate": d["coef"], "p_value": d["p_value"]})

    return pd.DataFrame(rows).dropna(subset=["p_value"])


def main() -> None:
    fam = collect()
    bh = benjamini_hochberg(fam["p_value"].to_numpy())
    out = pd.concat([fam.reset_index(drop=True), bh[["q_value", "reject"]]], axis=1)
    out.to_csv(TAB / "multiple_testing.csv", index=False)
    print(out.to_string(index=False))
    n = int(out["reject"].sum())
    print(f"\n{n} of {len(out)} primary tests survive BH at "
          f"{100*(1-config.get('stats.confidence_level')):.0f}% FDR.")


if __name__ == "__main__":
    main()
