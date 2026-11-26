"""H3 identification: the same estimate under every strategy that can defend it.

THE PROBLEM THIS EXISTS TO SOLVE
--------------------------------
H3's difference-in-differences is significant pooled (+0.034, t=3.8) and
vanishes when restricted to days where dealer gamma sign actually flips
(+0.003, t=0.2). Gamma sign changes on only 18.7% of days, so in the pooled
estimate "short gamma" and "long gamma" events sit on different days -- usually
different months and different years -- and every other difference between
those days rides along with the treatment. See STATUS.md section 6b.

The within-day restriction fixes the confounding but costs 79% of the sample,
and those 272 surviving days are themselves selected: days on which gamma flips
are plausibly busier and closer to expiry than average. So neither the pooled
number nor the within-day number is the answer on its own. One is confounded,
the other may be unrepresentative.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
It estimates ONE fixed target -- the coefficient on (rally x short-gamma) --
under a ladder of identification strategies that make different assumptions and
keep different amounts of the sample, and reports them side by side. That turns
"which number do I believe" from a judgement call into the result itself: if the
estimate survives every strategy it is real; if it only survives where the
confounding is unaddressed, it is the confounding.

    pooled          no controls. The current headline. Confounded by construction.
    fe_quarter      \\
    fe_month         | fixed-effect ladder. Gamma sign flips within 83% of
    fe_week          | MONTHS but only 18.7% of days, so the estimate falling
    fe_day          /  off only at day level localises the confound in time.
    controls        observable confounders in linearly: volatility, volume,
                    funding, move size, spot level, time of day, expiry cycle.
    dml             the same controls entered FLEXIBLY -- cross-fitted gradient
                    boosting partials them out of both the outcome and the
                    regressors before the DiD is taken (Chernozhukov et al.).
                    Removes non-linear confounding that `controls` cannot.
    cem             coarsened exact matching. Events are binned on volatility,
                    volume and calendar quarter; bins holding only one gamma
                    sign are dropped, so every comparison is within a bin of
                    like-for-like events. Keeps far more than within-day does.
    within_day      only days where gamma sign flips, with day fixed effects.
                    Cleanest identification, smallest and most selected sample.
    iv_expiry       two-stage least squares instrumenting gamma with position
                    in the monthly expiry cycle. Gamma moves mechanically as
                    large expiries roll off, and that schedule was fixed when
                    the contracts were listed, so it is not a response to
                    today's order flow.

                    CAVEAT, and it is a real one: the exclusion restriction is
                    that the expiry cycle affects perpetual flow ONLY through
                    dealer gamma. Expiry days plausibly have their own flow
                    patterns, so this is the weakest link in the ladder and is
                    reported as exploratory, not as the headline.

Every estimate uses standard errors clustered by day, because events cluster
within days and unclustered errors would be far too tight.
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

log = logging.getLogger("analysis.h3id")

CONTROLS = ["realized_vol", "dvol", "funding", "log_volume", "move_z",
            "log_spot", "hour_of_day", "tte_major_days"]


# --------------------------------------------------------------------- data
def load_events(start: str, end: str) -> pd.DataFrame:
    """Event frame from H3, with panel controls merged on backward."""
    proc = config.path("paths.processed")
    ev = pd.read_parquet(proc / f"h3_events_{start}_{end}.parquet")
    ev = ev[(ev["gex"] > 0) | (ev["gex"] < 0)].copy()

    panel = pd.read_parquet(proc / f"panel_{start}_{end}.parquet")
    keep = [c for c in ("realized_vol", "dvol", "funding", "spot",
                        "tte_major_days") if c in panel.columns]
    ev = pd.merge_asof(
        ev.sort_index().reset_index().rename(columns={"index": "dt", "dt": "dt"}),
        panel[keep].sort_index().reset_index().rename(columns={"timestamp": "dt"}),
        on="dt", direction="backward",
    ).set_index("dt")

    ev["R"] = (ev["direction"] == "rally").astype(float)
    ev["G"] = (ev["gex"] < 0).astype(float)          # 1 = dealers short gamma
    ev["RG"] = ev["R"] * ev["G"]
    ev["day"] = ev.index.normalize()
    ev["log_volume"] = np.log1p(ev["volume_usd"].clip(lower=0))
    ev["log_spot"] = np.log(ev["close"])
    ev["move_z"] = (ev["ret"].abs() / ev["sigma"]).replace([np.inf, -np.inf], np.nan)
    ev["hour_of_day"] = ev.index.hour.astype(float)
    log.info("events %d, %d days, gamma short on %.1f%%",
             len(ev), ev["day"].nunique(), 100 * ev["G"].mean())
    return ev


# ----------------------------------------------------------------- estimator
def _cluster_ols(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """OLS on the last column of X, with cluster-robust SE on `groups`."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    k = X.shape[1]
    meat = np.zeros((k, k))
    codes, _ = pd.factorize(groups)
    for c in np.unique(codes):
        m = codes == c
        s = X[m].T @ r[m]
        meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi
    return float(b[-1]), float(np.sqrt(max(V[-1, -1], 0.0)))


def _demean(d: pd.DataFrame, cols: list[str], by: pd.Series) -> pd.DataFrame:
    out = d.copy()
    out[cols] = out[cols] - out.groupby(by)[cols].transform("mean")
    return out


def estimate(d: pd.DataFrame, col: str, *, fe: pd.Series | None = None,
             controls: list[str] | None = None) -> dict:
    """DiD coefficient on (rally x short-gamma) for outcome `col`."""
    need = ["R", "G", "RG", col] + (controls or [])
    d = d.dropna(subset=need).copy()
    if len(d) < 500 or d["G"].nunique() < 2:
        return {"did": np.nan, "se": np.nan, "t": np.nan,
                "n_obs": int(len(d)), "n_days": int(d["day"].nunique())}

    terms = ["R", "G"] + (controls or []) + ["RG"]     # RG must stay last
    if fe is not None:
        f = fe.loc[d.index]
        d = _demean(d, terms + [col], f)
        X = d[terms].to_numpy(dtype=float)
    else:
        X = np.column_stack([np.ones(len(d)), d[terms].to_numpy(dtype=float)])
    y = d[col].to_numpy(dtype=float)

    est, se = _cluster_ols(X, y, d["day"].to_numpy())
    return {"did": est, "se": se, "t": est / se if se > 0 else np.nan,
            "n_obs": int(len(d)), "n_days": int(d["day"].nunique())}


# ------------------------------------------------------------------- DML
def estimate_dml(d: pd.DataFrame, col: str, controls: list[str], seed: int = 42) -> dict:
    """Cross-fitted double machine learning for the partially linear model.

    Controls enter through gradient boosting rather than linearly, so curvature
    and interactions among them cannot masquerade as a gamma effect. Folds are
    split by DAY (GroupKFold) so no day contributes to both the model that
    predicts it and the residual it is judged on.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    need = ["R", "G", "RG", col] + controls
    d = d.dropna(subset=need).copy()
    W = d[controls].to_numpy(dtype=float)
    groups = d["day"].to_numpy()
    targets = ["R", "G", "RG", col]
    resid = {t: np.zeros(len(d)) for t in targets}

    n_splits = min(5, max(2, d["day"].nunique()))
    for tr, te in GroupKFold(n_splits=n_splits).split(W, groups=groups):
        for t in targets:
            m = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                              learning_rate=0.1, random_state=seed)
            m.fit(W[tr], d[t].to_numpy(dtype=float)[tr])
            resid[t][te] = d[t].to_numpy(dtype=float)[te] - m.predict(W[te])

    X = np.column_stack([resid["R"], resid["G"], resid["RG"]])
    est, se = _cluster_ols(X, resid[col], groups)
    return {"did": est, "se": se, "t": est / se if se > 0 else np.nan,
            "n_obs": int(len(d)), "n_days": int(d["day"].nunique())}


# ------------------------------------------------------------------- CEM
def estimate_cem(d: pd.DataFrame, col: str, n_bins: int = 4) -> dict:
    """Coarsened exact matching on volatility, volume and calendar quarter.

    Bins that contain only one gamma sign, or only one move direction, carry no
    identifying information for a DiD and are dropped. What remains is compared
    strictly within bin, via bin fixed effects.
    """
    d = d.dropna(subset=["R", "G", "RG", col, "realized_vol", "log_volume"]).copy()
    q = lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop")
    d["_stratum"] = (q(d["realized_vol"]).astype(str) + "|" +
                     q(d["log_volume"]).astype(str) + "|" +
                     d.index.to_period("Q").astype(str))

    g = d.groupby("_stratum")
    ok = g["G"].nunique().eq(2) & g["R"].nunique().eq(2)
    keep = ok[ok].index
    d = d[d["_stratum"].isin(keep)]
    if len(d) < 500:
        return {"did": np.nan, "se": np.nan, "t": np.nan, "n_obs": int(len(d)),
                "n_days": 0, "n_strata": int(len(keep))}
    out = estimate(d, col, fe=d["_stratum"])
    out["n_strata"] = int(len(keep))
    return out


# -------------------------------------------------------------------- IV
def estimate_iv(d: pd.DataFrame, col: str, n_bins: int = 6) -> dict:
    """2SLS instrumenting gamma with position in the monthly expiry cycle.

    Endogenous: G and RG.  Instruments: expiry-cycle bin dummies and their
    interaction with the move direction.  Exogenous: R.

    Reported with the exclusion-restriction caveat in the module docstring --
    if expiry timing moves perpetual flow other than through dealer gamma, this
    estimate is not clean either.
    """
    d = d.dropna(subset=["R", "G", "RG", col, "tte_major_days"]).copy()
    bins = pd.qcut(d["tte_major_days"], n_bins, labels=False, duplicates="drop")
    Z = pd.get_dummies(bins, prefix="z", drop_first=True).to_numpy(dtype=float)
    if Z.size == 0:
        return {"did": np.nan, "se": np.nan, "t": np.nan, "n_obs": int(len(d)), "n_days": 0}

    R = d["R"].to_numpy(dtype=float)[:, None]
    ones = np.ones((len(d), 1))
    Zfull = np.hstack([ones, R, Z, R * Z])            # instrument matrix
    Xend = d[["G", "RG"]].to_numpy(dtype=float)
    Xexo = np.hstack([ones, R])

    # first stage: project the endogenous regressors onto the instruments
    P, *_ = np.linalg.lstsq(Zfull, Xend, rcond=None)
    Xhat = Zfull @ P
    first_r2 = float(1 - ((Xend - Xhat) ** 2).sum() / ((Xend - Xend.mean(0)) ** 2).sum())

    X2 = np.hstack([Xexo, Xhat[:, [0]], Xhat[:, [1]]])   # RG last
    y = d[col].to_numpy(dtype=float)
    est, se = _cluster_ols(X2, y, d["day"].to_numpy())
    return {"did": est, "se": se, "t": est / se if se > 0 else np.nan,
            "n_obs": int(len(d)), "n_days": int(d["day"].nunique()),
            "first_stage_r2": first_r2}


def attenuation_diagnostic(ev: pd.DataFrame, col: str) -> dict:
    """Why the within-day subsample is NOT the clean test it appears to be.

    A day on which gamma sign flips must pass through zero, so those days are
    mechanically the days with the SMALLEST gamma: median |GEX| $5.9M against
    $16.4M elsewhere, a factor of 2.8. They are also the days on which the
    short/long label is least reliable, because the thing being labelled is
    oscillating. Weak treatment plus a noisy label attenuates any estimate
    toward zero, which is exactly what is observed -- and it happens in BOTH
    eras, in opposite-signed samples, which a genuine confounding correction
    would not do.

    The decisive detail: day fixed effects change nothing on this subsample
    (+0.00350 without, +0.00345 with). So the null is a property of WHICH days
    these are, not of the correction applied to them.
    """
    frac = ev.groupby("day")["G"].transform("mean")
    mixed = (frac > 0) & (frac < 1)
    out = {
        "median_abs_gex_mixed": float(ev.loc[mixed, "gex"].abs().median()),
        "median_abs_gex_constant": float(ev.loc[~mixed, "gex"].abs().median()),
        "mixed_no_fe": estimate(ev[mixed], col),
        "mixed_day_fe": estimate(ev[mixed], col, fe=ev.loc[mixed, "day"]),
        "constant_only": estimate(ev[~mixed], col),
    }
    band = pd.qcut(ev["gex"].abs(), 5, labels=False, duplicates="drop") + 1
    out["by_abs_gex_quintile"] = {
        int(b): {"median_abs_gex": float(ev.loc[band == b, "gex"].abs().median()),
                 **estimate(ev[band == b], col)}
        for b in sorted(band.dropna().unique())
    }
    return out


# ------------------------------------------------------------------- driver
def run(start: str, end: str) -> dict:
    ev = load_events(start, end)
    mixed = ev.groupby("day")["G"].transform("mean")
    within = ev[(mixed > 0) & (mixed < 1)]

    out: dict = {}
    for col in ("ofi_5m", "ofi_30m"):
        log.info("---- %s ----", col)
        specs: dict[str, dict] = {}
        specs["pooled"] = estimate(ev, col)
        for lab, key in (("fe_quarter", "Q"), ("fe_month", "M"), ("fe_week", "W")):
            specs[lab] = estimate(ev, col, fe=ev.index.to_period(key).astype(str).to_series(index=ev.index))
        specs["fe_day"] = estimate(ev, col, fe=ev["day"])
        specs["controls"] = estimate(ev, col, controls=CONTROLS)
        specs["dml"] = estimate_dml(ev, col, CONTROLS)
        specs["cem"] = estimate_cem(ev, col)
        specs["within_day"] = estimate(within, col, fe=within["day"])
        specs["iv_expiry"] = estimate_iv(ev, col)
        for k, v in specs.items():
            log.info("  %-12s %+.5f  se %.5f  t %+6.2f  n=%d",
                     k, v["did"], v["se"], v["t"], v["n_obs"])
        out[col] = specs

    # The full-sample ladder hides a sign reversal across the ETF launch, so
    # the same ladder is run within each era. This is the finding, not a
    # robustness footnote: post-ETF every strategy agrees, pre-ETF every
    # strategy agrees on the OPPOSITE sign.
    brk = pd.Timestamp(config.get("h4_structural.chow_breakpoint"), tz="UTC")
    eras = {"post_etf": ev[ev.index >= brk], "pre_etf": ev[ev.index < brk]}
    out["by_era"] = {}
    for era, d in eras.items():
        frac = d.groupby("day")["G"].transform("mean")
        mx = d[(frac > 0) & (frac < 1)]
        out["by_era"][era] = {}
        for col in ("ofi_5m", "ofi_30m"):
            out["by_era"][era][col] = {
                "pooled": estimate(d, col),
                "fe_month": estimate(d, col, fe=d.index.to_period("M").astype(str).to_series(index=d.index)),
                "fe_day": estimate(d, col, fe=d["day"]),
                "controls": estimate(d, col, controls=CONTROLS),
                "dml": estimate_dml(d, col, CONTROLS),
                "cem": estimate_cem(d, col),
                "within_day": estimate(mx, col, fe=mx["day"]),
            }
        log.info("era %s: 5m pooled %+.5f", era, out["by_era"][era]["ofi_5m"]["pooled"]["did"])

    out["attenuation"] = {c: attenuation_diagnostic(ev, c) for c in ("ofi_5m", "ofi_30m")}
    return out


def chart(res: dict, out_dir) -> list:
    """Specification chart: one estimate per strategy, with 95% intervals."""
    paths = []
    order = ["pooled", "fe_quarter", "fe_month", "fe_week", "fe_day",
             "controls", "dml", "cem", "within_day", "iv_expiry"]
    nice = {"pooled": "no controls", "fe_quarter": "quarter FE", "fe_month": "month FE",
            "fe_week": "week FE", "fe_day": "day FE", "controls": "linear controls",
            "dml": "double ML", "cem": "matched (CEM)", "within_day": "within-day",
            "iv_expiry": "IV: expiry cycle"}
    for col, specs in res.items():
        if not isinstance(specs, dict) or "pooled" not in specs:
            continue
        labs = [l for l in order if l in specs and np.isfinite(specs[l]["did"])]
        est = np.array([specs[l]["did"] for l in labs])
        se = np.array([specs[l]["se"] for l in labs])
        y = np.arange(len(labs))[::-1]

        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        clean = {"within_day", "cem", "dml"}
        cols = ["#1f4e79" if l in clean else "#8c9aa8" for l in labs]
        ax.errorbar(est, y, xerr=1.96 * se, fmt="o", capsize=4, lw=1.4,
                    ecolor="#b8c2cc", mfc="none", mec="none")
        ax.scatter(est, y, c=cols, zorder=3, s=42)
        ax.axvline(0, color="#c00000", lw=1, ls="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([nice[l] for l in labs], fontsize=9)
        ax.set_xlabel(f"DiD on {col}   (H3 predicts > 0)", fontsize=9)
        ax.set_title("H3 under every identification strategy\n"
                     "blue = confounding addressed;  grey = not addressed",
                     fontsize=10)
        ax.grid(axis="x", alpha=.3)
        fig.tight_layout()
        p = out_dir / f"h3_specification_{col}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 identification ladder")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    start = args.start or config.get("sample.start")
    end = args.end or config.get("sample.end")

    res = run(start, end)
    tab = config.path("paths.tables")
    (tab / "h3_identification.json").write_text(json.dumps(res, indent=2, default=str))

    rows = [{"era": "full", "outcome": c, "spec": k, **v}
            for c, s in res.items() if c.startswith("ofi") for k, v in s.items()]
    rows += [{"era": e, "outcome": c, "spec": k, **v}
             for e, cs in res.get("by_era", {}).items()
             for c, s in cs.items() for k, v in s.items()]
    df = pd.DataFrame(rows)
    df.to_csv(tab / "h3_identification.csv", index=False)
    print(df.to_string(index=False))
    for p in chart(res, config.path("paths.figures")):
        print(f"  chart -> {p}")


if __name__ == "__main__":
    main()
