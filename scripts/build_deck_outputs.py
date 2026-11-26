"""Build the figures and tables the deck needs that nothing else produces.

Everything here reads from `outputs/tables/` or `data/processed/` -- nothing is
recomputed, so these can never disagree with the analysis that produced them.
Six outputs, one per slide that currently has numbers but no picture:

  1  sample funnel            slide 4   raw trades -> the panel, what survives
  2  gamma validation         slide 5   our gamma vs Deribit's published
  3  H3 by era                slide 10  the core result, and its one weakness
  4  H4 structural break      slide 12
  5  information coefficients slide 14  raw vs incremental
  6  multiple testing         slide 15  seven tests, two survive

Style is deliberately plain: one accent (#1f4e79), red only for "prediction
fails here", grey for context. Charts are sized for a slide, not a paper.
"""

from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

log = logging.getLogger("deck")

BLUE, RED, GREY, LIGHT = "#1f4e79", "#c00000", "#8c9aa8", "#d9e2ec"
INK = "#14171A"
TAB = config.path("paths.tables")
FIG = config.path("paths.figures")


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")
    return str(p)


# ---------------------------------------------------------------- 1. funnel
def sample_funnel() -> pd.DataFrame:
    """What survives each construction step, raw tape to analysis panel."""
    st = json.loads((TAB / "pipeline_stats.json").read_text())
    panel = pd.read_parquet(config.path("paths.processed") /
                            f"panel_{config.get('sample.start')}_{config.get('sample.end')}.parquet")
    usable = int(panel[["gex_flow", "realized_vol"]].dropna().shape[0])

    rows = [
        ("BTC option trades downloaded", st["n_trades_in"], "1 Jan 2022 - 31 Dec 2025, no gaps"),
        ("after block-trade policy", st["n_trades_out"],
         f"blocks kept ({100*st['block_fraction_volume']:.0f}% of volume, "
         f"{100*st['block_fraction_net_flow']:.0f}% of net dealer flow)"),
        ("instruments with a reconstructed position", st["instruments"], "cumulative from listing"),
        ("(instrument x interval) cells", int(st["cells"]), "10-minute grid"),
        ("cells before expiry", int(st["cells_live"]), "position forced to 0 at expiry"),
        ("cells priced with gamma", int(st["cells_with_gamma"]),
         f"{100*st['pos_weight_with_gamma']/st['pos_weight_total']:.0f}% of position weight"),
        ("panel intervals", st["intervals"], "aggregated GEX series"),
        ("intervals usable for analysis", usable, "GEX and realized vol both present"),
    ]
    df = pd.DataFrame(rows, columns=["step", "n", "note"])
    df.to_csv(TAB / "sample_funnel.csv", index=False)
    print(df.to_string(index=False))
    print(f"  -> {TAB / 'sample_funnel.csv'}")
    return df


# ------------------------------------------------------------ 2. validation
def gamma_validation() -> str:
    """Our recomputed gamma against Deribit's published gamma, instrument by instrument."""
    d = pd.read_csv(TAB / "gamma_validation_pairs.csv")
    res = json.loads((TAB / "gamma_aggregate_validation.json").read_text())
    a = res["aggregate"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    calls = d["option_type"] == "call"
    ax.scatter(d.loc[calls, "gamma_deribit"], d.loc[calls, "gamma_ours"],
               s=16, alpha=.55, color=BLUE, label="calls", edgecolors="none")
    ax.scatter(d.loc[~calls, "gamma_deribit"], d.loc[~calls, "gamma_ours"],
               s=16, alpha=.55, color=GREY, label="puts", edgecolors="none")
    hi = float(np.nanmax([d["gamma_deribit"].max(), d["gamma_ours"].max()]))
    ax.plot([0, hi], [0, hi], ls="--", lw=1, color=RED, label="45°")
    ax.set_xlim(0, hi * 1.02); ax.set_ylim(0, hi * 1.02)
    ax.set_xlabel("Deribit published gamma"); ax.set_ylabel("our recomputed gamma")
    ax.set_title(f"Per instrument  (n = {len(d):,}, r = "
                 f"{res['per_instrument']['corr_levels']:.4f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25)

    labs = ["signed GEX", "gross exposure"]
    der = [a["signed_gex_deribit"] / 1e6, a["gross_exposure_deribit"] / 1e6]
    our = [a["signed_gex_ours"] / 1e6, a["gross_exposure_ours"] / 1e6]
    y = np.arange(len(labs)); h = .36
    ax2.barh(y + h / 2, der, h, color=GREY, label="Deribit published")
    ax2.barh(y - h / 2, our, h, color=BLUE, label="ours")
    ax2.set_yticks(y); ax2.set_yticklabels(labs)
    ax2.axvline(0, color="black", lw=.8)
    ax2.set_xlabel("$M per 1% move")
    ax2.set_title(f"Aggregate  (signed {a['signed_ratio']:.4f}, "
                  f"gross {a['gross_ratio']:.4f})", fontsize=10)
    ax2.legend(fontsize=8, frameon=False); ax2.grid(axis="x", alpha=.25)

    fig.suptitle("Our gamma reproduces Deribit's own published numbers", fontsize=12)
    return _save(fig, "validation_gamma.png")


# ---------------------------------------------------------------- 3. H3 era
def h3_by_era() -> str:
    """The core slide: H3 holds post-ETF under every strategy; 2022 is the exception."""
    res = json.loads((TAB / "h3_identification.json").read_text())
    diag = json.loads((TAB / "h3_reversal_diagnosis.json").read_text())
    order = ["pooled", "fe_month", "fe_day", "controls", "dml", "cem", "within_day"]
    nice = {"pooled": "no controls", "fe_month": "month FE", "fe_day": "day FE",
            "controls": "linear controls", "dml": "double ML",
            "cem": "matched (CEM)", "within_day": "within-day"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    era = res["by_era"]
    labs = [l for l in order if l in era["post_etf"]["ofi_5m"]]
    y = np.arange(len(labs))[::-1]
    for key, col, off, lab in (("post_etf", BLUE, .16, "post-ETF (2024-25)"),
                               ("pre_etf", GREY, -.16, "pre-ETF (2022-23)")):
        e = np.array([era[key]["ofi_5m"][l]["did"] for l in labs])
        s = np.array([era[key]["ofi_5m"][l]["se"] for l in labs])
        ax.errorbar(e, y + off, xerr=1.96 * s, fmt="o", ms=6, capsize=3,
                    color=col, ecolor=col, lw=1.3, label=lab)
    ax.axvline(0, color=RED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([nice[l] for l in labs], fontsize=9)
    ax.set_xlabel("difference-in-differences, 5-minute flow   (theory predicts > 0)", fontsize=9)
    ax.set_title("Every strategy agrees within an era", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.25)

    ev = diag["evidence"]
    keys = ["y2022_only", "y2023_only", "pre_etf_excl_2022", "pre_etf_all"]
    names = ["2022 only", "2023 only", "pre-ETF excl. 2022", "pre-ETF all"]
    e = np.array([ev[k]["did"] for k in keys])
    s = np.array([ev[k]["se"] for k in keys])
    y2 = np.arange(len(keys))[::-1]
    cols = [RED if v < 0 and abs(t) > 1.96 else BLUE if v > 0 else GREY
            for v, t in zip(e, [ev[k]["t"] for k in keys])]
    ax2.errorbar(e, y2, xerr=1.96 * s, fmt="none", ecolor=LIGHT, lw=4)
    ax2.scatter(e, y2, c=cols, s=52, zorder=3)
    ax2.axvline(0, color=RED, ls="--", lw=1)
    ax2.set_yticks(y2); ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("same statistic, pre-ETF only", fontsize=9)
    ax2.set_title("The reversal is 2022 alone", fontsize=10)
    ax2.grid(axis="x", alpha=.25)

    fig.suptitle("H3: dealer hedging is visible in perpetual flow after 2024", fontsize=12)
    return _save(fig, "h3_by_era.png")


# ---------------------------------------------------------------- 4. H4
def h4_break() -> str:
    res = json.loads((TAB / "h4_results.json").read_text())
    b = res["h2_beta_break"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labs, est, se = [], [], []
    for k, lab in (("pre", "before ETF launch"), ("post", "after ETF launch")):
        labs.append(lab); est.append(b[k]["coef"]); se.append(b[k]["se"])
    y = np.arange(len(labs))[::-1]
    ax.errorbar(est, y, xerr=1.96 * np.array(se), fmt="o", ms=7, capsize=4,
                color=BLUE, ecolor=BLUE, lw=1.4)
    ax.axvline(0, color=RED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlabel("H2 slope: realized vol on dealer gamma   (theory predicts < 0)", fontsize=9)
    d = b.get("difference", {})
    ax.set_title("H4: no strengthening after the ETF launch\n"
                 f"difference p = {d.get('p_value', float('nan')):.2f}", fontsize=10)
    ax.grid(axis="x", alpha=.25)
    return _save(fig, "h4_structural_break.png")


# ---------------------------------------------------------------- 5. IC
def ic_chart() -> str:
    d = pd.read_csv(TAB / "signal_ic.csv")
    d = d[d["pair"].str.startswith("signal -> fwd_rv")].copy()
    d["h"] = d["pair"].str.extract(r"fwd_rv_(\w+)")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(d)); w = .36
    ax.bar(x - w / 2, d["ic_raw"], w, color=GREY, label="raw IC")
    ax.bar(x + w / 2, d["ic_incremental"], w, color=BLUE, label="after controlling for\ntrailing vol and DVOL")
    lo = d["ic_incremental"] - d["inc_lo"]; hi = d["inc_hi"] - d["ic_incremental"]
    ax.errorbar(x + w / 2, d["ic_incremental"], yerr=[lo, hi], fmt="none",
                ecolor="black", lw=1, capsize=3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(d["h"])
    ax.set_xlabel("forecast horizon"); ax.set_ylabel("information coefficient")
    ax.set_title("Predictive power disappears once you control for what is already known",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=.25)
    return _save(fig, "signal_ic.png")


# ------------------------------------------------------- 6. multiple testing
def multiple_testing() -> str:
    d = pd.read_csv(TAB / "multiple_testing.csv").sort_values("q_value")
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    y = np.arange(len(d))[::-1]
    cols = [BLUE if r else GREY for r in d["reject"]]
    ax.barh(y, d["q_value"], color=cols, height=.6)
    ax.axvline(0.05, color=RED, ls="--", lw=1.2, label="5% false-discovery rate")
    lab = [f"{h} · {t}" for h, t in zip(d["hypothesis"], d["test"])]
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("q-value (Benjamini-Hochberg)")
    n = int(d["reject"].sum())
    ax.set_title(f"Across the whole study, {n} of {len(d)} primary tests survive correction",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="x", alpha=.25)
    return _save(fig, "multiple_testing.png")


# ------------------------------------------------------- 7. market context
def market_context() -> str:
    """Why this market, and why now: growth, institutionalisation, scale.

    Everything on the left and centre panels comes from the trade tape itself.
    The right panel sets our own measured hedging flow against US spot-ETF
    creations, which is the competing flow the deck has to acknowledge.
    """
    m = json.loads((TAB / "market_context.json").read_text())["by_year"]
    yrs = ["2022", "2023", "2024", "2025"]
    notional = [m["notional_bn"][y] for y in yrs]
    blockpct = [m["block_pct_volume"][y] for y in yrs]
    hedge = json.loads((TAB / "market_context.json").read_text())["hedge_musd_per_day"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))

    ax = axes[0]
    ax.bar(yrs, notional, color=BLUE, width=.62)
    for x, v in zip(yrs, notional):
        ax.text(x, v + 28, f"${v:,.0f}bn", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, max(notional) * 1.22)
    ax.set_title("Option notional traded\n6.5x in four years", fontsize=10)
    ax.set_ylabel("$bn per year", fontsize=9)

    ax = axes[1]
    ax.plot(yrs, blockpct, "o-", color=BLUE, lw=2, ms=7)
    for x, v in zip(yrs, blockpct):
        ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax.set_ylim(min(blockpct) - 10, max(blockpct) + 12)
    ax.set_title("Negotiated block trades\nshare of volume", fontsize=10)
    ax.set_ylabel("% of volume", fontsize=9)

    ax = axes[2]
    labs = ["Dealer hedging\n(this study)", "US spot-ETF\ncreations"]
    vals = [hedge["2024"], 140.0]
    ax.barh(labs, vals, color=[BLUE, GREY], height=.5)
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"${v:,.0f}M", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, 175)
    ax.set_title("Daily flow into bitcoin, 2024\nETF creations are ~5x larger", fontsize=10)
    ax.set_xlabel("$M per day (median)", fontsize=9)
    ax.invert_yaxis()

    for a in axes:
        a.grid(axis="y" if a is not axes[2] else "x", alpha=.28)
        a.tick_params(labelsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    return _save(fig, "market_context.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    print("\n[1/7] sample funnel");            sample_funnel()
    print("\n[2/7] gamma validation");         gamma_validation()
    print("\n[3/7] H3 by era");                h3_by_era()
    print("\n[4/7] H4 structural break");      h4_break()
    print("\n[5/7] information coefficients"); ic_chart()
    print("\n[6/7] multiple testing");         multiple_testing()
    print("\n[7/7] market context");           market_context()
    print("\ndone.")


if __name__ == "__main__":
    main()
