#!/usr/bin/env python3
"""Does substituting our Black-Scholes gamma change GEX at the PORTFOLIO level?

WHY THIS EXISTS
---------------
`src/build/black_scholes.py` records a per-instrument calibration: correlation
0.9976 against Deribit's published gamma on 161 live contracts. That number is
weaker evidence than it looks, for a reason worth stating plainly:

    Deribit rounds published gamma to 5 decimal places. Typical live gamma is
    ~2e-5, so the rounding granularity is ~50% of the value. On a recent
    snapshot, 724 priced instruments carried only 25 DISTINCT gamma values and
    42% of them sat at exactly 1e-5, the smallest representable number.

A cross-sectional correlation against a yardstick that coarse is dominated by
the ATM-vs-OTM ordering, not by agreement. And it is the wrong question anyway:
nothing in this project uses a single instrument's gamma. GEX is a SUM, so what
matters is whether the substitution survives aggregation -- where independent
rounding errors partly cancel and position weights decide everything.

WHAT THIS MEASURES
------------------
Positions are held FIXED and only gamma is swapped:

    GEX_deribit = SUM_i  w_i * gamma_published_i * S^2 * 0.01
    GEX_ours    = SUM_i  w_i * gamma_blackscholes_i * S^2 * 0.01

with w_i the snapshot's open interest under the naive dealer sign convention
(short calls, long puts). Deribit publishes no GEX of its own, so this is the
only like-for-like comparison available: same book, same weights, same instant,
one input changed.

This is NOT a validation of GEX_flow. It validates the gamma substitution and
nothing else -- positions, the taker sign convention and netting are untouched
here and are what H3 tests.

Strike, expiry and option type are parsed from the instrument name rather than
joined from instruments.parquet, which was built for the 2022-25 sample and
contains only ~11% of today's live contracts.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402  -- must follow the sys.path shim above
from src.build.black_scholes import (  # noqa: E402
    greeks as bs_greeks,
    time_to_expiry_years,
)

log = logging.getLogger("validate.gamma")

# Deribit option symbols: BTC-26MAR27-104000-C
NAME_RE = r"^(?P<ccy>[A-Z]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"


def parse_names(names: pd.Series) -> pd.DataFrame:
    """Strike / expiry / type straight from the symbol.

    Deribit expiries settle at 08:00 UTC, which is what makes the parsed
    expiry exact rather than a date approximation.
    """
    m = names.str.extract(NAME_RE)
    out = pd.DataFrame(index=names.index)
    out["strike"] = pd.to_numeric(m["strike"], errors="coerce")
    out["option_type"] = m["cp"].map({"C": "call", "P": "put"})
    out["expiry_ts"] = pd.to_datetime(m["exp"], format="%d%b%y", utc=True, errors="coerce") \
        + pd.Timedelta(hours=8)
    return out


def load_snapshot(path: str | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    snap_dir = config.path("paths.oi_snapshots")
    files = sorted(glob.glob(str(snap_dir / "book_summary_*.parquet")))
    if not files:
        raise SystemExit(f"no full snapshots under {snap_dir}")
    f = path or files[-1]
    d = pd.read_parquet(f)
    if "gamma" not in d.columns:
        raise SystemExit(f"{f} is a LIGHT snapshot (no greeks) -- pass a full one")
    return d, pd.Timestamp(d["snapshot_ts"].iloc[0])


def compare(d: pd.DataFrame, ts: pd.Timestamp) -> dict:
    d = d.join(parse_names(d["instrument_name"]))
    d = d.dropna(subset=["strike", "option_type", "expiry_ts", "mark_iv", "underlying_price"])
    d = d[(d["mark_iv"] > 0) & (d["expiry_ts"] > ts)].copy()

    tte = time_to_expiry_years(pd.Series([ts] * len(d), index=d.index), d["expiry_ts"])
    g = bs_greeks(d["underlying_price"].to_numpy(), d["strike"].to_numpy(),
                  tte.to_numpy(), (d["mark_iv"] / 100.0).to_numpy(),
                  d["option_type"].to_numpy())
    d["gamma_ours"] = g["gamma"].to_numpy()
    d["tte_days"] = tte.to_numpy() * 365.25
    d = d.dropna(subset=["gamma_ours"])

    # naive dealer sign: short calls, long puts. Used ONLY to weight the two
    # aggregates identically -- this is not the flow-based position.
    sign = np.where(d["option_type"] == "call", -1.0, 1.0)
    w = sign * d["open_interest"].fillna(0.0)
    s2 = d["underlying_price"] ** 2 * 0.01

    gex_deribit = float((w * d["gamma"] * s2).sum())
    gex_ours = float((w * d["gamma_ours"] * s2).sum())

    # gross (unsigned) exposure: the aggregate that does not depend on the sign
    # convention, so a mismatch here is purely a gamma problem
    gross_deribit = float((d["open_interest"] * d["gamma"] * s2).sum())
    gross_ours = float((d["open_interest"] * d["gamma_ours"] * s2).sum())

    zero_pub = d["gamma"] <= 0
    lost = float((d.loc[zero_pub, "open_interest"]
                  * d.loc[zero_pub, "gamma_ours"] * s2[zero_pub]).sum())

    # Dump the per-instrument pairs so the deck can SHOW the agreement rather
    # than assert a correlation. Deribit rounds gamma to 5 decimals, so the
    # zero-gamma instruments are rounding casualties, not disagreements.
    from src import config as _cfg
    _pairs = d.loc[~zero_pub, ["instrument_name", "strike", "option_type",
                               "tte_days", "open_interest", "gamma", "gamma_ours"]].copy()
    _pairs = _pairs.rename(columns={"gamma": "gamma_deribit"})
    _pairs["moneyness"] = _pairs["strike"] / float(d["underlying_price"].median())
    _pdest = _cfg.path("paths.tables") / "gamma_validation_pairs.csv"
    _pairs.to_csv(_pdest, index=False)

    r = (d["gamma_ours"] / d["gamma"]).replace([np.inf, -np.inf], np.nan)
    res = {
        "snapshot_ts": str(ts),
        "n_instruments": int(len(d)),
        "spot": float(d["underlying_price"].median()),
        "per_instrument": {
            "n_with_published_gamma": int((~zero_pub).sum()),
            "n_published_gamma_zero": int(zero_pub.sum()),
            "distinct_published_gammas": int(d.loc[~zero_pub, "gamma"].nunique()),
            "median_ratio": float(r[~zero_pub].median()),
            "corr_levels": float(d.loc[~zero_pub, "gamma_ours"].corr(d.loc[~zero_pub, "gamma"])),
        },
        "aggregate": {
            "signed_gex_deribit": gex_deribit,
            "signed_gex_ours": gex_ours,
            "signed_ratio": gex_ours / gex_deribit if gex_deribit else np.nan,
            "gross_exposure_deribit": gross_deribit,
            "gross_exposure_ours": gross_ours,
            "gross_ratio": gross_ours / gross_deribit if gross_deribit else np.nan,
            "exposure_deribit_rounds_to_zero": lost,
            "share_of_gross_lost_to_rounding": lost / gross_ours if gross_ours else np.nan,
        },
    }

    by = []
    for lab, lo, hi in [("<7d", 0, 7), ("7-30d", 7, 30), ("30-90d", 30, 90), (">90d", 90, 1e9)]:
        m = (d["tte_days"] >= lo) & (d["tte_days"] < hi)
        if not m.any():
            continue
        gd = float((d.loc[m, "open_interest"] * d.loc[m, "gamma"] * s2[m]).sum())
        go = float((d.loc[m, "open_interest"] * d.loc[m, "gamma_ours"] * s2[m]).sum())
        by.append({"bucket": lab, "n": int(m.sum()),
                   "gross_deribit": gd, "gross_ours": go,
                   "ratio": go / gd if gd else np.nan,
                   "pub_zero": int((zero_pub & m).sum())})
    res["by_time_to_expiry"] = by
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate gamma validation vs Deribit")
    ap.add_argument("--snapshot", default=None, help="path to a FULL book_summary parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    d, ts = load_snapshot(args.snapshot)
    res = compare(d, ts)
    a, p = res["aggregate"], res["per_instrument"]

    print(f"\nsnapshot {res['snapshot_ts']}   spot ${res['spot']:,.0f}   "
          f"{res['n_instruments']} live options\n")
    print("PER INSTRUMENT (contaminated by Deribit's 5dp rounding)")
    print(f"  published gamma == 0      {p['n_published_gamma_zero']} of {res['n_instruments']}")
    print(f"  distinct published values {p['distinct_published_gammas']}")
    print(f"  median ratio ours/theirs  {p['median_ratio']:.4f}")
    print(f"  correlation (levels)      {p['corr_levels']:.4f}")
    print("\nAGGREGATE -- same book, same weights, only gamma swapped")
    print(f"  gross exposure  Deribit   ${a['gross_exposure_deribit']:,.0f} per 1% move")
    print(f"  gross exposure  ours      ${a['gross_exposure_ours']:,.0f} per 1% move")
    print(f"  ratio                     {a['gross_ratio']:.4f}   "
          f"({100*(a['gross_ratio']-1):+.2f}%)")
    print(f"  signed (naive-OI) Deribit ${a['signed_gex_deribit']:,.0f}")
    print(f"  signed (naive-OI) ours    ${a['signed_gex_ours']:,.0f}")
    print(f"  ratio                     {a['signed_ratio']:.4f}")
    print(f"\n  exposure Deribit rounds to ZERO: ${a['exposure_deribit_rounds_to_zero']:,.0f} "
          f"({100*a['share_of_gross_lost_to_rounding']:.2f}% of gross)")
    print("\nBY TIME TO EXPIRY (gross)")
    print(f"  {'bucket':8s}{'n':>5s}{'pub=0':>7s}{'Deribit':>16s}{'ours':>16s}{'ratio':>9s}")
    for b in res["by_time_to_expiry"]:
        print(f"  {b['bucket']:8s}{b['n']:>5d}{b['pub_zero']:>7d}"
              f"{b['gross_deribit']:>16,.0f}{b['gross_ours']:>16,.0f}{b['ratio']:>9.4f}")

    dest = config.path("paths.tables") / "gamma_aggregate_validation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
