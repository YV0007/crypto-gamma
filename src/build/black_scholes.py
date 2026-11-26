"""Closed-form option greeks from Deribit's published implied volatility.

WHY THIS EXISTS
---------------
CONTEXT.md §6 says greeks are pulled, never recomputed. That is impossible for
a backdated sample: Deribit stores no historical greeks. Verified exhaustively
2026-08-31 across both hosts --

    ticker(expired)                  -> "instrument is not active"
    get_book_summary_by_*(expired)   -> "instrument is not open"
    get_book_summary_by_*(LIVE)      -> carries no greeks columns at all
    get_order_book(expired)          -> returns a FROZEN settlement snapshot:
                                        gamma 0.0, mark_price 0.0,
                                        estimated_delivery_price "expired"
    get_mark_price_history(expired)  -> "instrument is not active"
    historical trades                -> carry `iv` and `index_price`, no greeks

What Deribit does retain historically is its own mark implied volatility, on
every single trade (100% non-null across all 555,316 March-2024 trades). So
the volatility surface stays Deribit-published; only the closed form is
recomputed.

CALIBRATION (this is what makes the substitution defensible)
------------------------------------------------------------
Recomputed gamma was compared against Deribit's PUBLISHED gamma on 161 live
instruments, using their published mark_iv as the input:

    correlation           0.9976
    median ratio          1.0138
    corr(ratio, spot)     0.0116
    corr(ratio, moneyness) -0.0804
    corr(ratio, maturity)  0.0129

No systematic dependence on any covariate, so Deribit uses the standard USD
convention and NOT an inverse/BTC-margined rescaling, despite the contracts
being inverse. Residual dispersion is explained by Deribit rounding gamma to 5
decimals, which is severe when gamma is ~1e-5.

The quanto/inverse adjustment (CONTEXT.md §12) is therefore a genuine
extension rather than a correction to these numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config

YEAR_SECONDS = 365.25 * 86400


def d1(S, K, T, sigma, r=0.0):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    option_type: np.ndarray,
    r: float | None = None,
) -> pd.DataFrame:
    """Vectorised BS greeks in the USD convention Deribit publishes.

    S      spot / index price, USD
    K      strike, USD
    T      time to expiry, YEARS
    sigma  implied volatility as a DECIMAL (0.65), not percent
    """
    cfg = config.get("construction.greeks.black_scholes")
    r = cfg["risk_free_rate"] if r is None else r

    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float); sigma = np.asarray(sigma, dtype=float)

    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    out = {k: np.full(S.shape, np.nan) for k in ("gamma", "delta", "vega", "theta")}
    if not valid.any():
        return pd.DataFrame(out)

    Sv, Kv, Tv, sv = S[valid], K[valid], T[valid], sigma[valid]
    sqrtT = np.sqrt(Tv)
    D1 = d1(Sv, Kv, Tv, sv, r)
    D2 = D1 - sv * sqrtT
    pdf = norm.pdf(D1)

    out["gamma"][valid] = pdf / (Sv * sv * sqrtT)
    out["vega"][valid] = Sv * pdf * sqrtT / 100.0          # per 1 vol point
    is_call = np.asarray(option_type)[valid] == "call"
    out["delta"][valid] = np.where(is_call, norm.cdf(D1), norm.cdf(D1) - 1.0)
    theta_common = -(Sv * pdf * sv) / (2 * sqrtT)
    out["theta"][valid] = np.where(
        is_call,
        theta_common - r * Kv * np.exp(-r * Tv) * norm.cdf(D2),
        theta_common + r * Kv * np.exp(-r * Tv) * norm.cdf(-D2),
    ) / 365.25                                              # per day
    return pd.DataFrame(out)


def clean_iv(iv_pct: pd.Series) -> pd.Series:
    """Drop the sentinel/garbage values Deribit carries in the trade `iv` field.

    The raw field spans 0.00 to 999.00 in March 2024; both extremes are
    non-economic. Bounds are config parameters, not magic numbers.
    """
    cfg = config.get("construction.greeks.black_scholes")
    out = pd.to_numeric(iv_pct, errors="coerce")
    return out.where((out >= cfg["iv_min_pct"]) & (out <= cfg["iv_max_pct"]))


def time_to_expiry_years(timestamp: pd.Series, expiry: pd.Series) -> pd.Series:
    cfg = config.get("construction.greeks.black_scholes")
    tte = (expiry - timestamp).dt.total_seconds() / YEAR_SECONDS
    return tte.where(tte > cfg["min_tte_days"] / 365.25)
