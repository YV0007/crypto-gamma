"""Statistical helpers: HAC (Newey-West) inference and Sharpe-ratio deflation.

Everything a number has to pass through before it goes on a slide.

Two problems this module exists to solve.

1. AUTOCORRELATION. The panel is 10-minute observations of trailing-window
   quantities, so consecutive rows overlap by construction. Ordinary standard
   errors on such a series are wrong by a large factor and always in the
   flattering direction. Every mean and every regression coefficient reported
   from this project goes through `nw_mean` or `hac_ols`, which use a
   Bartlett-kernel HAC estimator with the bandwidth declared in config
   (`stats.newey_west_lags`, "auto" -> floor(4*(T/100)^(2/9))).

2. MULTIPLE TESTING. A study that tries fifty specifications and reports the
   best one finds a Sharpe ratio of 1 in pure noise. The defence is the
   Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), which needs to know
   how many DISTINCT specifications were tried and how widely their Sharpes
   were spread. That bookkeeping cannot be reconstructed afterwards, so
   `src/utils/speclog.py` records each specification as it runs and
   `deflate_from_log` reads the trial count straight out of that log. The
   deflation therefore tightens on its own as the study grows, with no
   number to remember to update -- which is the whole point of it.

CONVENTION, and the easiest thing in here to get wrong: every Sharpe ratio is
PER OBSERVATION, never annualised. The DSR algebra is defined on the raw
per-observation moments and returns a quietly wrong answer if handed an
annualised figure. `annualise` exists for display and nothing else.

References
    Newey & West (1987), Econometrica 55(3).
    Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal.
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio
        Management 40(5).
    Benjamini & Hochberg (1995), JRSS-B 57(1).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from src import config

log = logging.getLogger("utils.stats")

EULER_GAMMA = 0.577_215_664_901_532_9


# ---------------------------------------------------------------------------
# bandwidth
# ---------------------------------------------------------------------------

def nw_lag(n_obs: int, setting: int | str | None = None) -> int:
    """Newey-West truncation lag.

    `setting` defaults to config `stats.newey_west_lags`. The string "auto"
    is the Newey-West (1994) plug-in rule floor(4*(T/100)^(2/9)) -- the same
    default Stata and statsmodels use, declared in config so a reader can see
    it was not chosen after seeing the answer.
    """
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    if setting is None:
        setting = config.get("stats.newey_west_lags")
    if isinstance(setting, str):
        if setting != "auto":
            raise ValueError(f"unknown newey_west_lags setting: {setting!r}")
        return int(math.floor(4 * (n_obs / 100) ** (2 / 9)))
    lag = int(setting)
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return lag


# ---------------------------------------------------------------------------
# HAC long-run covariance
# ---------------------------------------------------------------------------

def long_run_cov(scores: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel long-run covariance of a score matrix, times n.

    `scores` is (n, k): row t is the observation's contribution to the moment
    condition. Returns S = G0 + sum_j w_j (Gj + Gj'), w_j = 1 - j/(lags+1),
    which is n times the usual long-run covariance estimate. Keeping the
    factor of n out makes the sandwich formulas below read like the textbook.

    lags=0 collapses to the plain outer-product estimator, i.e. White.
    """
    g = np.asarray(scores, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    n = g.shape[0]
    if lags >= n:
        raise ValueError(f"lag {lags} >= n_obs {n}: no overlapping pairs left")
    s = g.T @ g
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        gamma_j = g[j:].T @ g[:-j]
        s = s + w * (gamma_j + gamma_j.T)
    return s


# ---------------------------------------------------------------------------
# OLS with HAC standard errors
# ---------------------------------------------------------------------------

@dataclass
class HACResult:
    """One HAC-corrected regression or mean."""
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    n_obs: int
    lags: int
    r2: float = float("nan")
    cov: np.ndarray = field(default=None, repr=False)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.coef, "se": self.se, "t_stat": self.t_stat,
                "p_value": self.p_value, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            },
            index=pd.Index(self.names, name="term"),
        )

    def get(self, term: str) -> dict[str, float]:
        i = self.names.index(term)
        return {
            "coef": float(self.coef[i]), "se": float(self.se[i]),
            "t_stat": float(self.t_stat[i]), "p_value": float(self.p_value[i]),
            "ci_lo": float(self.ci_lo[i]), "ci_hi": float(self.ci_hi[i]),
        }

    def __str__(self) -> str:
        return (f"HAC OLS  n={self.n_obs}  lags={self.lags}  R2={self.r2:.4f}\n"
                + self.to_frame().to_string())


def hac_ols(
    y,
    X=None,
    *,
    add_const: bool = True,
    lags: int | str | None = None,
    level: float | None = None,
    names: list[str] | None = None,
) -> HACResult:
    """OLS with Newey-West standard errors.

    Inference is asymptotic-normal, not t: the HAC sandwich has no exact
    small-sample distribution, and pretending otherwise by using t(n-k) would
    only widen intervals by a rounding error at these sample sizes.

    Rows with any missing value in y or X are dropped listwise BEFORE the lag
    structure is built. On a gappy series that quietly changes what "lag 1"
    means, so pass a series you have already made contiguous when the gaps are
    large; see `nw_mean` for the common case.
    """
    level = config.get("stats.confidence_level") if level is None else level

    y_arr = np.asarray(y, dtype=float).ravel()
    if X is None:
        x_arr = np.empty((y_arr.size, 0))
        names_in: list[str] = []
    elif isinstance(X, pd.DataFrame):
        names_in = list(X.columns)
        x_arr = X.to_numpy(dtype=float)
    else:
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        names_in = names or [f"x{i + 1}" for i in range(x_arr.shape[1])]

    if x_arr.shape[0] != y_arr.size:
        raise ValueError("y and X have different lengths")

    keep = np.isfinite(y_arr)
    if x_arr.shape[1]:
        keep &= np.isfinite(x_arr).all(axis=1)
    y_arr, x_arr = y_arr[keep], x_arr[keep]

    if add_const:
        x_arr = np.column_stack([np.ones(y_arr.size), x_arr])
        names_out = ["const", *names_in]
    else:
        names_out = list(names_in)

    n, k = x_arr.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")

    xtx_inv = np.linalg.pinv(x_arr.T @ x_arr)
    coef = xtx_inv @ (x_arr.T @ y_arr)
    resid = y_arr - x_arr @ coef

    lag = nw_lag(n, lags)
    meat = long_run_cov(x_arr * resid[:, None], lag)
    cov = xtx_inv @ meat @ xtx_inv

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, coef / se, np.nan)
    p_value = 2.0 * sps.norm.sf(np.abs(t_stat))
    z = sps.norm.ppf(0.5 + level / 2.0)

    tss = float(((y_arr - y_arr.mean()) ** 2).sum()) if add_const else float((y_arr**2).sum())
    r2 = 1.0 - float((resid**2).sum()) / tss if tss > 0 else float("nan")

    return HACResult(
        names=names_out, coef=coef, se=se, t_stat=t_stat, p_value=p_value,
        ci_lo=coef - z * se, ci_hi=coef + z * se,
        n_obs=n, lags=lag, r2=r2, cov=cov,
    )


def nw_mean(x, *, lags: int | str | None = None, level: float | None = None) -> HACResult:
    """The mean of an autocorrelated series, with a HAC standard error.

    This is `hac_ols` on a constant, which is exactly what a Newey-West mean
    is; the separate name is so calling code reads as what it means.
    """
    res = hac_ols(x, None, add_const=True, lags=lags, level=level)
    res.names = ["mean"]
    return res


def nw_diff(a, b, *, lags: int | str | None = None, level: float | None = None) -> HACResult:
    """Difference in means between two INDEPENDENT groups, HAC on each side.

    Used for the quintile-1 vs quintile-5 contrasts. Each group carries its
    own serial correlation, so the variances are estimated separately and
    added; that assumes the two groups are not cross-correlated, which holds
    only loosely for quintiles carved out of one time series. Treat the
    interval as indicative, and prefer the block bootstrap when the contrast
    is the headline number.
    """
    level = config.get("stats.confidence_level") if level is None else level
    ra, rb = nw_mean(a, lags=lags, level=level), nw_mean(b, lags=lags, level=level)
    diff = float(ra.coef[0] - rb.coef[0])
    se = float(np.hypot(ra.se[0], rb.se[0]))
    t = diff / se if se > 0 else np.nan
    z = sps.norm.ppf(0.5 + level / 2.0)
    return HACResult(
        names=["diff"], coef=np.array([diff]), se=np.array([se]),
        t_stat=np.array([t]), p_value=np.array([2.0 * sps.norm.sf(abs(t))]),
        ci_lo=np.array([diff - z * se]), ci_hi=np.array([diff + z * se]),
        n_obs=ra.n_obs + rb.n_obs, lags=max(ra.lags, rb.lags),
    )


# ---------------------------------------------------------------------------
# moving block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    values: np.ndarray,
    blocks: np.ndarray,
    stat=np.mean,
    reps: int | None = None,
    level: float | None = None,
    seed: int | None = None,
) -> tuple[float, float]:
    """Block bootstrap confidence interval, resampling whole blocks.

    `blocks` labels each observation with the block it belongs to (a calendar
    day, in this project). Whole blocks are drawn with replacement, so the
    within-block dependence is carried into every resample instead of being
    destroyed by an iid draw. On 10-minute data an iid bootstrap understates
    the error bars by roughly the square root of the block length, which is
    the difference between a result and an artefact.

    Returns (nan, nan) rather than raising when there is nothing to resample,
    so a sparse quintile shows up as a missing bar rather than a crash.
    """
    reps = config.get("stats.bootstrap.reps") if reps is None else reps
    level = config.get("stats.confidence_level") if level is None else level
    seed = config.get("seed") if seed is None else seed

    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    ok = np.isfinite(values)
    values, blocks = values[ok], blocks[ok]

    uniq = np.unique(blocks)
    if len(uniq) < 2 or values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    by = {b: values[blocks == b] for b in uniq}
    draws = np.empty(reps)
    for i in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws[i] = stat(np.concatenate([by[b] for b in pick]))
    a = (1 - level) / 2
    return float(np.nanquantile(draws, a)), float(np.nanquantile(draws, 1 - a))


def block_bootstrap_contrast(
    a_values: np.ndarray,
    a_blocks: np.ndarray,
    b_values: np.ndarray,
    b_blocks: np.ndarray,
    stat=np.mean,
    reps: int | None = None,
    level: float | None = None,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Block bootstrap interval for stat(a) - stat(b) across two groups.

    Returns (point_estimate, ci_lo, ci_hi).

    Days are drawn ONCE per replication and applied to both groups, not drawn
    independently. That matters here: the two groups are quintiles carved out
    of a single time series, so a day that was unusually volatile contributes
    to both sides at once. Resampling them independently would break that
    link and understate the uncertainty on the difference -- which is exactly
    the number a quintile-sort result rests on.

    Days on which one side has no observations are simply skipped for that
    side, so a replication that empties a group is dropped rather than
    counted as a zero.
    """
    reps = config.get("stats.bootstrap.reps") if reps is None else reps
    level = config.get("stats.confidence_level") if level is None else level
    seed = config.get("seed") if seed is None else seed

    a_values, b_values = np.asarray(a_values, float), np.asarray(b_values, float)
    a_blocks, b_blocks = np.asarray(a_blocks), np.asarray(b_blocks)
    ka, kb = np.isfinite(a_values), np.isfinite(b_values)
    a_values, a_blocks = a_values[ka], a_blocks[ka]
    b_values, b_blocks = b_values[kb], b_blocks[kb]

    point = float(stat(a_values) - stat(b_values))
    days = np.union1d(np.unique(a_blocks), np.unique(b_blocks))
    if len(days) < 2 or a_values.size == 0 or b_values.size == 0:
        return point, float("nan"), float("nan")

    by_a = {d: a_values[a_blocks == d] for d in days}
    by_b = {d: b_values[b_blocks == d] for d in days}
    rng = np.random.default_rng(seed)
    draws = np.full(reps, np.nan)
    for i in range(reps):
        pick = rng.choice(days, size=len(days), replace=True)
        sa = np.concatenate([by_a[d] for d in pick if by_a[d].size])
        sb = np.concatenate([by_b[d] for d in pick if by_b[d].size])
        if sa.size and sb.size:
            draws[i] = stat(sa) - stat(sb)
    a = (1 - level) / 2
    return point, float(np.nanquantile(draws, a)), float(np.nanquantile(draws, 1 - a))


# ---------------------------------------------------------------------------
# Sharpe ratios -- per observation unless explicitly annualised
# ---------------------------------------------------------------------------

def periods_per_year(interval: str | None = None) -> float:
    """Observations per year on the analysis grid, derived from config.

    Never hardcoded: the grid is `construction.interval`, and the robustness
    battery sweeps it to 1h and 1d. A stale constant here would rescale every
    annualised Sharpe in the deck by a silent factor.
    """
    interval = config.get("construction.interval") if interval is None else interval
    return pd.Timedelta("365D") / pd.Timedelta(interval)


def sharpe(returns, *, rf: float = 0.0) -> float:
    """Sharpe ratio PER OBSERVATION. Not annualised. See module docstring."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] - rf
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def annualise(sr_per_obs: float, interval: str | None = None) -> float:
    """Scale a per-observation Sharpe to annual, for DISPLAY ONLY.

    The sqrt-of-time scaling assumes iid returns. It is wrong for a strategy
    with autocorrelated returns and is never the input to any test in this
    module -- `deflated_sharpe` and `psr` take the per-observation figure.
    """
    return float(sr_per_obs * math.sqrt(periods_per_year(interval)))


def sharpe_se(returns, *, lags: int | str | None = None, iid: bool = False) -> float:
    """Standard error of a per-observation Sharpe ratio (Lo 2002).

    Default is the general non-iid case: the delta method applied to the HAC
    covariance of the moment conditions (r - mu, (r - mu)^2 - sigma^2). With
    `iid=True` this reduces analytically to sqrt((1 + SR^2/2)/n), which the
    tests check as a limiting case.

    Autocorrelated returns are the norm for a gamma-timing strategy, so the
    iid form flatters it; it is here to quantify by how much, not to report.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return float("nan")
    mu = r.mean()
    var = r.var(ddof=0)
    sd = math.sqrt(var)
    if sd <= 0:
        return float("nan")
    sr = mu / sd

    if iid:
        return float(math.sqrt((1.0 + 0.5 * sr**2) / n))

    g = np.column_stack([r - mu, (r - mu) ** 2 - var])
    s = long_run_cov(g, nw_lag(n, lags))
    grad = np.array([1.0 / sd, -mu / (2.0 * sd**3)])
    v = float(grad @ s @ grad)
    return float(math.sqrt(max(v, 0.0)) / n)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """P(true Sharpe > benchmark), adjusting for skew and fat tails.

    `sr` and `sr_benchmark` are PER OBSERVATION. `kurtosis` is raw, not
    excess (3.0 for a normal). Negative skew and fat tails both inflate the
    denominator, which is why a strategy that sells volatility needs a higher
    measured Sharpe to clear the same bar.
    """
    denom_sq = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr**2
    if not np.isfinite(denom_sq) or denom_sq <= 0 or n_obs < 2:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """The Sharpe you would expect from the BEST of n_trials pure-noise tries.

    Bailey & Lopez de Prado's SR0: the expected maximum of n draws from a
    mean-zero normal with variance `sr_variance`, via the Gumbel
    approximation. This is the bar a reported Sharpe has to clear to mean
    anything, and it grows with every extra specification tried.
    """
    if n_trials < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    n = float(n_trials)
    a = sps.norm.ppf(1.0 - 1.0 / n)
    b = sps.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


@dataclass
class DSRResult:
    sr: float                 # per observation
    sr_annual: float          # display only
    sr_benchmark: float       # SR0, per observation
    dsr: float                # P(true SR > SR0)
    psr_vs_zero: float        # P(true SR > 0), the undeflated comparison
    n_obs: int
    n_trials: int
    sr_variance: float
    skew: float
    kurtosis: float
    min_track_record: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__]).T.rename(columns={0: "value"})

    def __str__(self) -> str:
        verdict = "survives" if self.dsr >= 0.95 else "does NOT survive"
        return (
            f"Deflated Sharpe Ratio\n"
            f"  observations        {self.n_obs:,}\n"
            f"  distinct trials     {self.n_trials}\n"
            f"  Sharpe (per obs)    {self.sr:.5f}   (annualised {self.sr_annual:.3f})\n"
            f"  skew / kurtosis     {self.skew:.3f} / {self.kurtosis:.3f}\n"
            f"  selection bar SR0   {self.sr_benchmark:.5f}   "
            f"(annualised {annualise(self.sr_benchmark):.3f})\n"
            f"  PSR vs zero         {self.psr_vs_zero:.4f}\n"
            f"  DSR vs SR0          {self.dsr:.4f}  -> {verdict} deflation at 0.95\n"
            f"  min track record    {self.min_track_record:,.0f} observations"
        )


def min_track_record_length(
    sr: float, sr_benchmark: float, skew: float, kurtosis: float, prob: float | None = None
) -> float:
    """Observations needed for `sr` to beat `sr_benchmark` at confidence `prob`.

    The honest answer to "is the backtest long enough?". If this exceeds the
    sample, the result is not yet evidence however good the point estimate is.
    """
    prob = config.get("stats.confidence_level") if prob is None else prob
    edge = sr - sr_benchmark
    if edge <= 0 or not np.isfinite(edge):
        return float("inf")
    denom_sq = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr**2
    if denom_sq <= 0:
        return float("nan")
    return float(1.0 + denom_sq * (sps.norm.ppf(prob) / edge) ** 2)


def deflated_sharpe(
    returns,
    *,
    n_trials: int,
    sr_variance: float,
    rf: float = 0.0,
) -> DSRResult:
    """Deflated Sharpe Ratio of a return series.

    `n_trials` is the number of DISTINCT specifications the study tried, and
    `sr_variance` the variance of their Sharpes -- both per observation. Do
    not pass these by hand for a headline number: `deflate_from_log` reads
    them out of the specification log, which is the only record that cannot
    be tidied up after the fact.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] - rf
    n = r.size
    sr = sharpe(r)
    skew = float(sps.skew(r, bias=False)) if n > 2 else float("nan")
    kurt = float(sps.kurtosis(r, fisher=False, bias=False)) if n > 3 else float("nan")
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return DSRResult(
        sr=sr,
        sr_annual=annualise(sr),
        sr_benchmark=sr0,
        dsr=probabilistic_sharpe(sr, sr0, n, skew, kurt),
        psr_vs_zero=probabilistic_sharpe(sr, 0.0, n, skew, kurt),
        n_obs=n,
        n_trials=int(n_trials),
        sr_variance=float(sr_variance),
        skew=skew,
        kurtosis=kurt,
        min_track_record=min_track_record_length(sr, sr0, skew, kurt),
    )


def deflate_from_log(returns, *, hypothesis: str | None = None, rf: float = 0.0) -> DSRResult:
    """Deflated Sharpe with the trial count taken from the specification log.

    This is the function that makes the deflation self-updating: every run
    that records a specification widens the noise bar the headline Sharpe has
    to clear, automatically, with nothing to remember to bump by hand.

    `hypothesis` restricts the trial family. Leaving it None counts every
    specification the study has ever run, which is the conservative reading
    and the one to put on the slide.
    """
    from src.utils import speclog

    trials = speclog.trials(hypothesis=hypothesis)
    sr_col = trials["sharpe"].dropna() if "sharpe" in trials else pd.Series(dtype=float)
    n_trials = max(int(len(trials)), 1)
    sr_var = float(sr_col.var(ddof=1)) if len(sr_col) > 1 else float("nan")

    if not np.isfinite(sr_var):
        # No spread to estimate from: fall back to the variance of the series
        # itself under the null, which is the standard iid approximation
        # 1/n. Reported so the reader can see the assumption was needed.
        n = int(np.isfinite(np.asarray(returns, dtype=float)).sum())
        sr_var = 1.0 / n if n > 1 else float("nan")
        log.warning("spec log has <2 logged Sharpes; using iid SR variance 1/n = %.3g", sr_var)

    return deflated_sharpe(returns, n_trials=n_trials, sr_variance=sr_var, rf=rf)


# ---------------------------------------------------------------------------
# multiple testing across the hypothesis battery
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values, level: float | None = None) -> pd.DataFrame:
    """Benjamini-Hochberg FDR control over a family of p-values.

    Six hypotheses, each with several sub-tests, means the family-wise error
    rate is not 5%. BH controls the expected share of false positives among
    the rejections, which is the right target for an exploratory study; a
    Bonferroni correction here would be so conservative it would reject
    nothing and say nothing.
    """
    level = config.get("stats.confidence_level") if level is None else level
    alpha = 1.0 - level
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)
    out_q = np.empty(m)
    out_q[order] = q
    return pd.DataFrame({"p_value": p, "q_value": out_q, "reject": out_q <= alpha})
