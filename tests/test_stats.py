"""Tests for the statistics helpers.

The HAC estimator is checked against statsmodels rather than against a
hand-computed constant: statsmodels is already a declared dependency and is
the reference implementation everyone else in the literature uses, so an exact
match is a stronger claim than any number I could write down here.

The Sharpe machinery is checked against its own analytic limiting cases --
the general non-iid standard error must collapse to Lo's iid formula on iid
data, PSR at its own benchmark must be exactly one half, and so on. Those are
the places a sign error or a misplaced sqrt would hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src import config
from src.utils import speclog, stats


def ar1(n: int, rho: float, seed: int = 0, sd: float = 1.0, mean: float = 0.0) -> np.ndarray:
    """AR(1) series -- the failure mode HAC exists to handle."""
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sd, size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x + mean


# ---------------------------------------------------------------------------
# bandwidth
# ---------------------------------------------------------------------------

class TestNWLag:
    def test_auto_matches_newey_west_plug_in_rule(self):
        for n, expected in [(100, 4), (500, 5), (10_000, 11), (200_000, 21)]:
            assert stats.nw_lag(n, "auto") == expected

    def test_auto_matches_statsmodels_default_bandwidth(self):
        # statsmodels uses the same rule for cov_type="HAC" with maxlags=None
        n = 5_000
        assert stats.nw_lag(n, "auto") == int(np.floor(4 * (n / 100) ** (2 / 9)))

    def test_default_comes_from_config_not_from_code(self):
        assert stats.nw_lag(1_000) == stats.nw_lag(1_000, config.get("stats.newey_west_lags"))

    def test_explicit_integer_is_honoured(self):
        assert stats.nw_lag(1_000, 7) == 7

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            stats.nw_lag(0)
        with pytest.raises(ValueError):
            stats.nw_lag(100, -1)
        with pytest.raises(ValueError):
            stats.nw_lag(100, "plug_in")


# ---------------------------------------------------------------------------
# long-run covariance
# ---------------------------------------------------------------------------

class TestLongRunCov:
    def test_zero_lags_is_the_outer_product_estimator(self):
        g = np.random.default_rng(1).normal(size=(200, 3))
        assert np.allclose(stats.long_run_cov(g, 0), g.T @ g)

    def test_result_is_symmetric(self):
        g = np.random.default_rng(2).normal(size=(300, 4))
        s = stats.long_run_cov(g, 6)
        assert np.allclose(s, s.T)

    def test_positive_autocorrelation_inflates_the_variance(self):
        """The entire reason this module exists, in one assertion."""
        x = ar1(2_000, rho=0.8, seed=3)[:, None]
        assert stats.long_run_cov(x, 12)[0, 0] > 2 * stats.long_run_cov(x, 0)[0, 0]

    def test_accepts_a_flat_series(self):
        assert stats.long_run_cov(np.arange(50.0), 3).shape == (1, 1)

    def test_rejects_a_lag_longer_than_the_sample(self):
        with pytest.raises(ValueError):
            stats.long_run_cov(np.ones((10, 1)), 10)


# ---------------------------------------------------------------------------
# HAC OLS -- checked against statsmodels
# ---------------------------------------------------------------------------

class TestHACOls:
    @staticmethod
    def _data(n=500, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + ar1(n, 0.6, seed=seed + 1)
        return y, x

    def test_standard_errors_match_statsmodels_exactly(self):
        y, x = self._data()
        lag = stats.nw_lag(len(y))
        mine = stats.hac_ols(y, x, lags=lag)
        ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        assert np.allclose(mine.coef, ref.params)
        assert np.allclose(mine.se, ref.bse)
        assert mine.r2 == pytest.approx(ref.rsquared)

    def test_matches_statsmodels_at_every_bandwidth(self):
        y, x = self._data(seed=7)
        for lag in (0, 1, 5, 20):
            mine = stats.hac_ols(y, x, lags=lag)
            ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
            assert np.allclose(mine.se, ref.bse), f"mismatch at lag {lag}"

    def test_zero_lag_reduces_to_white_robust(self):
        y, x = self._data(seed=11)
        mine = stats.hac_ols(y, x, lags=0)
        ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC0")
        assert np.allclose(mine.se, ref.bse)

    def test_recovers_the_true_coefficients(self):
        y, x = self._data(n=5_000, seed=5)
        res = stats.hac_ols(y, x)
        assert res.get("x1")["coef"] == pytest.approx(2.0, abs=0.05)
        assert res.get("x2")["coef"] == pytest.approx(-0.5, abs=0.05)

    def test_dataframe_input_keeps_column_names(self):
        y, x = self._data()
        res = stats.hac_ols(y, pd.DataFrame(x, columns=["dvol", "funding"]))
        assert res.names == ["const", "dvol", "funding"]
        assert set(res.to_frame().columns) == {"coef", "se", "t_stat", "p_value",
                                               "ci_lo", "ci_hi"}

    def test_missing_rows_are_dropped_listwise(self):
        y, x = self._data(n=300)
        y_gap = y.copy()
        y_gap[[10, 20, 30]] = np.nan
        assert stats.hac_ols(y_gap, x).n_obs == 297

    def test_confidence_interval_brackets_the_estimate(self):
        y, x = self._data()
        res = stats.hac_ols(y, x)
        assert np.all(res.ci_lo < res.coef)
        assert np.all(res.coef < res.ci_hi)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            stats.hac_ols(np.zeros(10), np.zeros((9, 1)))

    def test_rejects_more_regressors_than_observations(self):
        with pytest.raises(ValueError):
            stats.hac_ols(np.zeros(3), np.zeros((3, 5)))


class TestNWMean:
    def test_iid_case_reproduces_the_textbook_standard_error(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=4_000)
        res = stats.nw_mean(x, lags=0)
        assert res.se[0] == pytest.approx(x.std(ddof=0) / np.sqrt(x.size), rel=1e-10)

    def test_autocorrelated_series_gets_a_materially_wider_error_bar(self):
        """A naive standard error on this series would be optimistic by ~2x."""
        x = ar1(5_000, rho=0.8, seed=6)
        naive = x.std(ddof=1) / np.sqrt(x.size)
        assert stats.nw_mean(x).se[0] > 2 * naive

    def test_reports_the_mean_under_a_readable_name(self):
        res = stats.nw_mean(ar1(500, 0.5, seed=8, mean=3.0))
        assert res.names == ["mean"]
        assert res.coef[0] == pytest.approx(np.mean(ar1(500, 0.5, seed=8, mean=3.0)))

    def test_records_the_bandwidth_actually_used(self):
        res = stats.nw_mean(np.random.default_rng(9).normal(size=1_000))
        assert res.lags == stats.nw_lag(1_000)


class TestNWDiff:
    def test_no_difference_is_not_significant(self):
        rng = np.random.default_rng(12)
        res = stats.nw_diff(rng.normal(size=3_000), rng.normal(size=3_000))
        assert abs(res.t_stat[0]) < 3.0
        assert res.ci_lo[0] < 0 < res.ci_hi[0]

    def test_a_real_difference_is_detected_with_the_right_sign(self):
        rng = np.random.default_rng(13)
        a = rng.normal(loc=1.0, size=3_000)
        b = rng.normal(loc=0.0, size=3_000)
        res = stats.nw_diff(a, b)
        assert res.coef[0] == pytest.approx(1.0, abs=0.1)
        assert res.p_value[0] < 0.001


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    @staticmethod
    def _daily_blocks(n_days=40, per_day=48, rho=0.9, seed=20):
        x = ar1(n_days * per_day, rho=rho, seed=seed)
        days = np.repeat(np.arange(n_days), per_day)
        return x, days

    def test_interval_contains_the_sample_mean(self):
        x, days = self._daily_blocks()
        lo, hi = stats.block_bootstrap_ci(x, days, reps=200)
        assert lo < x.mean() < hi

    def test_wider_than_an_iid_bootstrap_on_dependent_data(self):
        """Blocks preserve within-day dependence; shuffled labels destroy it."""
        x, days = self._daily_blocks()
        lo_b, hi_b = stats.block_bootstrap_ci(x, days, reps=200)
        singleton = np.arange(x.size)  # every observation its own block == iid
        lo_i, hi_i = stats.block_bootstrap_ci(x, singleton, reps=200)
        assert (hi_b - lo_b) > 1.5 * (hi_i - lo_i)

    def test_is_reproducible_from_the_config_seed(self):
        x, days = self._daily_blocks(n_days=20)
        assert stats.block_bootstrap_ci(x, days, reps=100) == \
               stats.block_bootstrap_ci(x, days, reps=100)

    def test_different_seeds_give_different_intervals(self):
        x, days = self._daily_blocks(n_days=20)
        assert stats.block_bootstrap_ci(x, days, reps=100, seed=1) != \
               stats.block_bootstrap_ci(x, days, reps=100, seed=2)

    def test_degenerate_input_returns_nan_rather_than_raising(self):
        lo, hi = stats.block_bootstrap_ci(np.array([1.0, 2.0]), np.array([0, 0]), reps=50)
        assert np.isnan(lo) and np.isnan(hi)

    def test_non_finite_values_are_dropped(self):
        x, days = self._daily_blocks(n_days=10)
        x[::50] = np.nan
        lo, hi = stats.block_bootstrap_ci(x, days, reps=100)
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_works_for_statistics_other_than_the_mean(self):
        x, days = self._daily_blocks(n_days=30)
        lo, hi = stats.block_bootstrap_ci(x, days, stat=np.std, reps=200)
        assert lo < x.std() < hi


class TestBlockBootstrapContrast:
    @staticmethod
    def _two_groups(shift=0.0, n_days=40, per_day=48, seed=25):
        """Two groups drawn from the same days, optionally with a real gap."""
        rng = np.random.default_rng(seed)
        days = np.repeat(np.arange(n_days), per_day)
        day_effect = np.repeat(rng.normal(scale=2.0, size=n_days), per_day)
        a = day_effect + rng.normal(size=days.size) + shift
        b = day_effect + rng.normal(size=days.size)
        return a, days, b, days

    def test_point_estimate_is_the_plain_difference(self):
        a, ad, b, bd = self._two_groups(shift=1.0)
        point, _, _ = stats.block_bootstrap_contrast(a, ad, b, bd, reps=100)
        assert point == pytest.approx(a.mean() - b.mean())

    def test_no_real_difference_gives_an_interval_covering_zero(self):
        a, ad, b, bd = self._two_groups(shift=0.0)
        _, lo, hi = stats.block_bootstrap_contrast(a, ad, b, bd, reps=300)
        assert lo < 0 < hi

    def test_a_real_difference_is_detected(self):
        a, ad, b, bd = self._two_groups(shift=1.0)
        _, lo, hi = stats.block_bootstrap_contrast(a, ad, b, bd, reps=300)
        assert lo > 0

    def test_shared_day_effects_are_differenced_out(self):
        """The reason days are drawn once and applied to both groups.

        A common day-level shock inflates each group's own error bar but
        cancels in the difference. Independent resampling would miss that and
        report a needlessly wide interval.
        """
        a, ad, b, bd = self._two_groups(shift=0.0)
        _, c_lo, c_hi = stats.block_bootstrap_contrast(a, ad, b, bd, reps=300)
        a_lo, a_hi = stats.block_bootstrap_ci(a, ad, reps=300)
        assert (c_hi - c_lo) < (a_hi - a_lo)

    def test_is_reproducible_from_the_config_seed(self):
        a, ad, b, bd = self._two_groups()
        assert stats.block_bootstrap_contrast(a, ad, b, bd, reps=100) == \
               stats.block_bootstrap_contrast(a, ad, b, bd, reps=100)

    def test_degenerate_input_returns_the_point_estimate_and_nan_bounds(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 1.0])
        d = np.array([0, 0])
        point, lo, hi = stats.block_bootstrap_contrast(a, d, b, d, reps=50)
        assert point == pytest.approx(1.0)
        assert np.isnan(lo) and np.isnan(hi)

    def test_groups_on_disjoint_days_still_work(self):
        """Quintiles need not share days; the union is resampled."""
        rng = np.random.default_rng(26)
        a, ad = rng.normal(size=500), np.repeat(np.arange(10), 50)
        b, bd = rng.normal(size=500), np.repeat(np.arange(10, 20), 50)
        _, lo, hi = stats.block_bootstrap_contrast(a, ad, b, bd, reps=200)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


# ---------------------------------------------------------------------------
# Sharpe ratios
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_matches_the_definition(self):
        r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
        assert stats.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))

    def test_risk_free_shifts_the_numerator(self):
        r = np.full(100, 0.02) + np.random.default_rng(21).normal(scale=0.01, size=100)
        assert stats.sharpe(r, rf=0.02) < stats.sharpe(r)

    def test_constant_series_has_no_sharpe(self):
        assert np.isnan(stats.sharpe(np.ones(50)))

    def test_too_short_returns_nan(self):
        assert np.isnan(stats.sharpe(np.array([0.01])))

    def test_periods_per_year_follows_the_config_grid(self):
        assert stats.periods_per_year("10min") == pytest.approx(365 * 24 * 6)
        assert stats.periods_per_year("1d") == pytest.approx(365)
        assert stats.periods_per_year() == stats.periods_per_year(
            config.get("construction.interval"))

    def test_annualisation_is_square_root_of_time(self):
        assert stats.annualise(0.01, "1d") == pytest.approx(0.01 * np.sqrt(365))


class TestSharpeStandardError:
    def test_iid_flag_is_the_lo_2002_closed_form(self):
        r = np.random.default_rng(22).normal(0.05, 1.0, size=1_000)
        sr = stats.sharpe(r)
        expected = np.sqrt((1 + 0.5 * sr**2) / r.size)
        assert stats.sharpe_se(r, iid=True) == pytest.approx(expected)

    def test_hac_version_collapses_to_the_iid_form_on_iid_data(self):
        """The general estimator must not pay a price when there is no dependence."""
        r = np.random.default_rng(23).normal(0.02, 1.0, size=50_000)
        assert stats.sharpe_se(r) == pytest.approx(stats.sharpe_se(r, iid=True), rel=0.05)

    def test_autocorrelated_returns_get_a_wider_error_bar(self):
        r = ar1(20_000, rho=0.5, seed=24, mean=0.05)
        assert stats.sharpe_se(r) > 1.3 * stats.sharpe_se(r, iid=True)

    def test_too_short_returns_nan(self):
        assert np.isnan(stats.sharpe_se(np.array([0.01, 0.02])))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

class TestProbabilisticSharpe:
    def test_at_its_own_benchmark_it_is_a_coin_flip(self):
        assert stats.probabilistic_sharpe(0.05, 0.05, 1_000, 0.0, 3.0) == pytest.approx(0.5)

    def test_increases_with_the_measured_sharpe(self):
        vals = [stats.probabilistic_sharpe(s, 0.0, 1_000, 0.0, 3.0)
                for s in (0.00, 0.02, 0.05, 0.10)]
        assert vals == sorted(vals)

    def test_increases_with_sample_length(self):
        short = stats.probabilistic_sharpe(0.05, 0.0, 250, 0.0, 3.0)
        long_ = stats.probabilistic_sharpe(0.05, 0.0, 10_000, 0.0, 3.0)
        assert long_ > short

    def test_negative_skew_is_penalised(self):
        """Selling volatility looks good until the moments are accounted for."""
        neutral = stats.probabilistic_sharpe(0.10, 0.0, 1_000, 0.0, 3.0)
        skewed = stats.probabilistic_sharpe(0.10, 0.0, 1_000, -2.0, 3.0)
        assert skewed < neutral

    def test_fat_tails_are_penalised(self):
        normal = stats.probabilistic_sharpe(0.10, 0.0, 1_000, 0.0, 3.0)
        fat = stats.probabilistic_sharpe(0.10, 0.0, 1_000, 0.0, 12.0)
        assert fat < normal


class TestExpectedMaxSharpe:
    def test_a_single_trial_carries_no_selection_bias(self):
        assert stats.expected_max_sharpe(1, 0.05) == 0.0

    def test_the_bar_rises_with_the_number_of_trials(self):
        bars = [stats.expected_max_sharpe(n, 0.01) for n in (2, 5, 20, 100, 1_000)]
        assert bars == sorted(bars)
        assert all(np.isfinite(bars))

    def test_scales_with_the_standard_deviation_of_trial_sharpes(self):
        a = stats.expected_max_sharpe(50, 0.01)
        b = stats.expected_max_sharpe(50, 0.04)   # 2x the sd
        assert b == pytest.approx(2 * a)

    def test_zero_spread_means_no_bar(self):
        assert stats.expected_max_sharpe(100, 0.0) == 0.0


class TestDeflatedSharpe:
    @staticmethod
    def _returns(n=10_000, mu=0.01, seed=30):
        return np.random.default_rng(seed).normal(mu, 1.0, size=n)

    def test_deflation_never_flatters_the_undeflated_number(self):
        res = stats.deflated_sharpe(self._returns(), n_trials=20, sr_variance=0.001)
        assert res.dsr <= res.psr_vs_zero

    def test_more_trials_means_a_lower_deflated_sharpe(self):
        r = self._returns()
        few = stats.deflated_sharpe(r, n_trials=3, sr_variance=0.001)
        many = stats.deflated_sharpe(r, n_trials=500, sr_variance=0.001)
        assert many.dsr < few.dsr
        assert many.sr_benchmark > few.sr_benchmark
        assert many.sr == few.sr   # the point estimate is untouched

    def test_a_pure_noise_strategy_does_not_survive(self):
        noise = self._returns(mu=0.0, seed=31)
        res = stats.deflated_sharpe(noise, n_trials=50, sr_variance=0.001)
        assert res.dsr < config.get("stats.deflated_sharpe.survival_threshold")

    def test_a_strong_genuine_edge_does_survive(self):
        strong = self._returns(mu=0.06, n=20_000, seed=32)
        res = stats.deflated_sharpe(strong, n_trials=50, sr_variance=0.0001)
        assert res.dsr > config.get("stats.deflated_sharpe.survival_threshold")
        assert res.min_track_record < res.n_obs   # the sample is long enough

    def test_reports_the_moments_it_used(self):
        res = stats.deflated_sharpe(self._returns(), n_trials=10, sr_variance=0.001)
        assert res.kurtosis == pytest.approx(3.0, abs=0.3)   # normal draws
        assert abs(res.skew) < 0.2
        assert res.n_obs == 10_000

    def test_annualised_figure_is_display_only_and_consistent(self):
        res = stats.deflated_sharpe(self._returns(), n_trials=10, sr_variance=0.001)
        assert res.sr_annual == pytest.approx(stats.annualise(res.sr))

    def test_summary_string_states_the_verdict(self):
        res = stats.deflated_sharpe(self._returns(), n_trials=10, sr_variance=0.001)
        assert "Deflated Sharpe Ratio" in str(res)
        assert ("survives" in str(res)) or ("does NOT survive" in str(res))


class TestMinTrackRecordLength:
    def test_no_edge_means_no_track_record_is_ever_long_enough(self):
        assert stats.min_track_record_length(0.01, 0.02, 0.0, 3.0) == float("inf")

    def test_a_bigger_edge_needs_less_data(self):
        small = stats.min_track_record_length(0.03, 0.02, 0.0, 3.0)
        big = stats.min_track_record_length(0.20, 0.02, 0.0, 3.0)
        assert big < small

    def test_fat_tails_demand_a_longer_record(self):
        thin = stats.min_track_record_length(0.10, 0.0, 0.0, 3.0)
        fat = stats.min_track_record_length(0.10, 0.0, 0.0, 15.0)
        assert fat > thin


# ---------------------------------------------------------------------------
# multiple testing
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_a_single_p_value_is_unchanged(self):
        out = stats.benjamini_hochberg([0.03])
        assert out["q_value"].iloc[0] == pytest.approx(0.03)

    def test_q_values_are_monotone_in_p(self):
        p = [0.001, 0.01, 0.02, 0.04, 0.3, 0.8]
        q = stats.benjamini_hochberg(p)["q_value"].to_numpy()
        assert np.all(np.diff(q) >= -1e-12)

    def test_q_never_falls_below_p(self):
        p = [0.001, 0.01, 0.02, 0.04, 0.3, 0.8]
        out = stats.benjamini_hochberg(p)
        assert np.all(out["q_value"] >= out["p_value"] - 1e-12)

    def test_order_of_input_is_preserved(self):
        p = [0.8, 0.001, 0.3]
        out = stats.benjamini_hochberg(p)
        assert list(out["p_value"]) == p
        assert out["q_value"].iloc[1] < out["q_value"].iloc[0]

    def test_all_null_family_rejects_almost_nothing(self):
        p = np.random.default_rng(40).uniform(size=200)   # uniform under the null
        assert stats.benjamini_hochberg(p)["reject"].sum() <= 2

    def test_a_clear_signal_survives_correction(self):
        p = np.concatenate([np.full(5, 1e-6), np.random.default_rng(41).uniform(size=95)])
        assert stats.benjamini_hochberg(p)["reject"].sum() >= 5


# ---------------------------------------------------------------------------
# specification log
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Isolate the spec log so tests never touch outputs/tables/."""
    p = tmp_path / "specification_log.csv"
    monkeypatch.setenv("CRYPTO_GAMMA_SPEC_LOG", str(p))
    return p


class TestSpecId:
    def test_is_stable_across_calls(self):
        a = speclog.spec_id("h2", "quintile_sort", {"n_q": 5})
        b = speclog.spec_id("h2", "quintile_sort", {"n_q": 5})
        assert a == b

    def test_ignores_the_order_keys_were_written_in(self):
        a = speclog.spec_id("h2", "sort", {"n_q": 5, "window": "1h"})
        b = speclog.spec_id("h2", "sort", {"window": "1h", "n_q": 5})
        assert a == b

    def test_a_changed_parameter_is_a_new_trial(self):
        a = speclog.spec_id("h2", "sort", {"n_q": 5})
        b = speclog.spec_id("h2", "sort", {"n_q": 10})
        assert a != b

    def test_the_same_name_under_a_different_hypothesis_is_a_new_trial(self):
        assert speclog.spec_id("h2", "sort", {}) != speclog.spec_id("h3", "sort", {})


class TestSpecLog:
    def test_round_trips_what_was_written(self, log_file):
        speclog.record("h2", "quintile_sort", params={"n_q": 5},
                       n_obs=1_000, statistic="mean_rv", estimate=0.42,
                       se=0.01, t_stat=42.0, p_value=0.0, sharpe=0.012)
        df = speclog.load()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["hypothesis"] == "h2"
        assert row["estimate"] == pytest.approx(0.42)
        assert row["sharpe"] == pytest.approx(0.012)
        assert row["interval"] == config.get("construction.interval")

    def test_creates_the_file_and_its_parent_directory(self, tmp_path, monkeypatch):
        p = tmp_path / "nested" / "deeper" / "log.csv"
        monkeypatch.setenv("CRYPTO_GAMMA_SPEC_LOG", str(p))
        speclog.record("h3", "event_window", params={"k": 5})
        assert p.exists()

    def test_appends_rather_than_overwrites(self, log_file):
        speclog.record("h2", "a", params={"i": 1})
        speclog.record("h2", "b", params={"i": 2})
        assert len(speclog.load()) == 2
        assert log_file.read_text().count("\n") == 3   # header + 2 rows

    def test_missing_log_reads_as_an_empty_frame_not_an_error(self, log_file):
        df = speclog.load()
        assert df.empty
        assert list(df.columns) == speclog.COLUMNS

    def test_stamps_a_config_fingerprint_on_every_row(self, log_file):
        speclog.record("h2", "a")
        assert speclog.load()["config_sha"].iloc[0] == speclog.config_sha()

    def test_defaults_the_sample_window_from_config(self, log_file):
        speclog.record("h2", "a")
        row = speclog.load().iloc[0]
        assert row["sample_start"] == config.get("sample.start")
        assert row["sample_end"] == config.get("sample.end")


class TestTrialCounting:
    def test_rerunning_the_same_spec_is_still_one_trial(self, log_file):
        """The rule that keeps the deflation honest in both directions."""
        for _ in range(5):
            speclog.record("h2", "quintile_sort", params={"n_q": 5}, sharpe=0.01)
        assert len(speclog.load()) == 5      # full audit trail kept
        assert len(speclog.trials()) == 1    # but one thing was tried

    def test_a_rerun_keeps_the_most_recent_numbers(self, log_file):
        speclog.record("h2", "s", params={"n_q": 5}, estimate=1.0)
        speclog.record("h2", "s", params={"n_q": 5}, estimate=2.0)
        assert speclog.trials()["estimate"].iloc[0] == pytest.approx(2.0)

    def test_changing_a_parameter_adds_a_trial(self, log_file):
        speclog.record("h2", "s", params={"n_q": 5})
        speclog.record("h2", "s", params={"n_q": 10})
        assert len(speclog.trials()) == 2

    def test_can_be_restricted_to_one_hypothesis(self, log_file):
        speclog.record("h2", "a", params={"i": 1})
        speclog.record("h3", "b", params={"i": 2})
        speclog.record("h3", "c", params={"i": 3})
        assert len(speclog.trials(hypothesis="h3")) == 2
        assert len(speclog.trials()) == 3

    def test_unknown_hypothesis_gives_an_empty_frame(self, log_file):
        speclog.record("h2", "a")
        assert speclog.trials(hypothesis="h9").empty


class TestSummaryAndDrift:
    def test_summary_counts_trials_runs_and_sharpe_spread(self, log_file):
        speclog.record("h2", "a", params={"i": 1}, sharpe=0.01)
        speclog.record("h2", "b", params={"i": 2}, sharpe=0.03)
        speclog.record("h6", "c", params={"i": 3}, sharpe=0.02)
        s = speclog.summary().set_index("hypothesis")
        assert s.loc["h2", "trials"] == 2
        assert s.loc["ALL", "trials"] == 3
        assert s.loc["h2", "sharpe_var"] == pytest.approx(np.var([0.01, 0.03], ddof=1))
        assert s.loc["ALL", "sharpe_max"] == pytest.approx(0.03)

    def test_summary_of_an_empty_log_is_empty_not_an_error(self, log_file):
        assert speclog.summary().empty

    def test_config_drift_is_silent_when_nothing_changed(self, log_file):
        speclog.record("h2", "a", params={"i": 1})
        speclog.record("h2", "a", params={"i": 1})
        assert speclog.config_drift().empty

    def test_config_drift_flags_a_spec_run_under_two_configurations(self, log_file):
        speclog.record("h2", "a", params={"i": 1})
        # simulate the same spec having been run before config.yaml was edited
        df = speclog.load()
        df.loc[0, "config_sha"] = "0000000000"
        df.to_csv(log_file, index=False)
        speclog.record("h2", "a", params={"i": 1})
        drift = speclog.config_drift()
        assert len(drift) == 1
        assert drift["name"].iloc[0] == "a"


class TestDeflateFromLog:
    """The self-updating behaviour: every logged spec raises the bar."""

    @staticmethod
    def _returns(seed=50):
        return np.random.default_rng(seed).normal(0.02, 1.0, size=10_000)

    def test_trial_count_is_read_from_the_log(self, log_file):
        for i in range(7):
            speclog.record("h6", f"spec_{i}", params={"i": i}, sharpe=0.01 * i)
        assert stats.deflate_from_log(self._returns(), hypothesis="h6").n_trials == 7

    def test_logging_more_specs_raises_the_bar_on_the_same_returns(self, log_file):
        r = self._returns()
        for i in range(3):
            speclog.record("h6", f"spec_{i}", params={"i": i}, sharpe=0.005 * i)
        before = stats.deflate_from_log(r, hypothesis="h6")
        for i in range(3, 40):
            speclog.record("h6", f"spec_{i}", params={"i": i}, sharpe=0.005 * (i % 5))
        after = stats.deflate_from_log(r, hypothesis="h6")
        assert after.n_trials > before.n_trials
        assert after.sr_benchmark > before.sr_benchmark
        assert after.dsr < before.dsr

    def test_an_empty_log_falls_back_to_the_iid_variance_with_a_warning(self, log_file, caplog):
        r = self._returns()
        with caplog.at_level("WARNING"):
            res = stats.deflate_from_log(r)
        assert res.n_trials == 1
        assert res.sr_variance == pytest.approx(1.0 / r.size)
        assert "iid SR variance" in caplog.text

    def test_hypothesis_filter_narrows_the_trial_family(self, log_file):
        for i in range(10):
            speclog.record("h2", f"a_{i}", params={"i": i}, sharpe=0.01)
        for i in range(3):
            speclog.record("h6", f"b_{i}", params={"i": i}, sharpe=0.02 * i)
        r = self._returns()
        assert stats.deflate_from_log(r, hypothesis="h6").n_trials == 3
        assert stats.deflate_from_log(r).n_trials == 13
