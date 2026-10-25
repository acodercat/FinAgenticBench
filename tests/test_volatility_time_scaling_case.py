"""Boundaries for the volatility time-scaling and autocorrelation case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

from math import sqrt

import pytest

from evals.comparison_ranking import volatility_time_scaling_autocorrelation_regimes as case


def _clear():
    for cached in (case._daily, case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert round(answers["early_market_autocorrelation"], 4) == 0.1681
    assert round(answers["early_market_daily_scaled_volatility"], 4) == 0.1319
    assert round(answers["early_market_monthly_scaled_volatility"], 4) == 0.1516
    assert round(answers["late_market_autocorrelation"], 4) == -0.0834
    assert round(answers["late_market_daily_scaled_volatility"], 4) == 0.1966
    assert round(answers["late_market_monthly_scaled_volatility"], 4) == 0.1567
    assert answers["early_largest_ratio_industry"] == "utils_pct"
    assert round(answers["early_largest_ratio"], 4) == 1.4213
    assert answers["late_smallest_ratio_industry"] == "utils_pct"
    assert round(answers["late_smallest_ratio"], 4) == 0.7766
    assert answers["late_ratio_above_one_count"] == 1
    assert round(answers["early_daily_sharpe"], 4) == 0.5192
    assert round(answers["early_monthly_sharpe"], 4) == 0.4699
    assert round(answers["late_daily_sharpe"], 4) == 0.4088
    assert round(answers["late_monthly_sharpe"], 4) == 0.4675


# --- properties (no literals) ----------------------------------------------

def test_the_scaling_gap_follows_the_sign_of_the_autocorrelation():
    """The premise: positive serial correlation puts the monthly-scaled figure
    above the daily-scaled one, negative serial correlation below it."""
    answers = case.analysis()
    for window in ("early", "late"):
        correlation = answers[f"{window}_market_autocorrelation"]
        gap = (
            answers[f"{window}_market_monthly_scaled_volatility"]
            - answers[f"{window}_market_daily_scaled_volatility"]
        )
        assert correlation * gap > 0


def test_the_two_windows_sit_on_opposite_sides():
    answers = case.analysis()
    assert answers["early_market_autocorrelation"] > 0 > answers["late_market_autocorrelation"]


def test_the_distortion_is_material_in_both_windows():
    """A gap inside a few percent would not carry the case; both are large."""
    answers = case.analysis()
    for window in ("early", "late"):
        ratio = (
            answers[f"{window}_market_monthly_scaled_volatility"]
            / answers[f"{window}_market_daily_scaled_volatility"]
        )
        assert abs(ratio - 1) > 0.1


def test_the_industry_extremes_bracket_one():
    answers = case.analysis()
    assert answers["early_largest_ratio"] > 1 > answers["late_smallest_ratio"]


def test_the_later_window_leaves_few_industries_above_one():
    count = case.analysis()["late_ratio_above_one_count"]
    assert 0 < count < len(case.industry_columns()) / 2


def test_the_sharpe_bias_opposes_the_volatility_bias():
    """Daily annualisation overstates the ratio where it understates risk and
    understates it where it overstates risk."""
    answers = case.analysis()
    assert answers["early_daily_sharpe"] > answers["early_monthly_sharpe"]
    assert answers["late_daily_sharpe"] < answers["late_monthly_sharpe"]


def test_every_industry_agrees_with_the_market_in_the_earlier_window():
    early = case._window(case.EARLY)
    assert case._ratios(early).gt(1).all()


def test_no_ranking_needs_its_tie_rule():
    """House rule: a stated tie rule that does not bind is kept, and the case
    asserts it does not, so a later data change is visible."""
    for bounds in (case.EARLY, case.LATE):
        ratios = case._ratios(case._window(bounds)).sort_values()
        assert not ratios.duplicated().any()
        assert ratios.iloc[1] - ratios.iloc[0] > 1e-3
        assert ratios.iloc[-1] - ratios.iloc[-2] > 1e-3


# --- the conventions the query states ----------------------------------------

def test_the_two_tables_share_every_trading_date():
    industries = case.load_ken_french_table("daily_industry_returns")
    factors = case.load_ken_french_table("daily_factors")
    assert set(industries.date.astype(str)) == set(factors.date.astype(str))


def test_log_returns_are_a_different_answer():
    """Log returns are a common convention for volatility work; the query
    states simple returns because the log reading moves every market output."""
    import numpy as np

    for label, bounds in (("early", case.EARLY), ("late", case.LATE)):
        window = case._window(bounds)
        logged = np.log1p(window.market)
        outside = _reported()
        outside[f"{label}_market_autocorrelation"] = round(float(logged.autocorr(1)), 4)
        outside[f"{label}_market_monthly_scaled_volatility"] = round(
            float(logged.groupby(window.month).sum().std(ddof=1) * sqrt(case.MONTHS)), 4
        )
        assert not case.validate(outside).success


def test_the_population_divisor_is_a_different_answer():
    window = case._window(case.EARLY)
    monthly = case._monthly(window, window.market).std(ddof=0) * sqrt(case.MONTHS)
    outside = _reported()
    outside["early_market_monthly_scaled_volatility"] = round(float(monthly), 4)
    assert not case.validate(outside).success


def test_summing_instead_of_compounding_is_a_different_answer():
    window = case._window(case.EARLY)
    summed = window.market.groupby(window.month).sum().std(ddof=1) * sqrt(case.MONTHS)
    outside = _reported()
    outside["early_market_monthly_scaled_volatility"] = round(float(summed), 4)
    assert not case.validate(outside).success


def test_subtracting_the_risk_free_rate_twice_is_a_different_answer():
    """The factor table's market column is already an excess return."""
    for label, bounds in (("early", case.EARLY), ("late", case.LATE)):
        window = case._window(bounds)
        twice = window.market_excess - window.risk_free
        sharpe = twice.mean() * case.TRADING_DAYS / (twice.std(ddof=1) * sqrt(case.TRADING_DAYS))
        outside = _reported()
        outside[f"{label}_daily_sharpe"] = round(float(sharpe), 4)
        assert not case.validate(outside).success


def test_compounding_the_daily_excess_return_is_a_different_answer():
    for label, bounds in (("early", case.EARLY), ("late", case.LATE)):
        window = case._window(bounds)
        compounded = case._monthly(window, window.market_excess)
        sharpe = compounded.mean() * case.MONTHS / (compounded.std(ddof=1) * sqrt(case.MONTHS))
        outside = _reported()
        outside[f"{label}_monthly_sharpe"] = round(float(sharpe), 4)
        assert not case.validate(outside).success


def test_the_autocorrelation_estimator_does_not_move_the_answer():
    """The textbook estimator is stated against rather than relied on; it
    differs by less than the reported precision here."""
    for label, bounds in (("early", case.EARLY), ("late", case.LATE)):
        returns = case._window(bounds).market.to_numpy(dtype=float)
        centred = returns - returns.mean()
        textbook = float((centred[1:] * centred[:-1]).sum() / (centred ** 2).sum())
        assert round(textbook, 4) == round(case.analysis()[f"{label}_market_autocorrelation"], 4)


# --- rejections -------------------------------------------------------------

def test_strict_display_boundary_both_sides():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        if precision is None:
            continue
        for direction in (1, -1):
            outside = dict(reported)
            step = 0.7 * 10 ** (-precision) if precision else 1
            outside[variable.name] = answers[variable.name] + direction * step
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


def test_industry_labels_are_matched_verbatim():
    outside = _reported()
    outside["early_largest_ratio_industry"] = "Utils"
    assert not case.validate(outside).success


# --- source dependence ------------------------------------------------------

@pytest.mark.parametrize("missing", ["daily_industry_returns", "daily_factors"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()
    real = case.load_ken_french_table

    def guarded(name):
        if name == missing:
            raise RuntimeError(f"{name} unavailable")
        return real(name)

    try:
        monkeypatch.setattr(case, "load_ken_french_table", guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
