import json
from pathlib import Path

import numpy as np

from evals.calculation_reconciliation import treasury_par_bootstrap_auction_hedge as treasury
from evals.scenario_analysis import industry_erc_ofr_tail_stress as industry
from scripts.audit_case_design import has_closed_text_representation


ROOT = Path(__file__).resolve().parents[1]
CASE_PATHS = (
    ROOT / "evals/calculation_reconciliation/treasury_par_bootstrap_auction_hedge.json",
    ROOT / "evals/scenario_analysis/industry_erc_ofr_tail_stress.json",
)


def _truth(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def _assert_closed_text_outputs(module):
    for variable, value in zip(module.variables, module.ground_truth()):
        if isinstance(value, str):
            assert has_closed_text_representation(variable.description), variable.name


def test_quantitative_cases_have_four_stateful_turns_and_partition_outputs():
    for path in CASE_PATHS:
        spec = json.loads(path.read_text())
        turns = spec["conversations"][0]["turns"]
        assert len(turns) >= 4
        stored = [name for turn in turns for name in turn["stores"]]
        module = treasury if spec["name"].startswith("treasury") else industry
        assert stored == [variable.name for variable in module.variables]
        assert len(stored) == len(set(stored))
        assert len(set(spec["data_sources"])) >= 2


def test_quantitative_ground_truths_pass_all_turn_and_full_validators():
    for module, turn_names in (
        (treasury, (treasury.TURN_1_NAMES, treasury.TURN_2_NAMES, treasury.TURN_3_NAMES, treasury.TURN_4_NAMES)),
        (industry, (industry.TURN_1_NAMES, industry.TURN_2_NAMES, industry.TURN_3_NAMES, industry.TURN_4_NAMES)),
    ):
        truth = _truth(module)
        _assert_closed_text_outputs(module)
        assert module.validate(truth).success
        for index, names in enumerate(turn_names, start=1):
            validator = getattr(module, f"validate_turn_{index}")
            assert validator({name: truth[name] for name in names}).success


def test_treasury_triangular_system_matches_recursive_bootstrap():
    _, _, node_yields, par_rates, recursive = treasury._curve_state()
    count = len(par_rates)
    system = np.zeros((count, count))
    target = np.full(count, 100.0)
    for row, par_rate in enumerate(par_rates):
        coupon = 100.0 * par_rate / 2.0
        system[row, :row] = coupon
        system[row, row] = 100.0 + coupon
    independent = np.linalg.solve(system, target)
    assert np.max(np.abs(independent - recursive)) < 1e-13
    assert np.all(np.diff(independent) < 0)
    assert treasury._maximum_repricing_residual(par_rates, independent) < 1e-11
    assert np.isfinite(node_yields).all()


def test_treasury_curve_convention_and_hedge_invariants_are_live():
    truth = _truth(treasury)
    _, selected = treasury._selected_auction()
    _, _, node_yields, _, discount_factors = treasury._curve_state()
    interpolated = np.interp(
        treasury.GRID_MATURITIES, treasury.NODE_MATURITIES, node_yields
    ) / 100.0
    naive_discount_factors = (
        1.0 + interpolated / 2.0
    ) ** (-2.0 * treasury.GRID_MATURITIES)
    naive_price = treasury._price(float(selected.int_rate), 10.0, naive_discount_factors)
    assert abs(naive_price - truth["target_model_price_per100"]) > 0.05
    assert truth["maximum_par_repricing_residual_per100"] < 1e-10
    assert truth["maximum_post_hedge_key_rate_residual_per100"] < 1e-10
    assert np.isclose(
        truth["five_year_hedge_face_per100_target"],
        0.31786470945525125,
    )
    assert np.isclose(
        truth["ten_year_hedge_face_per100_target"],
        -99.69133057819928,
    )
    assert abs(truth["next_day_hedged_price_change_per100"]) < abs(
        truth["next_day_unhedged_price_change_per100"]
    )


def _erc_newton(covariance, initial=None):
    count = len(covariance)
    budgets = np.full(count, 1.0 / count)
    solution = (
        np.ones(count)
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )

    def objective(point):
        return 0.5 * point @ covariance @ point - budgets @ np.log(point)

    for _ in range(100):
        gradient = covariance @ solution - budgets / solution
        if np.max(np.abs(gradient)) < 1e-12:
            break
        hessian = covariance + np.diag(budgets / solution**2)
        step = np.linalg.solve(hessian, gradient)
        step_size = 1.0
        while (
            np.any(solution - step_size * step <= 0)
            or objective(solution - step_size * step)
            > objective(solution) - 1e-4 * step_size * float(gradient @ step)
        ):
            step_size *= 0.5
        solution -= step_size * step
    else:
        raise AssertionError("independent ERC Newton solve did not converge")
    return solution / solution.sum()


def test_industry_newton_matches_coordinate_descent_and_risk_invariants():
    covariance, eigenvalues, _ = industry._estimation_state()
    coordinate_weights, variance, risk_shares = industry._erc_coordinate_descent(covariance)
    newton_weights = _erc_newton(covariance)
    assert eigenvalues[0] > 0
    assert np.max(np.abs(coordinate_weights - newton_weights)) < 1e-12
    assert np.isclose(coordinate_weights.sum(), 1.0)
    assert np.all(coordinate_weights > 0)
    assert variance > 0
    assert np.max(np.abs(risk_shares - 0.2)) < 1e-12
    for initial in (
        np.full(len(covariance), 0.1),
        np.linspace(0.2, 1.2, len(covariance)),
        np.array([2.0, 0.3, 1.4, 0.6, 3.0]),
    ):
        assert np.max(
            np.abs(_erc_newton(covariance, initial) - coordinate_weights)
        ) < 1e-12


def test_industry_tail_additivity_and_equal_weight_counterfactual():
    truth = _truth(industry)
    assert truth["stress_tail_observation_count"] == 2
    assert abs(truth["erc_es_additivity_residual_pct"]) < 1e-12
    assert truth["largest_erc_es_contributor_industry"] == "Manuf"
    assert not np.isclose(
        truth["erc_stress_expected_shortfall_pct"],
        truth["equal_weight_stress_expected_shortfall_pct"],
        atol=1e-6,
    )


def test_industry_text_outputs_use_exact_canonical_tokens():
    truth = _truth(industry)
    label_names = (
        *industry.TURN_1_NAMES[7:12],
        industry.TURN_2_NAMES[2],
        industry.TURN_2_NAMES[4],
        industry.TURN_4_NAMES[7],
    )
    descriptions = {variable.name: variable.description for variable in industry.variables}
    assert all("canonical Ken French" in descriptions[name] for name in label_names)

    candidate = {name: truth[name] for name in industry.TURN_1_NAMES}
    candidate["volatility_rank_1_industry"] = "Energy"
    assert not industry.validate_turn_1(candidate).success


def test_treasury_greeks_are_pinned_to_the_unrounded_price_path():
    """The greeks are scored strictly against one stated computation.

    Their descriptions pin the convention: compute from the unrounded model prices,
    not from the rounded figures reported for them. So a value recomputed from the
    full-precision prices reproduces the pinned figure exactly, while carrying the
    rounded prices forward drifts outside the shared tolerance. No acceptance path
    is consulted for either.
    """
    truth = _truth(treasury)
    candidate = {name: truth[name] for name in treasury.TURN_3_NAMES}
    base = truth["target_model_price_per100"]
    up = truth["target_parallel_up_price_per100"]
    down = truth["target_parallel_down_price_per100"]
    shock = treasury.BUMP_BP / 10_000.0
    candidate["target_effective_duration_years"] = round(
        (down - up) / (2.0 * base * shock), 6
    )
    candidate["target_effective_convexity"] = round(
        (down + up - 2.0 * base) / (base * shock * shock), 6
    )
    result = treasury.validate_turn_3(candidate)
    assert result.success
    assert result.variable_results["target_effective_convexity"]["correct"] is True

    rounded_base, rounded_up, rounded_down = (
        round(base, 6), round(up, 6), round(down, 6)
    )
    carried = dict(candidate)
    carried["target_effective_duration_years"] = round(
        (rounded_down - rounded_up) / (2.0 * rounded_base * shock), 6
    )
    carried["target_effective_convexity"] = round(
        (rounded_down + rounded_up - 2.0 * rounded_base)
        / (rounded_base * shock * shock), 6
    )
    assert not treasury.validate_turn_3(carried).success

    candidate["target_effective_convexity"] += 0.01
    assert not treasury.validate_turn_3(candidate).success


def test_industry_erc_residual_is_pinned_at_zero_not_a_solver_fingerprint():
    """A convergent solver's residual reports as zero; a non-convergent one does not.

    The residual is pinned at 5 decimals rather than scored against a one-sided bound,
    so the shared rounding-boundary tolerance is what absorbs the difference between
    coordinate descent and any other convergent method. That tolerance is symmetric:
    it cannot reject a value just above a threshold, and it does not reject a small
    negative report even though a maximum absolute deviation cannot be negative. What
    it does reject is a residual large enough to mean the solve did not converge.
    """
    truth = _truth(industry)
    candidate = {name: truth[name] for name in industry.TURN_3_NAMES}
    candidate["maximum_erc_share_deviation_pp"] = 4.548e-7
    assert industry.validate_turn_3(candidate).success

    candidate["maximum_erc_share_deviation_pp"] = 1e-4
    assert not industry.validate_turn_3(candidate).success


def test_quantitative_cases_are_deterministic_and_sources_are_indispensable():
    for module in (treasury, industry):
        first = module.ground_truth()
        module.ground_truth.cache_clear()
        second = module.ground_truth()
        assert first == second

    treasury_spec = json.loads(CASE_PATHS[0].read_text())
    industry_spec = json.loads(CASE_PATHS[1].read_text())
    assert set(treasury_spec["data_sources"]) == {
        "treasury_auctions", "treasury_yield_curve",
    }
    assert set(industry_spec["data_sources"]) == {
        "ofr_market_stress", "ken_french_10_industry_portfolios",
        "ken_french_daily_factors",
    }
    # These source-owned outputs make removal non-cosmetic: no repository
    # substitute can recover the selected security/episode, return covariance,
    # or factor overlay from the remaining declared sources.
    source_owned_outputs = {
        "treasury_auctions": treasury.TURN_1_NAMES,
        "treasury_yield_curve": treasury.TURN_2_NAMES + treasury.TURN_3_NAMES + treasury.TURN_4_NAMES,
        "ofr_market_stress": industry.TURN_1_NAMES[:7],
        "ken_french_10_industry_portfolios": industry.TURN_1_NAMES[7:] + industry.TURN_3_NAMES,
        "ken_french_daily_factors": industry.TURN_2_NAMES[2:] + industry.TURN_3_NAMES[7:],
    }
    assert all(source_owned_outputs.values())


def test_each_quantitative_variable_is_rejected_when_materially_wrong():
    for module in (treasury, industry):
        truth = _truth(module)
        for variable in module.variables:
            candidate = dict(truth)
            value = candidate[variable.name]
            candidate[variable.name] = (
                value + 1_000_000 if isinstance(value, (int, float))
                else "__FINBENCH_WRONG__"
            )
            assert not module.validate(candidate).success, variable.name


def test_public_contract_pins_daily_rebalancing_and_baseline_classification():
    treasury_spec = json.loads(CASE_PATHS[0].read_text())
    industry_spec = json.loads(CASE_PATHS[1].read_text())
    assert treasury_spec["case_type"] == "baseline"
    final_query = industry_spec["conversations"][0]["turns"][3]["query"]
    assert "daily rebalanced" in final_query
    assert "without rebalancing" not in final_query

    # The stress window opens at the training end, not at the peak. Two models
    # read the earlier wording the other way, took 40 observations starting
    # after the peak and lost both tail dates with it.
    opening_query = industry_spec["conversations"][0]["turns"][0]["query"]
    assert (
        "the disjoint stress window from the common date immediately after that "
        "training end" in opening_query
    )
    truth = dict(
        zip((variable.name for variable in industry.variables), industry.ground_truth())
    )
    assert truth["stress_observation_count"] == 60
