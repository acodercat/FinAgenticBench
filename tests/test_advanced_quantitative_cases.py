import json
from pathlib import Path

import numpy as np

from evals.temporal_path import bis_currency_credit_cointegration_branch as bis
from evals.temporal_path import treasury_dns_inversion_forecast as dns
from scripts.audit_case_design import has_closed_text_representation


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ROOT / "evals/temporal_path/treasury_dns_inversion_forecast.json",
    ROOT / "evals/temporal_path/bis_currency_credit_cointegration_branch.json",
)


def _truth(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def test_group_one_has_four_stateful_multisource_turns():
    for path, module in zip(CASES, (dns, bis)):
        spec = json.loads(path.read_text())
        turns = spec["conversations"][0]["turns"]
        assert len(turns) >= 4
        stored = [name for turn in turns for name in turn["stores"]]
        assert stored == [variable.name for variable in module.variables]
        assert len(stored) == len(set(stored))
        assert len(set(spec["data_sources"])) >= 2


def test_group_one_truth_passes_each_turn_and_full_validator():
    for module in (dns, bis):
        truth = _truth(module)
        assert all(
            has_closed_text_representation(variable.description)
            for variable, value in zip(module.variables, module.ground_truth())
            if isinstance(value, str)
        )
        assert module.validate(truth).success
        for index in range(1, 5):
            names = getattr(module, f"TURN_{index}_NAMES")
            assert getattr(module, f"validate_turn_{index}")(
                {name: truth[name] for name in names}
            ).success


def test_dns_bootstrap_matches_independent_triangular_solve():
    training, _, _, _, _, _, _ = dns._dns_state()
    for _, row in training.iloc[[0, -1]].iterrows():
        node = np.array([float(row[column]) for column in dns.NODE_COLUMNS])
        par = np.interp(dns.GRID, dns.NODE_MATURITIES, node) / 100.0
        system = np.zeros((len(par), len(par)))
        for index, rate in enumerate(par):
            coupon = rate / 2.0
            system[index, :index] = coupon
            system[index, index] = 1.0 + coupon
        independent = np.linalg.solve(system, np.ones(len(par)))
        recursive_zero, residual = dns._bootstrap(row)
        independent_zero = -np.log(independent) / dns.GRID * 100
        assert np.max(np.abs(independent_zero - recursive_zero)) < 1e-12
        assert residual < 1e-11


def test_dns_shared_decay_is_local_minimum_and_ols_moments_hold():
    _, panel, decay, loadings, factors, fitted, _ = dns._dns_state()

    def objective(value):
        matrix = dns._loadings(value)
        coefficients = np.linalg.solve(
            matrix.T @ matrix, matrix.T @ panel.T
        ).T
        return float(np.mean((panel - coefficients @ matrix.T) ** 2))

    assert objective(decay) < objective(decay - 1e-4)
    assert objective(decay) < objective(decay + 1e-4)
    assert np.max(np.abs(loadings.T @ (panel - fitted).T)) < 1e-10
    independent_factors = np.linalg.solve(
        loadings.T @ loadings, loadings.T @ panel.T
    ).T
    assert np.max(np.abs(independent_factors - factors)) < 1e-11


def test_dns_forecast_invariants_and_deterministic_replay():
    truth = _truth(dns)
    assert truth["dns_var_spectral_radius"] < 1
    first = dns.ground_truth()
    dns.ground_truth.cache_clear()
    dns._dns_state.cache_clear()
    second = dns.ground_truth()
    assert first == second


def test_bis_component_identity_and_independent_normal_equations():
    panel, training, long_coef, residual, _, _, rejects, label, coef, _, moment, _ = bis._model_state()
    identity = (
        panel["B: Credit (loans & debt securities)"]
        - panel["D: Debt securities"]
        - panel["G: Loans and deposits"]
    )
    assert np.max(np.abs(identity)) < 0.002
    loans = np.log(training["G: Loans and deposits"].to_numpy(dtype=float))
    debt = np.log(training["D: Debt securities"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(training)), loans])
    independent = np.linalg.solve(design.T @ design, design.T @ debt)
    assert np.max(np.abs(independent - long_coef)) < 1e-11
    assert not rejects
    assert label == "first_difference_ardl"
    assert len(coef) == 4
    assert moment < 1e-10
    assert np.std(residual, ddof=1) > 0


def test_bis_branch_boundary_and_forecast_arithmetic_are_live():
    truth = _truth(bis)
    assert truth["residual_adf_rejects_unit_root"] == (
        truth["residual_adf_t_stat"] < bis.ADF_CRITICAL
    )
    assert truth["conditional_test_observation_count"] == 12
    assert not truth["conditional_prediction_beats_no_change"]


def test_bis_discrete_text_contracts_are_exact_and_closed():
    truth = _truth(bis)
    turn_one = {name: truth[name] for name in bis.TURN_1_NAMES}
    turn_one["selected_reference_area"] = "Japan"
    assert not bis.validate_turn_1(turn_one).success

    turn_three = {name: truth[name] for name in bis.TURN_3_NAMES}
    turn_three["selected_dynamic_model"] = (
        "first-difference ardl: debt change on lagged debt change, current "
        "and lagged loan changes"
    )
    assert not bis.validate_turn_3(turn_three).success
    descriptions = {variable.name: variable.description for variable in bis.variables}
    assert "verbatim" in descriptions["selected_reference_area"]
    assert "canonical branch token" in descriptions["selected_dynamic_model"]


def test_group_one_materially_wrong_numbers_fail_and_replay_is_exact():
    for module in (dns, bis):
        truth = _truth(module)
        for index in range(1, 5):
            names = getattr(module, f"TURN_{index}_NAMES")
            candidate = {name: truth[name] for name in names}
            numeric = next(
                name for name in names
                if isinstance(candidate[name], (int, float))
                and not isinstance(candidate[name], bool)
            )
            candidate[numeric] = float(candidate[numeric]) + 1.0
            assert not getattr(module, f"validate_turn_{index}")(candidate).success
        first = module.ground_truth()
        module.ground_truth.cache_clear()
        second = module.ground_truth()
        assert first == second
