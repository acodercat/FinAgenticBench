"""Focused author checks for the unregistered disclosure/mortgage candidates."""

import importlib

import numpy as np

CASES = (
    (
        "disclosure",
        "evals.temporal_path.insider_disclosure_lineage_market_windows",
        (
            "validate_sale_filing_selection", "validate_disclosure_lineage",
            "validate_market_window_coverage", "validate_post_window_activity",
        ),
    ),
    (
        "mortgage",
        "evals.comparison_ranking.mortgage_denial_support_institution_influence",
        (
            "validate_linked_decision_population", "validate_common_support",
            "validate_direct_standardization", "validate_lender_influence",
        ),
    ),
)


def _truth(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def test_disclosure_boolean_contract_rejects_wrong_types_and_values():
    module = importlib.import_module(
        "evals.temporal_path.insider_disclosure_lineage_market_windows"
    )
    truth = _truth(module)
    outputs = {name: truth[name] for name in module.TURN_3_NAMES}
    for name in ("matched_complete_window", "selected_complete_window"):
        for value in (0, 1, np.int64(0), np.int64(1), "false", "true", "", not truth[name], np.bool_(not truth[name])):
            verdict = module.validate_turn_3({**outputs, name: value})
            assert not verdict.success
            assert not verdict.variables_not_set
            assert verdict.variable_results[name]["is_set"]
            assert not verdict.variable_results[name]["correct"]


def test_disclosure_accepts_python_and_numpy_booleans():
    module = importlib.import_module(
        "evals.temporal_path.insider_disclosure_lineage_market_windows"
    )
    truth = _truth(module)
    outputs = {name: truth[name] for name in module.TURN_3_NAMES}
    for name in ("matched_complete_window", "selected_complete_window"):
        for value in (bool(truth[name]), np.bool_(truth[name])):
            assert module.validate_turn_3({**outputs, name: value}).success


def test_disclosure_missing_boolean_retains_unset_diagnostics():
    module = importlib.import_module(
        "evals.temporal_path.insider_disclosure_lineage_market_windows"
    )
    truth = _truth(module)
    outputs = {name: truth[name] for name in module.TURN_3_NAMES}
    for name in ("matched_complete_window", "selected_complete_window"):
        missing_key = {key: value for key, value in outputs.items() if key != name}
        missing_value = {**outputs, name: None}
        for candidate in (missing_key, missing_value):
            verdict = module.validate_turn_3(candidate)
            assert not verdict.success
            assert verdict.variables_not_set
            assert verdict.variable_results[name] == {
                "is_set": False, "correct": False, "message": "not set",
            }


def test_candidate_anchor_values_remain_stable():
    disclosure = importlib.import_module(
        "evals.temporal_path.insider_disclosure_lineage_market_windows"
    )
    d = _truth(disclosure)
    assert d["selected_accession"] == "0001883850-24-000012"
    assert d["matched_accession"] == "0001883850-24-000007"
    assert (d["matched_pre_days"], d["matched_complete_window"]) == (3, False)
    assert round(d["matched_post_hidden_pct"], 4) == 47.6949

    mortgage = importlib.import_module(
        "evals.comparison_ranking.mortgage_denial_support_institution_influence"
    )
    m = _truth(mortgage)
    assert (m["linked_bank_count"], m["support_cell_count"]) == (175, 7)
    assert round(m["standardized_gap_pp"], 4) == 6.4440
    assert m["influential_lei"] == "593C3GZG957YOJPS2Z63"
