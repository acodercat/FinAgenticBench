"""Source-name outputs must use the exact stored source spelling."""

import importlib

import pytest

from core.data import load_insider_table


@pytest.mark.parametrize(
    ("module_name", "field"),
    [
        ("evals.calculation_reconciliation.bank_year_to_date_flow_derivation", "leader_name"),
        ("evals.calculation_reconciliation.insider_option_exercise_intrinsic_capture", "leading_issuer_name"),
        ("evals.calculation_reconciliation.stress_capital_buffer_minimum_versus_ending", "widest_exercise"),
        ("evals.calculation_reconciliation.xbrl_dimensional_revenue_double_count", "anchor_name"),
        ("evals.calculation_reconciliation.thirteen_f_reporting_unit_break_ownership", "leading_issuer_name"),
        ("evals.comparison_ranking.cblr_election_leverage_capital_comparability", "cblr_below_threshold_name"),
    ],
)
def test_source_name_rejects_case_and_spacing_variants(monkeypatch, module_name, field):
    case = importlib.import_module(module_name)
    names = [variable.name for variable in case.variables]
    expected = [0] * len(names)
    expected[names.index(field)] = "MiXeD  Name, Inc."
    monkeypatch.setattr(case, "ground_truth", lambda: tuple(expected))

    outputs = dict(zip(names, expected))
    assert case.validate(outputs).success
    for replacement in ("mixed  name, inc.", "MiXeD Name, Inc.", " MiXeD  Name, Inc. "):
        assert not case.validate({**outputs, field: replacement}).success


@pytest.mark.runtime_data
@pytest.mark.runtime_data   # reads the insider runtime tables
def test_insider_population_uses_row_form_type_and_iso_transaction_date():
    derivatives = load_insider_table("deriv_trans")
    submissions = load_insider_table("submission")
    dated = derivatives.loc[derivatives.trans_date.astype(str).str.fullmatch(r"2024-\d{2}-\d{2}")]
    amended = dated.merge(
        submissions[["accession_number", "document_type"]], on="accession_number", how="left"
    )
    assert len(amended.loc[(amended.trans_form_type == 4) & amended.document_type.eq("4/A")]) > 0
    assert len(derivatives.loc[derivatives.trans_date.astype(str).str.match(r"24-\d{2}-\d{2}")]) > 0

    case = importlib.import_module(
        "evals.calculation_reconciliation.insider_option_exercise_intrinsic_capture"
    )
    assert case.ground_truth()[0] == len(
        dated.loc[(dated.trans_form_type == 4) & dated.trans_code.eq("M")]
    )
