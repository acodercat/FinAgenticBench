"""Regression checks for contingent funding and RC-N state migration."""

from evals.scenario_analysis import bank_unused_commitment_draw_provision as rcl
from evals.scenario_analysis import bank_past_due_nonaccrual_migration as rcn
from core.data import load_expansion_table


def _reported(module):
    full = module.analysis()
    return {v.name: round(full[v.name], module.PRECISION[v.name])
            if module.PRECISION[v.name] is not None else full[v.name]
            for v in module.variables}


def test_rcl_cash_bridge_and_provision_are_separate():
    assert load_expansion_table("ffiec_call_reports_commitments").filing_type.eq("031").all()
    result = rcl.analysis()
    assert result["eligible_bank_count"] == 32
    assert result["selected_rssd_id"] == "1394676"
    assert result["selected_commitment_to_equity_pct"] > 3 * result["runner_up_commitment_to_equity_pct"]
    assert 0 < result["bridge_principal_m"] < 0.15 * result["unused_card_m"] + 0.30 * result["unused_ci_m"]
    assert 0 < result["hypothetical_provision_m"] < result["bridge_principal_m"]
    assert rcl.validate(_reported(rcl)).success
    wrong = _reported(rcl)
    wrong["bridge_principal_m"] = round(0.15 * result["unused_card_m"] + 0.30 * result["unused_ci_m"], 4)
    assert not rcl.validate(wrong).success
    wrong = _reported(rcl)
    wrong["ending_book_equity_m"] = round(result["ending_book_equity_m"] - result["bridge_principal_m"], 4)
    assert not rcl.validate(wrong).success
    wrong = _reported(rcl)
    wrong["selected_rssd_id"] = "American Express National Bank"
    assert not rcl.validate(wrong).success


def test_rcn_opening_cohorts_and_new_cohort_cure():
    assert load_expansion_table("ffiec_call_reports_past_due").filing_type.eq("031").all()
    result = rcn.analysis()
    assert result["eligible_bank_count"] == 37
    assert result["selected_rssd_id"] == "940311"
    assert result["post_migration_rate_pct"] > result["selected_noncurrent_to_net_loans_leases_pct"]
    assert result["selected_noncurrent_to_net_loans_leases_pct"] < result["post_cure_rate_pct"] < result["post_migration_rate_pct"]
    assert result["post_migration_nonaccrual_m"] - result["opening_nonaccrual_m"] == 0.4 * result["opening_90_plus_m"]
    assert result["cured_new_90_plus_m"] == 0.25 * 0.2 * result["opening_30_89_m"]
    assert rcn.validate(_reported(rcn)).success
    wrong = _reported(rcn)
    wrong["post_migration_90_plus_m"] = round(result["opening_90_plus_m"] + 0.2 * result["opening_30_89_m"], 4)
    assert not rcn.validate(wrong).success
    wrong = _reported(rcn)
    wrong["cured_new_90_plus_m"] = round(0.25 * result["post_migration_90_plus_m"], 4)
    assert not rcn.validate(wrong).success
    wrong = _reported(rcn)
    wrong["selected_rssd_id"] = "Banco Popular de Puerto Rico"
    assert not rcn.validate(wrong).success
