"""Credit-status union and pension target/asset-timing regression checks."""

import pytest

from evals.risk_screening import nport_arrears_default_watchlist_redemption as fund
from evals.scenario_analysis import pension_at_risk_target_asset_timing_plan as pension


def _reported(case):
    return {name: round(value, case.PRECISION[name])
            if isinstance(value, float) and case.PRECISION[name] is not None else value
            for name, value in case.analysis().items()}


def test_nport_status_union_changes_limit_decision_after_stress():
    found = fund.analysis()
    assert found["watchlist_accession"] == "0002071691-25-004164"
    assert found["watchlist_status_rows"] == 3610
    assert (found["watchlist_arrears_count"], found["watchlist_default_count"],
            found["watchlist_overlap_count"]) == (72, 1, 0)
    assert found["watchlist_union_value_m"] == pytest.approx(208.15228433)
    assert found["watchlist_original_breach"] is False
    assert found["watchlist_stress_breach"] is True
    assert found["watchlist_required_sale_m"] == pytest.approx(0.854232282378)
    assert fund.validate(_reported(fund)).success
    wrong = _reported(fund)
    wrong["watchlist_union_value_m"] = round(found["watchlist_default_value_m"], 4)
    assert not fund.validate(wrong).success
    wrong = _reported(fund)
    wrong["watchlist_stress_breach"] = False
    assert not fund.validate(wrong).success
    wrong = _reported(fund)
    wrong["watchlist_required_sale_m"] = 0.0
    assert not fund.validate(wrong).success


def test_pension_target_basis_and_asset_date_affect_contribution():
    found = pension.analysis()
    assert found["at_risk_eligible_plans"] == 12
    assert found["at_risk_selected_filing"] == "20251002153843NAL0000663025001"
    assert found["at_risk_valuation_date"] == "2024-01-01"
    assert found["at_risk_target_contribution_usd"] == 330606450
    assert found["at_risk_without_contribution_usd"] == 323215325
    assert pension.validate(_reported(pension)).success
    wrong = _reported(pension)
    wrong["at_risk_target_contribution_usd"] = (
        found["at_risk_target_usd"] + found["at_risk_repeated_expense_usd"]
        - found["at_risk_h_begin_assets_usd"])
    assert not pension.validate(wrong).success
    wrong = _reported(pension)
    wrong["at_risk_without_contribution_usd"] = found["at_risk_target_contribution_usd"]
    assert not pension.validate(wrong).success
