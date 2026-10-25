"""Financial-result and validator checks for the stress/tax-yield batch."""

import pytest

from evals.calculation_reconciliation import stress_ppnr_pretax_bridge_revenue_sensitivity as stress
from evals.comparison_ranking import money_market_tax_equivalent_yield_switch as tax_yield


def _reported(case):
    return {name: round(value, case.PRECISION[name])
            if isinstance(value, float) and case.PRECISION[name] is not None else value
            for name, value in case.analysis().items()}


def test_stress_income_bridge_flips_simple_screen():
    found = stress.analysis()
    assert (found["stress_bridge_matched_firms"], found["stress_bridge_reversal_firms"]) == (22, 3)
    assert found["stress_bridge_rssd"] == "5006575"
    assert found["stress_bridge_residual_b"] == pytest.approx(-1.5)
    assert found["stress_bridge_restored_firms"] == 1
    assert found["stress_bridge_selected_stressed_pretax_b"] == pytest.approx(0.378715)
    assert stress.validate(_reported(stress)).success
    wrong = _reported(stress)
    wrong["stress_bridge_residual_b"] = 0.0
    assert not stress.validate(wrong).success
    wrong = _reported(stress)
    wrong["stress_bridge_restored_firms"] = 3
    assert not stress.validate(wrong).success


def test_tax_yield_preference_switch_and_sensitivity():
    found = tax_yield.analysis()
    assert (found["tax_yield_government_classes"], found["tax_yield_exempt_classes"]) == (729, 76)
    assert found["tax_yield_government_net_pct"] == pytest.approx(4.1556016308)
    assert found["tax_yield_exempt_net_pct"] == pytest.approx(3.2049488356)
    assert (found["tax_yield_choice_22"], found["tax_yield_choice_24"]) == (
        "GOVERNMENT", "OTHER_TAX_EXEMPT")
    assert (found["tax_yield_shock_choice_22"], found["tax_yield_shock_choice_24"]) == (
        "OTHER_TAX_EXEMPT", "OTHER_TAX_EXEMPT")
    assert tax_yield.validate(_reported(tax_yield)).success
    wrong = _reported(tax_yield)
    wrong["tax_yield_choice_24"] = "GOVERNMENT"
    assert not tax_yield.validate(wrong).success
    wrong = _reported(tax_yield)
    wrong["tax_yield_government_net_pct"] = round(found["tax_yield_government_net_pct"] * 100, 4)
    assert not tax_yield.validate(wrong).success


@pytest.mark.parametrize("replacement", ["Barclays US LLC", "RSSD 5006575", "5006575.0"])
def test_stress_rssd_requires_exact_identifier(replacement):
    answer = _reported(stress)
    assert stress.validate({**answer, "stress_bridge_rssd": "5006575"}).success
    assert not stress.validate({**answer, "stress_bridge_rssd": replacement}).success


@pytest.mark.parametrize("replacement", ["Government", "GOV", "government"])
def test_tax_category_requires_canonical_token(replacement):
    answer = _reported(tax_yield)
    assert tax_yield.validate({**answer, "tax_yield_choice_22": "GOVERNMENT"}).success
    assert not tax_yield.validate({**answer, "tax_yield_choice_22": replacement}).success
