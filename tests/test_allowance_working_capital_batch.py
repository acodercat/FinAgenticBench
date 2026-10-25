"""Case-local contract and economically meaningful wrong-path regressions."""

from evals.scenario_analysis import (
    bank_allowance_chargeoff_capital_path as bank,
    corporate_working_capital_credit_terms as working,
)


def _answers(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def test_bank_states_and_double_chargeoff_rejected():
    answer = _answers(bank)
    assert bank.validate(answer).success
    assert 1433 < answer["opening_net_loans_bn"] < 1434
    assert 15 < answer["opening_cet1_pct"] < 16
    assert answer["stress_provision_bn"] > 0
    assert answer["ending_topup_provision_bn"] > 0
    assert answer["new_cet1_required_bn"] > 0
    # The book already recognized this part of the loss when it provisioned.
    wrong = dict(answer)
    wrong["ending_cet1_pct"] -= answer["charged_off_bn"] * .75 / 1856.46605 * 100
    assert not bank.validate(wrong).success
    assert not bank.validate_turn_4(wrong).success


def test_working_capital_funding_and_action_contract():
    answer = _answers(working)
    assert working.validate(answer).success
    assert answer["reported_cash_bn"] == 8.022
    assert answer["initial_revolver_draw_bn"] > 0
    assert answer["base_funding_interest_bn"] > 0
    assert answer["chosen_action"] == "COLLECTIONS"
    # A valid-looking other action and a rate-unit error must both fail PV.
    wrong_action = dict(answer, chosen_action="SUPPLIER")
    assert not working.validate_turn_4(wrong_action).success
    for noncanonical in ("collections", "Collections", "COLLECTION", "CUSTOMER COLLECTIONS", "SUPPLIER "):
        assert not working.validate_turn_4(dict(answer, chosen_action=noncanonical)).success
    wrong_rate = dict(answer, base_funding_interest_bn=answer["base_funding_interest_bn"] * 100)
    assert not working.validate_turn_3(wrong_rate).success
