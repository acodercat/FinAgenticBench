"""Decision-boundary checks for the branch-runoff and Reg CF cases."""

from decimal import Decimal

from evals.scenario_analysis import (
    liberty_branch_runoff_funding_equity_headroom as bank,
    reg_cf_deadline_bridge_cash_shortfall as crowd,
)


def _answer(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def _rounded_answer(module):
    answer = _answer(module)
    for variable, decimals in zip(module.variables, module.DECIMALS):
        if decimals is not None and decimals > 0:
            answer[variable.name] = round(answer[variable.name], decimals)
    return answer


def test_bank_funding_and_equity_boundary():
    answer = _answer(bank)
    assert bank.validate(answer).success
    assert bank.validate(_rounded_answer(bank)).success
    assert answer["deposit_reconciliation"] == "MATCH"
    assert not bank.validate(dict(answer, deposit_reconciliation="MISMATCH")).success
    assert answer["max_ct_runoff_pct"] > 20
    assert answer["max_ct_runoff_pct"] < 25
    assert answer["year_end_equity_k"] > 1_095_000
    assert answer["extra_stress_equity_k"] < 1_095_000
    # Selling at book instead of 98% leaves this much funding unaccounted for.
    wrong_book_funding = dict(answer, initial_borrowing_k=answer["initial_borrowing_k"] - answer["securities_loss_k"])
    assert not bank.validate(wrong_book_funding).success
    assert not bank.validate(dict(answer, year_end_interest_k=0)).success
    assert not bank.validate(dict(answer, max_ct_runoff_pct=answer["max_ct_runoff_pct"] / 100)).success
    assert bank.validators["validate_turn_5"] is not None


def test_crowdfunding_preclose_cash_and_whole_share_boundary():
    answer = _answer(crowd)
    assert crowd.validate(answer).success
    assert crowd.validate(_rounded_answer(crowd)).success
    assert answer["first_negative_cash_date"] == "2025-09-06"
    assert abs(answer["prior_close_cash_usd"] - 23970.42793) < 1e-8
    assert answer["minimum_subscription_shares"] == 118_349
    assert answer["stressed_subscription_shares"] == 126_294
    assert answer["stressed_bridge_usd"] > answer["preclose_shortfall_usd"]
    assert answer["minimum_subscription_shares"] * 2.93 > answer["target_offering_usd"]
    assert answer["stressed_subscription_shares"] * 2.93 < answer["maximum_offering_usd"]
    for field in ("minimum_subscription_shares", "stressed_subscription_shares"):
        assert not crowd.validate(dict(answer, **{field: answer[field] - 1})).success
    assert not crowd.validate(dict(answer, preclose_shortfall_usd=0)).success
    assert not crowd.validate(dict(answer, first_negative_cash_date="2025-9-6")).success
    # The same-day effective fixing has not yet been published at the daily reset.
    assert not crowd.validate(dict(answer, prior_close_cash_usd=23971.03173888889)).success


def test_crowdfunding_one_share_too_few_breaks_reserve():
    answer = _answer(crowd)
    cash, debt, _, _ = crowd._filing_state()
    debt_factor = crowd._factor("2025-06-18", "2025-10-16", "0.03")
    bridge_factor = crowd._factor("2025-08-31", "2025-10-16", "0.05")
    for shares, bridge in (
        (answer["minimum_subscription_shares"], answer["preclose_shortfall_usd"]),
        (answer["stressed_subscription_shares"], answer["stressed_bridge_usd"]),
    ):
        obligations = debt * (1 + debt_factor) + Decimal(str(bridge)) * (1 + bridge_factor)
        one_less_cash = Decimal(shares - 1) * Decimal("2.93") * Decimal("0.90") - obligations
        enough_cash = Decimal(shares) * Decimal("2.93") * Decimal("0.90") - obligations
        assert one_less_cash < Decimal(50_000) <= enough_cash
