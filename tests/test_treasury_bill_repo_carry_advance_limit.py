"""Financing-boundary checks for the bill and daily repo case."""

from evals.scenario_analysis import treasury_bill_repo_carry_advance_limit as case


def test_bill_repo_carry_and_advance_boundaries():
    answer = dict(zip((variable.name for variable in case.variables), case.ground_truth()))
    assert case.validate(answer).success
    rounded = dict(answer)
    for variable, decimals in zip(case.variables, case.DECIMALS):
        if decimals is not None and decimals > 0:
            rounded[variable.name] = round(rounded[variable.name], decimals)
    assert case.validate(rounded).success
    assert answer["bill_cusip"] == "912797QS9"
    assert answer["bank_discount_rate_pct"] < answer["investment_yield_pct"]
    discount = 10_000_000 - answer["purchase_cash_usd"]
    assert answer["repo_interest_usd"] < discount < answer["stressed_repo_interest_usd"]
    assert answer["maximum_advance_pct"] < 90
    for name, wrong in (
        ("investment_yield_pct", answer["bank_discount_rate_pct"]),
        ("repo_interest_usd", answer["repo_interest_usd"] * 360 / 365),
        ("maximum_advance_pct", 90),
    ):
        assert not case.validate(dict(answer, **{name: wrong})).success
