"""Contract and failure-path checks for the refinance and bank-run pair."""

from evals.comparison_ranking import mortgage_refinance_horizon_sunk_cost_replanning as mortgage
from evals.risk_screening import bank_run_liquidity_equity_frontier as bank


def test_mortgage_executed_choice_and_payoff_acceptance():
    values = mortgage.analysis()
    assert values["eligible_loan_count"] == 69993
    assert tuple(values[f"band_{span}_action"] for span in (
        "3_to_5", "5_to_6", "6_to_7", "7_to_10"
    )) == ("HOLD", "HOLD", "B", "A")
    assert values["offer_b_monthly_payment_mn"] < values["offer_a_monthly_payment_mn"]
    assert values["all_b_saving_mn"] < values["all_a_saving_mn"]
    assert values["early_exit_action"] == "KEEP"
    assert values["restored_horizon_action"] == "PAYDOWN"
    assert mortgage.validate(values).success
    wrong = dict(values, early_exit_payoff_mn=values["early_exit_payoff_mn"] - 1)
    assert not mortgage.validate(wrong).success
    wrong = dict(values, band_7_to_10_action="B")
    assert not mortgage.validate(wrong).success


def test_bank_joint_constraints_and_wrong_capital_paths():
    values = bank.analysis()
    assert values["selected_certificate"] == "622"
    assert values["first_repo_principal_mn"] == 1000
    assert values["frontier_bindings"] == "CASH_COLLATERAL_CAPITAL"
    assert values["second_new_equity_mn"] > 0
    assert values["third_new_equity_mn"] > 0
    assert bank.validate(values).success
    wrong = dict(values, first_ending_equity_assets_pct=values["opening_equity_assets_pct"])
    assert not bank.validate(wrong).success
    wrong = dict(values, second_sale_book_mn=values["second_sale_book_mn"] + 1)
    assert not bank.validate(wrong).success
    wrong = dict(values, frontier_bindings="CASH_ONLY")
    assert not bank.validate(wrong).success


def test_display_precision_boundary_and_canonical_tokens():
    for module in (mortgage, bank):
        full = module.analysis()
        reported = {
            variable.name: (round(full[variable.name], decimals)
                            if decimals is not None else full[variable.name])
            for variable, decimals in zip(module.variables, module.DECIMALS)
        }
        assert module.validate(reported).success
        for name, decimals in zip((v.name for v in module.variables), module.DECIMALS):
            if decimals is None or decimals == 0:
                continue
            outside = dict(reported)
            outside[name] = full[name] + .7 * 10 ** (-decimals)
            assert not module.validate(outside).success, name
