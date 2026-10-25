"""Financial-state and numerical-boundary checks for the flood/factoring pair."""

import numpy as np

from evals.scenario_analysis import corporate_factoring_revolver_cash_waterfall as cash
from evals.scenario_analysis import flood_parametric_basis_risk_frozen_contract as flood


def _reported(module):
    full = module.analysis()
    return {
        variable.name: round(full[variable.name], precision)
        if precision is not None else full[variable.name]
        for variable, precision in zip(module.variables, module.DECIMALS)
    }


def test_flood_training_only_policy_and_validation_gap():
    panel = flood._panel()
    values = flood.analysis()
    train = panel.loc[panel.month.le("2022-12")]
    validation = panel.loc[panel.month.ge("2023-01")]
    assert len(train) == 240 and len(validation) == 120
    assert panel["index"].ge(0).all()
    original = np.array([values[f"initial_county_{i}_benefit_mn"] for i in range(1, 6)])
    revised = np.array([values[f"revised_county_{i}_benefit_mn"] for i in range(1, 6)])
    assert np.count_nonzero(original > 1e-8) == 3
    assert np.count_nonzero(revised > 1e-8) == 5
    assert values["worst_gap_index_mn"] > 0
    assert values["worst_gap_reference_mn"] > 50
    assert values["worst_gap_contract_mn"] < 1e-8
    assert values["validation_negative_basis_mn"] > values["revised_validation_negative_basis_mn"]
    assert flood.validate(_reported(flood)).success
    assert not flood.validate(dict(_reported(flood), worst_gap_county_fips="JEFFERSON")).success
    assert not flood.validate(dict(_reported(flood), validation_mse_winner="new")).success


def test_flood_reestimated_budget_and_trigger_support(monkeypatch):
    panel = flood._panel()
    train = panel.loc[panel.month.le("2022-12")]
    removed = train.loc[train.month.ne("2021-05")]
    thresholds, caps, policy, premium, score = flood._fit(removed)
    assert all((removed.loc[removed.county_code.eq(c), "index"] >= t).sum() >= 2 for c, t in zip(flood.FIPS, thresholds))
    assert np.all(policy >= 0) and np.all(policy <= caps + 1e-8)
    assert premium < flood.BUDGET_SHARE * removed.reference.sum() / 47
    assert score < 140
    assert train.groupby("month").reference.sum().idxmax() == "2021-05"
    assert len(removed) == 235
    original = flood._fit(train)[2]
    monkeypatch.setattr(flood, "BUDGET_SHARE", .151)
    perturbed = flood._fit(train)[2]
    assert abs(perturbed[-1] - original[-1]) > .1
    assert np.all(np.isfinite(perturbed))


def test_factoring_rights_and_credit_cures(monkeypatch):
    values = cash.analysis()
    state = cash._state()
    assert values["usable_cash_mn"] == 1351
    assert values["cash_including_restricted_mn"] == 1400
    assert values["current_restricted_cash_mn"] == 49
    assert values["cash_including_restricted_mn"] - values["current_restricted_cash_mn"] == values["usable_cash_mn"]
    assert values["already_sold_face_mn"] == 159
    assert values["day_zero_mar_lot"] == "SELL"
    assert values["day_zero_feb_lot"] == "KEEP"
    april = values["day_zero_april_face_mn"]
    extra = values["additional_april_factor_face_mn"]
    assert april + extra < .2 * values["owned_receivables_mn"]
    assert abs(665 - 159 - 200 - april - extra - values["factor_remaining_headroom_mn"]) < 1e-8
    day_zero = cash._path(state, mar_lot=200, april_sale=april)
    factored = cash._path(state, mar_lot=200, april_sale=april, delay=.75, extra_sale=extra)
    amended = cash._path(state, mar_lot=200, april_sale=april, delay=.75, amend_fee=15)
    assert abs(day_zero["ledger"][1][1] - 600) < 1e-8
    assert factored["peak"] <= 600 + 1e-8
    assert amended["peak"] > values["stressed_march_credit_demand_mn"]
    assert factored["debt"] == amended["debt"] == 0
    assert factored["cost"] < amended["cost"]
    assert cash.validate(_reported(cash)).success
    assert not cash.validate(dict(_reported(cash), day_zero_mar_lot="SOLD")).success
    assert not cash.validate(dict(_reported(cash), chosen_cure="FACTORING")).success
    rate_stress = (state[0], state[1], state[2], state[3] + .0025, state[4])
    _, stressed_feb, stressed_mar, stressed_april, _ = cash._day_zero_best(rate_stress)
    assert stressed_feb == 0 and stressed_mar == 200
    assert abs(stressed_april - april) < 1e-6
    spend = cash.SPENDING_MN.copy()
    monkeypatch.setattr(cash, "SPENDING_MN", spend + np.array([0, 1, 0, 0, 0, 0]))
    _, higher_spend_feb, higher_spend_mar, higher_spend_april, _ = cash._day_zero_best(state)
    assert higher_spend_feb == 0 and higher_spend_mar == 200
    assert higher_spend_april > april


def test_interest_beginning_debt_and_cash_reconciliation():
    state = cash._state()
    _, _, _, rate, days = state[:5]
    path = cash._path(state, mar_lot=200, april_sale=29.1666666666667)
    debt = [0] + [row[1] for row in path["ledger"]]
    for i, (_, _, interest) in enumerate(path["ledger"]):
        assert abs(interest - debt[i] * rate[i] * days[i] / 360) < 1e-10
    assert abs(sum(row[2] for row in path["ledger"]) - path["interest"]) < 1e-10
    assert abs(path["cash"] - (1351 + 3874 - sum(cash.SPENDING_MN) - .03*200 - .04*29.1666666666667 - 10 - path["interest"])) < 1e-8


def test_strict_display_boundary_each_numeric_output():
    for module in (flood, cash):
        full = module.analysis()
        reported = _reported(module)
        for variable, precision in zip(module.variables, module.DECIMALS):
            if precision in (None, 0):
                continue
            wrong = dict(reported)
            wrong[variable.name] = full[variable.name] + .7 * 10 ** (-precision)
            assert not module.validate(wrong).success, variable.name


def test_exact_token_surface_rejects_aliases():
    alternatives = {
        flood: {
            "county_1_fips": "22019.0",
            "county_2_fips": "East Baton Rouge Parish",
            "county_3_fips": "LA-22051",
            "county_4_fips": "22103 ",
            "county_5_fips": "089",
            "worst_gap_county_fips": "Jefferson",
            "worst_gap_month": "September 2024",
            "deleted_train_month": "May 2021",
            "validation_mse_winner": "revised",
        },
        cash: {
            "issuer_cik": "0001103982",
            "original_accession": "000110398225000030",
            "first_no_factor_breach_month": "Feb 2025",
            "day_zero_feb_lot": "NO",
            "day_zero_mar_lot": "YES",
            "stressed_first_breach_month": "March 2025",
            "chosen_cure": "factoring",
        },
    }
    for module, alias_map in alternatives.items():
        reported = _reported(module)
        for name, alias in alias_map.items():
            assert not module.validate(dict(reported, **{name: alias})).success, name
