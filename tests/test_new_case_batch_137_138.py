"""Financial-state checks for the issuer-note and margin-account pair."""

import numpy as np

from evals.temporal_path import bill_refinancing_settlement_face_cap_replanning as bill
from evals.temporal_path import industry_margin_liquidation_path_replanning as margin


def _reported(module):
    full = module.analysis()
    return {
        v.name: round(full[v.name], digits) if digits is not None else full[v.name]
        for v, digits in zip(module.variables, module.DECIMALS)
    }


def test_bill_cusip_grain_and_executed_branch_conservation():
    due, maturity, prices, events, factors = bill._inputs()
    assert len(due) == 25 and abs(sum(due.values()) - 4.3841507164) < 1e-10
    assert len(events) == 26 and len(factors) == len(events)
    assert maturity[-1] == "2025-04-01" and all(0 < p < 1 for p in prices)
    original = bill._best()
    tightened = bill._best(prefix=original["lots"][:2], face_cap=2.3, emergency=True)
    restored = bill._best(prefix=tightened["lots"][:3], face_cap=2.3, emergency=True, restore=True)
    assert original["lots"] == (4, 19, 6, 19)
    assert tightened["lots"][:2] == original["lots"][:2]
    assert restored["lots"][:3] == tightened["lots"][:3]
    assert original["rows"]["2025-01-28"] == tightened["rows"]["2025-01-28"]
    assert tightened["rows"]["2025-02-27"] == restored["rows"]["2025-02-27"]
    assert tightened["fee"] == restored["fee"] == .01
    assert tightened["first_extra_date"] == restored["first_extra_date"] == "2025-02-27"
    assert tightened["peak_extra"] > .16
    assert bill._simulate(tightened["lots"], face_cap=2.3, emergency=False) is None
    for path in (original, tightened, restored):
        assert abs(
            path["cash"] - path["base"] - path["extra"] - path["live_face"]
            - (bill.OPENING_CASH_MN - sum(due.values()) - path["cost"])
        ) < 1e-8


def test_bill_output_boundaries_and_exact_dates():
    values = _reported(bill)
    assert bill.validate(values).success
    assert not bill.validate(dict(values, jan_7_lots=4.1)).success
    assert not bill.validate(dict(values, first_emergency_draw_date="February 27, 2025")).success
    assert not bill.validate(dict(values, last_legacy_maturity_date="2025-03-28")).success
    for v, digits in zip(bill.variables, bill.DECIMALS):
        if digits is None:
            continue
        wrong = dict(values)
        wrong[v.name] = bill.analysis()[v.name] + .7 * 10 ** (-digits)
        assert not bill.validate(wrong).success, v.name


def test_margin_irreversible_sale_and_outside_capital():
    no_sale = margin._path()
    liquidated = margin._path(policy="liquidate")
    injected = margin._path(policy="inject")
    tightened = margin._path(policy="inject", tightened=True)
    assert no_sale["tokens"] == ("Durbl", "Enrgy", "HiTec")
    assert no_sale["low_day"] == liquidated["first_sale"] == "2023-01-03"
    assert liquidated["sale_sum"] > 5 and liquidated["terminal_assets"] < no_sale["terminal_assets"]
    assert liquidated["terminal_debt"] < no_sale["terminal_debt"]
    assert abs(injected["injection"] - .0285873270605137) < 1e-10
    assert injected["sales"] == []
    assert injected["adjusted_terminal_equity"] < injected["terminal_equity"]
    assert tightened["first_sale"] == "2023-03-09"
    assert tightened["sale_sum"] > 3
    assert tightened["adjusted_terminal_equity"] < injected["adjusted_terminal_equity"]


def test_margin_full_grid_boundary_and_strict_output_tokens():
    assert margin._path(leverage=2.3, policy="liquidate", tightened=True)["sales"] == []
    higher = margin._path(leverage=2.35, policy="liquidate", tightened=True)
    assert higher["sales"] and higher["sales"][0][0] == "2023-03-15"
    values = _reported(margin)
    assert margin.validate(values).success
    for name, alias in {
        "industry_1": "Durable Goods",
        "industry_2": "ENRGY",
        "industry_3": "High Tech",
        "unliquidated_low_margin_day": "Jan 3, 2023",
        "first_liquidation_day": "2023/01/03",
        "tightened_first_sale_day": "2023-03-10",
    }.items():
        assert not margin.validate(dict(values, **{name: alias})).success, name
    for v, digits in zip(margin.variables, margin.DECIMALS):
        if digits is None:
            continue
        wrong = dict(values)
        wrong[v.name] = margin.analysis()[v.name] + .7 * 10 ** (-digits)
        assert not margin.validate(wrong).success, v.name
    assert np.isfinite(margin.analysis()["best_no_sale_terminal_equity_mn"])
