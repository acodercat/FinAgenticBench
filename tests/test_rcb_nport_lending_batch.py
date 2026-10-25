"""Meaningful accounting and exposure boundaries for the two-case batch."""

from core.data import load_expansion_table
from evals.scenario_analysis import bank_rcb_pledge_allocation_runoff as bank
from evals.scenario_analysis import nport_lending_reinvestment_redemption_bridge as fund


def _reported(module):
    full = module.analysis()
    return {v.name: round(full[v.name], module.PRECISION[v.name])
            if module.PRECISION[v.name] is not None else full[v.name]
            for v in module.variables}


def test_bank_unknown_pledge_allocation_changes_sale_accounting():
    result = bank.analysis()
    assert result["selected_rssd_id"] == "619877"
    assert result["minimum_unpledged_afs_m"] < result["cash_shortfall_m"]
    assert result["maximum_unpledged_afs_m"] > result["cash_shortfall_m"]
    assert result["afs_only_feasibility"] == "allocation-dependent"
    assert 0 < result["pessimistic_sale_loss_m"] < result["htm_gap_m"]
    assert bank.validate(_reported(bank)).success
    naive = _reported(bank)
    naive["pessimistic_sale_loss_m"] = round(result["pessimistic_sale_loss_m"] + result["afs_gap_m"], 3)
    assert not bank.validate(naive).success
    securities = load_expansion_table("ffiec_call_reports_securities")
    usaa = securities.loc[securities.bank_rssd_id.eq("619877")].iloc[0]
    assert usaa.reporting_basis == "consolidated"


def test_fund_partial_lending_and_collateral_reinvestment_are_distinct():
    result = fund.analysis()
    holdings = load_expansion_table("sec_nport_lending_cohort")
    equities = holdings.loc[holdings.asset_category.eq("EC")]
    loaned = equities.loc[equities.is_loan_by_fund.eq("Y")]
    entirely_unloaned_equity = equities.loc[equities.is_loan_by_fund.eq("N"), "currency_value_usd"].sum() / 1e6
    assert result["unloaned_equity_value_m"] > entirely_unloaned_equity + 100
    assert result["immediate_funding_gap_m"] > 0
    assert result["immediate_funding_gap_m"] < result["cash_collateral_reinvestment_m"]
    assert len(loaned) == result["loaned_holding_count"]
    assert fund.validate(_reported(fund)).success
    naive = _reported(fund)
    naive["immediate_funding_gap_m"] = round(max(0, result["immediate_funding_gap_m"] - result["cash_collateral_reinvestment_m"]), 3)
    assert not fund.validate(naive).success
