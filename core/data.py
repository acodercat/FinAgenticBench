from collections.abc import Iterable

import pandas as pd
from cave_agent import Variable

from core.dataset_layers import load_runtime_table


EXPANSION_TABLES = {
    "philadelphia_fed_rtdsm": "rtdsm.revisions",
    "sec_nmfp": "sec_nmfp.series_filings",
    "sec_nmfp_flows": "sec_nmfp.daily_shareholder_flows",
    "sec_nmfp_liquidity": "sec_nmfp.daily_liquidity",
    "sec_nmfp_class_yields": "sec_nmfp.share_class_yields",
    "sec_nport": "sec_nport.fund_reports",
    "sec_nport_interest_rate_risk": "sec_nport.interest_rate_risk",
    "sec_nport_holding_details": "sec_nport.selected_holding_details",
    "sec_nport_lending_cohort": "sec_nport.lending_cohort",
    "sec_nport_repo_collateral": "sec_nport.repo_collateral",
    "hmda": "hmda.loan_applications",
    "sec_notes": "sec_notes.numeric_facts",
    "sec_notes_submissions": "sec_notes.submissions",
    "sec_notes_dimensional": "sec_notes.dimensional_facts",
    "sec_notes_dimensions": "sec_notes.dimensions",
    "sec_13f_filings": "sec_13f.filings",
    "sec_13f_amendment_chain": "sec_13f.amendment_chain",
    "sec_13f_holdings": "sec_13f.holdings_by_cusip",
    "fed_z1_vintages": "fed_z1_vintages.series_observations",
    "sec_ftd": "sec_ftd.fails_to_deliver",
    "treasury_yield_curve": "treasury_rates.yield_curve",
    "sec_form_c": "sec_form_c.filings",
    "sec_form_d": "sec_form_d.filings",
    "bis_credit_gap": "bis.credit_gap",
    "bis_debt_service": "bis.debt_service_ratios",
    "bis_effective_exchange_rates": "bis.effective_exchange_rates",
    "bis_global_liquidity": "bis.global_liquidity",
    "bis_otc_derivatives": "bis.otc_derivatives",
    "bls_qcew": "bls_qcew.area_industry_annual",
    "census_counties": "census_geo_crosswalk.counties_2024",
    "census_zcta_county_relationships": "census_geo_crosswalk.zcta_county_relationships_2020",
    "census_zcta_primary_counties": "census_geo_crosswalk.zcta_primary_counties_2020",
    "cftc_cot": "cftc_cot.financial_futures",
    "fed_stress_tests": "fed_stress_tests.results",
    "ffiec_call_reports_capital": "ffiec_call_reports.bank_capital",
    "ffiec_call_reports_balance": "ffiec_call_reports.bank_balance_sheets",
    "ffiec_call_reports_securities": "ffiec_call_reports.bank_securities_rcb",
    "ffiec_call_reports_commitments": "ffiec_call_reports.bank_unused_commitments_rcl",
    "ffiec_call_reports_past_due": "ffiec_call_reports.bank_past_due_rcn",
    "ffiec_call_reports_recourse": "ffiec_call_reports.bank_sold_recourse_rcs",
    "ffiec_call_reports_time_deposits": "ffiec_call_reports.bank_time_deposits_rce",
    "ffiec_nic_institutions": "ffiec_nic_structure.institution_identifiers_current",
    "ffiec_nic_ownership": "ffiec_nic_structure.ownership_relationships_current",
    "ffiec_y9c": "ffiec_y9c.holding_company_balance_sheets",
    "fhfa_hpi": "fhfa_hpi.house_price_indexes",
    "fdic_sod": "fdic_sod.branches",
    "fdic_bankfind": "fdic.bank_financials",
    "finra_short_volume": "finra_short_volume.daily",
    "eia_steo_vintages": "eia_steo_vintages.wti_monthly_vintages",
    "eia_bulk_wti": "eia_bulk.wti_monthly_spot_prices",
    "eia_bulk_wti_daily": "eia_bulk.wti_daily_spot_prices",
    "fema_disasters": "fema_disasters.declaration_areas",
    "noaa_storm_events": "noaa_storm_events.details",
    "fema_nfip": "fema_nfip.claims",
    "nyfed_reference_rates": "nyfed.reference_rates",
    "ofr_market_stress": "ofr_market_stress.financial_stress_index",
    "sec_bdc": "sec_bdc.investment_schedule",
    "sec_bdc_filings": "sec_bdc.filings",
    "sba_7a": "sba_7a_504.seven_a_loans",
    "sba_504": "sba_7a_504.five_o_four_loans",
    "ncua_call_reports": "ncua_call_reports.institutions",
    "sec_midas_security_exchange": "sec_midas.daily_security",
    "fed_scf": "fed_scf.household_summary",
    "dol_form5500_financials": "dol_form5500.plan_financials",
    "dol_form5500_schedule_sb": "dol_form5500.schedule_sb",
    "treasury_auctions": "treasury_auctions.auctions",
    "treasury_debt_to_penny": "treasury_debt.debt_to_penny",
    "treasury_marketable_securities": "treasury_debt.marketable_securities",
    "treasury_usfr_reconciliations": "treasury_usfr.reconciliations",
    "treasury_tic": "treasury_tic.major_foreign_holders",
    "treasury_dts_operating_cash": "treasury_dts.operating_cash_balance",
    "treasury_dts_public_debt": "treasury_dts.public_debt_transactions",
}


def load_facts() -> pd.DataFrame:
    return load_runtime_table("sec.company_facts")


def load_filings() -> pd.DataFrame:
    return load_runtime_table("sec.filings")


def load_companies() -> pd.DataFrame:
    return load_runtime_table("sec.companies")


def load_financial_statements() -> pd.DataFrame:
    return load_runtime_table("sec.financial_statements")


def load_insider_table(name: str) -> pd.DataFrame:
    tables = {
        "submission": "sec_insider.submissions",
        "reportingowner": "sec_insider.reporting_owners",
        "nonderiv_trans": "sec_insider.nonderivative_transactions",
        "nonderiv_holding": "sec_insider.nonderivative_holdings",
        "deriv_trans": "sec_insider.derivative_transactions",
        "deriv_holding": "sec_insider.derivative_holdings",
        "footnotes": "sec_insider.footnotes",
    }
    try:
        return load_runtime_table(tables[name])
    except KeyError as error:
        raise KeyError(f"unknown SEC insider table: {name}") from error


def load_ken_french_table(name: str) -> pd.DataFrame:
    tables = {
        "daily_factors": "ken_french.daily_factors",
        "daily_industry_returns": "ken_french.daily_industry_returns",
    }
    try:
        return load_runtime_table(tables[name])
    except KeyError as error:
        raise KeyError(f"unknown Ken French table: {name}") from error


def qcew_state_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """The state all-industry annual totals (total-covered and private ownership).

    `bls_qcew.area_industry_annual` carries county and NAICS-sector rows as
    well; the cases that only need the state totals keep the shape of the
    retired `state_annual_totals` table through this projection.
    """
    return frame.loc[
        frame["industry_code"].astype(str).eq("10")
        & frame["aggregation_level"].astype(str).isin(["state_total", "state_by_ownership"])
    ].copy()


def largest_fdic_institutions(
    financials: pd.DataFrame, report_date: str, count: int = 12
) -> pd.DataFrame:
    """The `count` largest FDIC-insured institutions by total assets at one report date.

    `fdic.bank_financials` covers every insured institution, 2019-2025. Cases
    that were authored on the 12-bank cohort (the twelve largest by `ASSET`)
    state this rule in their query and select the cohort through this helper;
    ties break on certificate ascending.
    """
    snapshot = financials.loc[financials.REPDTE.astype(str).eq(report_date)]
    return snapshot.sort_values(["ASSET", "CERT"], ascending=[False, True]).head(count).copy()


def load_expansion_table(source: str) -> pd.DataFrame:
    try:
        return load_runtime_table(EXPANSION_TABLES[source])
    except KeyError as error:
        raise KeyError(f"unknown expansion table alias: {source}") from error


def runtime_variables(data_sources: Iterable[str]) -> list[Variable]:
    """Eager runtime injection; see ``core.runtime_catalog`` for the registry."""
    # Imported here because the catalog's loaders live in this module.
    from core.runtime_catalog import runtime_variables as _runtime_variables

    return _runtime_variables(data_sources)


def one_fact(
    facts: pd.DataFrame,
    *,
    ticker: str,
    accession: str,
    concept: str,
    end_date: str,
    start_date: str | None = None,
    unit: str = "USD",
) -> float:
    mask = (
        facts["ticker"].eq(ticker)
        & facts["accession"].eq(accession)
        & facts["concept"].eq(concept)
        & facts["end_date"].eq(end_date)
        & facts["unit"].eq(unit)
    )
    mask &= facts["start_date"].isna() if start_date is None else facts["start_date"].eq(start_date)
    values = facts.loc[mask, "value"].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"expected exactly one value for {ticker}/{accession}/{concept}; "
            f"found {values.tolist()}"
        )
    return float(values.iloc[0])
