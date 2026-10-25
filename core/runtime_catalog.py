"""The registry of governed tables a case can expose to its runtime.

One entry per runtime table: the name the agent sees, the ``data_sources``
labels that expose it, the loader that materialises it, and the description
the runtime hands to the agent. Everything the evaluator needs to build either
an eager runtime (every declared table loaded up front as a DataFrame) or a
lazy one (a ``DataStore`` handle that loads on request) derives from this
tuple, so a new table is registered in exactly one place.

Labels are what case JSON declares. A label may expose several tables (a
source family such as ``sec_insider``), and a table may be reachable through
several labels (``companies_df`` accompanies every SEC label). Never make this
module import from ``cases/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd
from cave_agent import Variable

from core.data import (
    load_companies,
    load_expansion_table,
    load_facts,
    load_filings,
    load_financial_statements,
    load_insider_table,
    load_ken_french_table,
)


@dataclass(frozen=True)
class RuntimeTable:
    """A governed table as the agent runtime exposes it."""

    name: str
    labels: tuple[str, ...]
    loader: Callable[[], pd.DataFrame]
    description: str

    @property
    def summary(self) -> str:
        """The first sentence of the description, for table listings."""
        head, _, _ = self.description.partition(". ")
        return head.rstrip(".") + "."


# A family label stands for every table of that source; declaring the family
# is the same as declaring each member.
SOURCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "ffiec_call_reports": ("ffiec_call_reports_capital", "ffiec_call_reports_balance",
                          "ffiec_call_reports_commitments", "ffiec_call_reports_past_due",
                          "ffiec_call_reports_securities", "ffiec_call_reports_recourse",
                          "ffiec_call_reports_time_deposits"),
    "treasury_dts": ("treasury_dts_operating_cash", "treasury_dts_public_debt"),
}


RUNTIME_TABLES: tuple[RuntimeTable, ...] = (
    RuntimeTable(
        name="bis_credit_gap_df",
        labels=("bis_credit_gap",),
        loader=lambda: load_expansion_table("bis_credit_gap"),
        description=(
            "pandas DataFrame of BIS quarterly credit-to-GDP data through 2025-Q4. "
            "Dimensions identify borrower country and sector, lender sector, and "
            "whether the observation is the actual ratio, HP-filter trend or "
            "actual-minus-trend gap; units are supplied in dedicated columns."
        ),
    ),
    RuntimeTable(
        name="bis_debt_service_df",
        labels=("bis_debt_service",),
        loader=lambda: load_expansion_table("bis_debt_service"),
        description=(
            "pandas DataFrame of BIS quarterly debt-service ratios through 2025-Q4. "
            "Dimensions identify borrower country and whether the ratio covers "
            "households, non-financial corporations or the private non-financial "
            "sector; values are percentages."
        ),
    ),
    RuntimeTable(
        name="bis_effective_exchange_rates_df",
        labels=("bis_effective_exchange_rates",),
        loader=lambda: load_expansion_table("bis_effective_exchange_rates"),
        description=(
            "pandas DataFrame of BIS monthly broad effective exchange-rate indexes "
            "for every month from January 2019 through December 2025. It identifies "
            "nominal versus real series and reference economies; indexes use "
            "2020=100. Each reference economy's effective index is measured against "
            "its own broad currency basket, not bilaterally against the U.S. dollar. "
            "The frozen panel is a retrospective snapshot rather than a December-2025 "
            "as-known release."
        ),
    ),
    RuntimeTable(
        name="bis_global_liquidity_df",
        labels=("bis_global_liquidity",),
        loader=lambda: load_expansion_table("bis_global_liquidity"),
        description=(
            "pandas DataFrame of BIS quarterly global-liquidity indicators, all "
            "quarters through 2025-Q4, for USD-, EUR- and JPY-denominated credit to "
            "non-bank borrowers in emerging market and developing economies. Currency "
            "amounts are millions of the named denomination and bank-loan, "
            "debt-security and total-credit rows are separate."
        ),
    ),
    RuntimeTable(
        name="bis_otc_derivatives_df",
        labels=("bis_otc_derivatives",),
        loader=lambda: load_expansion_table("bis_otc_derivatives"),
        description=(
            "pandas DataFrame of BIS semi-annual OTC derivatives outstanding, every "
            "half-year through 2025, covering total instruments, counterparties, "
            "currencies, maturities and ratings. Notional amounts and gross market "
            "values are separate measures in USD millions; risk-category totals can "
            "overlap nested categories."
        ),
    ),
    RuntimeTable(
        name="bls_qcew_state_annual_df",
        labels=("bls_qcew",),
        loader=lambda: load_expansion_table("bls_qcew"),
        description=(
            "pandas DataFrame of BLS QCEW annual averages for 2019 through 2025 at "
            "the state and county grain: aggregation_level is state_total, "
            "state_by_ownership, state_naics_sector, county_total, "
            "county_by_ownership or county_naics_sector; ownership_scope is "
            "total_covered or private; industry_code is 10 for all industries or a "
            "NAICS sector code. Columns: area_fips, state_fips, year, "
            "annual_avg_establishments, annual_avg_employment, "
            "total_annual_wages_usd, annual_avg_weekly_wage_usd, "
            "average_annual_pay_usd, employment_location_quotient, "
            "over_year_employment_change_pct, "
            "over_year_average_annual_pay_change_pct, disclosure_code (N = "
            "suppressed), over_year_disclosure_code, source_archive."
        ),
    ),
    RuntimeTable(
        name="census_counties_df",
        labels=("census_counties", "census_geo_crosswalk"),
        loader=lambda: load_expansion_table("census_counties"),
        description=(
            "pandas DataFrame of Census 2024 county Gazetteer rows. Columns include "
            "USPS state abbreviation and five-digit county GEOID; the first two GEOID "
            "digits provide the state FIPS bridge needed for code-based state joins. "
            "Multiple county rows can share the same state-level USPS/FIPS pair."
        ),
    ),
    RuntimeTable(
        name="census_zcta_county_relationships_df",
        labels=("census_geo_crosswalk", "census_zcta_county_relationships"),
        loader=lambda: load_expansion_table("census_zcta_county_relationships"),
        description=(
            "pandas DataFrame of Census 2020 ZCTA-to-county relationship rows "
            "(GEOID_ZCTA5_20, GEOID_COUNTY_20, names, AREALAND_PART, AREAWATER_PART). "
            "A ZCTA that straddles counties appears once per county with the "
            "overlapping land area, so this is an allocation table, not a one-to-one "
            "map."
        ),
    ),
    RuntimeTable(
        name="census_zcta_primary_counties_df",
        labels=("census_geo_crosswalk", "census_zcta_primary_counties"),
        loader=lambda: load_expansion_table("census_zcta_primary_counties"),
        description=(
            "pandas DataFrame mapping each 2020 ZCTA (zcta5) to one primary county "
            "(county_geoid) chosen by the largest overlapping land area; "
            "selection_rule records the rule. Use it for one-to-one ZIP-to-county "
            "joins."
        ),
    ),
    RuntimeTable(
        name="cftc_financial_futures_df",
        labels=("cftc_cot",),
        loader=lambda: load_expansion_table("cftc_cot"),
        description=(
            "pandas DataFrame of CFTC financial-futures Commitments of Traders "
            "reports, 2019 through 2025, in both futures-only and "
            "futures-and-options-combined scopes. Rows report market, date, contract "
            "code, open interest and leveraged-money long, short and spreading "
            "positions."
        ),
    ),
    RuntimeTable(
        name="form5500_plan_financials_df",
        labels=("dol_form5500_financials",),
        loader=lambda: load_expansion_table("dol_form5500_financials"),
        description=(
            "pandas DataFrame of DOL Form 5500 Schedule H plan financial rows for "
            "form year 2024, limited to filings received by 2025-12-31. Rows retain "
            "filing identifiers, amendment indicators, sponsor and plan identifiers, "
            "participant counts, beginning and ending net assets, income, expenses "
            "and transfers to and from the plan."
        ),
    ),
    RuntimeTable(
        name="form5500_schedule_sb_df",
        labels=("dol_form5500_schedule_sb",),
        loader=lambda: load_expansion_table("dol_form5500_schedule_sb"),
        description=(
            "pandas DataFrame of DOL Form 5500 Schedule SB actuarial funding rows for "
            "form year 2024, limited to filings received by 2025-12-31. Rows retain "
            "filing and plan identifiers, valuation dates, current and actuarial "
            "asset values, funding targets, funding balances and reported FTAP "
            "measures."
        ),
    ),
    RuntimeTable(
        name="eia_wti_monthly_spot_prices_df",
        labels=("eia_bulk_wti",),
        loader=lambda: load_expansion_table("eia_bulk_wti"),
        description=(
            "pandas DataFrame of EIA PET.RWTC.M monthly Cushing WTI spot prices from "
            "1986 through 2025. This is a current revised bulk snapshot downloaded in "
            "2026, not a historical as-known vintage; values are USD per barrel."
        ),
    ),
    RuntimeTable(
        name="eia_wti_daily_spot_prices_df",
        labels=("eia_bulk_wti", "eia_bulk_wti_daily"),
        loader=lambda: load_expansion_table("eia_bulk_wti_daily"),
        description=(
            "pandas DataFrame of EIA PET.RWTC.D daily Cushing WTI spot prices from "
            "1986 through 2025. Rows contain only reported observation dates; "
            "weekends, holidays and other non-observation dates are not synthesized. "
            "This is the same current revised 2026 bulk snapshot family as "
            "PET.RWTC.M, not a historical as-known vintage; values are USD per "
            "barrel."
        ),
    ),
    RuntimeTable(
        name="eia_steo_wti_vintages_df",
        labels=("eia_steo_vintages",),
        loader=lambda: load_expansion_table("eia_steo_vintages"),
        description=(
            "pandas DataFrame of EIA Short-Term Energy Outlook monthly WTI "
            "spot-average values from 24 immutable 2024-2025 publication workbooks. "
            "Rows are keyed by publication date and target month and state whether "
            "that value was forecast or history in that vintage; later-vintage "
            "history must not replace an earlier forecast."
        ),
    ),
    RuntimeTable(
        name="fdic_bank_financials_df",
        labels=("fdic_bankfind",),
        loader=lambda: load_expansion_table("fdic_bankfind"),
        description=(
            "pandas DataFrame of FDIC BankFind quarterly financial fields for every "
            "FDIC-insured institution, quarter-ends 2019-03-31 through 2025-12-31. "
            "REPDTE is report date; CERT is the unpadded FDIC certificate; DEP and "
            "DEPDOM are total and domestic deposits; ASSET is total assets; LNLSNET "
            "is net loans and leases; and NCLNLS is noncurrent loans. These "
            "balance-sheet amount fields are in thousand USD. The table also retains "
            "equity, income, charge-off, profitability, margin and regulatory-capital "
            "fields."
        ),
    ),
    RuntimeTable(
        name="fdic_sod_branches_df",
        labels=("fdic_sod",),
        loader=lambda: load_expansion_table("fdic_sod"),
        description=(
            "pandas DataFrame of FDIC Summary of Deposits branch rows for each annual "
            "snapshot 2019 through 2025 (report_date is the June 30 survey date of "
            "that year). Each row identifies an insured bank and branch, branch state "
            "and county, and branch-reported deposits in thousand USD. "
            "Institution-level balance-sheet fields repeated in the source file are "
            "not projected into this branch table; use BankFind for independent "
            "institution totals."
        ),
    ),
    RuntimeTable(
        name="scf_summary_df",
        labels=("fed_scf",),
        loader=lambda: load_expansion_table("fed_scf"),
        description=(
            "pandas DataFrame of Federal Reserve 2022 Survey of Consumer Finances "
            "public summary extract. Each sampled household has five implicate rows. "
            "survey_weight is the summary macro's WGT, defined as the final analysis "
            "weight X42001 divided by five; summing it across all five implicate rows "
            "estimates the population once. Other columns identify household, record, "
            "implicate, reference-person age and net worth in dollars."
        ),
    ),
    RuntimeTable(
        name="fed_stress_results_df",
        labels=("fed_stress_tests",),
        loader=lambda: load_expansion_table("fed_stress_tests"),
        description=(
            "pandas DataFrame of Federal Reserve supervisory stress-test public "
            "results for every exercise from DFAST 2013 through the 2025 stress test "
            "(exercise_name), one row per exercise x scenario x disclosed entity. "
            "Rows identify the entity by RSSD and disclosure_legal_name; "
            "scenario_name distinguishes Supervisory Severely Adverse, Adverse and "
            "Alternative Severe. Amount columns (_amt) are billions of USD over the "
            "nine-quarter horizon; _rate columns are percent loss rates; the CET1 "
            "columns are actual, end-of-horizon and minimum ratios in percent. These "
            "are conditional supervisory results under hypothetical scenarios, not "
            "forecasts; amount fields are billion USD."
        ),
    ),
    RuntimeTable(
        name="fed_z1_vintages_df",
        labels=("fed_z1_vintages",),
        loader=lambda: load_expansion_table("fed_z1_vintages"),
        description=(
            "pandas DataFrame of Federal Reserve Z.1 Financial Accounts, seven "
            "release vintages from 2024-03-07 through 2025-09-11, unpivoted to one "
            "row per vintage x table x series x date for observations from 2015 "
            "onward. frequency is Q or A; values are in the series' published units "
            "(millions of USD for most levels and flows). Compare the same series "
            "across vintages to measure revisions."
        ),
    ),
    RuntimeTable(
        name="fema_declaration_areas_df",
        labels=("fema_disasters",),
        loader=lambda: load_expansion_table("fema_disasters"),
        description=(
            "pandas DataFrame of OpenFEMA disaster declarations at declared-area "
            "grain, declaration dates 2019-01-01 through 2025-12-31 (current "
            "snapshot). Columns: declaration_area_id, disaster_number, state_usps, "
            "declaration_type (DR/EM/FM), declaration_date, incident_type, "
            "incident_begin_date, incident_end_date, tribal_request, fips_state_code, "
            "fips_county_code, county_fips, designated_area, the four program flags, "
            "snapshot_last_refresh, source_file. One disaster appears once per "
            "designated area; statewide rows carry county code 000. A county can "
            "appear in multiple disaster declarations, so rows are not unique per "
            "county."
        ),
    ),
    RuntimeTable(
        name="nfip_claims_df",
        labels=("fema_nfip",),
        loader=lambda: load_expansion_table("fema_nfip"),
        description=(
            "pandas DataFrame of OpenFEMA NFIP claim records with dates of loss from 2019-01-01 through 2025-12-31. "
            "Each physical row retains its source claim-record ID and reported "
            "policy_count, which can exceed one; row count is not represented policy "
            "count. Gross and signed net building, contents and ICC payment fields "
            "remain separate, as do building and contents coverage. Negative net "
            "payments and missing gross payments are preserved. This is a "
            "retrospective 2026 snapshot."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_capital_df",
        labels=("ffiec_call_reports", "ffiec_call_reports_capital"),
        loader=lambda: load_expansion_table("ffiec_call_reports_capital"),
        description=(
            "pandas DataFrame of FFIEC Call Report regulatory-capital rows for all "
            "reporting banks from 2024-Q1 through 2025-Q3 in the retained official "
            "bulk snapshots. Rows include bank identity, standardized- and "
            "advanced-approaches risk-weighted assets, common equity tier 1 capital, "
            "both reported CET1 ratios, and leverage ratios. Items are coalesced "
            "consolidated-first then domestic (capital_reporting_basis records which); "
            "cblr_election marks community-bank-leverage-ratio electors, who report a "
            "leverage ratio but no risk-weighted assets or CET1 ratio, so those blanks "
            "are not missing data. Monetary values are thousands of dollars; ratios "
            "are percentages."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_balance_sheets_df",
        labels=("ffiec_call_reports", "ffiec_call_reports_balance"),
        loader=lambda: load_expansion_table("ffiec_call_reports_balance"),
        description=(
            "pandas DataFrame of FFIEC Call Report balance-sheet rows for all "
            "reporting banks from 2024-Q1 through 2025-Q3. Bank RSSD identity and "
            "consolidated RCFD versus domestic-office RCON total assets, liabilities "
            "and equity remain separate, with coalesced total_assets/total_liabilities/"
            "total_equity_thousand_usd columns beside them; values are thousand USD. "
            "Historical bulk "
            "files can be revised in place, so this is a retrospective frozen panel "
            "rather than a strict as-known archive."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_securities_rcb_df",
        labels=("ffiec_call_reports_securities",),
        loader=lambda: load_expansion_table("ffiec_call_reports_securities"),
        description=(
            "pandas DataFrame of 2025-09-30 FFIEC Schedule RC-B bank securities "
            "totals by RSSD ID: held-to-maturity and available-for-sale "
            "amortized cost and fair value, plus pledged securities book value. "
            "Amounts are thousand USD; consolidated RCFD is preferred where "
            "reported, otherwise domestic RCON. reporting_basis identifies the "
            "selected scope. This is a retrospective bulk snapshot."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_unused_commitments_df",
        labels=("ffiec_call_reports_commitments",),
        loader=lambda: load_expansion_table("ffiec_call_reports_commitments"),
        description=(
            "pandas DataFrame of FFIEC 031 filer consolidated RCFD 2025-09-30 RC-L unused "
            "credit-card lines and commercial-and-industrial loan commitments by "
            "bank RSSD ID, linked to a contemporaneous FDIC certificate from the "
            "same raw ZIP's POR identity table. Values are thousand USD of unused commitment face amount, "
            "not funded loans, expected draws, losses or regulatory credit equivalents. "
            "Only complete FFIEC 031 RCFD rows are retained from the frozen bulk ZIP; filing_type is explicit."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_past_due_rcn_df",
        labels=("ffiec_call_reports_past_due",),
        loader=lambda: load_expansion_table("ffiec_call_reports_past_due"),
        description=(
            "pandas DataFrame of FFIEC 031 filer consolidated RCFD 2025-09-30 RC-N total "
            "loans and leases in three mutually exclusive reported states: 30-89 "
            "days past due and still accruing, 90 or more days past due and still "
            "accruing, and nonaccrual. Each RSSD ID carries its contemporaneous "
            "FDIC certificate from the same raw ZIP's POR identity table. Values are full outstanding balances in "
            "thousand USD, not missed-installment amounts or charge-offs. Only "
            "complete FFIEC 031 RCFD rows are retained from the frozen bulk ZIP; filing_type is explicit."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_sold_recourse_rcs_df",
        labels=("ffiec_call_reports_recourse",),
        loader=lambda: load_expansion_table("ffiec_call_reports_recourse"),
        description=(
            "pandas DataFrame of consolidated FFIEC 031 Schedule RC-S item 11 sold-but-not-"
            "bank-securitized unpaid principal and item 12 remaining maximum contractual "
            "seller recourse exposure at 2025-09-30, for residential and all-other "
            "columns. Rows include RSSD and same-archive FDIC certificate. Amounts "
            "are thousand USD; maximum contractual exposure is not expected loss."
        ),
    ),
    RuntimeTable(
        name="ffiec_bank_time_deposits_rce_df",
        labels=("ffiec_call_reports_time_deposits",),
        loader=lambda: load_expansion_table("ffiec_call_reports_time_deposits"),
        description=(
            "pandas DataFrame of 2025-09-30 FFIEC 041 Schedule RC-E domestic-office "
            "time deposits split at USD 250,000. For each size class, two buckets "
            "report remaining maturity OR next repricing within one year, while "
            "a separate memorandum reports the subset with contractual remaining "
            "maturity within one year. Amounts are thousand USD; the bucket "
            "difference is long-maturity floating-rate deposits resetting soon, "
            "not cash due within one year. RSSD and FDIC certificate come from "
            "the same 2025Q3 raw archive."
        ),
    ),
    RuntimeTable(
        name="ffiec_nic_institutions_df",
        labels=("ffiec_nic_institutions", "ffiec_nic_structure"),
        loader=lambda: load_expansion_table("ffiec_nic_institutions"),
        description=(
            "pandas DataFrame of current FFIEC NIC institution identifiers observed "
            "2026-08-13. Rows identify legal entities by RSSD ID, legal name, entity "
            "type, location and selected regulator identifiers. This is a current "
            "structural snapshot, not a 2025-12-31 point-in-time institution file."
        ),
    ),
    RuntimeTable(
        name="ffiec_nic_ownership_df",
        labels=("ffiec_nic_ownership", "ffiec_nic_structure"),
        loader=lambda: load_expansion_table("ffiec_nic_ownership"),
        description=(
            "pandas DataFrame of current FFIEC NIC parent-offspring relationships "
            "observed 2026-08-13. Rows identify parent and offspring RSSD IDs and "
            "legal names, relationship start and level, control and equity fields. "
            "Relationship level 1 is direct; level 2 is indirect through an "
            "intervening entity. This is a current structural snapshot, not a "
            "2025-12-31 point-in-time relationship file."
        ),
    ),
    RuntimeTable(
        name="ffiec_y9c_balance_sheets_df",
        labels=("ffiec_y9c",),
        loader=lambda: load_expansion_table("ffiec_y9c"),
        description=(
            "pandas DataFrame of FFIEC holding-company financial rows with nonmissing "
            "Y-9C BHCK consolidated total assets from 2024-Q1 through 2025-Q3. Rows "
            "retain reporter RSSD identity, legal name, total assets, liabilities and "
            "equity in thousand USD. A holding-company consolidated amount can "
            "already include subsidiary-bank amounts and must not be added to a bank "
            "Call Report value. Historical financial downloads can be revised in "
            "place."
        ),
    ),
    RuntimeTable(
        name="fhfa_hpi_df",
        labels=("fhfa_hpi",),
        loader=lambda: load_expansion_table("fhfa_hpi"),
        description=(
            "pandas DataFrame of FHFA house-price indexes through 2025. Rows identify "
            "index type and flavor, frequency, geographic level and place, year and "
            "period, seasonally adjusted and unadjusted values where available, "
            "standard error and notes."
        ),
    ),
    RuntimeTable(
        name="finra_short_volume_df",
        labels=("finra_short_volume",),
        loader=lambda: load_expansion_table("finra_short_volume"),
        description=(
            "pandas DataFrame of FINRA consolidated daily short-volume rows for every "
            "trading date from 2024-01-02 through 2025-12-31. Rows are unique by ISO "
            "date and symbol and report ShortVolume, ShortExemptVolume and "
            "TotalVolume in shares plus market codes. ShortVolume includes executed "
            "short-sale and short-sale-exempt volume; ShortExemptVolume is the "
            "separately reported exempt subset and must not be added again. "
            "Source-file count trailers are excluded. Daily short volume is publicly "
            "disseminated off-exchange transaction volume reported to FINRA "
            "facilities, not short interest."
        ),
    ),
    RuntimeTable(
        name="hmda_lar_df",
        labels=("hmda",),
        loader=lambda: load_expansion_table("hmda"),
        description=(
            "pandas DataFrame of The complete public 2024 HMDA loan/application "
            "register, every reporting institution and state (about 13 million "
            "records). Columns: activity_year, lei, institution_name, state_code, "
            "county_code (five-digit FIPS), derived_msa_md, derived "
            "product/dwelling/ethnicity/race/sex fields, action_taken (1 = "
            "originated, 3 = denied, ...), purchaser_type, loan_type, loan_purpose, "
            "lien_status, business_or_commercial_purpose, loan_amount (USD), "
            "loan_to_value_ratio, interest_rate, rate_spread, property_value, income "
            "(thousands of USD), debt_to_income_ratio (bucketed text), "
            "construction_method, occupancy_type, total_units, applicant_age, "
            "denial_reason_1, tract_minority_population_percent, "
            "tract_to_msa_income_percentage. Code values follow the public HMDA LAR "
            "schema; NA and Exempt are stored as missing in the numeric columns."
        ),
    ),
    RuntimeTable(
        name="ff_industry_daily_df",
        labels=("ken_french_10_industry_portfolios",),
        loader=lambda: load_ken_french_table("daily_industry_returns"),
        description=(
            "pandas DataFrame of daily value-weighted returns for the Kenneth French "
            "10-industry portfolios through the 2025-12-31 observation cutoff, from a "
            "later retrospective source snapshot based on the 202606 CRSP database. "
            "Columns: date and nodur_pct, durbl_pct, manuf_pct, enrgy_pct, hitec_pct, "
            "telcm_pct, shops_pct, hlth_pct, utils_pct and other_pct. Values are "
            "simple daily percentages, not decimals."
        ),
    ),
    RuntimeTable(
        name="ff_daily_factors_df",
        labels=("ken_french_daily_factors",),
        loader=lambda: load_ken_french_table("daily_factors"),
        description=(
            "pandas DataFrame of daily Fama-French research factors frozen at "
            "2025-12-31. Columns: date, mkt_rf_pct, smb_pct, hml_pct and rf_pct. All "
            "return values are simple daily percentages, not decimal fractions; "
            "mkt_rf_pct is the market return minus the risk-free return."
        ),
    ),
    RuntimeTable(
        name="ncua_call_reports_df",
        labels=("ncua_call_reports",),
        loader=lambda: load_expansion_table("ncua_call_reports"),
        description=(
            "pandas DataFrame of NCUA credit-union call-report population for every "
            "quarter-end from 2019-03-31 through 2025-12-31. Each row is one credit "
            "union at one report_date and reports its state, total loans and leases, "
            "and loans and leases delinquent for two or more months; monetary values "
            "are reported dollars."
        ),
    ),
    RuntimeTable(
        name="noaa_storm_event_details_df",
        labels=("noaa_storm_events",),
        loader=lambda: load_expansion_table("noaa_storm_events"),
        description=(
            "pandas DataFrame of NOAA Storm Events database detail records for 2019 "
            "through 2025, one row per event. Columns: episode_id, event_id, "
            "state_name, state_fips, state_usps, event_type, county_zone_type (C = "
            "county, Z = forecast zone), county_zone_fips, county_zone_name, "
            "begin_datetime, end_datetime, property_damage_source_token, "
            "property_damage_usd, crop_damage_source_token, crop_damage_usd, "
            "flood_cause, data_source, source_file. Damage tokens such as 10.00K are "
            "parsed to USD in the _usd columns. Missing damage is distinct from "
            "reported zero: a missing _usd value means no amount was reported."
        ),
    ),
    RuntimeTable(
        name="nyfed_reference_rates_df",
        labels=("nyfed_reference_rates",),
        loader=lambda: load_expansion_table("nyfed_reference_rates"),
        description=(
            "pandas DataFrame of New York Fed reference rates and SOFR averages/index "
            "from 2021 through 2025. Rows are keyed by effective date and rate type; "
            "SOFR rates and SOFR Index observations occupy different rate-type rows."
        ),
    ),
    RuntimeTable(
        name="ofr_financial_stress_index_df",
        labels=("ofr_market_stress",),
        loader=lambda: load_expansion_table("ofr_market_stress"),
        description=(
            "pandas DataFrame of OFR daily Financial Stress Index through 2025-12-31. "
            "Rows contain the composite FSI plus five market-category and three "
            "regional contributions. Dates are ISO strings and unique. The table is a "
            "current revised history observed 2026-08-13, not a historical as-known "
            "vintage."
        ),
    ),
    RuntimeTable(
        name="rtdsm_revisions_df",
        labels=("philadelphia_fed_rtdsm",),
        loader=lambda: load_expansion_table("philadelphia_fed_rtdsm"),
        description=(
            "pandas DataFrame of Philadelphia Fed real-time data revisions through "
            "2025. Columns: series, period, first_release, second_release, "
            "third_release, most_recent. Release-stage values are distinct from the "
            "current most-recent revision."
        ),
    ),
    RuntimeTable(
        name="sba_five_o_four_loans_df",
        labels=("sba_504",),
        loader=lambda: load_expansion_table("sba_504"),
        description=(
            "pandas DataFrame of Privacy-reduced SBA 504 public loan-record rows for "
            "approval fiscal years 2020 through 2025 from the 2026-06-30 public "
            "snapshot. It retains state, CDC and third-party lender, SBA and "
            "third-party financing, approval, program, dated performance and "
            "aggregate business fields but excludes borrower names and addresses. The "
            "extract has no guaranteed unique loan identifier; exact-looking rows are "
            "preserved and must not be silently deduplicated."
        ),
    ),
    RuntimeTable(
        name="sba_seven_a_loans_df",
        labels=("sba_7a",),
        loader=lambda: load_expansion_table("sba_7a"),
        description=(
            "pandas DataFrame of Privacy-reduced SBA 7(a) public loan-record rows for "
            "approval fiscal years 2020 through 2025 from the 2026-06-30 public "
            "snapshot. It retains state, lender, approval, program, dated performance "
            "events and aggregate business fields but excludes borrower names and "
            "addresses. Source-native LocationID and five-character BankFDICNumber "
            "strings retain leading zeroes; fdic_certificate is the governed unpadded "
            "certificate for cross-source joins. The extract has no guaranteed unique "
            "public loan identifier; exact-looking rows are preserved and must not be "
            "silently deduplicated."
        ),
    ),
    RuntimeTable(
        name="sec_13f_filings_df",
        labels=("sec_13f_filings",),
        loader=lambda: load_expansion_table("sec_13f_filings"),
        description=(
            "pandas DataFrame of Every Form 13F filing in the SEC 13F data sets "
            "2019q1 through 2025q4, filed by 2025-12-31. One row per accession: "
            "filing_date, submissiontype (13F-HR, 13F-HR/A, 13F-NT, ...), cik, "
            "periodofreport, cover-page manager name and state, amendment flags, "
            "summary-page tableentrytotal and tablevaluetotal, and "
            "value_reporting_convention (thousands_of_dollars before 2023-01-03 "
            "filings, dollars after)."
        ),
    ),
    RuntimeTable(
        name="sec_13f_amendment_chain_df",
        labels=("sec_13f_amendment_chain",),
        loader=lambda: load_expansion_table("sec_13f_amendment_chain"),
        description=(
            "pandas DataFrame of source-native Form 13F information-table rows "
            "for four Palogic Value Management 2024-12-31 quarter filings: an "
            "original, two restatements and one added-holdings amendment. "
            "Each row retains accession, SEC row key, issuer, CUSIP, value in "
            "USD, security quantity, SH/PRN type, option side and discretion. "
            "Amendment scope comes from the separate 13F filings table; a "
            "PRN quantity is not a count of common shares."
        ),
    ),
    RuntimeTable(
        name="sec_13f_holdings_df",
        labels=("sec_13f_holdings",),
        loader=lambda: load_expansion_table("sec_13f_holdings"),
        description=(
            "pandas DataFrame of Institutional holdings aggregated over original "
            "13F-HR information tables: one row per period_of_report x cusip with "
            "issuer_name, title_of_class, holding_rows, filer_count (distinct "
            "filers), value_usd_total, share_value_usd_total and share_amount_total "
            "(SH rows without put/call), put_value_usd_total, call_value_usd_total, "
            "sole_voting_shares_total, and value_reporting_convention. All "
            "*_usd_total columns are in dollars: each accession's reporting unit "
            "(thousands of dollars before the 2023 rule change, with filers on both "
            "conventions around it) is inferred from its implied prices per share and "
            "rescaled before aggregation; value_reporting_convention records whether "
            "a period's filings were thousands_of_dollars, dollars or mixed. Sums can "
            "exceed an issuer's market capitalization because affiliated managers "
            "file overlapping holdings; they are reported-holdings totals, not "
            "ownership shares."
        ),
    ),
    RuntimeTable(
        name="bdc_investment_schedule_df",
        labels=("sec_bdc",),
        loader=lambda: load_expansion_table("sec_bdc"),
        description=(
            "pandas DataFrame of business-development-company investment schedules "
            "across 1,119 accessions from 170 filers, report dates 2013-04-30 "
            "through 2025-09-30. One row per reported investment, carrying "
            "accession, cik, filer_name, report_date, form, filed_date, "
            "investment_id, industry_sector, issuer_affiliation, investment_type, "
            "principal, cost, fair value, total interest rate and paid-in-kind rate "
            "where supplied. investment_id is a composite text label and is not "
            "unique within an accession."
        ),
    ),
    RuntimeTable(
        name="bdc_filings_df",
        labels=("sec_bdc", "sec_bdc_filings"),
        loader=lambda: load_expansion_table("sec_bdc_filings"),
        description=(
            "pandas DataFrame of every submission in the SEC BDC data sets "
            "2024q1-2025q4: accession, cik, filer_name, form, period, fiscal_year, "
            "fiscal_period, filed_date, fiscal_year_end and source_archive."
        ),
    ),
    RuntimeTable(
        name="facts_df",
        labels=("sec_companyfacts",),
        loader=load_facts,
        description=(
            "pandas DataFrame of every numeric 10-K/10-Q XBRL fact for the 500 "
            "largest SEC filers by latest fiscal-year total assets (the companies "
            "listed in companies_df), periods ending 2019 onward, frozen at "
            "2025-12-31. Columns: cik, ticker, company_name, taxonomy, concept, "
            "label, description, unit, value, start_date, end_date, filed_date, form, "
            "fiscal_year, fiscal_period, frame, accession. Values are in the raw unit "
            "named by unit, not pre-scaled."
        ),
    ),
    RuntimeTable(
        name="companies_df",
        labels=("sec_companyfacts", "sec_filings", "sec_insider", "sec_submissions"),
        loader=load_companies,
        description=(
            "pandas DataFrame of the 500 largest SEC filers by latest fiscal-year "
            "total assets, one row per company. Columns: ticker, cik, name, exchange, "
            "sic, sic_description, state_of_incorporation, fiscal_year_end (MMDD), "
            "universe_rank, total_assets_usd, assets_period_end, assets_filed."
        ),
    ),
    RuntimeTable(
        name="filings_df",
        labels=("sec_filings", "sec_submissions"),
        loader=load_filings,
        description=(
            "pandas DataFrame of 10-K, 10-Q, 20-F and 40-F filing metadata (including "
            "amendments) filed 2019 through 2025-12-31 by every SEC XBRL filer. "
            "Columns: cik, ticker, company_name, accession, form, filing_date, "
            "report_date, acceptance_datetime, fiscal_year_end, primary_document, "
            "is_amendment, source_url."
        ),
    ),
    RuntimeTable(
        name="statements_df",
        labels=("sec_financial_statements",),
        loader=load_financial_statements,
        description=(
            "pandas DataFrame with one row per 10-K/10-Q filing (amendments included) "
            "for every SEC XBRL filer, 2019 through 2025-12-31, carrying 42 standard "
            "us-gaap/dei concepts as columns. Instant concepts (e.g. Assets) hold the "
            "value at the filing's report date; duration concepts appear twice: "
            "<concept>_ytd is the period ending at the report date with the earliest "
            "start (fiscal year on a 10-K, year-to-date on a 10-Q) and <concept>_q is "
            "the 80-100 day period ending at the report date (a single quarter, often "
            "absent on 10-Ks). Empty means the filer did not tag that concept. Header "
            "columns: cik, ticker, company_name, accession, form, fiscal_year, "
            "fiscal_period, report_date, filed_date, sic. Money in USD, shares in "
            "shares, EPS in USD per share."
        ),
    ),
    RuntimeTable(
        name="sec_form_c_filings_df",
        labels=("sec_form_c",),
        loader=lambda: load_expansion_table("sec_form_c"),
        description=(
            "pandas DataFrame of SEC Regulation Crowdfunding filings made 2019 through 2025. "
            "Rows contain accession, form type, filing date, issuer/file identifiers, "
            "amendment description, target and maximum offering amounts and selected "
            "issuer financial fields. C, C/A, C-U and C-AR have different roles."
        ),
    ),
    RuntimeTable(
        name="sec_form_d_filings_df",
        labels=("sec_form_d",),
        loader=lambda: load_expansion_table("sec_form_d"),
        description=(
            "pandas DataFrame of SEC Form D filings made 2019 through 2025. Rows contain "
            "accession, filing date, issuer, amendment link, offering type, total "
            "offering, amount sold and remaining. Monetary fields can contain the "
            "literal value Indefinite."
        ),
    ),
    RuntimeTable(
        name="sec_ftd_df",
        labels=("sec_ftd",),
        loader=lambda: load_expansion_table("sec_ftd"),
        description=(
            "pandas DataFrame of SEC fails-to-deliver data, every semi-monthly file "
            "for 2024 through 2025: settlement_date, cusip, symbol, quantity_fails "
            "(shares), description, price (USD, prior-day close) and source_file."
        ),
    ),
    RuntimeTable(
        name="insider_submissions_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("submission"),
        description=(
            "pandas DataFrame of SEC Forms 3/4/5 submission metadata for the 500-company "
            "universe, filed 2019 through 2025. It identifies issuer, filing date, report period, "
            "document type, original-submission date, ticker and Rule 10b5-1 "
            "indicator."
        ),
    ),
    RuntimeTable(
        name="insider_owners_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("reportingowner"),
        description=(
            "pandas DataFrame of reporting-owner identity and relationship data. Join "
            "to insider_submissions_df on accession_number."
        ),
    ),
    RuntimeTable(
        name="insider_nonderivative_transactions_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("nonderiv_trans"),
        description=(
            "pandas DataFrame of non-derivative insider transactions. Join on "
            "accession_number. Transaction code, acquisition/disposition flag, "
            "direct/indirect ownership, shares and price are separate fields; blanks "
            "are not zero."
        ),
    ),
    RuntimeTable(
        name="insider_nonderivative_holdings_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("nonderiv_holding"),
        description=(
            "pandas DataFrame of non-derivative securities reported as holdings on "
            "SEC Forms 3/4/5. Join on accession_number. A holding row is a reported "
            "position, not an additional transaction; direct and indirect ownership "
            "are separate."
        ),
    ),
    RuntimeTable(
        name="insider_derivative_transactions_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("deriv_trans"),
        description=(
            "pandas DataFrame of derivative insider transactions. Join on "
            "accession_number; exercise/conversion terms and underlying-security "
            "fields must not be confused with non-derivative open-market trades."
        ),
    ),
    RuntimeTable(
        name="insider_derivative_holdings_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("deriv_holding"),
        description=(
            "pandas DataFrame of derivative securities reported as holdings on SEC "
            "Forms 3/4/5. Join on accession_number; underlying-share fields must not "
            "be added to non-derivative positions without reconciling the instrument "
            "relationship."
        ),
    ),
    RuntimeTable(
        name="insider_footnotes_df",
        labels=("sec_insider",),
        loader=lambda: load_insider_table("footnotes"),
        description=(
            "pandas DataFrame of as-filed Form 3/4/5 footnotes, keyed by "
            "accession_number and footnote_id. Use footnote-reference columns in "
            "transaction tables when a numeric field or transaction meaning is "
            "qualified."
        ),
    ),
    RuntimeTable(
        name="midas_security_exchange_df",
        labels=("sec_midas_security_exchange",),
        loader=lambda: load_expansion_table("sec_midas_security_exchange"),
        description=(
            "pandas DataFrame of SEC MIDAS individual-security market-quality "
            "metrics, 2024-01-02 through 2025-12-31, aggregated to one row per date x "
            "security_type x ticker by summing the exchange-level counts and volumes; "
            "exchange_count is the number of exchange rows summed. The four SEC "
            "decile ranks (mcap_rank, turnover_rank, volatility_rank, price_rank) are "
            "security-level. Volume columns are in thousands of shares; odd_lots, "
            "hidden and lit_trades are trade counts."
        ),
    ),
    RuntimeTable(
        name="nmfp_series_filings_df",
        labels=("sec_nmfp", "sec_nmfp_series_filings"),
        loader=lambda: load_expansion_table("sec_nmfp"),
        description=(
            "pandas DataFrame of SEC Form N-MFP series-level filings accepted by "
            "2025-12-31. Rows identify accession, filing and report dates, form, fund "
            "series, maturity, assets, liabilities and net assets. Multiple "
            "accessions may report the same series and report date."
        ),
    ),
    RuntimeTable(
        name="nmfp_daily_flows_df",
        labels=("sec_nmfp", "sec_nmfp_flows"),
        loader=lambda: load_expansion_table("sec_nmfp_flows"),
        description=(
            "pandas DataFrame of SEC N-MFP class-level daily gross subscriptions and "
            "redemptions for filings accepted by 2025-12-31. Physical source rows are "
            "retained with source archive and row number; the frozen files include "
            "one exact zero-flow business-row duplicate. Multiple physical or "
            "share-class rows can describe one series-date and must be aggregated "
            "only at the grain explicitly requested by the task."
        ),
    ),
    RuntimeTable(
        name="nmfp_daily_liquidity_df",
        labels=("sec_nmfp", "sec_nmfp_liquidity"),
        loader=lambda: load_expansion_table("sec_nmfp_liquidity"),
        description=(
            "pandas DataFrame of SEC N-MFP daily and weekly liquid-asset values and "
            "fractions, keyed by accession and liquidity date."
        ),
    ),
    RuntimeTable(
        name="nmfp_class_yields_df",
        labels=("sec_nmfp", "sec_nmfp_class_yields"),
        loader=lambda: load_expansion_table("sec_nmfp_class_yields"),
        description=(
            "pandas DataFrame of share-class seven-day net yields for every money "
            "market fund filing in the N-MFP archives that carry the N-MFP3 yield "
            "schedule (the 2025 monthly data sets), joined to the same-date "
            "series-level gross yield. Columns: accession, class_id, class_name, "
            "class_net_assets_usd, yield_date, seven_day_net_yield, "
            "seven_day_gross_yield, source_archive; yields are decimal rates."
        ),
    ),
    RuntimeTable(
        name="sec_notes_numeric_df",
        labels=("sec_notes",),
        loader=lambda: load_expansion_table("sec_notes"),
        description=(
            "pandas DataFrame of SEC Financial Statement and Notes numeric facts "
            "(num.tsv) for the 500 largest 10-K filers by total assets, every filing "
            "in the archives on hand, filed through 2025-12-31. Unlike CompanyFacts "
            "this retains dimensional facts: dimh is the dimension hash (0x00000000 = "
            "no dimensions) and coreg the co-registrant. Columns: adsh (accession), "
            "tag, version (taxonomy), ddate (period end, YYYYMMDD), qtrs (0 = "
            "instant, 1 = one quarter, 4 = one year), uom, dimh, iprx, value, "
            "footnote, dimn, coreg, durp, datp, dcml, cik, name, form, filed. Values "
            "are as filed in the unit named by uom."
        ),
    ),
    RuntimeTable(
        name="sec_notes_dimensional_facts_df",
        labels=("sec_notes_dimensional",),
        loader=lambda: load_expansion_table("sec_notes_dimensional"),
        description=(
            "pandas DataFrame of SEC Financial Statement and Notes single-axis "
            "dimensional numeric facts for the 500 largest 10-K filers, every "
            "10-K/10-Q (and amendments) filed 2019-2025. Each row is a value broken "
            "down along exactly one XBRL axis (axis, member: e.g. axis "
            "StatementBusinessSegments / member a segment name, or ProductOrService, "
            "StatementGeographical, ClassOfStock). Columns: adsh, cik, name, form, "
            "filed, tag, version, ddate (period end), qtrs (0 = instant, 1 = quarter, "
            "4 = year), uom, dimh (dimension hash), axis, member, iprx, value, coreg, "
            "dcml, source_archive. Facts with two or more axes are not included; "
            "totals without dimensions live in CompanyFacts."
        ),
    ),
    RuntimeTable(
        name="sec_notes_dimensions_df",
        labels=("sec_notes_dimensions",),
        loader=lambda: load_expansion_table("sec_notes_dimensions"),
        description=(
            "pandas DataFrame of Decode table for the dimh hashes used by the notes "
            "tables: dimh, axis_count, axis and member (filled for single-axis "
            "hashes), the full segments string (Axis=Member; pairs) and segment_type."
        ),
    ),
    RuntimeTable(
        name="sec_notes_submissions_df",
        labels=("sec_notes_submissions",),
        loader=lambda: load_expansion_table("sec_notes_submissions"),
        description=(
            "pandas DataFrame of SEC Financial Statement and Notes submissions "
            "(sub.tsv) for the same 500-filer universe. Columns: adsh, cik, name, "
            "sic, form, period, fy, fp, filed, prevrpt, detail, source_archive."
        ),
    ),
    RuntimeTable(
        name="nport_fund_reports_df",
        labels=("sec_nport",),
        loader=lambda: load_expansion_table("sec_nport"),
        description=(
            "pandas DataFrame of SEC Form N-PORT fund-level reports filed by "
            "2025-12-31. Rows identify accession, report date, registrant and series, "
            "total assets, liabilities, net assets, monthly non-derivative gains and "
            "fund flows. Amended accessions may repeat a series and report date."
        ),
    ),
    RuntimeTable(
        name="nport_interest_rate_risk_df",
        labels=("sec_nport", "sec_nport_interest_rate_risk"),
        loader=lambda: load_expansion_table("sec_nport_interest_rate_risk"),
        description=(
            "pandas DataFrame of SEC N-PORT interest-rate-risk rows from every "
            "quarterly data set 2019q4-2025q4, filed by 2025-12-31. Each "
            "accession-currency row reports DV01 separately for the 3-month, 1-year, "
            "5-year, 10-year and 30-year tenor shocks; positive and negative tenor "
            "values can offset within a currency."
        ),
    ),
    RuntimeTable(
        name="nport_selected_holding_details_df",
        labels=("sec_nport_holding_details",),
        loader=lambda: load_expansion_table("sec_nport_holding_details"),
        description=(
            "pandas DataFrame of SEC N-PORT reported holding values and joined "
            "debt-security terms for two frozen 2025 filings. Each row carries "
            "accession, holding_id, asset_category, currency_value_usd, "
            "maturity_date, coupon_type and filed default/arrears flags. "
            "This is a selected-accession table, not a full-fund universe."
        ),
    ),
    RuntimeTable(
        name="nport_lending_cohort_df",
        labels=("sec_nport_lending_cohort",),
        loader=lambda: load_expansion_table("sec_nport_lending_cohort"),
        description=(
            "pandas DataFrame containing every reported holding of accession "
            "0001410368-25-031694, with source holding ID, asset category, "
            "reported holding value, restricted flag, and SEC N-PORT C.12 "
            "securities-lending, cash-collateral reinvestment and non-cash "
            "collateral-asset flags and values. Cash-collateral reinvestment "
            "value is not the total collateral received."
        ),
    ),
    RuntimeTable(
        name="nport_repo_collateral_df",
        labels=("sec_nport_repo_collateral",),
        loader=lambda: load_expansion_table("sec_nport_repo_collateral"),
        description=(
            "pandas DataFrame of every repo and reverse-repo position in the original "
            "PIMCO Total Return ESG Fund 2025-06-30 N-PORT filing. Each row is a "
            "reported collateral security, so holdings can repeat; holding balance "
            "and rate belong to the agreement, while collateral_security_principal "
            "belongs to the posted or received security. Currency amounts are USD "
            "and rates are percentages. The filing was submitted 2025-08-28."
        ),
    ),
    RuntimeTable(
        name="treasury_auctions_df",
        labels=("treasury_auctions",),
        loader=lambda: load_expansion_table("treasury_auctions"),
        description=(
            "pandas DataFrame of Treasury auction records through 2025, including "
            "security type and term, CUSIP, auction/issue dates, reopening and "
            "inflation-index flags, yields, offering/tender/accepted amounts, bidder "
            "allotments and bid-to-cover."
        ),
    ),
    RuntimeTable(
        name="treasury_debt_to_penny_df",
        labels=("treasury_debt_to_penny",),
        loader=lambda: load_expansion_table("treasury_debt_to_penny"),
        description=(
            "pandas DataFrame of Treasury Debt to the Penny daily observations for "
            "2019 through 2025. Rows report debt held by the public, "
            "intragovernmental holdings and total public debt outstanding in dollars."
        ),
    ),
    RuntimeTable(
        name="treasury_dts_operating_cash_df",
        labels=("treasury_dts", "treasury_dts_operating_cash"),
        loader=lambda: load_expansion_table("treasury_dts_operating_cash"),
        description=(
            "pandas DataFrame of Daily Treasury Statement operating-cash rows for "
            "2019 through 2025. Each date has TGA opening balance, total deposits, "
            "total withdrawals and closing balance account types; "
            "today_amount_million_usd is the source's generic daily value field for "
            "that row. Amounts are million USD and the table is a daily "
            "cash-flow/stock bridge."
        ),
    ),
    RuntimeTable(
        name="treasury_dts_public_debt_transactions_df",
        labels=("treasury_dts", "treasury_dts_public_debt"),
        loader=lambda: load_expansion_table("treasury_dts_public_debt"),
        description=(
            "pandas DataFrame of Daily Treasury Statement public-debt transaction "
            "rows for 2019 through 2025. Rows distinguish Issues from Redemptions and "
            "Marketable from Nonmarketable security types; "
            "transaction_today_amount_million_usd is a daily flow and can include "
            "signed source adjustments."
        ),
    ),
    RuntimeTable(
        name="treasury_marketable_securities_df",
        labels=("treasury_marketable_securities",),
        loader=lambda: load_expansion_table("treasury_marketable_securities"),
        description=(
            "pandas DataFrame of Treasury MSPD marketable-security detail for month "
            "ends from 2019 through 2025. Rows include security class, CUSIP or total "
            "label, stated interest rate, issue and maturity dates, and issued, "
            "redeemed and outstanding amounts in millions of dollars. Detail and "
            "total rows coexist."
        ),
    ),
    RuntimeTable(
        name="treasury_tic_major_foreign_holders_df",
        labels=("treasury_tic",),
        loader=lambda: load_expansion_table("treasury_tic"),
        description=(
            "pandas DataFrame of Treasury International Capital Major Foreign Holders "
            "monthly positions, every year block of the TIC history file through 2025-12. Rows distinguish named countries from source "
            "aggregate rows and report end-of-month Treasury-security holdings in "
            "billion USD. The geography is residence-based; positions are stocks "
            "affected by transactions, valuation, custody and other changes, not "
            "bilateral trade flows or investor nationality."
        ),
    ),
    RuntimeTable(
        name="treasury_usfr_reconciliations_df",
        labels=("treasury_usfr_reconciliations",),
        loader=lambda: load_expansion_table("treasury_usfr_reconciliations"),
        description=(
            "pandas DataFrame of U.S. Financial Report reconciliation statements for "
            "fiscal years reported from 2020 through 2025. Rows preserve statement "
            "vintage, restatement flag, hierarchical account/component/line "
            "descriptions, signed amounts in billions of dollars and source line "
            "numbers; detail rows and subtotals coexist."
        ),
    ),
    RuntimeTable(
        name="treasury_yield_curve_df",
        labels=("treasury_yield_curve",),
        loader=lambda: load_expansion_table("treasury_yield_curve"),
        description=(
            "pandas DataFrame of Daily U.S. Treasury par yield curve observations for "
            "2019 through 2025. Columns are Date and standard tenors from one month "
            "through 30 years; not every tenor exists in every source year."
        ),
    ),
)


def expand_labels(data_sources: Iterable[str]) -> set[str]:
    """The declared labels plus the members of any declared family."""
    selected = set(data_sources)
    for family, members in SOURCE_FAMILIES.items():
        if family in selected:
            selected.update(members)
    return selected


def select_runtime_tables(data_sources: Iterable[str]) -> list[RuntimeTable]:
    """The tables a case's declared ``data_sources`` expose, in registry order.

    Raises ValueError for an empty declaration or a label no table answers to,
    so a typo in case JSON fails loudly instead of silently exposing nothing.
    """
    selected = expand_labels(data_sources)
    if not selected:
        raise ValueError("data_sources must contain at least one declared source")
    known = {label for table in RUNTIME_TABLES for label in table.labels} | set(SOURCE_FAMILIES)
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"unknown data_sources label(s): {unknown}")
    return [table for table in RUNTIME_TABLES if selected.intersection(table.labels)]


def runtime_variables(data_sources: Iterable[str]) -> list[Variable]:
    """Eager injection: every declared table loaded as a DataFrame variable."""
    return [
        Variable(table.name, table.loader(), table.description)
        for table in select_runtime_tables(data_sources)
    ]
