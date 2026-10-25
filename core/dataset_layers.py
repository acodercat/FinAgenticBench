"""Build and load the governed FinAgenticBench dataset layers.

The source-native files below ``datasets/*/raw`` are immutable inputs and are
never touched here. Source-specific builders first create reviewed staging
tables, which this module publishes into two explicit layers:

``normalized``
    Conservative semantic cleaning: whitespace/empty-value normalization,
    stable identifier strings, ISO dates and a recorded schema.

``runtime``
    A byte-for-byte copy of the normalized table for the current release. A later
    release may project columns/rows per benchmark protocol without changing the
    normalized asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from config import DATASETS_DIR, PROJECT_ROOT


BENCHMARK_CUTOFF = "2025-12-31"
# Build inputs and intermediates live outside the distributed tree: `datasets/`
# holds only what a benchmark run reads (the runtime tables and their metadata),
# while the reviewed staging inputs and the normalized layer sit under
# BUILD_DIR. Override with FINBENCH_BUILD_DIR to build against another copy.
BUILD_DIR = Path(os.environ.get("FINBENCH_BUILD_DIR", PROJECT_ROOT / "datasets_build"))
# The source-native archives. Acquisition and staging scripts read them from here;
# nothing a benchmark run touches lives in this tree.
RAW_DIR = Path(os.environ.get("FINBENCH_RAW_DIR", PROJECT_ROOT / "datasets_raw"))


def raw_dir(source_id: str) -> Path:
    """The raw archive directory of one source."""
    return RAW_DIR / source_id
CATALOG_PATH = DATASETS_DIR / "catalog.json"


@dataclass(frozen=True)
class TableDefinition:
    table_id: str
    reviewed_input_path: str
    primary_key: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    identifier_columns: tuple[str, ...] = ()
    cutoff_column: str | None = None
    target_name: str | None = None
    # Text columns with few distinct values relative to rows. They are stored as
    # plain strings in the CSV and loaded as pandas categories, which is what
    # keeps a multi-million-row facts table inside a runtime's memory budget.
    categorical_columns: tuple[str, ...] = ()
    # Columns the source publishes but never fills for this table's population.
    # They are dropped during cleaning so the runtime does not advertise fields
    # that can only ever return nothing. Each entry was verified against the raw
    # archive before being listed here; a column that carries any value stays.
    empty_at_source_columns: tuple[str, ...] = ()

    @property
    def source_id(self) -> str:
        return Path(self.reviewed_input_path).parts[0]

    @property
    def output_name(self) -> str:
        return self.target_name or f"{self.table_id.split('.', 1)[1]}.csv.gz"

    @property
    def normalized_path(self) -> str:
        return f"{self.source_id}/normalized/{self.output_name}"

    @property
    def runtime_path(self) -> str:
        return f"{self.source_id}/runtime/{self.output_name}"


# These are the reviewed tables exposed by the current FinAgenticBench runtime.  The
# registry is deliberately explicit: filesystem discovery must never promote a
# newly downloaded raw file into the benchmark by accident.
TABLES: tuple[TableDefinition, ...] = (
    TableDefinition("sec.company_facts", "sec/facts.csv.gz", ("cik", "accession", "taxonomy", "concept", "unit", "start_date", "end_date", "frame"), ("start_date", "end_date", "filed_date"), ("cik", "ticker", "accession"), "filed_date", categorical_columns=("ticker", "company_name", "taxonomy", "concept", "label", "description", "unit", "form", "fiscal_period", "frame")),
    TableDefinition("sec.financial_statements", "sec/financial_statements.csv.gz", ("cik", "accession"), ("report_date", "filed_date"), ("cik", "ticker", "accession"), "filed_date", categorical_columns=("ticker", "company_name", "form", "fiscal_period")),
    TableDefinition("sec.filings", "sec/filings.csv", ("accession", "cik"), ("filing_date", "report_date", "acceptance_datetime"), ("cik", "ticker", "accession"), "filing_date", categorical_columns=("company_name", "form", "fiscal_year_end")),
    TableDefinition("sec.companies", "sec/companies.csv", ("cik",), (), ("cik", "ticker")),
    TableDefinition("sec_insider.submissions", "sec_insider/submission.csv.gz", ("accession_number",), ("filing_date", "period_of_report", "date_of_orig_sub"), ("accession_number", "issuercik", "issuertradingsymbol"), "filing_date"),
    TableDefinition("sec_insider.reporting_owners", "sec_insider/reportingowner.csv.gz", ("accession_number", "rptownercik"), (), ("accession_number", "rptownercik", "file_number")),
    TableDefinition("sec_insider.nonderivative_transactions", "sec_insider/nonderiv_trans.csv.gz", ("accession_number", "nonderiv_trans_sk"), ("trans_date", "deemed_execution_date"), ("accession_number", "nonderiv_trans_sk"), empty_at_source_columns=('valu_ownd_folwng_trans_fn',)),
    TableDefinition("sec_insider.nonderivative_holdings", "sec_insider/nonderiv_holding.csv.gz", ("accession_number", "nonderiv_holding_sk"), (), ("accession_number", "nonderiv_holding_sk"), empty_at_source_columns=('trans_form_type_fn',)),
    TableDefinition("sec_insider.derivative_transactions", "sec_insider/deriv_trans.csv.gz", ("accession_number", "deriv_trans_sk"), ("trans_date", "deemed_execution_date", "excercise_date", "expiration_date"), ("accession_number", "deriv_trans_sk"), empty_at_source_columns=('trans_acquired_disp_cd_fn',)),
    TableDefinition("sec_insider.derivative_holdings", "sec_insider/deriv_holding.csv.gz", ("accession_number", "deriv_holding_sk"), ("exercise_date", "expiration_date"), ("accession_number", "deriv_holding_sk"), empty_at_source_columns=('trans_form_type_fn', 'undlyng_sec_value_fn')),
    TableDefinition("sec_insider.footnotes", "sec_insider/footnotes.csv.gz", ("accession_number", "footnote_id"), (), ("accession_number", "footnote_id")),
    TableDefinition("ken_french.daily_factors", "ken_french/daily_factors.csv.gz", ("date",), ("date",), (), "date"),
    TableDefinition("ken_french.daily_industry_returns", "ken_french/daily_industry_returns.csv.gz", ("date",), ("date",), (), "date"),
    TableDefinition("rtdsm.revisions", "rtdsm/revisions.csv.gz", ("series", "period"), (), (), "period"),
    TableDefinition("fed_stress_tests.results", "fed_stress_tests/results.csv.gz", ("exercise_name", "scenario_name", "disclosure_legal_name"), (), ("id_rssd",), "dt_exercise_quarter"),
    TableDefinition("sec_nmfp.series_filings", "sec_nmfp/series_filings.csv.gz", ("accession",), ("filing_date", "report_date"), ("accession", "cik", "series_id"), "filing_date"),
    TableDefinition("sec_nmfp.daily_shareholder_flows", "sec_nmfp/daily_shareholder_flows.csv.gz", ("source_archive", "source_row_number"), ("flow_date",), ("accession", "class_id")),
    TableDefinition("sec_nmfp.daily_liquidity", "sec_nmfp/daily_liquidity.csv.gz", ("accession", "liquidity_date"), ("liquidity_date",), ("accession",)),
    TableDefinition("sec_nmfp.share_class_yields", "sec_nmfp/share_class_yields.csv.gz", ("accession", "class_id", "yield_date"), ("yield_date",), ("accession", "class_id"), categorical_columns=("source_archive",)),
    TableDefinition("sec_nport.fund_reports", "sec_nport/fund_reports.csv.gz", ("accession",), ("filing_date", "report_date"), ("accession", "cik", "series_id"), "filing_date"),
    TableDefinition("sec_nport.interest_rate_risk", "sec_nport/interest_rate_risk.csv.gz", ("accession", "currency_code"), ("filing_date", "report_date"), ("accession", "series_id"), "filing_date"),
    TableDefinition("sec_nport.selected_holding_details", "sec_nport/selected_holding_details.csv.gz", ("accession", "holding_id"), ("maturity_date",), ("accession", "holding_id"), None, categorical_columns=("asset_category", "coupon_type", "payoff_profile", "default_flag", "interest_arrears_flag", "pik_flag")),
    TableDefinition("sec_nport.lending_cohort", "sec_nport/lending_cohort_2025q3.csv.gz", ("accession", "holding_id"), (), ("accession", "holding_id", "issuer_cusip"), None, categorical_columns=("asset_category", "is_restricted_security", "fair_value_level", "is_cash_collateral_reinvestment", "is_non_cash_collateral_asset", "is_loan_by_fund")),
    TableDefinition("sec_nport.repo_collateral", "sec_nport/repo_collateral_2025q2.csv.gz", ("accession", "holding_id", "collateral_id"), ("report_date", "filing_date", "maturity_date"), ("accession", "holding_id", "collateral_id", "series_id"), "filing_date", categorical_columns=("transaction_type", "unit", "holding_currency", "collateral_principal_currency", "collateral_currency", "collateral_category", "payoff_profile")),
    TableDefinition("hmda.loan_applications", "hmda/lar_2024.csv.gz", (), (), ("lei", "state_code", "county_code"), "activity_year", categorical_columns=("activity_year", "lei", "institution_name", "state_code", "county_code", "derived_msa_md", "derived_loan_product_type", "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex", "action_taken", "purchaser_type", "loan_type", "loan_purpose", "lien_status", "business_or_commercial_purpose", "debt_to_income_ratio", "construction_method", "occupancy_type", "total_units", "applicant_age", "denial_reason_1")),
    TableDefinition("sec_notes.numeric_facts", "sec_notes/numeric_facts.csv.gz", ("adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx", "coreg"), ("ddate", "filed"), ("cik",), "filed", categorical_columns=("adsh", "tag", "version", "uom", "dimh", "coreg", "name", "form", "cik")),
    TableDefinition("sec_notes.dimensional_facts", "sec_notes/dimensional_facts.csv.gz", ("adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx", "coreg"), ("ddate", "filed"), ("cik",), "filed", categorical_columns=("adsh", "cik", "name", "form", "tag", "version", "uom", "dimh", "axis", "member", "coreg", "source_archive")),
    TableDefinition("sec_notes.dimensions", "sec_notes/dimensions.csv.gz", ("dimh",), (), ("dimh",), None, categorical_columns=("axis", "segment_type")),
    TableDefinition("sec_notes.submissions", "sec_notes/submissions.csv.gz", ("adsh",), ("period", "filed"), ("adsh", "cik"), "filed", categorical_columns=("form", "fp", "sic", "source_archive")),
    TableDefinition("sec_13f.filings", "sec_13f/filings.csv.gz", ("accession_number",), ("filing_date", "periodofreport"), ("accession_number", "cik"), "filing_date", categorical_columns=("submissiontype", "isamendment", "amendmenttype", "reporttype", "filingmanager_stateorcountry", "isconfidentialomitted", "source_archive", "value_reporting_convention")),
    TableDefinition("sec_13f.amendment_chain", "sec_13f/amendment_chain_2024q4.csv.gz", ("accession", "source_row_key"), (), ("accession", "source_row_key", "cusip"), None, categorical_columns=("issuer_name", "title_of_class", "quantity_type", "option_side", "investment_discretion", "source_archive")),
    TableDefinition("sec_13f.holdings_by_cusip", "sec_13f/holdings_by_cusip.csv.gz", ("period_of_report", "cusip"), ("period_of_report",), ("cusip",), "period_of_report", categorical_columns=("title_of_class", "value_reporting_convention")),
    TableDefinition("fed_z1_vintages.series_observations", "fed_z1_vintages/series_observations.csv.gz", ("vintage", "table", "series", "date"), ("vintage", "date"), ("series",), "date", categorical_columns=("vintage", "table", "series", "frequency")),
    TableDefinition("sec_ftd.fails_to_deliver", "sec_ftd/fails_to_deliver.csv.gz", ("settlement_date", "cusip", "symbol"), ("settlement_date",), ("cusip", "symbol"), "settlement_date", categorical_columns=("source_file",)),
    TableDefinition("nyfed.reference_rates", "nyfed_rates/reference_rates.csv.gz", ("Effective Date", "Rate Type"), ("Effective Date",), ("Footnote ID",), "Effective Date", empty_at_source_columns=('Intra Day - High (%)', 'Intra Day - Low (%)', 'Revision Indicator (Y/N)', 'Standard Deviation (%)', 'Volume ($Billions)')),
    TableDefinition("treasury_auctions.auctions", "treasury_auctions/auctions.csv.gz", ("record_date", "cusip", "auction_date", "issue_date"), ("record_date", "auction_date", "issue_date", "maturity_date", "announcemt_date", "dated_date", "original_dated_date", "original_issue_date", "mat_date"), ("cusip", "original_cusip"), "record_date"),
    TableDefinition("treasury_rates.yield_curve", "treasury_rates/yield_curve.csv.gz", ("Date",), ("Date",), (), "Date"),
    TableDefinition("sec_form_c.filings", "sec_form_c/filings.csv.gz", ("ACCESSION_NUMBER",), ("FILING_DATE", "DEADLINEDATE", "PERIOD"), ("ACCESSION_NUMBER", "CIK", "FILE_NUMBER"), "FILING_DATE"),
    TableDefinition("sec_form_d.filings", "sec_form_d/filings.csv.gz", ("ACCESSIONNUMBER",), ("FILING_DATE", "SALE_DATE"), ("ACCESSIONNUMBER", "PREVIOUSACCESSIONNUMBER", "CIK"), "FILING_DATE"),
    TableDefinition("bis.credit_gap", "bis/credit_gap.csv.gz", ("BORROWERS_CTY:Borrowers' country", "TC_BORROWERS:Borrowing sector", "TC_LENDERS:Lending sector", "CG_DTYPE:Credit gap data type", "TIME_PERIOD:Time period or range"), (), (), "TIME_PERIOD:Time period or range", empty_at_source_columns=('OBS_PRE_BREAK:Pre-Break Observation', 'TIME_FORMAT:Time Format', 'TITLE_TS:Title (tseries level)')),
    TableDefinition("bis.debt_service_ratios", "bis/debt_service_ratios.csv.gz", ("BORROWERS_CTY:Borrowers' country", "DSR_BORROWERS:Borrowers", "TIME_PERIOD:Time period or range"), (), (), "TIME_PERIOD:Time period or range", empty_at_source_columns=('COLLECTION:Collection Indicator', 'OBS_PRE_BREAK:Pre-Break Observation')),
    TableDefinition("bis.effective_exchange_rates", "bis/broad_effective_exchange_rates.csv.gz", ("EER_TYPE:Type", "EER_BASKET:Basket", "REF_AREA:Reference area", "TIME_PERIOD:Time period or range"), (), (), "TIME_PERIOD:Time period or range", empty_at_source_columns=('UNIT_MEASURE:Unit of measure',)),
    TableDefinition("bis.global_liquidity", "bis/global_liquidity_emde.csv.gz", ("CURR_DENOM:Currency of denomination", "BORROWERS_CTY:Borrowers' country", "BORROWERS_SECTOR:Borrowers' sector", "LENDERS_SECTOR:Lending sector", "L_POS_TYPE:Position type", "L_INSTR:Type of instruments", "TIME_PERIOD:Time period or range"), (), (), "TIME_PERIOD:Time period or range"),
    TableDefinition("bis.otc_derivatives", "bis/otc_derivatives.csv.gz", ("DER_TYPE:Measure", "DER_INSTR:Instrument", "DER_RISK:Risk category", "DER_SECTOR_CPY:Counterparty sector", "DER_BASIS:Basis", "TIME_PERIOD:Time period or range"), (), (), "TIME_PERIOD:Time period or range"),
    TableDefinition("fdic.bank_financials", "fdic_bankfind/bank_financials.csv.gz", ("CERT", "REPDTE"), ("REPDTE",), ("CERT", "ID"), "REPDTE"),
    TableDefinition("fdic_sod.branches", "fdic_sod/branches.csv.gz", ("report_date", "branch_id"), ("report_date",), ("fdic_certificate", "bank_rssd_id", "branch_id", "branch_number", "branch_county_fips"), "report_date", categorical_columns=("bank_name", "branch_state", "main_office_indicator", "source_file")),
    TableDefinition("sec_bdc.investment_schedule", "sec_bdc/investment_schedule.csv.gz", ("accession", "report_date", "investment_id"), ("report_date", "filed_date", "maturity_date"), ("accession", "cik", "investment_id"), "filed_date", categorical_columns=("cik", "filer_name", "form", "investment_type", "industry_sector", "issuer_affiliation", "source_archive")),
    TableDefinition("sec_bdc.filings", "sec_bdc/filings.csv.gz", ("accession",), ("period", "filed_date"), ("accession", "cik"), "filed_date", categorical_columns=("form", "fiscal_period", "source_archive")),
    TableDefinition("sba_7a_504.seven_a_loans", "sba_7a_504/seven_a_2020_2025.csv.gz", (), ("AsOfDate", "ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"), ("LocationID", "BankFDICNumber", "fdic_certificate", "NaicsCode"), "ApprovalFY"),
    TableDefinition("sba_7a_504.five_o_four_loans", "sba_7a_504/five_o_four_2020_2025.csv.gz", (), ("AsOfDate", "ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"), ("LocationID", "NaicsCode"), "ApprovalFY"),
    TableDefinition("cftc_cot.financial_futures", "cftc_cot/financial_futures.csv.gz", ("report_scope", "CFTC_Contract_Market_Code", "Report_Date_as_YYYY-MM-DD"), ("Report_Date_as_YYYY-MM-DD",), ("CFTC_Contract_Market_Code",), "Report_Date_as_YYYY-MM-DD"),
    TableDefinition("fhfa_hpi.house_price_indexes", "fhfa_hpi/hpi_through_2025.csv.gz", ("hpi_type", "hpi_flavor", "frequency", "level", "place_id", "yr", "period"), (), ("place_id",), "yr"),
    TableDefinition("ncua_call_reports.institutions", "ncua_call_reports/institutions.csv.gz", ("credit_union_number", "report_date"), ("report_date",), ("credit_union_number",), "report_date", categorical_columns=("credit_union_name", "state")),
    TableDefinition("sec_midas.daily_security", "sec_midas_security_exchange/daily_security.csv.gz", ("date", "security_type", "ticker"), ("date",), ("ticker",), "date", categorical_columns=("security_type",)),
    TableDefinition("finra_short_volume.daily", "finra_short_volume/daily_short_volume.csv.gz", ("date", "symbol"), ("date",), ("symbol",), "date", categorical_columns=("market_codes", "source_file")),
    TableDefinition("fed_scf.household_summary", "fed_scf/summary_2022.csv.gz", ("record_id",), (), ("household_id", "record_id")),
    TableDefinition("dol_form5500.plan_financials", "dol_form5500/plan_financials.csv.gz", ("filing_id",), ("plan_year_end", "date_received"), ("filing_id", "plan_number", "sponsor_ein"), "date_received"),
    TableDefinition("dol_form5500.schedule_sb", "dol_form5500/schedule_sb.csv.gz", ("filing_id",), ("plan_year_end", "date_received", "valuation_date"), ("filing_id", "plan_number", "sponsor_ein"), "date_received"),
    TableDefinition("ffiec_call_reports.bank_capital", "ffiec_call_reports/bank_capital.csv.gz", ("bank_id", "report_date", "filing_type", "last_submission_update"), ("report_date", "last_submission_update"), ("bank_id", "filing_type"), "report_date"),
    TableDefinition("ffiec_call_reports.bank_balance_sheets", "ffiec_call_reports/bank_balance_sheets.csv.gz", ("bank_rssd_id", "report_date"), ("report_date", "last_submission_update"), ("bank_rssd_id", "fdic_certificate", "filing_type"), "report_date"),
    TableDefinition("ffiec_call_reports.bank_securities_rcb", "ffiec_call_reports/bank_securities_rcb_2025q3.csv.gz", ("bank_rssd_id", "report_date"), ("report_date",), ("bank_rssd_id",), "report_date"),
    TableDefinition("ffiec_call_reports.bank_unused_commitments_rcl", "ffiec_call_reports/bank_unused_commitments_rcl_2025q3.csv.gz", ("bank_rssd_id", "report_date"), ("report_date",), ("bank_rssd_id", "fdic_certificate", "filing_type"), "report_date"),
    TableDefinition("ffiec_call_reports.bank_past_due_rcn", "ffiec_call_reports/bank_past_due_rcn_2025q3.csv.gz", ("bank_rssd_id", "report_date"), ("report_date",), ("bank_rssd_id", "fdic_certificate", "filing_type"), "report_date"),
    TableDefinition("ffiec_call_reports.bank_sold_recourse_rcs", "ffiec_call_reports/bank_sold_recourse_rcs_2025q3.csv.gz", ("bank_rssd_id", "report_date"), ("report_date",), ("bank_rssd_id", "fdic_certificate", "filing_type"), "report_date"),
    TableDefinition("ffiec_call_reports.bank_time_deposits_rce", "ffiec_call_reports/bank_time_deposits_rce_2025q3.csv.gz", ("bank_rssd_id", "report_date"), ("report_date",), ("bank_rssd_id", "fdic_certificate", "filing_type"), "report_date"),
    TableDefinition("treasury_debt.debt_to_penny", "treasury_debt/debt_to_penny.csv.gz", ("record_date",), ("record_date",), (), "record_date"),
    TableDefinition("treasury_debt.marketable_securities", "treasury_debt/marketable_securities.csv.gz", ("record_date", "source_line_number"), ("record_date", "issue_date", "maturity_date"), ("security_identifier_or_total_label",), "record_date", empty_at_source_columns=('series_code',)),
    TableDefinition("treasury_usfr.reconciliations", "treasury_usfr/reconciliations.csv.gz", ("record_date", "statement_fiscal_year", "source_line_number"), ("record_date",), (), "record_date"),
    TableDefinition("treasury_tic.major_foreign_holders", "treasury_tic/major_foreign_holders.csv.gz", ("country_name", "observation_month"), (), (), "observation_month"),
    TableDefinition("treasury_dts.operating_cash_balance", "treasury_dts/operating_cash_balance.csv.gz", ("record_date", "account_type"), ("record_date",), (), "record_date"),
    TableDefinition("treasury_dts.public_debt_transactions", "treasury_dts/public_debt_transactions.csv.gz", ("record_date", "transaction_type", "security_market", "security_type", "src_line_nbr"), ("record_date",), (), "record_date"),
    TableDefinition("bls_qcew.area_industry_annual", "bls_qcew/area_industry_annual.csv.gz", ("area_fips", "aggregation_level", "ownership_scope", "industry_code", "year"), (), ("area_fips", "state_fips", "industry_code"), None, categorical_columns=("state_fips", "aggregation_level", "ownership_scope", "industry_code", "disclosure_code", "over_year_disclosure_code", "source_archive")),
    TableDefinition("ofr_market_stress.financial_stress_index", "ofr_market_stress/financial_stress_index.csv.gz", ("date",), ("date",), (), "date"),
    TableDefinition("census_geo_crosswalk.counties_2024", "census_geo_crosswalk/normalized/county_fips_2024.csv.gz", ("GEOID",), (), ("GEOID", "ANSICODE"), None, "county_fips_2024.csv.gz"),
    TableDefinition("census_geo_crosswalk.zcta_county_relationships_2020", "census_geo_crosswalk/normalized/zcta_county_relationship_2020.csv.gz", ("GEOID_ZCTA5_20", "GEOID_COUNTY_20"), (), ("GEOID_ZCTA5_20", "GEOID_COUNTY_20"), None, "zcta_county_relationship_2020.csv.gz"),
    TableDefinition("census_geo_crosswalk.zcta_primary_counties_2020", "census_geo_crosswalk/normalized/zcta_primary_county_2020.csv.gz", ("zcta5",), (), ("zcta5", "county_geoid"), None, "zcta_primary_county_2020.csv.gz"),
    TableDefinition("ffiec_nic_structure.institution_identifiers_current", "ffiec_nic_structure/normalized/institution_identifiers_current.csv.gz", ("rssd_id",), (), ("rssd_id", "county_code", "zip_code", "fdic_certificate", "occ_id", "ncua_id", "aba_primary", "cusip_id", "lei", "tax_id"), None, "institution_identifiers_current.csv.gz"),
    TableDefinition("ffiec_nic_structure.ownership_relationships_current", "ffiec_nic_structure/normalized/ownership_relationships_current.csv.gz", ("parent_rssd_id", "offspring_rssd_id", "relationship_start", "relationship_level"), ("relationship_start",), ("parent_rssd_id", "offspring_rssd_id"), None, "ownership_relationships_current.csv.gz"),
    TableDefinition("ffiec_y9c.holding_company_balance_sheets", "ffiec_y9c/holding_company_balance_sheets.csv.gz", ("reporter_rssd_id", "report_date"), ("report_date",), ("reporter_rssd_id",), "report_date"),
    TableDefinition("eia_steo_vintages.wti_monthly_vintages", "eia_steo_vintages/wti_monthly_vintages.csv.gz", ("publication_date", "target_month"), ("publication_date",), ("series_code",), "publication_date"),
    TableDefinition("eia_bulk.wti_monthly_spot_prices", "eia_bulk/wti_monthly_spot_prices.csv.gz", ("observation_month",), (), ("series_id",), "observation_month"),
    TableDefinition("eia_bulk.wti_daily_spot_prices", "eia_bulk/wti_daily_spot_prices.csv.gz", ("observation_date",), ("observation_date",), ("series_id",), "observation_date"),
    TableDefinition("fema_disasters.declaration_areas", "fema_disasters/declaration_areas.csv.gz", ("declaration_area_id",), ("declaration_date", "incident_begin_date", "incident_end_date", "snapshot_last_refresh"), ("declaration_area_id", "disaster_number", "fips_state_code", "fips_county_code", "county_fips"), "declaration_date"),
    TableDefinition("noaa_storm_events.details", "noaa_storm_events/storm_event_details.csv.gz", ("event_id",), ("begin_datetime", "end_datetime"), ("episode_id", "event_id", "state_fips", "state_usps", "county_zone_fips"), "begin_datetime"),
    TableDefinition("fema_nfip.claims", "fema_nfip/claims.csv.gz", ("claim_record_id",), ("snapshot_as_of_date", "date_of_loss", "open_date"), ("claim_record_id", "state_usps", "county_code", "census_geoid"), "date_of_loss"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision_policy(table_id: str) -> str:
    if table_id.startswith(("sec.", "sec_insider.", "sec_nmfp.", "sec_nport.", "sec_form_")):
        return "preserve all accessions/amendments; downstream selection must apply filing-time and amendment semantics"
    if table_id.startswith(("rtdsm.", "sec_bdc.", "treasury_usfr.")):
        return "preserve all published vintages/periods; do not silently replace historical observations"
    return "frozen benchmark snapshot; rebuilds must retain the declared cutoff and source provenance"


def _unit_for(table_id: str, column: str) -> str | None:
    lower = column.lower()
    if table_id == "fed_scf.household_summary" and column == "survey_weight":
        return "household weight; public-summary WGT equals X42001 divided by five"
    if lower.endswith("_thousand_usd"):
        return "thousand USD"
    if lower.endswith("_million_usd"):
        return "million USD"
    if lower.endswith("_billion_usd") or "($billions)" in lower:
        return "billion USD"
    if lower.endswith("_usd") or "dollaramount" in lower or lower in {
        "offeringamount", "maximumofferingamount", "totalofferingamount",
        "totalamountsold", "totalremaining", "grossapproval",
        "sbaguaranteedapproval", "thirdpartydollars", "grosschargeoffamount",
        "loan_amount",
    }:
        return "USD"
    if lower.endswith("_pct") or "(%)" in column or lower.endswith("_rat"):
        return "percent"
    if lower.endswith("_fraction"):
        return "decimal fraction"
    if lower.endswith("_thousands"):
        return "thousands of the named measure"
    if table_id == "fdic.bank_financials" and column in {
        "ASSET", "CHBAL", "DEP", "DEPDOM", "EINTEXP", "EQ", "FREPO",
        "INTINC", "LNATRES", "LNCI", "LNCON", "LNLSNET", "LNRE",
        "NCLNLS", "NETINC", "NONII", "NONIX", "NTLNLS",
        "RBCT1J", "RBCT2",
    }:
        return "thousand USD"
    if table_id == "fed_stress_tests.results" and lower.endswith("_amt"):
        return "billion USD"
    if table_id == "fed_stress_tests.results" and lower.endswith("_rate"):
        return "percent"
    if table_id.startswith("sec_insider.") and lower in {
        "trans_pricepershare", "conv_exercise_price",
    }:
        return "USD per share unless the filing footnote declares another currency or basis"
    if table_id.startswith("sec_insider.") and lower in {
        "trans_total_value", "undlyng_sec_value", "valu_ownd_folwng_trans",
    }:
        return "source-reported monetary value; filing footnotes control the currency and basis"
    if table_id == "treasury_auctions.auctions" and lower in {
        "price_per100", "adj_price", "avg_med_price", "high_price",
        "low_price", "unadj_price",
    }:
        return "USD per 100 USD face value"
    if table_id == "treasury_auctions.auctions" and lower == "offering_amt":
        return "USD"
    if table_id == "sec_form_c.filings" and lower in {
        "totalassetmostrecentfiscalyear", "cashequimostrecentfiscalyear",
        "shorttermdebtmrecentfiscalyear", "longtermdebtrecentfiscalyear",
        "revenuemostrecentfiscalyear", "netincomemostrecentfiscalyear",
    }:
        return "USD"
    if table_id.startswith("sec_bdc.") and lower in {
        "total_interest_rate", "pik_interest_rate",
    }:
        return "decimal rate"
    if table_id.startswith("sba_7a_504.") and lower == "initialinterestrate":
        return "percent"
    if table_id == "fdic.bank_financials" and column in {
        "NIMY", "RBC1AAJ", "RBCRWAJ", "ROA", "ROE",
    }:
        return "percent"
    if table_id == "bis.global_liquidity" and column == "OBS_VALUE:Observation Value":
        return "millions of the currency named by CURR_DENOM; this governed projection retains amount rows only"
    if table_id == "bis.effective_exchange_rates" and column == "OBS_VALUE:Observation Value":
        return "index points, 2020=100; this governed projection retains monthly broad EER rows"
    if table_id.startswith("bis.") and column == "OBS_VALUE:Observation Value":
        return "row-specific; see UNIT_MEASURE and UNIT_MULT columns"
    if table_id == "sec.financial_statements":
        if column in {"cik", "ticker", "company_name", "accession", "form", "fiscal_year",
                      "fiscal_period", "report_date", "filed_date", "sic"}:
            return None
        base = column.removesuffix("_ytd").removesuffix("_q")
        if base.startswith("EarningsPerShare"):
            return "USD per share"
        if "Shares" in base:
            return "shares"
        return "USD"
    if table_id == "sec.company_facts" and column == "value":
        return "row-specific; see unit column"
    if table_id in {"sec_notes.numeric_facts", "sec_notes.dimensional_facts"} and column == "value":
        return "row-specific; see uom column"
    if table_id == "fed_z1_vintages.series_observations" and column == "value":
        return "row-specific; Z.1 series units (millions of USD for most levels and flows)"
    if table_id == "sec_13f.holdings_by_cusip" and lower.endswith("_usd_total"):
        return "USD (normalised from the filing-date reporting unit)"
    if table_id == "sec_13f.filings" and column == "tablevaluetotal":
        # The cover page carries its own scale, which is NOT the one
        # value_reporting_convention describes: that column is inferred from the
        # holding rows. Form 13F moved the cover-page total to whole dollars for
        # periods from 2022Q4, and roughly one filing in eleven sits on the other
        # side of that break in either direction, so the scale is per filing.
        return "as reported; cover-page scale, thousands of USD before the 2022Q4 whole-dollar change and USD after, with per-filing exceptions"
    if table_id == "sec_13f.filings" and column == "value_reporting_convention":
        return "inferred unit of this filing's holding rows, not of tablevaluetotal"
    if table_id == "sec_midas.daily_security" and lower.endswith("_thousands"):
        return "thousands of shares"
    if table_id == "sec_ftd.fails_to_deliver":
        if column == "price":
            return "USD per share"
        if column == "quantity_fails":
            return "shares"
    if table_id == "sec_nmfp.share_class_yields" and "yield" in lower:
        return "decimal rate"
    if table_id.startswith("ken_french.") and lower.endswith("_pct"):
        return "percent"
    if table_id == "nyfed.reference_rates" and column == "Volume ($Billions)":
        return "billion USD"
    if table_id in {
        "eia_steo_vintages.wti_monthly_vintages",
        "eia_bulk.wti_monthly_spot_prices",
        "eia_bulk.wti_daily_spot_prices",
    } and lower.endswith("_usd_per_barrel"):
        return "USD per barrel"
    if table_id == "noaa_storm_events.details" and lower.endswith("_damage_usd"):
        return "USD"
    if table_id == "treasury_tic.major_foreign_holders" and lower.endswith("_usd_billions"):
        return "billion USD"
    if table_id.startswith("treasury_dts.") and lower.endswith("_million_usd"):
        return "million USD"
    if table_id == "fema_nfip.claims" and lower.endswith("_usd"):
        return "USD"
    return None


def _string_cleanup(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""), pd.NA)


def _identifier_cleanup(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series.dtype):
        def render(value: Any) -> Any:
            if pd.isna(value):
                return pd.NA
            numeric = float(value)
            return str(int(numeric)) if math.isfinite(numeric) and numeric.is_integer() else str(value)
        return series.map(render).astype("string")
    return _string_cleanup(series)


def _normalize_date(series: pd.Series, *, table_id: str, column: str) -> pd.Series:
    cleaned = _string_cleanup(series)
    nonnull = cleaned.dropna()
    if nonnull.empty:
        return cleaned
    parsed = pd.to_datetime(nonnull, errors="coerce", format="mixed")
    failures = nonnull[parsed.isna()]
    if not failures.empty:
        examples = failures.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"{table_id}.{column}: unparseable dates {examples}")
    has_time = bool((parsed.dt.hour.ne(0) | parsed.dt.minute.ne(0) | parsed.dt.second.ne(0)).any())
    formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S" if has_time else "%Y-%m-%d")
    result = cleaned.copy()
    result.loc[nonnull.index] = formatted
    return result.astype("string")


def _cutoff_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    years = values.str.extract(r"((?:19|20)\d{2})", expand=False).astype("Int64")
    unknown = values.notna() & years.isna()
    if unknown.any():
        examples = values.loc[unknown].drop_duplicates().head(5).tolist()
        raise ValueError(f"cannot derive cutoff year from {examples}")
    return years.isna() | years.le(int(BENCHMARK_CUTOFF[:4]))


def _storage_type(
    series: pd.Series, date: bool, identifier: bool, categorical: bool = False,
) -> tuple[str, str]:
    if date:
        return "string", "date_or_datetime"
    if identifier:
        return "string", "identifier"
    if categorical:
        return "category", "string"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean", "boolean"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "integer", "integer"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "number", "number"
    return "string", "string"


def clean_table(
    definition: TableDefinition,
    datasets_dir: Path = DATASETS_DIR,
    build_dir: Path | None = None,
) -> dict[str, Any]:
    build_dir = BUILD_DIR if build_dir is None else build_dir
    reviewed_input = build_dir / definition.reviewed_input_path
    if not reviewed_input.is_file():
        raise FileNotFoundError(
            f"missing reviewed input for {definition.table_id}: {reviewed_input}"
        )
    input_sha256 = _sha256(reviewed_input)
    # Only the empty field is missing: pandas' default NA tokens would silently
    # turn real values into nulls (ticker "NA" for Nano Labs, country code "NA"
    # for Namibia, "NULL"/"None" category labels).
    frame = pd.read_csv(
        reviewed_input,
        low_memory=False,
        keep_default_na=False,
        na_values=[""],
        dtype={
            column: "string"
            for column in (*definition.identifier_columns, *definition.date_columns)
        },
    )
    original_rows = len(frame)
    for column in definition.empty_at_source_columns:
        if column not in frame.columns:
            raise ValueError(f"{definition.table_id}: unknown empty-at-source column {column}")
        if frame[column].notna().any() and frame[column].astype("string").str.strip().ne("").any():
            raise ValueError(
                f"{definition.table_id}: {column} is registered as empty at source but carries values"
            )
    frame = frame.drop(columns=list(definition.empty_at_source_columns))
    missing_dates = set(definition.date_columns) - set(frame.columns)
    missing_ids = set(definition.identifier_columns) - set(frame.columns)
    missing_key = set(definition.primary_key) - set(frame.columns)
    if missing_dates or missing_ids or missing_key:
        raise ValueError(
            f"{definition.table_id}: registry columns missing; dates={sorted(missing_dates)}, "
            f"identifiers={sorted(missing_ids)}, key={sorted(missing_key)}"
        )

    for column in frame.columns:
        if column in definition.identifier_columns:
            frame[column] = _identifier_cleanup(frame[column])
        elif column in definition.date_columns:
            frame[column] = _normalize_date(
                frame[column], table_id=definition.table_id, column=column
            )
        elif pd.api.types.is_object_dtype(frame[column].dtype) or isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = _string_cleanup(frame[column])

    filtered_rows = 0
    observed_cutoff_max: str | None = None
    if definition.cutoff_column:
        if definition.cutoff_column not in frame:
            raise ValueError(f"{definition.table_id}: missing cutoff column {definition.cutoff_column}")
        keep = _cutoff_mask(frame[definition.cutoff_column])
        filtered_rows = int((~keep).sum())
        frame = frame.loc[keep].copy()
        nonnull = frame[definition.cutoff_column].dropna().astype(str)
        observed_cutoff_max = nonnull.max() if not nonnull.empty else None

    exact_duplicate_rows = int(frame.duplicated(keep=False).sum())
    key_duplicate_rows = (
        int(frame.duplicated(list(definition.primary_key), keep=False).sum())
        if definition.primary_key else None
    )

    normalized = build_dir / definition.normalized_path
    runtime = datasets_dir / definition.runtime_path
    normalized.parent.mkdir(parents=True, exist_ok=True)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        normalized, index=False, na_rep="",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    dtype_map = {}
    for column in frame.columns:
        dtype_map[column] = (
            "string" if column in definition.date_columns or column in definition.identifier_columns
            or pd.api.types.is_object_dtype(frame[column].dtype)
            or isinstance(frame[column].dtype, pd.StringDtype)
            else "boolean" if pd.api.types.is_bool_dtype(frame[column].dtype)
            else "Int64" if pd.api.types.is_integer_dtype(frame[column].dtype)
            else "Float64"
        )
    roundtrip = pd.read_csv(
        normalized, dtype=dtype_map, keep_default_na=False,
        na_values=[""], low_memory=False,
    )
    if list(roundtrip.columns) != list(frame.columns) or len(roundtrip) != len(frame):
        raise ValueError(f"{definition.table_id}: row/column loss after UTF-8 CSV round trip")
    for column in frame.columns:
        left = frame[column].reset_index(drop=True)
        right = roundtrip[column].reset_index(drop=True)
        if not left.isna().equals(right.isna()):
            raise ValueError(
                f"{definition.table_id}.{column}: missing/zero distinction changed "
                "after CSV round trip"
            )
        present = left.notna()
        if pd.api.types.is_numeric_dtype(left.dtype):
            equal = np.allclose(
                pd.to_numeric(left[present]).astype(float),
                pd.to_numeric(right[present]).astype(float),
                rtol=1e-12, atol=1e-12,
            )
        else:
            equal = left[present].astype(str).tolist() == right[present].astype(str).tolist()
        if not equal:
            raise ValueError(
                f"{definition.table_id}.{column}: value or text corruption after "
                "CSV round trip"
            )
    shutil.copyfile(normalized, runtime)

    columns = []
    for column in frame.columns:
        storage_type, logical_type = _storage_type(
            frame[column], column in definition.date_columns,
            column in definition.identifier_columns,
            column in definition.categorical_columns,
        )
        columns.append({
            "name": column,
            "storage_type": storage_type,
            "logical_type": logical_type,
            "nullable": bool(frame[column].isna().any()),
            "missing_count": int(frame[column].isna().sum()),
            "unit": _unit_for(definition.table_id, column),
        })

    duplicate_policy = {
        "rule": "preserve source rows; never convert duplicates to zero or silently drop them",
        "exact_duplicate_rows_observed": exact_duplicate_rows,
        "resolution": "table-specific downstream logic must select amendments/vintages or aggregate the declared grain",
    }
    if definition.table_id == "sec_nmfp.daily_shareholder_flows":
        business_columns = [
            "accession", "class_id", "gross_subscriptions_usd",
            "gross_redemptions_usd", "flow_date",
        ]
        duplicated = frame.duplicated(business_columns, keep=False)
        duplicate_policy.update({
            "business_row_columns": business_columns,
            "business_duplicate_rows_observed": int(duplicated.sum()),
            "business_duplicate_groups_observed": int(
                frame.loc[duplicated, business_columns].drop_duplicates().shape[0]
            ),
            "resolution": (
                "preserve every physical row with source_archive/source_row_number; "
                "case logic must aggregate the explicitly requested accession/class/date grain"
            ),
        })

    return {
        "table_id": definition.table_id,
        "source_id": definition.source_id,
        "status": "active",
        "provenance": {
            "manifest_path": f"{definition.source_id}/sources.json",
            "reviewed_input_path": definition.reviewed_input_path,
        },
        "layers": {
            "normalized": {
                "path": definition.normalized_path,
                "bytes": normalized.stat().st_size,
                "sha256": _sha256(normalized),
            },
            "runtime": {
                "path": definition.runtime_path,
                "bytes": runtime.stat().st_size,
                "sha256": _sha256(runtime),
            },
        },
        "row_count": len(frame),
        "source_row_count": original_rows,
        "column_count": len(frame.columns),
        "columns": columns,
        "primary_key": {
            "columns": list(definition.primary_key),
            "kind": (
                "natural_or_source_key" if definition.primary_key
                else "none_declared; source extract lacks a guaranteed unique row identifier"
            ),
            "is_unique": key_duplicate_rows == 0 if key_duplicate_rows is not None else False,
            "duplicate_rows_observed": key_duplicate_rows,
        },
        "duplicate_policy": duplicate_policy,
        "revision_policy": _revision_policy(definition.table_id),
        "identifiers": list(definition.identifier_columns),
        "cutoff": {
            "benchmark_cutoff": BENCHMARK_CUTOFF,
            "column": definition.cutoff_column,
            "observed_max": observed_cutoff_max,
            "rows_removed_above_cutoff": filtered_rows,
            "inherited_from_parent": definition.cutoff_column is None,
        },
        "missing_value_policy": "empty CSV field means missing; numeric zero remains numeric zero",
        "transforms": [
            "trim string whitespace and map empty strings to missing",
            "store declared identifiers as strings",
            "format declared dates as ISO 8601",
            "apply only the declared benchmark cutoff column",
            "preserve row order and all source rows within cutoff",
        ],
        "input_sha256": input_sha256,
    }


def build_layers(datasets_dir: Path = DATASETS_DIR) -> dict[str, Any]:
    tables = [clean_table(definition, datasets_dir) for definition in TABLES]
    catalog = {
        "schema_version": 1,
        "dataset_release": "2026-08-13-layered-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_cutoff": BENCHMARK_CUTOFF,
        "raw_policy": "datasets/*/raw is immutable and excluded from runtime discovery",
        "normalized_policy": "conservative semantic cleaning with declared schemas; no silent deduplication",
        "runtime_policy": "v1 mirrors normalized tables; later releases may create documented projections",
        "table_count": len(tables),
        "tables": {table["table_id"]: table for table in tables},
    }
    catalog_path = datasets_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        by_source.setdefault(table["source_id"], []).append(table)
    for source_id, source_tables in by_source.items():
        payload = {
            "schema_version": 1,
            "source_id": source_id,
            "benchmark_cutoff": BENCHMARK_CUTOFF,
            "tables": {table["table_id"]: table for table in source_tables},
        }
        (datasets_dir / source_id / "schema.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
    return catalog


def build_selected_layers(
    table_ids: list[str], datasets_dir: Path = DATASETS_DIR,
) -> dict[str, Any]:
    """Rebuild explicit tables while retaining the existing governed catalog."""
    requested = list(dict.fromkeys(table_ids))
    if not requested:
        raise ValueError("at least one table ID is required")
    definitions = {definition.table_id: definition for definition in TABLES}
    unknown = sorted(set(requested) - set(definitions))
    if unknown:
        raise KeyError(f"unknown governed table IDs: {unknown}")

    catalog_path = datasets_dir / "catalog.json"
    catalog = load_catalog(catalog_path)
    affected_sources: set[str] = set()
    for table_id in requested:
        metadata = clean_table(definitions[table_id], datasets_dir)
        catalog["tables"][table_id] = metadata
        affected_sources.add(metadata["source_id"])
    catalog["table_count"] = len(catalog["tables"])
    catalog["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for source_id in sorted(affected_sources):
        schema_path = datasets_dir / source_id / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["tables"] = {
            table_id: metadata
            for table_id, metadata in catalog["tables"].items()
            if metadata["source_id"] == source_id
        }
        schema_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return catalog


def load_catalog(catalog_path: Path = CATALOG_PATH) -> dict[str, Any]:
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def load_runtime_table(
    table_id: str, *, datasets_dir: Path = DATASETS_DIR,
    catalog_path: Path | None = None,
) -> pd.DataFrame:
    """Load one governed runtime table using its recorded storage contract."""
    catalog = load_catalog(catalog_path or datasets_dir / "catalog.json")
    try:
        metadata = catalog["tables"][table_id]
    except KeyError as error:
        raise KeyError(f"unknown governed runtime table: {table_id}") from error
    dtype_map = {}
    for column in metadata["columns"]:
        dtype_map[column["name"]] = {
            "string": "string",
            "category": "category",
            "integer": "Int64",
            "number": "Float64",
            "boolean": "boolean",
        }[column["storage_type"]]
    return pd.read_csv(
        datasets_dir / metadata["layers"]["runtime"]["path"],
        dtype=dtype_map, keep_default_na=False, na_values=[""], low_memory=False,
    )


def validate_layers(
    datasets_dir: Path = DATASETS_DIR, *, verify_hashes: bool = True,
    build_dir: Path | None = None,
) -> list[str]:
    """Validate either a full maintainer install or the runtime-only release.

    Normalized tables live in ``build_dir`` when a maintainer build is present.
    The distributed ``dataset_new`` ZIP intentionally omits that layer; a
    completely absent normalized layer is therefore valid, while a partial one
    remains an error.
    """
    errors: list[str] = []
    catalog_path = datasets_dir / "catalog.json"
    if not catalog_path.is_file():
        return [f"missing {catalog_path}"]
    catalog = load_catalog(catalog_path)
    expected = {definition.table_id for definition in TABLES}
    tables = catalog.get("tables", {})
    actual = set(tables)
    if expected != actual:
        errors.append(f"catalog table IDs differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if catalog.get("table_count") != len(tables):
        errors.append("catalog table_count does not match registered tables")
    source_dirs = {path.name for path in datasets_dir.iterdir() if path.is_dir()}
    catalog_sources = set(catalog.get("sources", {}))
    if source_dirs != catalog_sources:
        errors.append(
            f"source coverage differs: missing={sorted(catalog_sources-source_dirs)}, "
            f"extra={sorted(source_dirs-catalog_sources)}"
        )
    build_dir = BUILD_DIR if build_dir is None else build_dir
    normalized_paths = [
        build_dir / metadata["layers"]["normalized"]["path"]
        for metadata in tables.values()
    ]
    normalized_present = [path.is_file() for path in normalized_paths]
    if any(normalized_present) and not all(normalized_present):
        errors.append("normalized layer is only partially materialized")
    layers = ("normalized", "runtime") if all(normalized_present) else ("runtime",)
    for table_id, metadata in tables.items():
        for layer in layers:
            info = metadata["layers"][layer]
            path = (build_dir if layer == "normalized" else datasets_dir) / info["path"]
            if not path.is_file():
                errors.append(f"{table_id}: missing {layer} file {info['path']}")
                continue
            if path.stat().st_size != info["bytes"]:
                errors.append(f"{table_id}: {layer} byte size changed")
            if verify_hashes and _sha256(path) != info["sha256"]:
                errors.append(f"{table_id}: {layer} sha256 changed")
        runtime_path = metadata["layers"]["runtime"]["path"]
        if "/raw/" in f"/{runtime_path}":
            errors.append(f"{table_id}: runtime path points into raw")
        if not metadata["columns"]:
            errors.append(f"{table_id}: empty schema")
        manifest_path = datasets_dir / metadata["provenance"]["manifest_path"]
        if not manifest_path.is_file():
            errors.append(f"{table_id}: missing provenance manifest {manifest_path}")
        if metadata["source_row_count"] - metadata["cutoff"]["rows_removed_above_cutoff"] != metadata["row_count"]:
            errors.append(f"{table_id}: cutoff row accounting mismatch")
    return errors
