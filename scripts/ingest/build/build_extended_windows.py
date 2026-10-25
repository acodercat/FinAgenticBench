"""Extend one-year and two-year governed tables to the 2019–2025 window.

The 2026-08-26 audit marked these sources "thin": complete populations but a
single year or two. Each rebuild below keeps the table's column contract and
widens its window from the raw files pulled on 2026-08-27
(`scripts/ingest/fetch/fetch_full_population.py`, `fetch_fiscal_history`):

- `treasury_debt.debt_to_penny` / `marketable_securities`: FiscalData pulls
  with record_date from 2019-01-01 (previously 2024-01-01).
- `treasury_dts.operating_cash_balance` / `public_debt_transactions`: the same
  two Daily Treasury Statement tables from 2019-01-01.
- `noaa_storm_events.details`: Storm Events details for 2019–2025 (previously
  2024 only); the file name drops the year.
- `fed_stress_tests.results`: every exercise in the Board's public results file
  (DFAST 2013 through the 2025 stress test). The earlier build filtered the
  same file down to the 2025 exercise.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from scripts.ingest.build.build_phase4_runtime import _numeric_strings
from scripts.ingest.build.build_phase11_cross_source_tables import STATE_FIPS_TO_USPS, _damage_usd


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "extended_windows_build.json"
FREEZE_DATE = "2025-12-31"
WINDOW_START = "2019-01-01"


def _write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, compression="gzip")
    print(f"  {path.relative_to(DATASETS)}: {len(frame):,} rows", flush=True)
    return {"file": str(path.relative_to(DATASETS)), "rows": len(frame), "columns": list(frame.columns)}


def build_treasury_debt() -> dict:
    raw = RAW_DIR / "treasury_debt"
    debt = pd.read_csv(raw / "debt_to_penny_2019_through_2025.csv", dtype={"record_date": str})
    debt = debt[["record_date", "debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"]].rename(columns={
        "debt_held_public_amt": "debt_held_by_public_usd",
        "intragov_hold_amt": "intragovernmental_holdings_usd",
        "tot_pub_debt_out_amt": "total_public_debt_outstanding_usd",
    })
    debt = debt.loc[debt["record_date"].between(WINDOW_START, FREEZE_DATE)].sort_values("record_date")

    market = pd.read_csv(raw / "mspd_table_3_market_2019_through_2025.csv", low_memory=False)
    market = market[[
        "record_date", "security_class1_desc", "security_class2_desc", "series_cd",
        "interest_rate_pct", "yield_pct", "issue_date", "maturity_date", "issued_amt",
        "inflation_adj_amt", "redeemed_amt", "outstanding_amt", "src_line_nbr",
    ]].copy()
    for column in ["interest_rate_pct", "yield_pct", "issued_amt", "inflation_adj_amt", "redeemed_amt", "outstanding_amt"]:
        market[column] = _numeric_strings(market[column])
    market = market.rename(columns={
        "security_class1_desc": "security_class",
        "security_class2_desc": "security_identifier_or_total_label",
        "series_cd": "series_code",
        "issued_amt": "issued_million_usd",
        "inflation_adj_amt": "inflation_adjustment_million_usd",
        "redeemed_amt": "redeemed_million_usd",
        "outstanding_amt": "outstanding_million_usd",
        "src_line_nbr": "source_line_number",
    })
    market = market.loc[market["record_date"].between(WINDOW_START, FREEZE_DATE)]
    market = market.sort_values(["record_date", "source_line_number"]).reset_index(drop=True)
    return {
        "debt_to_penny": _write(debt, DATASETS / "treasury_debt" / "debt_to_penny.csv.gz"),
        "marketable_securities": _write(market, DATASETS / "treasury_debt" / "marketable_securities.csv.gz"),
    }


def build_treasury_dts() -> dict:
    raw = RAW_DIR / "treasury_dts"
    operating = pd.read_csv(raw / "operating_cash_balance_2019_through_2025.csv", low_memory=False)
    operating = operating.rename(columns={
        "open_today_bal": "today_amount_million_usd",
        "open_month_bal": "month_to_date_amount_million_usd",
        "open_fiscal_year_bal": "fiscal_year_to_date_amount_million_usd",
    })[[
        "record_date", "account_type", "today_amount_million_usd",
        "month_to_date_amount_million_usd", "fiscal_year_to_date_amount_million_usd",
        "record_fiscal_year", "record_fiscal_quarter", "src_line_nbr",
    ]]
    operating = operating.loc[operating["record_date"].between(WINDOW_START, FREEZE_DATE)]
    operating = operating.sort_values(["record_date", "src_line_nbr"]).reset_index(drop=True)

    transactions = pd.read_csv(raw / "public_debt_transactions_2019_through_2025.csv",
                               dtype={"transaction_today_amt": "string"}, low_memory=False)
    transactions["transaction_today_amount_million_usd"] = pd.to_numeric(
        transactions.transaction_today_amt.str.replace("'", "", regex=False), errors="coerce")
    transactions = transactions[[
        "record_date", "transaction_type", "security_market", "security_type",
        "security_type_desc", "transaction_today_amount_million_usd",
        "record_fiscal_year", "record_fiscal_quarter", "src_line_nbr",
    ]]
    transactions = transactions.loc[transactions["record_date"].between(WINDOW_START, FREEZE_DATE)]
    transactions = transactions.sort_values(["record_date", "src_line_nbr"]).reset_index(drop=True)
    return {
        "operating_cash_balance": _write(operating, DATASETS / "treasury_dts" / "operating_cash_balance.csv.gz"),
        "public_debt_transactions": _write(transactions, DATASETS / "treasury_dts" / "public_debt_transactions.csv.gz"),
    }


NOAA_FIELDS = {
    "EPISODE_ID": "episode_id", "EVENT_ID": "event_id", "STATE": "state_name",
    "STATE_FIPS": "state_fips", "EVENT_TYPE": "event_type", "CZ_TYPE": "county_zone_type",
    "CZ_FIPS": "county_zone_fips", "CZ_NAME": "county_zone_name",
    "BEGIN_DATE_TIME": "begin_datetime", "END_DATE_TIME": "end_datetime",
    "DAMAGE_PROPERTY": "property_damage_source_token", "DAMAGE_CROPS": "crop_damage_source_token",
    "FLOOD_CAUSE": "flood_cause", "DATA_SOURCE": "data_source",
}


def build_noaa() -> dict:
    raw = RAW_DIR / "noaa_storm_events"
    parts = []
    for path in sorted(raw.glob("StormEvents_details-ftp_v1.0_d20*_c*.csv.gz")):
        year = int(re.search(r"_d(\d{4})_c", path.name).group(1))
        if not 2019 <= year <= 2025:
            continue
        source = pd.read_csv(path, low_memory=False, dtype={"STATE_FIPS": "string", "CZ_FIPS": "string"})
        result = source[list(NOAA_FIELDS)].rename(columns=NOAA_FIELDS)
        result["state_fips"] = result["state_fips"].str.zfill(2)
        result.insert(3, "state_usps", result["state_fips"].map(STATE_FIPS_TO_USPS))
        for column in ("begin_datetime", "end_datetime"):
            result[column] = pd.to_datetime(result[column], format="%d-%b-%y %H:%M:%S").dt.strftime("%Y-%m-%dT%H:%M:%S")
        result.insert(result.columns.get_loc("property_damage_source_token") + 1, "property_damage_usd",
                      result["property_damage_source_token"].map(_damage_usd))
        result.insert(result.columns.get_loc("crop_damage_source_token") + 1, "crop_damage_usd",
                      result["crop_damage_source_token"].map(_damage_usd))
        result["source_file"] = path.name
        parts.append(result)
        print(f"  {path.name}: {len(result):,} events", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame["begin_datetime"].le(FREEZE_DATE + "T23:59:59")]
    if frame["event_id"].duplicated().any():
        raise ValueError("duplicate NOAA event_id across yearly files")
    frame = frame.sort_values("event_id").reset_index(drop=True)
    return {"details": _write(frame, DATASETS / "noaa_storm_events" / "storm_event_details.csv.gz")}


DFAST_COLUMNS = [
    "exercise_name", "dt_exercise_quarter", "id_rssd", "disclosure_legal_name",
    "scenario_name", "common_equity_tier1_actual_rat", "common_equity_tier1_end_rat",
    "common_equity_tier1_min_rat", "loss_total_loan_amt", "loss_total_loan_rate",
    "loss_dom_first_mtg_amt", "loss_dom_first_mtg_rate", "loss_comml_ind_loan_amt",
    "loss_comml_ind_loan_rate", "loss_dom_cre_loan_amt", "loss_dom_cre_loan_rate",
    "loss_credit_card_loan_amt", "loss_credit_card_loan_rate",
    "revenue_preprovision_net_amt", "provision_amt", "income_pretax_net_amt",
]


def build_dfast() -> dict:
    raw = pd.read_csv(RAW_DIR / "fed_stress_tests" / "public_results_DFAST_2025.csv", dtype={"id_rssd": "string"})
    result = raw[DFAST_COLUMNS].copy()
    result["dt_exercise_quarter"] = pd.to_datetime(result["dt_exercise_quarter"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    result = result.drop_duplicates(["exercise_name", "scenario_name", "disclosure_legal_name"])
    result = result.sort_values(["dt_exercise_quarter", "exercise_name", "scenario_name", "disclosure_legal_name"]).reset_index(drop=True)
    info = _write(result, DATASETS / "fed_stress_tests" / "results.csv.gz")
    info["exercises"] = sorted(result["exercise_name"].unique())
    return {"results": info}


FEMA_FIELDS = {
    "id": "declaration_area_id", "disasterNumber": "disaster_number", "state": "state_usps",
    "declarationType": "declaration_type", "declarationDate": "declaration_date", "incidentType": "incident_type",
    "incidentBeginDate": "incident_begin_date", "incidentEndDate": "incident_end_date", "tribalRequest": "tribal_request",
    "fipsStateCode": "fips_state_code", "fipsCountyCode": "fips_county_code", "designatedArea": "designated_area",
    "ihProgramDeclared": "individuals_households_program_declared", "iaProgramDeclared": "individual_assistance_declared",
    "paProgramDeclared": "public_assistance_declared", "hmProgramDeclared": "hazard_mitigation_declared",
    "lastRefresh": "snapshot_last_refresh",
}


def build_fema_disasters() -> dict:
    """Declared-area rows for declaration dates 2019-2025 (previously 2024-2025)."""
    source_path = RAW_DIR / "fema_disasters" / "DisasterDeclarationsSummaries.parquet"
    source = pd.read_parquet(source_path, columns=list(FEMA_FIELDS))
    dates = pd.to_datetime(source["declarationDate"], errors="raise")
    selected = source.loc[dates.between(WINDOW_START, FREEZE_DATE)].rename(columns=FEMA_FIELDS).copy()
    selected["county_fips"] = selected["fips_state_code"].astype(str).str.zfill(2) + selected["fips_county_code"].astype(str).str.zfill(3)
    for column in ("declaration_date", "incident_begin_date", "incident_end_date"):
        selected[column] = pd.to_datetime(selected[column]).dt.strftime("%Y-%m-%d")
    selected["snapshot_last_refresh"] = pd.to_datetime(selected["snapshot_last_refresh"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    selected["source_file"] = source_path.name
    if selected["declaration_area_id"].duplicated().any():
        raise ValueError("duplicate FEMA declaration_area_id")
    selected = selected.sort_values(["declaration_date", "disaster_number", "declaration_area_id"]).reset_index(drop=True)
    return {"declaration_areas": _write(selected, DATASETS / "fema_disasters" / "declaration_areas.csv.gz")}


BUILDERS = {"treasury_debt": build_treasury_debt, "treasury_dts": build_treasury_dts,
            "noaa": build_noaa, "dfast": build_dfast, "fema": build_fema_disasters}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BUILDERS))
    args = parser.parse_args()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name, builder in BUILDERS.items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}", flush=True)
        report[name] = {"built_at": datetime.now(timezone.utc).isoformat(), "window": [WINDOW_START, FREEZE_DATE],
                        "tables": builder()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
