"""Rebuild the single-period and cohort slices as full 2019–2025 populations.

Acquisition and projection record for `docs/data_quality_audit.md`. Each
function keeps the column contract of the table it replaces and removes the
one restriction that made the table a slice: a single month, a single quarter,
a hand-picked cohort. Filters that encode a measurement definition (BIS basket,
position type, counterparty totals) are kept verbatim from the phase builders,
because those are what the case validators were written against.

| table                              | before                    | after                          |
|------------------------------------|---------------------------|--------------------------------|
| treasury_rates.yield_curve         | 2023–2025                 | 2019–2025 daily                |
| cftc_cot.financial_futures         | 2025                      | 2019–2025, futures and combined|
| bis.effective_exchange_rates       | Dec 2024 and Dec 2025     | every month, broad basket      |
| bis.global_liquidity               | 2025-Q2                   | every quarter                  |
| bis.otc_derivatives                | 2025-S1                   | every half-year                |
| fdic.bank_financials               | 12 banks, 2022–2025       | every institution, 2019–2025   |
| fdic_sod.branches                  | 2025 snapshot             | 2019–2025 annual snapshots     |
| ncua_call_reports.institutions     | 2025-Q3                   | every quarter 2019–2025        |
| finra_short_volume.daily           | April 2025                | every trading day 2024–2025    |
"""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT
from scripts.ingest.build.build_phase6_cross_source_tables import (
    FINRA_FIELDS, SOD_FIELDS, _clean, _write_deterministic_gzip_csv,
)


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
CUTOFF = pd.Timestamp("2025-12-31")
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "full_population_small_sources_build.json"


def _write(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return len(frame)


def build_treasury_yield_curve() -> int:
    curves = []
    for path in sorted((RAW_DIR / "treasury_rates").glob("daily_treasury_yield_curve_*.csv")):
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"])
        curves.append(frame)
    curve = (pd.concat(curves, ignore_index=True).drop_duplicates("Date")
             .sort_values("Date"))
    curve = curve[curve.Date.ge("2019-01-01") & curve.Date.le(CUTOFF)]
    return _write(curve, DATASETS / "treasury_rates" / "yield_curve.csv.gz")


def build_cftc() -> int:
    keep = [
        "Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code", "Open_Interest_All",
        "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All",
        "Lev_Money_Positions_Spread_All",
    ]
    parts = []
    for year in range(2019, 2026):
        for report_scope, prefix in [("futures_only", "fut_fin_txt"),
                                     ("futures_and_options_combined", "com_fin_txt")]:
            with zipfile.ZipFile(RAW_DIR / "cftc_cot" / f"{prefix}_{year}.zip") as zf:
                frame = pd.read_csv(zf.open(zf.namelist()[0]), usecols=keep, low_memory=False)
            frame.insert(0, "report_scope", report_scope)
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    result = result[pd.to_datetime(result["Report_Date_as_YYYY-MM-DD"]).le(CUTOFF)]
    return _write(result, DATASETS / "cftc_cot" / "financial_futures.csv.gz")


def _bis(name: str) -> pd.DataFrame:
    with zipfile.ZipFile(RAW_DIR / "bis" / f"{name}_csv_flat.zip") as zf:
        return pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)


def build_bis() -> dict[str, int]:
    period = "TIME_PERIOD:Time period or range"
    eer = _bis("WS_EER")
    eer = eer.loc[
        eer["FREQ:Frequency"].eq("M: Monthly")
        & eer["EER_BASKET:Basket"].eq("B: Broad (64 economies)")
        & eer[period].astype(str).ge("2019-01") & eer[period].astype(str).le("2025-12")
        & eer["OBS_VALUE:Observation Value"].notna(),
        ["EER_TYPE:Type", "EER_BASKET:Basket", "REF_AREA:Reference area", period,
         "OBS_VALUE:Observation Value", "UNIT_MEASURE:Unit of measure"],
    ].copy()
    gli = _bis("WS_GLI")
    gli = gli.loc[
        gli[period].astype(str).le("2025-Q4")
        & gli["CURR_DENOM:Currency of denomination"].isin(["USD: US dollar", "EUR: Euro", "JPY: Yen"])
        & gli["BORROWERS_CTY:Borrowers' country"].eq("4T: Emerging market and developing economies")
        & gli["BORROWERS_SECTOR:Borrowers' sector"].eq("N: Non-banks, total")
        & gli["L_POS_TYPE:Position type"].eq("I: Cross-border & Local in FCY")
        & gli["UNIT_MEASURE:Unit of measure"].isin(["USD: US dollar", "EUR: Euro", "JPY: Yen"]),
        ["CURR_DENOM:Currency of denomination", "BORROWERS_CTY:Borrowers' country",
         "BORROWERS_SECTOR:Borrowers' sector", "LENDERS_SECTOR:Lending sector",
         "L_POS_TYPE:Position type", "L_INSTR:Type of instruments",
         "UNIT_MEASURE:Unit of measure", period, "OBS_VALUE:Observation Value", "TITLE:Title"],
    ].copy()
    otc = _bis("WS_OTC_DERIV2")
    otc = otc.loc[
        otc[period].astype(str).le("2025-S2")
        & otc["DER_TYPE:Measure"].isin(["A: Outstanding - notional amounts", "D: Outstanding - gross market values"])
        & otc["DER_INSTR:Instrument"].eq("A: Total (all instruments)")
        & otc["DER_SECTOR_CPY:Counterparty sector"].eq("A: Total (all counterparties)")
        & otc["DER_CPC:Counterparty country"].eq("5J: All countries")
        & otc["DER_SECTOR_UDL:Underlying risk sector"].eq("A: Total (all sectors)")
        & otc["DER_CURR_LEG1:Currency leg 1"].eq("TO1: Total (all currencies)")
        & otc["DER_CURR_LEG2:Currency leg 2"].eq("TO1: Total (all currencies)")
        & otc["DER_ISSUE_MAT:Maturity"].eq("A: Total (all maturities)")
        & otc["DER_RATING:Rating (outstanding) or settlement (FX turnover)"].eq("A: Total (all ratings)")
        & otc["DER_EX_METHOD:Execution method"].eq("3: Total (all methods)")
        & otc["DER_BASIS:Basis"].eq("C: Net - net"),
        ["DER_TYPE:Measure", "DER_INSTR:Instrument", "DER_RISK:Risk category",
         "DER_SECTOR_CPY:Counterparty sector", "DER_BASIS:Basis", period,
         "OBS_VALUE:Observation Value", "UNIT_MEASURE:Unit of measure", "UNIT_MULT:Unit Multiplier"],
    ].copy()
    return {
        "effective_exchange_rates": _write(eer, DATASETS / "bis" / "broad_effective_exchange_rates.csv.gz"),
        "global_liquidity": _write(gli, DATASETS / "bis" / "global_liquidity_emde.csv.gz"),
        "otc_derivatives": _write(otc, DATASETS / "bis" / "otc_derivatives.csv.gz"),
    }


def build_fdic_financials() -> int:
    parts = [pd.read_csv(p, low_memory=False)
             for p in sorted((RAW_DIR / "fdic_bankfind").glob("all_banks_financials_*.csv"))]
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["CERT", "REPDTE"])
    data = data.sort_values(["REPDTE", "CERT"])
    return _write(data, DATASETS / "fdic_bankfind" / "bank_financials.csv.gz")


def build_fdic_sod() -> int:
    rows: list[dict[str, str]] = []
    for path in sorted((RAW_DIR / "fdic_sod").glob("sod_20*_offset*.csv")):
        year = path.name[4:8]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for source in reader:
                rows.append({
                    "report_date": f"{year}-06-30",
                    "fdic_certificate": _clean(source["CERT"]),
                    "bank_rssd_id": _clean(source["RSSDID"]),
                    "bank_name": _clean(source["NAMEFULL"]),
                    "branch_id": _clean(source["ID"]),
                    "branch_number": _clean(source["BRNUM"]),
                    "branch_name": _clean(source["NAMEBR"]),
                    "main_office_indicator": _clean(source["BKMO"]),
                    "branch_state": _clean(source["STALPBR"]),
                    "branch_county_fips": _clean(source["STCNTYBR"]).zfill(5),
                    "branch_deposits_thousand_usd": _clean(source["DEPSUMBR"]),
                    "source_file": path.name,
                })
    rows.sort(key=lambda row: (row["report_date"], row["fdic_certificate"], row["branch_id"]))
    keys = [(row["report_date"], row["branch_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (report_date, branch_id) in FDIC SOD pages")
    _write_deterministic_gzip_csv(DATASETS / "fdic_sod" / "branches.csv.gz", SOD_FIELDS, rows)
    return len(rows)


def build_ncua() -> int:
    parts = []
    for source in sorted((RAW_DIR / "ncua_call_reports").glob("call-report-data-20*.zip")):
        with ZipFile(source) as archive:
            # Member names are lower-case in the 2019 archives and upper-case after.
            members = {name.lower(): name for name in archive.namelist()}
            institutions = pd.read_csv(archive.open(members["foicu.txt"]),
                                       usecols=["CU_NUMBER", "CYCLE_DATE", "CU_NAME", "STATE"])
            financials = pd.read_csv(archive.open(members["fs220.txt"]),
                                     usecols=["CU_NUMBER", "ACCT_041B", "ACCT_025B"], low_memory=False)
        result = institutions.merge(financials, on="CU_NUMBER", validate="one_to_one")
        parts.append(result)
    result = pd.concat(parts, ignore_index=True).rename(columns={
        "CU_NUMBER": "credit_union_number", "CYCLE_DATE": "report_date",
        "CU_NAME": "credit_union_name", "STATE": "state",
        "ACCT_041B": "delinquent_loans_two_plus_months_usd",
        "ACCT_025B": "total_loans_and_leases_usd",
    })
    result["report_date"] = pd.to_datetime(result["report_date"]).dt.strftime("%Y-%m-%d")
    for column in ("delinquent_loans_two_plus_months_usd", "total_loans_and_leases_usd"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result[result["report_date"].le("2025-12-31")]
    if result.duplicated(["credit_union_number", "report_date"]).any():
        raise ValueError("duplicate credit union within a quarter")
    return _write(result.sort_values(["report_date", "credit_union_number"]),
                  DATASETS / "ncua_call_reports" / "institutions.csv.gz")


def build_finra_daily() -> int:
    rows: list[dict[str, str]] = []
    for source_path in sorted((RAW_DIR / "finra_short_volume").glob("CNMSshvol202[45]*.txt")):
        with source_path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.reader(handle, delimiter="|"))
        if not raw_rows or raw_rows[0] != ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]:
            raise ValueError(f"unexpected FINRA header in {source_path.name}")
        if len(raw_rows[-1]) != 1 or not raw_rows[-1][0].isdigit():
            raise ValueError(f"missing FINRA trailer in {source_path.name}")
        data_rows = raw_rows[1:-1]
        if int(raw_rows[-1][0]) != len(data_rows):
            raise ValueError(f"FINRA trailer count mismatch in {source_path.name}")
        expected_date = source_path.stem.removeprefix("CNMSshvol")
        for source in data_rows:
            if len(source) != 6 or source[0] != expected_date:
                raise ValueError(f"invalid FINRA data row in {source_path.name}")
            rows.append({
                "date": f"{source[0][:4]}-{source[0][4:6]}-{source[0][6:]}",
                "symbol": _clean(source[1]),
                "short_volume_shares": _clean(source[2]),
                "short_exempt_volume_shares": _clean(source[3]),
                "total_volume_shares": _clean(source[4]),
                "market_codes": _clean(source[5]),
                "source_file": source_path.name,
            })
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    keys = [(row["date"], row["symbol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate date-symbol rows in FINRA daily files")
    _write_deterministic_gzip_csv(DATASETS / "finra_short_volume" / "daily_short_volume.csv.gz", FINRA_FIELDS, rows)
    return len(rows)


def main() -> None:
    rows = {
        "treasury_rates.yield_curve": build_treasury_yield_curve(),
        "cftc_cot.financial_futures": build_cftc(),
        **{f"bis.{k}": v for k, v in build_bis().items()},
        "fdic.bank_financials": build_fdic_financials(),
        "fdic_sod.branches": build_fdic_sod(),
        "ncua_call_reports.institutions": build_ncua(),
        "finra_short_volume.daily": build_finra_daily(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(), "cutoff": "2025-12-31",
        "window_start": "2019-01-01", "rows": rows,
    }, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
