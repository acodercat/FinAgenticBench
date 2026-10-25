"""Replace the remaining single-entity and single-year slices with populations.

Third data round (2026-08-27, `docs/data_quality_audit.md` §6). Every table
below existed as one filer, one fund, one accession or one year; each is
rebuilt from the full raw population already on disk, keeping the column
contract of the table it replaces:

- `sec_nport.fund_reports` — every N-PORT filing in the 25 quarterly data
  sets 2019q4–2025q4 (previously 2025 H2); `sec_nport.interest_rate_risk`
  over the same quarters (previously 2025q4). The two-accession
  `holding_details` slice is retired.
- `sec_nmfp.*` — series filings over the quarterly data sets 2019q1–2024q4
  plus the 2025 monthly archives. The daily-flow, daily-liquidity and
  seven-day-yield tables exist only in archives that carry the N-MFP3
  schedules (2025 monthly files); the class-yield table now covers every
  fund and class in those archives instead of one BlackRock FedFund filing.
  The single-filing portfolio slice is retired.
- `sec_form_d.filings`, `sec_form_c.filings` — 2019q1–2025q4 (previously 2025).
- `sec_bdc.investment_schedule` — every BDC filer's schedule-of-investments
  rows from the BDC data sets 2024q1–2025q4 (previously one Ares Capital
  filing); `sec_bdc.filings` lists the underlying submissions. The SEC
  publishes these data sets from 2024q1 only.
- `treasury_tic.major_foreign_holders` — every year block in the TIC
  history file (previously 2025 only).
- `fema_nfip.claims` — every year of loss on disk (2019–2025 once the
  OpenFEMA pull completes; previously 2024).
- `dol_form5500.*` — form years 2023 and 2024 (previously 2024).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from scripts.ingest.build.build_expansion_runtime import _read_tsv
from scripts.ingest.build.build_phase4_runtime import _zip_csv


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "population_windows_build.json"
FREEZE_DATE = "2025-12-31"
CUTOFF = pd.Timestamp(FREEZE_DATE)


def _write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, compression="gzip")
    print(f"  {path.relative_to(DATASETS)}: {len(frame):,} rows", flush=True)
    return {"file": str(path.relative_to(DATASETS)), "rows": len(frame), "columns": list(frame.columns)}


def _iso(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series, format=fmt, errors="coerce").dt.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- N-PORT

NPORT_FUND_COLUMNS = {
    "ACCESSION_NUMBER": "accession", "filing_date": "filing_date", "report_date": "report_date",
    "SUB_TYPE": "form", "CIK": "cik", "REGISTRANT_NAME": "registrant_name",
    "SERIES_NAME": "series_name", "SERIES_ID": "series_id", "TOTAL_ASSETS": "total_assets_usd",
    "TOTAL_LIABILITIES": "total_liabilities_usd", "NET_ASSETS": "net_assets_usd",
    "CASH_NOT_RPTD_IN_C_OR_D": "cash_not_elsewhere_reported_usd",
    "NET_REALIZE_GAIN_NONDERIV_MON1": "realized_gain_month_1_usd",
    "NET_UNREALIZE_AP_NONDERIV_MON1": "unrealized_change_month_1_usd",
    "NET_REALIZE_GAIN_NONDERIV_MON2": "realized_gain_month_2_usd",
    "NET_UNREALIZE_AP_NONDERIV_MON2": "unrealized_change_month_2_usd",
    "NET_REALIZE_GAIN_NONDERIV_MON3": "realized_gain_month_3_usd",
    "NET_UNREALIZE_AP_NONDERIV_MON3": "unrealized_change_month_3_usd",
    "SALES_FLOW_MON1": "sales_month_1_usd", "REINVESTMENT_FLOW_MON1": "reinvestment_month_1_usd",
    "REDEMPTION_FLOW_MON1": "redemptions_month_1_usd",
    "SALES_FLOW_MON2": "sales_month_2_usd", "REINVESTMENT_FLOW_MON2": "reinvestment_month_2_usd",
    "REDEMPTION_FLOW_MON2": "redemptions_month_2_usd",
    "SALES_FLOW_MON3": "sales_month_3_usd", "REINVESTMENT_FLOW_MON3": "reinvestment_month_3_usd",
    "REDEMPTION_FLOW_MON3": "redemptions_month_3_usd", "source_archive": "source_archive",
}
NPORT_RISK_COLUMNS = {
    "ACCESSION_NUMBER": "accession", "filing_date": "filing_date", "report_date": "report_date",
    "SERIES_NAME": "series_name", "SERIES_ID": "series_id", "NET_ASSETS": "net_assets_usd",
    "CURRENCY_CODE": "currency_code", "INTRST_RATE_CHANGE_3MON_DV01": "dv01_3month",
    "INTRST_RATE_CHANGE_1YR_DV01": "dv01_1year", "INTRST_RATE_CHANGE_5YR_DV01": "dv01_5year",
    "INTRST_RATE_CHANGE_10YR_DV01": "dv01_10year", "INTRST_RATE_CHANGE_30YR_DV01": "dv01_30year",
}


def build_nport() -> dict:
    root = RAW_DIR / "sec_nport"
    funds, risks = [], []
    for path in sorted(root.glob("*_nport.zip")):
        with zipfile.ZipFile(path) as zf:
            sub = _read_tsv(zf, "SUBMISSION.tsv")
            reg = _read_tsv(zf, "REGISTRANT.tsv", dtype={"ACCESSION_NUMBER": "string", "CIK": "string"})
            fund = _read_tsv(zf, "FUND_REPORTED_INFO.tsv")
            risk = _read_tsv(zf, "INTEREST_RATE_RISK.tsv")
        sub["filing_date"] = _iso(sub.FILING_DATE, "%d-%b-%Y")
        sub["report_date"] = _iso(sub.REPORT_DATE, "%d-%b-%Y")
        sub = sub.loc[sub.filing_date.le(FREEZE_DATE)]
        joined = sub.merge(reg[["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"]], on="ACCESSION_NUMBER", how="left")
        joined = joined.merge(fund, on="ACCESSION_NUMBER", how="inner")
        joined["source_archive"] = path.name
        funds.append(joined[list(NPORT_FUND_COLUMNS)])
        risk = risk.merge(sub[["ACCESSION_NUMBER", "filing_date", "report_date"]], on="ACCESSION_NUMBER", how="inner")
        risk = risk.merge(fund[["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "NET_ASSETS"]], on="ACCESSION_NUMBER", how="left")
        risks.append(risk[list(NPORT_RISK_COLUMNS)])
        print(f"  {path.name}: {len(joined):,} fund reports, {len(risk):,} rate-risk rows", flush=True)
    fund_frame = pd.concat(funds, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    fund_frame = fund_frame.rename(columns=NPORT_FUND_COLUMNS)
    fund_frame["cik"] = fund_frame.cik.astype("string").str.zfill(10)
    fund_frame = fund_frame.sort_values(["filing_date", "accession"]).reset_index(drop=True)
    risk_frame = pd.concat(risks, ignore_index=True).rename(columns=NPORT_RISK_COLUMNS)
    risk_frame = risk_frame.drop_duplicates(["accession", "currency_code"]).sort_values(["filing_date", "accession", "currency_code"]).reset_index(drop=True)
    return {
        "fund_reports": _write(fund_frame, DATASETS / "sec_nport" / "fund_reports.csv.gz"),
        "interest_rate_risk": _write(risk_frame, DATASETS / "sec_nport" / "interest_rate_risk.csv.gz"),
    }


# --------------------------------------------------------------------------- N-MFP

NMFP_SERIES_COLUMNS = {
    "ACCESSION_NUMBER": "accession", "filing_date": "filing_date", "report_date": "report_date",
    "SUBMISSIONTYPE": "form", "CIK": "cik", "REGISTRANTFULLNAME": "registrant_name",
    "SERIES_NAME": "series_name", "SERIESID": "series_id", "MONEYMARKETFUNDCATEGORY": "fund_category",
    "AVERAGEPORTFOLIOMATURITY": "weighted_average_maturity_days", "AVERAGELIFEMATURITY": "weighted_average_life_days",
    "CASH": "cash_usd", "TOTALVALUEPORTFOLIOSECURITIES": "portfolio_securities_usd",
    "TOTALVALUEOTHERASSETS": "other_assets_usd", "TOTALVALUELIABILITIES": "liabilities_usd",
    "NETASSETOFSERIES": "net_assets_usd", "NUMBEROFSHARESOUTSTANDING": "shares_outstanding",
    "source_archive": "source_archive",
}


def _members(zf: zipfile.ZipFile) -> set[str]:
    return {Path(name).name for name in zf.namelist()}


def build_nmfp() -> dict:
    root = RAW_DIR / "sec_nmfp"
    series, flows, liquidity, yields = [], [], [], []
    for path in sorted(root.glob("*_nmfp.zip")):
        with zipfile.ZipFile(path) as zf:
            members = _members(zf)
            sub = _read_tsv(zf, "NMFP_SUBMISSION.tsv", dtype={"ACCESSION_NUMBER": "string", "CIK": "string", "SERIESID": "string"})
            info = _read_tsv(zf, "NMFP_SERIESLEVELINFO.tsv", dtype={"ACCESSION_NUMBER": "string"})
            sub["filing_date"] = _iso(sub.FILING_DATE, "%d-%b-%Y")
            sub["report_date"] = _iso(sub.REPORTDATE, "%d-%b-%Y")
            sub = sub.loc[sub.filing_date.le(FREEZE_DATE)]
            if "REGISTRANTFULLNAME" not in sub:
                sub["REGISTRANTFULLNAME"] = sub.get("REGISTRANT")
            eligible = set(sub.ACCESSION_NUMBER)
            joined = sub.merge(info, on="ACCESSION_NUMBER", how="inner", suffixes=("", "_info"))
            joined["source_archive"] = path.name
            series.append(joined[[c for c in NMFP_SERIES_COLUMNS if c in joined]])
            if "NMFP_DLYSHAREHOLDERFLOWREPORT.tsv" in members:
                f = _read_tsv(zf, "NMFP_DLYSHAREHOLDERFLOWREPORT.tsv", dtype={"ACCESSION_NUMBER": "string", "CLASSESID": "string"})
                f.insert(0, "source_row_number", range(1, len(f) + 1)); f.insert(0, "source_archive", path.name)
                flows.append(f.loc[f.ACCESSION_NUMBER.isin(eligible)])
            if "NMFP_LIQUIDASSETSDETAILS.tsv" in members:
                l = _read_tsv(zf, "NMFP_LIQUIDASSETSDETAILS.tsv", dtype={"ACCESSION_NUMBER": "string"})
                liquidity.append(l.loc[l.ACCESSION_NUMBER.isin(eligible)])
            if "NMFP_SEVENDAYNETYIELD.tsv" in members:
                classes = _read_tsv(zf, "NMFP_CLASSLEVELINFO.tsv", usecols=["ACCESSION_NUMBER", "CLASSESID", "CLASS_NAME", "NETASSETSOFCLASS"])
                gross = _read_tsv(zf, "NMFP_SEVENDAYGROSSYIELD.tsv", usecols=["ACCESSION_NUMBER", "SEVENDAYGROSSYIELDVALUE", "SEVENDAYGROSSYIELDDATE"])
                net = _read_tsv(zf, "NMFP_SEVENDAYNETYIELD.tsv", usecols=["ACCESSION_NUMBER", "CLASSESID", "SEVENDAYNETYIELDVALUE", "SEVENDAYNETYIELDDATE"])
                net = net.loc[net.ACCESSION_NUMBER.isin(eligible)]
                y = net.merge(classes, on=["ACCESSION_NUMBER", "CLASSESID"], how="left").merge(
                    gross.drop_duplicates(["ACCESSION_NUMBER", "SEVENDAYGROSSYIELDDATE"]),
                    left_on=["ACCESSION_NUMBER", "SEVENDAYNETYIELDDATE"],
                    right_on=["ACCESSION_NUMBER", "SEVENDAYGROSSYIELDDATE"], how="left")
                y["source_archive"] = path.name
                yields.append(y)
        print(f"  {path.name}: {len(joined):,} series filings", flush=True)
    series_frame = pd.concat(series, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    series_frame = series_frame.rename(columns=NMFP_SERIES_COLUMNS).sort_values(["filing_date", "accession"]).reset_index(drop=True)
    out = {"series_filings": _write(series_frame, DATASETS / "sec_nmfp" / "series_filings.csv.gz")}
    if flows:
        f = pd.concat(flows, ignore_index=True)
        f.columns = ["source_archive", "source_row_number", "accession", "class_id", "gross_subscriptions_usd", "gross_redemptions_usd", "flow_date"]
        out["daily_shareholder_flows"] = _write(f, DATASETS / "sec_nmfp" / "daily_shareholder_flows.csv.gz")
    if liquidity:
        l = pd.concat(liquidity, ignore_index=True).drop_duplicates()
        l.columns = ["accession", "daily_liquid_assets_usd", "weekly_liquid_assets_usd", "daily_liquid_assets_fraction", "weekly_liquid_assets_fraction", "liquidity_date"]
        out["daily_liquidity"] = _write(l, DATASETS / "sec_nmfp" / "daily_liquidity.csv.gz")
    if yields:
        y = pd.concat(yields, ignore_index=True)[["ACCESSION_NUMBER", "CLASSESID", "CLASS_NAME", "NETASSETSOFCLASS", "SEVENDAYNETYIELDDATE", "SEVENDAYNETYIELDVALUE", "SEVENDAYGROSSYIELDVALUE", "source_archive"]]
        y.columns = ["accession", "class_id", "class_name", "class_net_assets_usd", "yield_date", "seven_day_net_yield", "seven_day_gross_yield", "source_archive"]
        y = y.drop_duplicates(["accession", "class_id", "yield_date"]).sort_values(["accession", "class_id", "yield_date"]).reset_index(drop=True)
        out["share_class_yields"] = _write(y, DATASETS / "sec_nmfp" / "share_class_yields.csv.gz")
    return out


# --------------------------------------------------------------------------- Form D / Form C

FORM_D_KEEP = [
    "ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "ENTITYNAME", "INDUSTRYGROUPTYPE",
    "ISAMENDMENT", "PREVIOUSACCESSIONNUMBER", "SALE_DATE", "TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD",
    "TOTALREMAINING", "HASNONACCREDITEDINVESTORS", "NUMBERNONACCREDITEDINVESTORS",
    "TOTALNUMBERALREADYINVESTED", "SALESCOMM_DOLLARAMOUNT", "FINDERSFEE_DOLLARAMOUNT",
]
FORM_C_KEEP = [
    "ACCESSION_NUMBER", "SUBMISSION_TYPE", "FILING_DATE", "CIK", "FILE_NUMBER", "PERIOD", "ISAMENDMENT",
    "PROGRESSUPDATE", "NATUREOFAMENDMENT", "NAMEOFISSUER", "OFFERINGAMOUNT", "MAXIMUMOFFERINGAMOUNT",
    "DEADLINEDATE", "CURRENTEMPLOYEES", "TOTALASSETMOSTRECENTFISCALYEAR", "CASHEQUIMOSTRECENTFISCALYEAR",
    "SHORTTERMDEBTMRECENTFISCALYEAR", "LONGTERMDEBTRECENTFISCALYEAR", "REVENUEMOSTRECENTFISCALYEAR",
    "NETINCOMEMOSTRECENTFISCALYEAR",
]


def _root(zf: zipfile.ZipFile) -> str:
    first = zf.namelist()[0]
    return first.split("/")[0] + "/" if "/" in first else ""


def build_form_d() -> dict:
    parts = []
    for path in sorted((RAW_DIR / "sec_form_d").glob("*_d.zip")):
        with zipfile.ZipFile(path) as zf:
            root = _root(zf)
            sub = _read_tsv(zf, f"{root}FORMDSUBMISSION.tsv")
            offering = _read_tsv(zf, f"{root}OFFERING.tsv")
            issuers = _read_tsv(zf, f"{root}ISSUERS.tsv", dtype={"ACCESSIONNUMBER": "string", "CIK": "string"})
        primary = issuers[issuers.IS_PRIMARYISSUER_FLAG.eq("YES")][["ACCESSIONNUMBER", "CIK", "ENTITYNAME"]].drop_duplicates("ACCESSIONNUMBER")
        frame = sub.merge(offering, on="ACCESSIONNUMBER").merge(primary, on="ACCESSIONNUMBER", how="left")
        frame["source_archive"] = path.name
        parts.append(frame)
        print(f"  {path.name}: {len(frame):,} filings", flush=True)
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSIONNUMBER", keep="last")
    data["FILING_DATE"] = _iso(data.FILING_DATE, "%d-%b-%Y")
    data["CIK"] = data.CIK.astype("string").str.zfill(10)
    data = data.loc[data.FILING_DATE.le(FREEZE_DATE)]
    data = data[FORM_D_KEEP + ["source_archive"]].sort_values(["FILING_DATE", "ACCESSIONNUMBER"]).reset_index(drop=True)
    return {"filings": _write(data, DATASETS / "sec_form_d" / "filings.csv.gz")}


def build_form_c() -> dict:
    parts = []
    for path in sorted((RAW_DIR / "sec_form_c").glob("*_cf.zip")):
        with zipfile.ZipFile(path) as zf:
            root = _root(zf)
            sub = _read_tsv(zf, f"{root}FORM_C_SUBMISSION.tsv",
                            dtype={"ACCESSION_NUMBER": "string", "CIK": "string", "FILE_NUMBER": "string", "PERIOD": "string"})
            info = _read_tsv(zf, f"{root}FORM_C_ISSUER_INFORMATION.tsv")
            disclosure = _read_tsv(zf, f"{root}FORM_C_DISCLOSURE.tsv")
        frame = sub.merge(info, on="ACCESSION_NUMBER", how="left").merge(disclosure, on="ACCESSION_NUMBER", how="left")
        frame["source_archive"] = path.name
        parts.append(frame)
        print(f"  {path.name}: {len(frame):,} filings", flush=True)
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    data["FILING_DATE"] = _iso(data.FILING_DATE.astype(str).str.slice(0, 8), "%Y%m%d")
    data["CIK"] = data.CIK.astype("string").str.zfill(10)
    data["PERIOD"] = data.PERIOD.astype("string").str.replace(r"\.0$", "", regex=True)
    data = data.loc[data.FILING_DATE.le(FREEZE_DATE)]
    keep = [c for c in FORM_C_KEEP if c in data] + ["source_archive"]
    data = data[keep].sort_values(["FILING_DATE", "ACCESSION_NUMBER"]).reset_index(drop=True)
    return {"filings": _write(data, DATASETS / "sec_form_c" / "filings.csv.gz")}


# --------------------------------------------------------------------------- BDC

BDC_SOI_COLUMNS = {
    "adsh": "accession", "cik": "cik", "name": "filer_name", "ddate": "report_date", "form": "form",
    "filed": "filed_date", "Investment, Identifier Axis": "investment_id",
    "Investment Type Axis": "investment_type", "Industry Sector Axis": "industry_sector",
    "Investment, Issuer Affiliation Axis": "issuer_affiliation",
    # Filers tag the classification either on the axis above or, in the newer
    # taxonomy, as an extensible enumeration; the two are coalesced below.
    "Investment, Type [Extensible Enumeration]": "investment_type_enum",
    "Investment, Industry Sector [Extensible Enumeration]": "industry_sector_enum",
    "Investment, Issuer Affiliation [Extensible Enumeration]": "issuer_affiliation_enum", "Investment Interest Rate": "total_interest_rate",
    "Investment, Basis Spread, Variable Rate": "basis_spread", "Investment Maturity Date": "maturity_date",
    "Investment Owned, Balance, Principal Amount": "principal_amount_usd",
    # SEC leaves its "Investment Owned, Cost"/"Investment Owned, Fair Value" columns
    # empty in every quarter; the values sit under these two labels instead. Verified
    # against datasets/num.tsv: per filing and report date, the column sums equal the
    # InvestmentOwnedAtCost / InvestmentOwnedAtFairValue tag sums less the one
    # portfolio-total row that carries no identifier axis.
    "Adjusted cost basis": "cost_usd",
    "Initial fair value of Investment": "fair_value_usd",
    "Investment Owned, Net Assets, Percentage": "net_assets_pct",
    "Investment, Interest Rate, Paid in Kind": "pik_interest_rate",
}
BDC_SUB_COLUMNS = {"adsh": "accession", "cik": "cik", "name": "filer_name", "form": "form", "period": "period",
                   "fy": "fiscal_year", "fp": "fiscal_period", "filed": "filed_date", "fye": "fiscal_year_end"}


def build_bdc() -> dict:
    root = RAW_DIR / "sec_bdc"
    schedules, subs = [], []
    for path in sorted(root.glob("*_bdc.zip")):
        with zipfile.ZipFile(path) as zf:
            with zf.open("soi.tsv") as handle:
                header = handle.readline().decode("utf-8").rstrip("\n").split("\t")
            usecols = [c for c in BDC_SOI_COLUMNS if c in header]
            soi = _read_tsv(zf, "soi.tsv", usecols=usecols)
            sub_member = next(m for m in zf.namelist() if m.endswith("sub.tsv"))
            sub = _read_tsv(zf, sub_member, usecols=[c for c in BDC_SUB_COLUMNS])
        soi = soi.loc[soi["Investment, Identifier Axis"].notna() & soi["Investment, Identifier Axis"].ne("")].copy()
        soi["source_archive"] = path.name
        sub["source_archive"] = path.name
        schedules.append(soi)
        subs.append(sub)
        print(f"  {path.name}: {len(soi):,} schedule rows, {len(sub):,} submissions", flush=True)
    schedule = pd.concat(schedules, ignore_index=True).rename(columns=BDC_SOI_COLUMNS)
    for column in ("investment_type", "industry_sector", "issuer_affiliation"):
        enum = f"{column}_enum"
        if enum in schedule:
            schedule[column] = schedule[column].fillna(schedule.pop(enum))
    for column in ("total_interest_rate", "basis_spread", "principal_amount_usd", "cost_usd", "fair_value_usd", "net_assets_pct", "pik_interest_rate"):
        if column in schedule:
            schedule[column] = pd.to_numeric(schedule[column], errors="coerce")
    schedule = schedule.loc[schedule.filed_date.astype(str).le(FREEZE_DATE)]
    schedule = schedule.drop_duplicates(["accession", "report_date", "investment_id"]).sort_values(["filed_date", "accession", "investment_id"]).reset_index(drop=True)
    filings = pd.concat(subs, ignore_index=True).rename(columns=BDC_SUB_COLUMNS)
    filings = filings.loc[filings.filed_date.astype(str).le(FREEZE_DATE)].drop_duplicates("accession", keep="last")
    filings = filings.sort_values(["filed_date", "accession"]).reset_index(drop=True)
    return {
        "investment_schedule": _write(schedule, DATASETS / "sec_bdc" / "investment_schedule.csv.gz"),
        "filings": _write(filings, DATASETS / "sec_bdc" / "filings.csv.gz"),
    }


# --------------------------------------------------------------------------- TIC

def build_tic() -> dict:
    path = RAW_DIR / "treasury_tic" / "mfhhis01.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    records = []
    for index, row in enumerate(rows):
        if not row or row[0].strip() != "Country" or len(row) < 13:
            continue
        years = [cell.strip() for cell in row[1:13]]
        if not all(re.fullmatch(r"\d{4}", y) for y in years):
            continue
        months = [cell.strip() for cell in rows[index - 1][1:13]]
        for body in rows[index + 1:]:
            country = body[0].strip() if body else ""
            if not country:
                if any(r.get("block_year") == years[0] for r in records[-1:]):
                    break
                continue
            if country.startswith("-"):
                continue
            row_type = "aggregate" if country in {"All Other", "Grand Total"} else "country"
            for year, month, raw in zip(years, months, body[1:13]):
                value = pd.to_numeric(raw.strip() or None, errors="coerce")
                if pd.isna(value):
                    continue
                stamp = pd.Timestamp(f"{month} 1, {year}")
                records.append({"country_name": country, "row_type": row_type,
                                "observation_month": f"{stamp.year}-{stamp.month:02d}",
                                "treasury_holdings_usd_billions": float(value),
                                "source_table": "Major Foreign Holders of Treasury Securities", "block_year": year})
    frame = pd.DataFrame(records).drop(columns="block_year")
    frame = frame.loc[frame.observation_month.le(FREEZE_DATE[:7])]
    frame = frame.drop_duplicates(["observation_month", "country_name"]).sort_values(["observation_month", "row_type", "country_name"]).reset_index(drop=True)
    return {"major_foreign_holders": _write(frame, DATASETS / "treasury_tic" / "major_foreign_holders.csv.gz")}


# --------------------------------------------------------------------------- NFIP

NFIP_FIELDS = {
    "id": "claim_record_id", "asOfDate": "snapshot_as_of_date", "dateOfLoss": "date_of_loss", "openDate": "open_date",
    "state": "state_usps", "countyCode": "county_code", "censusGeoid": "census_geoid", "policyCount": "policy_count",
    "floodEvent": "flood_event", "causeOfDamage": "cause_of_damage",
    "amountPaidOnBuildingClaim": "gross_building_payment_usd", "amountPaidOnContentsClaim": "gross_contents_payment_usd",
    "amountPaidOnIncreasedCostOfComplianceClaim": "gross_icc_payment_usd",
    "netBuildingPaymentAmount": "net_building_payment_usd", "netContentsPaymentAmount": "net_contents_payment_usd",
    "netIccPaymentAmount": "net_icc_payment_usd", "totalBuildingInsuranceCoverage": "building_coverage_usd",
    "totalContentsInsuranceCoverage": "contents_coverage_usd",
}


def build_nfip() -> dict:
    root = RAW_DIR / "fema_nfip"
    parts = []
    for path in sorted(root.glob("NfipClaims_20??.parquet")):
        source = pd.read_parquet(path, columns=list(NFIP_FIELDS))
        result = source.rename(columns=NFIP_FIELDS)
        result["snapshot_as_of_date"] = pd.to_datetime(result["snapshot_as_of_date"], utc=True).dt.strftime("%Y-%m-%d")
        for column in ("date_of_loss", "open_date"):
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
        # The 2023 year came through the CSV endpoint, which renders the codes as
        # floats ("6081.0"); normalise every year to zero-padded digit strings.
        for column, width in (("county_code", 5), ("census_geoid", 12)):
            codes = result[column].astype("string").str.replace(r"\.0$", "", regex=True)
            result[column] = codes.where(codes.isna() | codes.eq(""), codes.str.zfill(width))
        result["source_file"] = path.name
        parts.append(result)
        print(f"  {path.name}: {len(result):,} claims", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame.date_of_loss.le(FREEZE_DATE)].drop_duplicates("claim_record_id")
    frame = frame.sort_values(["date_of_loss", "claim_record_id"]).reset_index(drop=True)
    return {"claims": _write(frame, DATASETS / "fema_nfip" / "claims.csv.gz")}


# --------------------------------------------------------------------------- Form 5500

F5500_MAIN = ["ACK_ID", "TYPE_PLAN_ENTITY_CD", "PLAN_NAME", "SPONSOR_DFE_NAME", "SPONS_DFE_EIN", "SPONS_DFE_PN",
              "FORM_TAX_PRD", "INITIAL_FILING_IND", "AMENDED_IND", "FILING_STATUS", "DATE_RECEIVED",
              "TOT_PARTCP_BOY_CNT", "TOT_ACTIVE_PARTCP_CNT", "PARTCP_ACCOUNT_BAL_CNT"]
F5500_H = ["ACK_ID", "NET_ASSETS_BOY_AMT", "NET_ASSETS_EOY_AMT", "TOT_INCOME_AMT", "TOT_EXPENSES_AMT",
           "NET_INCOME_AMT", "TOT_TRANSFERS_TO_AMT", "TOT_TRANSFERS_FROM_AMT"]
F5500_SB = ["ACK_ID", "SB_PLAN_TYPE_CODE", "SB_VALUE_DATE", "SB_CURR_VALUE_AST_01_AMT", "SB_ACTRL_VALUE_AST_AMT",
            "SB_TOT_FNDNG_TGT_AMT", "SB_PLAN_AT_RISK_IND", "SB_TGT_DISREGARD_ASSUMP_AMT", "SB_CARRYOVER_BOY_TOT_AMT",
            "SB_PRE_FNDNG_BOY_TOT_AMT", "SB_FNDNG_TGT_PRCNT", "SB_ADJ_FNDNG_TGT_PRCNT", "SB_PR_YR_FNDNG_PRCNT", "SB_TOT_PARTCP_CNT"]
F5500_RENAME = {
    "ACK_ID": "filing_id", "TYPE_PLAN_ENTITY_CD": "plan_entity_code", "PLAN_NAME": "plan_name",
    "SPONSOR_DFE_NAME": "sponsor_name", "SPONS_DFE_EIN": "sponsor_ein", "SPONS_DFE_PN": "plan_number",
    "FORM_TAX_PRD": "plan_year_end", "INITIAL_FILING_IND": "initial_filing_indicator", "AMENDED_IND": "amended_indicator",
    "FILING_STATUS": "filing_status", "DATE_RECEIVED": "date_received", "TOT_PARTCP_BOY_CNT": "participants_beginning_count",
    "TOT_ACTIVE_PARTCP_CNT": "active_participants_end_count", "PARTCP_ACCOUNT_BAL_CNT": "participants_with_account_balance_count",
    "NET_ASSETS_BOY_AMT": "net_assets_beginning_usd", "NET_ASSETS_EOY_AMT": "net_assets_ending_usd",
    "TOT_INCOME_AMT": "total_income_usd", "TOT_EXPENSES_AMT": "total_expenses_usd", "NET_INCOME_AMT": "net_income_usd",
    "TOT_TRANSFERS_TO_AMT": "transfers_to_plan_usd", "TOT_TRANSFERS_FROM_AMT": "transfers_from_plan_usd",
    "SB_PLAN_TYPE_CODE": "sb_plan_type_code", "SB_VALUE_DATE": "valuation_date",
    "SB_CURR_VALUE_AST_01_AMT": "current_value_assets_usd", "SB_ACTRL_VALUE_AST_AMT": "actuarial_value_assets_usd",
    "SB_TOT_FNDNG_TGT_AMT": "total_funding_target_usd", "SB_PLAN_AT_RISK_IND": "at_risk_indicator",
    "SB_TGT_DISREGARD_ASSUMP_AMT": "funding_target_without_at_risk_assumptions_usd",
    "SB_CARRYOVER_BOY_TOT_AMT": "funding_standard_carryover_balance_usd", "SB_PRE_FNDNG_BOY_TOT_AMT": "prefunding_balance_usd",
    "SB_FNDNG_TGT_PRCNT": "reported_ftap_pct", "SB_ADJ_FNDNG_TGT_PRCNT": "reported_adjusted_ftap_pct",
    "SB_PR_YR_FNDNG_PRCNT": "prior_year_ftap_pct", "SB_TOT_PARTCP_CNT": "schedule_sb_participant_count",
}


def build_form5500() -> dict:
    root = RAW_DIR / "dol_form5500"
    financial_parts, pension_parts = [], []
    for main_zip in sorted(root.glob("F_5500_20??_all.zip")):
        year = re.search(r"F_5500_(\d{4})_all", main_zip.name).group(1)
        main = _zip_csv(main_zip, F5500_MAIN, dtype={"ACK_ID": "string", "SPONS_DFE_EIN": "string", "SPONS_DFE_PN": "string"})
        main["ACK_ID"] = main["ACK_ID"].str.strip()
        main["SPONS_DFE_EIN"] = main["SPONS_DFE_EIN"].str.strip().str.zfill(9)
        main["SPONS_DFE_PN"] = main["SPONS_DFE_PN"].str.strip().str.zfill(3)
        main["DATE_RECEIVED"] = pd.to_datetime(main["DATE_RECEIVED"], errors="coerce")
        main = main.loc[main["DATE_RECEIVED"].le(CUTOFF)].copy()
        main["form_year"] = int(year)
        financials = _zip_csv(root / f"F_SCH_H_{year}_all.zip", F5500_H)
        financial_parts.append(main.merge(financials, on="ACK_ID", how="inner", validate="one_to_one"))
        sb = _zip_csv(root / f"F_SCH_SB_{year}_all.zip", F5500_SB)
        pension_parts.append(main.merge(sb, on="ACK_ID", how="inner", validate="one_to_one"))
        print(f"  form year {year}: {len(financial_parts[-1]):,} Schedule H, {len(pension_parts[-1]):,} Schedule SB", flush=True)
    out = {}
    for name, parts in (("plan_financials", financial_parts), ("schedule_sb", pension_parts)):
        frame = pd.concat(parts, ignore_index=True).rename(columns=F5500_RENAME)
        frame["date_received"] = frame["date_received"].dt.strftime("%Y-%m-%d")
        frame = frame.drop_duplicates("filing_id").sort_values(["form_year", "date_received", "filing_id"]).reset_index(drop=True)
        out[name] = _write(frame, DATASETS / "dol_form5500" / f"{name}.csv.gz")
    return out


BUILDERS = {"nport": build_nport, "nmfp": build_nmfp, "form_d": build_form_d, "form_c": build_form_c,
            "bdc": build_bdc, "tic": build_tic, "nfip": build_nfip, "form5500": build_form5500}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BUILDERS))
    args = parser.parse_args()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name in args.only or list(BUILDERS):
        print(f"== {name}", flush=True)
        report[name] = {"built_at": datetime.now(timezone.utc).isoformat(), "freeze_date": FREEZE_DATE, "tables": BUILDERS[name]()}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
