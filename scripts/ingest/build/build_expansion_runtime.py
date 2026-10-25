"""Build cutoff-safe runtime tables from the frozen expansion raw archives.

The raw downloads remain immutable. This script selects only fields that can be
redistributed to the Python runtime, removes direct applicant/borrower identity
fields, and enforces the benchmark knowledge cutoff of 2025-12-31.
"""

from __future__ import annotations

import json
import hashlib
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
CUTOFF = pd.Timestamp("2025-12-31")
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression=GZIP)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}: {len(df):,} rows")


def _read_tsv(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(zf.open(name), sep="\t", low_memory=False, **kwargs)


def _xlsx_data_sheet(path: Path) -> pd.DataFrame:
    """Read the simple worksheet headed by ``Date`` in an RTDSM workbook."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.iterfind(".//m:t", ns)))
        for sheet_name in sorted(
            name for name in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        ):
            root = ET.fromstring(zf.read(sheet_name))
            rows: list[dict[int, object]] = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values: dict[int, object] = {}
                for cell in row.findall("m:c", ns):
                    ref = cell.attrib["r"]
                    letters = re.match(r"[A-Z]+", ref).group(0)
                    col = 0
                    for letter in letters:
                        col = col * 26 + ord(letter) - 64
                    value_node = cell.find("m:v", ns)
                    inline = cell.find("m:is/m:t", ns)
                    raw = value_node.text if value_node is not None else inline.text if inline is not None else None
                    if raw is not None and cell.attrib.get("t") == "s":
                        raw = shared[int(raw)]
                    if raw is not None and cell.attrib.get("t") not in {"s", "str", "inlineStr"}:
                        try:
                            raw = float(raw)
                        except ValueError:
                            pass
                    values[col - 1] = raw
                rows.append(values)
            width = max(max(row, default=-1) for row in rows) + 1
            matrix = [[row.get(i) for i in range(width)] for row in rows]
            indices = [i for i, row in enumerate(matrix) if row and str(row[0]).strip() == "Date"]
            if indices:
                header_index = indices[0]
                headers = [str(value).strip() for value in matrix[header_index]]
                return pd.DataFrame(matrix[header_index + 1 :], columns=headers).dropna(how="all")
    raise ValueError(f"no worksheet headed by Date in {path}")


def build_rtdsm() -> None:
    labels = {
        "routput": "real_gdp_annualized_pct",
        "noutput": "nominal_gdp_annualized_pct",
        "employ_level": "payroll_change_thousands",
        "employ_pct_chg": "payroll_annualized_pct",
        "hstarts": "housing_starts_annualized_thousands",
        "pconx": "real_consumption_annualized_pct",
        "pcpi": "cpi_annualized_pct",
        "pcpix": "core_cpi_annualized_pct",
    }
    parts = []
    for path in sorted((RAW_DIR / "rtdsm").glob("*_first_second_third.xlsx")):
        key = path.name.removesuffix("_first_second_third.xlsx")
        frame = _xlsx_data_sheet(path)
        frame = frame[["Date", "First", "Second", "Third", "Most_Recent"]].copy()
        frame.columns = ["period", "first_release", "second_release", "third_release", "most_recent"]
        frame.insert(0, "series", labels[key])
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    for col in ["first_release", "second_release", "third_release", "most_recent"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result[result.period.astype(str).str.extract(r"(\d{4})", expand=False).astype(float).le(2025)]
    _write(result, DATASETS / "rtdsm" / "revisions.csv.gz")


def build_stress_tests() -> None:
    raw = pd.read_csv(
        RAW_DIR / "fed_stress_tests" / "public_results_DFAST_2025.csv",
        dtype={"id_rssd": "string"},
    )
    result = raw.loc[raw.exercise_name.eq("2025 Stress Test")].copy()
    keep = [
        "exercise_name", "dt_exercise_quarter", "id_rssd", "disclosure_legal_name",
        "scenario_name", "common_equity_tier1_actual_rat", "common_equity_tier1_end_rat",
        "common_equity_tier1_min_rat", "loss_total_loan_amt", "loss_total_loan_rate",
        "loss_dom_first_mtg_amt", "loss_dom_first_mtg_rate", "loss_comml_ind_loan_amt",
        "loss_comml_ind_loan_rate", "loss_dom_cre_loan_amt", "loss_dom_cre_loan_rate",
        "loss_credit_card_loan_amt", "loss_credit_card_loan_rate",
        "revenue_preprovision_net_amt", "provision_amt", "income_pretax_net_amt",
    ]
    _write(result[keep], DATASETS / "fed_stress_tests" / "results_2025.csv.gz")


def build_nmfp() -> None:
    parts = []
    flow_parts = []
    liquidity_parts = []
    for path in sorted((RAW_DIR / "sec_nmfp").glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            sub = _read_tsv(
                zf,
                "NMFP_SUBMISSION.tsv",
                dtype={
                    "ACCESSION_NUMBER": "string",
                    "CIK": "string",
                    "FILER_CIK": "string",
                    "SERIESID": "string",
                },
            )
            info = _read_tsv(
                zf,
                "NMFP_SERIESLEVELINFO.tsv",
                dtype={"ACCESSION_NUMBER": "string"},
            )
            flows = _read_tsv(
                zf,
                "NMFP_DLYSHAREHOLDERFLOWREPORT.tsv",
                dtype={"ACCESSION_NUMBER": "string", "CLASSESID": "string"},
            )
            liquidity = _read_tsv(
                zf,
                "NMFP_LIQUIDASSETSDETAILS.tsv",
                dtype={"ACCESSION_NUMBER": "string"},
            )
        filing_dates = sub[["ACCESSION_NUMBER", "FILING_DATE"]].copy()
        filing_dates["filing_date"] = pd.to_datetime(
            filing_dates.FILING_DATE, format="%d-%b-%Y", errors="coerce"
        )
        eligible = set(filing_dates.loc[filing_dates.filing_date.le(CUTOFF), "ACCESSION_NUMBER"])
        # Preserve source-native physical rows. One frozen file contains an
        # exact zero-flow duplicate, so a source ordinal is required to retain
        # provenance without pretending the class-day business key is unique.
        flows.insert(0, "source_row_number", range(1, len(flows) + 1))
        flows.insert(0, "source_archive", path.name)
        flow_parts.append(flows.loc[flows.ACCESSION_NUMBER.isin(eligible)])
        liquidity_parts.append(liquidity.loc[liquidity.ACCESSION_NUMBER.isin(eligible)])
        joined = sub.merge(info, on="ACCESSION_NUMBER", how="inner")
        joined["source_archive"] = path.name
        parts.append(joined)
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    data["filing_date"] = pd.to_datetime(data.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    data["report_date"] = pd.to_datetime(data.REPORTDATE, format="%d-%b-%Y", errors="coerce")
    data = data[data.filing_date.le(CUTOFF)]
    result = data[[
        "ACCESSION_NUMBER", "filing_date", "report_date", "SUBMISSIONTYPE", "CIK",
        "REGISTRANTFULLNAME", "SERIES_NAME", "SERIESID", "MONEYMARKETFUNDCATEGORY",
        "AVERAGEPORTFOLIOMATURITY", "AVERAGELIFEMATURITY", "CASH",
        "TOTALVALUEPORTFOLIOSECURITIES", "TOTALVALUEOTHERASSETS", "TOTALVALUELIABILITIES",
        "NETASSETOFSERIES", "NUMBEROFSHARESOUTSTANDING", "source_archive",
    ]].copy()
    result.columns = [
        "accession", "filing_date", "report_date", "form", "cik", "registrant_name",
        "series_name", "series_id", "fund_category", "weighted_average_maturity_days",
        "weighted_average_life_days", "cash_usd", "portfolio_securities_usd",
        "other_assets_usd", "liabilities_usd", "net_assets_usd", "shares_outstanding",
        "source_archive",
    ]
    _write(result, DATASETS / "sec_nmfp" / "series_filings.csv.gz")
    flows = pd.concat(flow_parts, ignore_index=True)
    flows.columns = [
        "source_archive", "source_row_number", "accession", "class_id",
        "gross_subscriptions_usd", "gross_redemptions_usd", "flow_date",
    ]
    _write(flows, DATASETS / "sec_nmfp" / "daily_shareholder_flows.csv.gz")
    liquidity = pd.concat(liquidity_parts, ignore_index=True).drop_duplicates()
    liquidity.columns = [
        "accession", "daily_liquid_assets_usd", "weekly_liquid_assets_usd",
        "daily_liquid_assets_fraction", "weekly_liquid_assets_fraction", "liquidity_date",
    ]
    _write(liquidity, DATASETS / "sec_nmfp" / "daily_liquidity.csv.gz")

    portfolio_accession = "0002071691-25-004249"
    path = RAW_DIR / "sec_nmfp" / "20251008-20251107_nmfp.zip"
    with zipfile.ZipFile(path) as zf:
        portfolio = _read_tsv(zf, "NMFP_SCHPORTFOLIOSECURITIES.tsv")
    portfolio = portfolio.loc[portfolio.ACCESSION_NUMBER.eq(portfolio_accession), [
        "ACCESSION_NUMBER", "SECURITY_ID", "NAMEOFISSUER", "TITLEOFISSUER",
        "INVESTMENTCATEGORY", "INVESTMENTMATURITYDATEWAM",
        "INVESTMENTMATURITYDATEWAL", "FINALLEGALINVESTMENTMATURITYDA",
        "EXCLUDINGVALUEOFANYSPONSORSUPP", "PERCENTAGEOFMONEYMARKETFUNDNET",
        "DAILYLIQUIDASSETSECURITYFLAG", "WEEKLYLIQUIDASSETSECURITYFLAG",
        "ILLIQUIDSECURITYFLAG",
    ]].copy()
    portfolio.columns = [
        "accession", "security_id", "issuer_name", "security_title",
        "investment_category", "wam_maturity_date", "wal_maturity_date",
        "final_legal_maturity_date", "reported_value_usd", "net_asset_fraction",
        "daily_liquid_flag", "weekly_liquid_flag", "illiquid_flag",
    ]
    _write(portfolio, DATASETS / "sec_nmfp" / "fedfund_2025_10_portfolio.csv.gz")

    with zipfile.ZipFile(path) as zf:
        classes = _read_tsv(
            zf, "NMFP_CLASSLEVELINFO.tsv",
            usecols=[
                "ACCESSION_NUMBER", "CLASSESID", "CLASS_NAME", "NETASSETSOFCLASS",
            ],
        )
        gross_yield = _read_tsv(
            zf, "NMFP_SEVENDAYGROSSYIELD.tsv",
            usecols=[
                "ACCESSION_NUMBER", "SEVENDAYGROSSYIELDVALUE",
                "SEVENDAYGROSSYIELDDATE",
            ],
        )
        net_yield = _read_tsv(
            zf, "NMFP_SEVENDAYNETYIELD.tsv",
            usecols=[
                "ACCESSION_NUMBER", "CLASSESID", "SEVENDAYNETYIELDVALUE",
                "SEVENDAYNETYIELDDATE",
            ],
        )
    classes = classes.loc[classes.ACCESSION_NUMBER.eq(portfolio_accession)]
    gross_yield = gross_yield.loc[gross_yield.ACCESSION_NUMBER.eq(portfolio_accession)]
    net_yield = net_yield.loc[net_yield.ACCESSION_NUMBER.eq(portfolio_accession)]
    class_yields = net_yield.merge(
        classes, on=["ACCESSION_NUMBER", "CLASSESID"], how="left", validate="many_to_one",
    ).merge(
        gross_yield,
        left_on=["ACCESSION_NUMBER", "SEVENDAYNETYIELDDATE"],
        right_on=["ACCESSION_NUMBER", "SEVENDAYGROSSYIELDDATE"],
        how="left", validate="many_to_one",
    )
    class_yields = class_yields[[
        "ACCESSION_NUMBER", "CLASSESID", "CLASS_NAME", "NETASSETSOFCLASS",
        "SEVENDAYNETYIELDDATE", "SEVENDAYNETYIELDVALUE",
        "SEVENDAYGROSSYIELDVALUE",
    ]]
    class_yields.columns = [
        "accession", "class_id", "class_name", "class_net_assets_usd", "yield_date",
        "seven_day_net_yield", "seven_day_gross_yield",
    ]
    _write(class_yields, DATASETS / "sec_nmfp" / "fedfund_2025_10_class_yields.csv.gz")


def build_nport() -> None:
    parts = []
    for path in sorted((RAW_DIR / "sec_nport").glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            sub = _read_tsv(zf, "SUBMISSION.tsv")
            reg = _read_tsv(
                zf, "REGISTRANT.tsv",
                dtype={"ACCESSION_NUMBER": "string", "CIK": "string"},
            )
            fund = _read_tsv(zf, "FUND_REPORTED_INFO.tsv")
        joined = sub.merge(reg[["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"]], on="ACCESSION_NUMBER")
        joined = joined.merge(fund, on="ACCESSION_NUMBER")
        joined["source_archive"] = path.name
        parts.append(joined)
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    data["filing_date"] = pd.to_datetime(data.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    data["report_date"] = pd.to_datetime(data.REPORT_DATE, format="%d-%b-%Y", errors="coerce")
    data = data[data.filing_date.le(CUTOFF)]
    keep = [
        "ACCESSION_NUMBER", "filing_date", "report_date", "SUB_TYPE", "CIK",
        "REGISTRANT_NAME", "SERIES_NAME", "SERIES_ID", "TOTAL_ASSETS",
        "TOTAL_LIABILITIES", "NET_ASSETS", "CASH_NOT_RPTD_IN_C_OR_D",
        "NET_REALIZE_GAIN_NONDERIV_MON1", "NET_UNREALIZE_AP_NONDERIV_MON1",
        "NET_REALIZE_GAIN_NONDERIV_MON2", "NET_UNREALIZE_AP_NONDERIV_MON2",
        "NET_REALIZE_GAIN_NONDERIV_MON3", "NET_UNREALIZE_AP_NONDERIV_MON3",
        "SALES_FLOW_MON1", "REINVESTMENT_FLOW_MON1", "REDEMPTION_FLOW_MON1",
        "SALES_FLOW_MON2", "REINVESTMENT_FLOW_MON2", "REDEMPTION_FLOW_MON2",
        "SALES_FLOW_MON3", "REINVESTMENT_FLOW_MON3", "REDEMPTION_FLOW_MON3",
        "source_archive",
    ]
    result = data[keep].copy()
    result.columns = [
        "accession", "filing_date", "report_date", "form", "cik", "registrant_name",
        "series_name", "series_id", "total_assets_usd", "total_liabilities_usd",
        "net_assets_usd", "cash_not_elsewhere_reported_usd",
        "realized_gain_month_1_usd", "unrealized_change_month_1_usd",
        "realized_gain_month_2_usd", "unrealized_change_month_2_usd",
        "realized_gain_month_3_usd", "unrealized_change_month_3_usd",
        "sales_month_1_usd", "reinvestment_month_1_usd", "redemptions_month_1_usd",
        "sales_month_2_usd", "reinvestment_month_2_usd", "redemptions_month_2_usd",
        "sales_month_3_usd", "reinvestment_month_3_usd", "redemptions_month_3_usd",
        "source_archive",
    ]
    result["cik"] = result.cik.astype("string").str.zfill(10)
    _write(result, DATASETS / "sec_nport" / "fund_reports.csv.gz")

    detail_accessions = {"0001049169-25-000764", "0002071691-25-004164"}
    path = RAW_DIR / "sec_nport" / "2025q4_nport.zip"
    holding_columns = [
        "ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_TITLE",
        "CURRENCY_VALUE", "PERCENTAGE", "PAYOFF_PROFILE", "ASSET_CAT",
        "DERIVATIVE_CAT",
    ]
    selected_holdings = []
    with zipfile.ZipFile(path) as zf:
        for chunk in pd.read_csv(
            zf.open("FUND_REPORTED_HOLDING.tsv"), sep="\t", usecols=holding_columns,
            chunksize=250_000, low_memory=False,
        ):
            selected = chunk.loc[chunk.ACCESSION_NUMBER.isin(detail_accessions)]
            if not selected.empty:
                selected_holdings.append(selected)
        holdings = pd.concat(selected_holdings, ignore_index=True)
        debt = _read_tsv(
            zf, "DEBT_SECURITY.tsv",
            usecols=[
                "HOLDING_ID", "MATURITY_DATE", "COUPON_TYPE", "ANNUALIZED_RATE",
                "IS_DEFAULT", "ARE_ANY_INTEREST_PAYMENT",
                "IS_ANY_PORTION_INTEREST_PAID",
            ],
        )
    debt = debt.loc[debt.HOLDING_ID.isin(holdings.HOLDING_ID)]
    details = holdings.merge(debt, on="HOLDING_ID", how="left")
    details.columns = [
        "accession", "holding_id", "issuer_name", "issuer_title",
        "currency_value_usd", "net_asset_pct", "payoff_profile",
        "asset_category", "derivative_category", "maturity_date", "coupon_type",
        "annualized_rate", "default_flag", "interest_arrears_flag", "pik_flag",
    ]
    _write(details, DATASETS / "sec_nport" / "selected_holding_details.csv.gz")

    with zipfile.ZipFile(path) as zf:
        interest_risk = _read_tsv(zf, "INTEREST_RATE_RISK.tsv")
        submissions = _read_tsv(
            zf, "SUBMISSION.tsv",
            usecols=["ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE"],
        )
        funds = _read_tsv(
            zf, "FUND_REPORTED_INFO.tsv",
            usecols=["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "NET_ASSETS"],
        )
    submissions["filing_date"] = pd.to_datetime(
        submissions.FILING_DATE, format="%d-%b-%Y", errors="coerce"
    )
    submissions["report_date"] = pd.to_datetime(
        submissions.REPORT_DATE, format="%d-%b-%Y", errors="coerce"
    )
    submissions = submissions.loc[submissions.filing_date.le(CUTOFF)]
    interest_risk = interest_risk.merge(
        submissions[["ACCESSION_NUMBER", "filing_date", "report_date"]],
        on="ACCESSION_NUMBER", how="inner", validate="many_to_one",
    ).merge(funds, on="ACCESSION_NUMBER", how="left", validate="many_to_one")
    interest_risk = interest_risk[[
        "ACCESSION_NUMBER", "filing_date", "report_date", "SERIES_NAME", "SERIES_ID",
        "NET_ASSETS", "CURRENCY_CODE", "INTRST_RATE_CHANGE_3MON_DV01",
        "INTRST_RATE_CHANGE_1YR_DV01", "INTRST_RATE_CHANGE_5YR_DV01",
        "INTRST_RATE_CHANGE_10YR_DV01", "INTRST_RATE_CHANGE_30YR_DV01",
    ]]
    interest_risk.columns = [
        "accession", "filing_date", "report_date", "series_name", "series_id",
        "net_assets_usd", "currency_code", "dv01_3month", "dv01_1year",
        "dv01_5year", "dv01_10year", "dv01_30year",
    ]
    _write(interest_risk, DATASETS / "sec_nport" / "interest_rate_risk_2025q4.csv.gz")


def build_hmda() -> None:
    filers = json.loads((RAW_DIR / "hmda" / "filers_2024.json").read_text())
    names = {item["lei"]: item["name"] for item in filers["institutions"]}
    usecols = [
        "activity_year", "lei", "derived_loan_product_type", "derived_dwelling_category",
        "action_taken", "loan_type", "loan_purpose", "lien_status",
        "business_or_commercial_purpose", "construction_method", "occupancy_type",
        "total_units", "loan_amount", "derived_race", "derived_ethnicity", "derived_sex",
    ]
    parts = []
    for path in sorted((RAW_DIR / "hmda").glob("2024_*.csv")):
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    result.insert(2, "institution_name", result.lei.map(names))
    _write(result, DATASETS / "hmda" / "lar_2024_top5.csv.gz")


def build_sec_notes() -> None:
    accession = "0000320193-25-000079"
    path = RAW_DIR / "sec_notes" / "2025_10_notes.zip"
    with zipfile.ZipFile(path) as zf:
        num = _read_tsv(zf, "num.tsv")
        tag = _read_tsv(zf, "tag.tsv")
    data = num[num.adsh.eq(accession)].copy()
    descriptions = tag[["tag", "version", "tlabel", "doc"]].drop_duplicates(["tag", "version"])
    data = data.merge(descriptions, on=["tag", "version"], how="left")
    keep = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "dimn", "value", "tlabel", "doc"]
    result = data[keep].copy()
    result.columns = ["accession", "concept", "taxonomy", "end_date", "quarters", "unit", "dimension_hash", "dimension_count", "value", "label", "description"]
    _write(result, DATASETS / "sec_notes" / "apple_fy2025_numeric.csv.gz")


def build_rates_and_auctions() -> None:
    rates = pd.read_csv(
        RAW_DIR / "nyfed_rates" / "reference_rates_2021_2025.csv",
        dtype={"Footnote ID": "string"},
    )
    rates["Effective Date"] = pd.to_datetime(rates["Effective Date"])
    rates = rates[rates["Effective Date"].le(CUTOFF)]
    _write(rates, DATASETS / "nyfed_rates" / "reference_rates.csv.gz")

    auctions = pd.concat([
        pd.read_csv(path, low_memory=False)
        for path in sorted((RAW_DIR / "treasury_auctions").glob("*.csv"))
    ], ignore_index=True).drop_duplicates()
    auctions["auction_date"] = pd.to_datetime(auctions["auction_date"], errors="coerce")
    auctions = auctions[auctions.auction_date.le(CUTOFF)]
    _write(auctions, DATASETS / "treasury_auctions" / "auctions.csv.gz")

    curves = []
    for path in sorted((RAW_DIR / "treasury_rates").glob("yield_curve_*.csv")):
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"])
        curves.append(frame)
    curve = pd.concat(curves, ignore_index=True).sort_values("Date")
    curve = curve[curve.Date.le(CUTOFF)]
    _write(curve, DATASETS / "treasury_rates" / "yield_curve.csv.gz")


def build_form_c() -> None:
    parts = []
    for path in sorted((RAW_DIR / "sec_form_c").glob("2025q*_cf.zip")):
        with zipfile.ZipFile(path) as zf:
            root = zf.namelist()[0].split("/")[0]
            sub = _read_tsv(
                zf, f"{root}/FORM_C_SUBMISSION.tsv",
                dtype={"ACCESSION_NUMBER":"string","CIK":"string","FILE_NUMBER":"string","PERIOD":"string"},
            )
            info = _read_tsv(zf, f"{root}/FORM_C_ISSUER_INFORMATION.tsv")
            disclosure = _read_tsv(zf, f"{root}/FORM_C_DISCLOSURE.tsv")
        parts.append(sub.merge(info, on="ACCESSION_NUMBER", how="left").merge(disclosure, on="ACCESSION_NUMBER", how="left"))
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    data["FILING_DATE"] = pd.to_datetime(data.FILING_DATE, format="%Y%m%d", errors="coerce")
    data["CIK"] = data.CIK.astype("string").str.zfill(10)
    data["PERIOD"] = data.PERIOD.astype("string").str.replace(r"\.0$", "", regex=True)
    data = data[data.FILING_DATE.le(CUTOFF)]
    keep = [
        "ACCESSION_NUMBER", "SUBMISSION_TYPE", "FILING_DATE", "CIK", "FILE_NUMBER",
        "PERIOD", "ISAMENDMENT", "PROGRESSUPDATE", "NATUREOFAMENDMENT", "NAMEOFISSUER",
        "OFFERINGAMOUNT", "MAXIMUMOFFERINGAMOUNT", "DEADLINEDATE", "CURRENTEMPLOYEES",
        "TOTALASSETMOSTRECENTFISCALYEAR", "CASHEQUIMOSTRECENTFISCALYEAR",
        "SHORTTERMDEBTMRECENTFISCALYEAR", "LONGTERMDEBTRECENTFISCALYEAR",
        "REVENUEMOSTRECENTFISCALYEAR", "NETINCOMEMOSTRECENTFISCALYEAR",
    ]
    _write(data[keep], DATASETS / "sec_form_c" / "filings_2025.csv.gz")


def build_form_d() -> None:
    parts = []
    for path in sorted((RAW_DIR / "sec_form_d").glob("2025q*_d.zip")):
        with zipfile.ZipFile(path) as zf:
            root = zf.namelist()[0].split("/")[0]
            sub = _read_tsv(zf, f"{root}/FORMDSUBMISSION.tsv")
            offering = _read_tsv(zf, f"{root}/OFFERING.tsv")
            issuers = _read_tsv(
                zf, f"{root}/ISSUERS.tsv",
                dtype={"ACCESSIONNUMBER":"string","CIK":"string"},
            )
        primary = issuers[issuers.IS_PRIMARYISSUER_FLAG.eq("YES")][["ACCESSIONNUMBER", "CIK", "ENTITYNAME"]].drop_duplicates("ACCESSIONNUMBER")
        parts.append(sub.merge(offering, on="ACCESSIONNUMBER").merge(primary, on="ACCESSIONNUMBER", how="left"))
    data = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSIONNUMBER", keep="last")
    data["FILING_DATE"] = pd.to_datetime(data.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    data["CIK"] = data.CIK.astype("string").str.zfill(10)
    data = data[data.FILING_DATE.le(CUTOFF)]
    keep = [
        "ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "ENTITYNAME",
        "INDUSTRYGROUPTYPE", "ISAMENDMENT", "PREVIOUSACCESSIONNUMBER", "SALE_DATE",
        "TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "TOTALREMAINING",
        "HASNONACCREDITEDINVESTORS", "NUMBERNONACCREDITEDINVESTORS",
        "TOTALNUMBERALREADYINVESTED", "SALESCOMM_DOLLARAMOUNT", "FINDERSFEE_DOLLARAMOUNT",
    ]
    _write(data[keep], DATASETS / "sec_form_d" / "filings_2025.csv.gz")


def build_bis_and_fdic() -> None:
    path = RAW_DIR / "bis" / "WS_CREDIT_GAP_csv_flat.zip"
    with zipfile.ZipFile(path) as zf:
        data = pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)
    data = data[data["TIME_PERIOD:Time period or range"].astype(str).le("2025-Q4")]
    _write(data, DATASETS / "bis" / "credit_gap.csv.gz")

    path = RAW_DIR / "bis" / "WS_DSR_csv_flat.zip"
    with zipfile.ZipFile(path) as zf:
        debt_service = pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)
    debt_service = debt_service.loc[
        debt_service["TIME_PERIOD:Time period or range"].astype(str).le("2025-Q4")
    ]
    _write(debt_service, DATASETS / "bis" / "debt_service_ratios.csv.gz")

    with zipfile.ZipFile(RAW_DIR / "bis" / "WS_EER_csv_flat.zip") as zf:
        eer = pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)
    eer = eer.loc[
        eer["FREQ:Frequency"].eq("M: Monthly")
        & eer["EER_BASKET:Basket"].eq("B: Broad (64 economies)")
        & eer["TIME_PERIOD:Time period or range"].isin(["2024-12", "2025-12"])
        & eer["OBS_VALUE:Observation Value"].notna(),
        [
            "EER_TYPE:Type", "EER_BASKET:Basket", "REF_AREA:Reference area",
            "TIME_PERIOD:Time period or range", "OBS_VALUE:Observation Value",
            "UNIT_MEASURE:Unit of measure",
        ],
    ].copy()
    _write(eer, DATASETS / "bis" / "broad_effective_exchange_rates.csv.gz")

    with zipfile.ZipFile(RAW_DIR / "bis" / "WS_GLI_csv_flat.zip") as zf:
        gli = pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)
    gli = gli.loc[
        gli["TIME_PERIOD:Time period or range"].eq("2025-Q2")
        & gli["CURR_DENOM:Currency of denomination"].isin(
            ["USD: US dollar", "EUR: Euro", "JPY: Yen"]
        )
        & gli["BORROWERS_CTY:Borrowers' country"].eq(
            "4T: Emerging market and developing economies"
        )
        & gli["BORROWERS_SECTOR:Borrowers' sector"].eq("N: Non-banks, total")
        & gli["L_POS_TYPE:Position type"].eq("I: Cross-border & Local in FCY")
        & gli["UNIT_MEASURE:Unit of measure"].isin(
            ["USD: US dollar", "EUR: Euro", "JPY: Yen"]
        ),
        [
            "CURR_DENOM:Currency of denomination", "BORROWERS_CTY:Borrowers' country",
            "BORROWERS_SECTOR:Borrowers' sector", "LENDERS_SECTOR:Lending sector",
            "L_POS_TYPE:Position type", "L_INSTR:Type of instruments",
            "UNIT_MEASURE:Unit of measure", "TIME_PERIOD:Time period or range",
            "OBS_VALUE:Observation Value", "TITLE:Title",
        ],
    ].copy()
    _write(gli, DATASETS / "bis" / "global_liquidity_emde_2025q2.csv.gz")

    with zipfile.ZipFile(RAW_DIR / "bis" / "WS_OTC_DERIV2_csv_flat.zip") as zf:
        otc = pd.read_csv(zf.open(zf.namelist()[0]), low_memory=False)
    otc = otc.loc[
        otc["TIME_PERIOD:Time period or range"].eq("2025-S1")
        & otc["DER_TYPE:Measure"].isin([
            "A: Outstanding - notional amounts",
            "D: Outstanding - gross market values",
        ])
        & otc["DER_INSTR:Instrument"].eq("A: Total (all instruments)")
        & otc["DER_SECTOR_CPY:Counterparty sector"].eq("A: Total (all counterparties)")
        & otc["DER_CPC:Counterparty country"].eq("5J: All countries")
        & otc["DER_SECTOR_UDL:Underlying risk sector"].eq("A: Total (all sectors)")
        & otc["DER_CURR_LEG1:Currency leg 1"].eq("TO1: Total (all currencies)")
        & otc["DER_CURR_LEG2:Currency leg 2"].eq("TO1: Total (all currencies)")
        & otc["DER_ISSUE_MAT:Maturity"].eq("A: Total (all maturities)")
        & otc["DER_RATING:Rating (outstanding) or settlement (FX turnover)"].eq(
            "A: Total (all ratings)"
        )
        & otc["DER_EX_METHOD:Execution method"].eq("3: Total (all methods)")
        & otc["DER_BASIS:Basis"].eq("C: Net - net"),
        [
            "DER_TYPE:Measure", "DER_INSTR:Instrument", "DER_RISK:Risk category",
            "DER_SECTOR_CPY:Counterparty sector", "DER_BASIS:Basis",
            "TIME_PERIOD:Time period or range", "OBS_VALUE:Observation Value",
            "UNIT_MEASURE:Unit of measure", "UNIT_MULT:Unit Multiplier",
        ],
    ].copy()
    _write(otc, DATASETS / "bis" / "otc_derivatives_2025s1.csv.gz")

    data = pd.read_csv(RAW_DIR / "fdic_bankfind" / "top12_banks_financials_2022_2025.csv")
    _write(data, DATASETS / "fdic_bankfind" / "top12_financials.csv.gz")


def build_bdc() -> None:
    path = RAW_DIR / "sec_bdc" / "2025q4_bdc.zip"
    accession = "0001287750-25-000046"
    columns = [
        "adsh", "ddate", "Investment, Identifier Axis", "Investment Interest Rate",
        "Investment Owned, Balance, Principal Amount", "Adjusted cost basis",
        "Initial fair value of Investment", "Investment, Interest Rate, Paid in Kind",
    ]
    with zipfile.ZipFile(path) as zf:
        data = _read_tsv(zf, "soi.tsv", usecols=columns)
    data = data.loc[
        data.adsh.eq(accession)
        & data.ddate.eq("2025-09-30")
        & data["Investment, Identifier Axis"].notna()
    ].copy()
    data = data.drop_duplicates("Investment, Identifier Axis")
    data.columns = [
        "accession", "report_date", "investment_id", "total_interest_rate",
        "principal_amount_usd", "cost_usd", "fair_value_usd", "pik_interest_rate",
    ]
    _write(data, DATASETS / "sec_bdc" / "ares_2025q3_schedule.csv.gz")

    history = []
    for archive, accession, report_date in [
        ("2025q3_bdc.zip", "0001287750-25-000038", "2025-06-30"),
        ("2025q4_bdc.zip", "0001287750-25-000046", "2025-09-30"),
    ]:
        with zipfile.ZipFile(RAW_DIR / "sec_bdc" / archive) as zf:
            frame = _read_tsv(zf, "soi.tsv", usecols=columns)
        frame = frame.loc[
            frame.adsh.eq(accession)
            & frame.ddate.eq(report_date)
            & frame["Investment, Identifier Axis"].notna()
        ].drop_duplicates("Investment, Identifier Axis").copy()
        frame.columns = [
            "accession", "report_date", "investment_id", "total_interest_rate",
            "principal_amount_usd", "cost_usd", "fair_value_usd", "pik_interest_rate",
        ]
        history.append(frame)
    _write(
        pd.concat(history, ignore_index=True),
        DATASETS / "sec_bdc" / "ares_2025q2_q3_schedule.csv.gz",
    )


def build_sba() -> None:
    path = RAW_DIR / "sba_7a_504" / "FOIA_7a_FY2020_Present_asof_260630.csv"
    keep = [
        "AsOfDate", "Program", "LocationID", "BankName", "BankFDICNumber",
        "GrossApproval", "SBAGuaranteedApproval", "ApprovalDate", "ApprovalFY",
        "FirstDisbursementDate", "ProcessingMethod", "InitialInterestRate",
        "FixedorVariableInterestInd", "TermInMonths", "NaicsCode", "NaicsDescription",
        "ProjectState", "BusinessType", "BusinessAge", "LoanStatus", "PaidInFullDate",
        "ChargeOffDate", "GrossChargeOffAmount", "RevolverStatus", "JobsSupported",
        "CollateralInd", "SoldSecMrktInd",
    ]
    data = pd.read_csv(
        path,
        usecols=keep,
        dtype={"LocationID": "string", "BankFDICNumber": "string"},
        low_memory=False,
    )
    data["LocationID"] = data.LocationID.str.strip()
    data["BankFDICNumber"] = data.BankFDICNumber.str.strip().str.zfill(5)
    data.insert(
        data.columns.get_loc("BankFDICNumber") + 1,
        "fdic_certificate",
        data.BankFDICNumber.str.lstrip("0").replace("", pd.NA),
    )
    data = data.loc[data.ApprovalFY.between(2020, 2025)].copy()
    _write(data, DATASETS / "sba_7a_504" / "seven_a_2020_2025.csv.gz")

    path = RAW_DIR / "sba_7a_504" / "FOIA_504_FY2010_Present_asof_260630.csv"
    keep = [
        "AsOfDate", "Program", "LocationID", "CDC_Name", "ThirdPartyLender_Name",
        "ThirdPartyDollars", "GrossApproval", "ApprovalDate", "ApprovalFY",
        "FirstDisbursementDate", "ProcessingMethod", "TermInMonths", "NaicsCode",
        "NaicsDescription", "ProjectState", "BusinessType", "BusinessAge",
        "LoanStatus", "PaidInFullDate", "ChargeOffDate", "GrossChargeOffAmount",
        "JobsSupported", "CollateralInd",
    ]
    data = pd.read_csv(path, usecols=keep, low_memory=False)
    data = data.loc[data.ApprovalFY.between(2020, 2025)].copy()
    _write(data, DATASETS / "sba_7a_504" / "five_o_four_2020_2025.csv.gz")


def build_fhfa() -> None:
    data = pd.read_csv(RAW_DIR / "fhfa_hpi" / "hpi_master.csv", low_memory=False)
    data = data.loc[data.yr.le(2025)].copy()
    _write(data, DATASETS / "fhfa_hpi" / "hpi_through_2025.csv.gz")


def build_treasury_tic() -> None:
    """Project the 2025 Major Foreign Holders table into a tidy monthly panel."""
    path = RAW_DIR / "treasury_tic" / "mfhhis01.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header_index = next(
        index
        for index, row in enumerate(rows)
        if row and row[0].strip() == "Country"
        and len(row) >= 13
        and all(cell.strip() == "2025" for cell in row[1:13])
    )
    months = [cell.strip() for cell in rows[header_index - 1][1:13]]
    if months != ["Dec", "Nov", "Oct", "Sep", "Aug", "Jul", "Jun", "May", "Apr", "Mar", "Feb", "Jan"]:
        raise ValueError(f"unexpected TIC month header: {months}")
    records = []
    for row in rows[header_index + 1 :]:
        country = row[0].strip() if row else ""
        if not country:
            if records:
                break
            continue
        if country.startswith("-"):
            continue
        if len(row) < 13:
            raise ValueError(f"short TIC country row: {row}")
        row_type = "aggregate" if country in {"All Other", "Grand Total"} else "country"
        for month, raw_value in zip(months, row[1:13], strict=True):
            records.append({
                "country_name": country,
                "row_type": row_type,
                "observation_month": f"2025-{pd.Timestamp(f'{month} 1, 2025').month:02d}",
                "treasury_holdings_usd_billions": pd.to_numeric(raw_value, errors="raise"),
                "source_table": "Major Foreign Holders of Treasury Securities",
            })
    result = pd.DataFrame(records).sort_values(
        ["observation_month", "row_type", "country_name"]
    ).reset_index(drop=True)
    _write(result, DATASETS / "treasury_tic" / "major_foreign_holders_2025.csv.gz")


def build_treasury_dts() -> None:
    """Normalize the two DTS tables used by the daily cash/debt bridge."""
    operating = pd.read_csv(
        RAW_DIR / "treasury_dts" / "operating_cash_balance_2024_through_2025.csv",
        low_memory=False,
    )
    # In this official API extract ``open_today_bal`` is the generic value field
    # for every account-type row; the nominal ``close_today_bal`` field is empty.
    operating = operating.rename(columns={
        "open_today_bal": "today_amount_million_usd",
        "open_month_bal": "month_to_date_amount_million_usd",
        "open_fiscal_year_bal": "fiscal_year_to_date_amount_million_usd",
    })[[
        "record_date", "account_type", "today_amount_million_usd",
        "month_to_date_amount_million_usd", "fiscal_year_to_date_amount_million_usd",
        "record_fiscal_year", "record_fiscal_quarter", "src_line_nbr",
    ]]
    _write(operating, DATASETS / "treasury_dts" / "operating_cash_balance.csv.gz")

    transactions = pd.read_csv(
        RAW_DIR / "treasury_dts" / "public_debt_transactions_2024_through_2025.csv",
        dtype={"transaction_today_amt": "string"}, low_memory=False,
    )
    transactions["transaction_today_amount_million_usd"] = pd.to_numeric(
        transactions.transaction_today_amt.str.replace("'", "", regex=False),
        errors="raise",
    )
    transactions = transactions[[
        "record_date", "transaction_type", "security_market", "security_type",
        "security_type_desc", "transaction_today_amount_million_usd",
        "record_fiscal_year", "record_fiscal_quarter", "src_line_nbr",
    ]]
    _write(transactions, DATASETS / "treasury_dts" / "public_debt_transactions.csv.gz")


def build_cftc() -> None:
    parts = []
    keep = [
        "Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code", "Open_Interest_All",
        "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All",
        "Lev_Money_Positions_Spread_All",
    ]
    for report_scope, filename in [
        ("futures_only", "fut_fin_txt_2025.zip"),
        ("futures_and_options_combined", "com_fin_txt_2025.zip"),
    ]:
        path = RAW_DIR / "cftc_cot" / filename
        with zipfile.ZipFile(path) as zf:
            frame = pd.read_csv(zf.open(zf.namelist()[0]), usecols=keep, low_memory=False)
        frame.insert(0, "report_scope", report_scope)
        parts.append(frame)
    _write(pd.concat(parts, ignore_index=True), DATASETS / "cftc_cot" / "financial_futures_2025.csv.gz")


def build_manifest() -> None:
    files = [
        "rtdsm/revisions.csv.gz", "fed_stress_tests/results_2025.csv.gz",
        "sec_nmfp/series_filings.csv.gz", "sec_nport/fund_reports.csv.gz",
        "hmda/lar_2024_top5.csv.gz", "sec_notes/apple_fy2025_numeric.csv.gz",
        "nyfed_rates/reference_rates.csv.gz", "treasury_auctions/auctions.csv.gz",
        "treasury_rates/yield_curve.csv.gz", "sec_form_c/filings_2025.csv.gz",
        "sec_form_d/filings_2025.csv.gz", "bis/credit_gap.csv.gz",
        "fdic_bankfind/top12_financials.csv.gz",
        "sec_nmfp/daily_shareholder_flows.csv.gz", "sec_nmfp/daily_liquidity.csv.gz",
        "sec_bdc/ares_2025q3_schedule.csv.gz", "sba_7a_504/seven_a_2020_2025.csv.gz",
        "cftc_cot/financial_futures_2025.csv.gz",
        "sec_nmfp/fedfund_2025_10_portfolio.csv.gz",
        "sec_nmfp/fedfund_2025_10_class_yields.csv.gz",
        "sec_nport/selected_holding_details.csv.gz",
        "sec_nport/interest_rate_risk_2025q4.csv.gz",
        "bis/debt_service_ratios.csv.gz",
        "bis/broad_effective_exchange_rates.csv.gz",
        "bis/global_liquidity_emde_2025q2.csv.gz",
        "bis/otc_derivatives_2025s1.csv.gz",
        "sec_bdc/ares_2025q2_q3_schedule.csv.gz",
        "sba_7a_504/five_o_four_2020_2025.csv.gz",
        "fhfa_hpi/hpi_through_2025.csv.gz",
        "treasury_tic/major_foreign_holders_2025.csv.gz",
        "treasury_dts/operating_cash_balance.csv.gz",
        "treasury_dts/public_debt_transactions.csv.gz",
    ]
    records = {}
    for relative in files:
        path = DATASETS / relative
        records[relative] = {
            "rows": len(pd.read_csv(path, low_memory=False)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "benchmark_cutoff": "2025-12-31",
        "builder": "python -m scripts.ingest.build.build_expansion_runtime",
        "privacy": (
            "HMDA runtime data excludes applicant identifiers and geographic fields; "
            "SBA runtime data excludes borrower names and addresses."
        ),
        "files": records,
    }
    report_path = PROJECT_ROOT / "metadata/build_reports/expansion_runtime_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {report_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    build_rtdsm()
    build_stress_tests()
    build_nmfp()
    build_nport()
    build_hmda()
    build_sec_notes()
    build_rates_and_auctions()
    build_form_c()
    build_form_d()
    build_bis_and_fdic()
    build_bdc()
    build_sba()
    build_cftc()
    build_fhfa()
    build_treasury_tic()
    build_treasury_dts()
    build_manifest()


if __name__ == "__main__":
    main()
