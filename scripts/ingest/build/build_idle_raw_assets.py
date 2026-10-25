"""Build governed tables from raw archives that were downloaded but never built.

The 2026-08-26 data-quality audit (`docs/data_quality_audit.md`) found five
sources whose full raw population sat idle while the runtime exposed either
nothing or a one-month slice:

- `sec_midas_security_exchange`: eight quarterly MIDAS archives (2024q1–2025q4,
  ~185M exchange-level rows). Projected to one row per security-date, summing
  the count and volume columns across exchanges; the four SEC ranks are
  security-level and carried through unchanged. Replaces the April-2025 slice.
- `sec_13f`: 29 quarterly Form 13F data sets, 2019q1–2025q4. Two tables — one
  row per filing (submission, cover page and summary page joined) and one row
  per report period × CUSIP aggregated over original 13F-HR holdings tables.
  SEC changed the VALUE unit from thousands of dollars to dollars for filings
  made on or after 2023-01-03; values are rescaled to dollars per accession
  before aggregation (`*_usd_total`), and the period's mix of reporting
  conventions is recorded in `value_reporting_convention`.
- `fed_z1_vintages`: seven Z.1 release vintages (2024-03 … 2025-09). Every
  table CSV is unpivoted to (vintage, table, series, date, value); the
  `all_sectors_*` convenience files duplicate the table files and are skipped.
  Observations before 2015 are dropped to keep the table inside one runtime.
- `sec_ftd`: 48 semi-monthly fails-to-deliver files, 2024–2025, concatenated.
- `sec_notes`: Financial Statement and Notes `num.tsv`, restricted to the
  deep-tier universe (`sec/companies.csv`) and joined to `sub.tsv`. Unlike
  CompanyFacts this keeps dimensional facts (`dimh`, `coreg`). Archives are
  quarterly through 2025q2 and monthly from 2025-07; `numeric_facts` keeps
  every fact of the two most recent years, `dimensional_facts` keeps the
  single-axis dimensional facts of every 10-K/10-Q for 2019-2025, and
  `dimensions` decodes the `dimh` hashes.

Two toy projections are rebuilt from raw populations the same way:
`bls_qcew` (state x county x NAICS-sector annual averages, 2019–2025, from the
annual singlefiles) and `hmda` (the complete public 2024 LAR fetched state by
state through the data-browser API).

Nothing here deletes a raw file. Tables are written next to the existing
governed tables and registered in `core/dataset_layers.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "idle_raw_assets_build.json"
FREEZE_DATE = "2025-12-31"
UNIVERSE = DATASETS / "sec" / "companies.csv"


def _universe_ciks() -> set[str]:
    companies = pd.read_csv(UNIVERSE, dtype={"cik": str})
    return set(companies["cik"].astype(str).str.lstrip("0"))


def _write(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)
    print(f"  {path.relative_to(DATASETS)}: {len(frame):,} rows", flush=True)
    return {"file": str(path.relative_to(DATASETS)), "rows": len(frame), "columns": list(frame.columns)}


# --------------------------------------------------------------------------- MIDAS

MIDAS_COLUMNS = {
    "Date": "date", "Security": "security_type", "Ticker": "ticker",
    "McapRank": "mcap_rank", "TurnRank": "turnover_rank",
    "VolatilityRank": "volatility_rank", "PriceRank": "price_rank",
    "Cancels": "cancels", "Trades": "trades", "LitTrades": "lit_trades",
    "OddLots": "odd_lots", "Hidden": "hidden", "TradesForHidden": "trades_for_hidden",
    "OrderVol('000)": "order_volume_thousands", "TradeVol('000)": "trade_volume_thousands",
    "LitVol('000)": "lit_volume_thousands", "OddLotVol('000)": "odd_lot_volume_thousands",
    "HiddenVol('000)": "hidden_volume_thousands",
    "TradeVolForHidden('000)": "trade_volume_for_hidden_thousands",
}
MIDAS_SUMS = [
    "cancels", "trades", "lit_trades", "odd_lots", "hidden", "trades_for_hidden",
    "order_volume_thousands", "trade_volume_thousands", "lit_volume_thousands",
    "odd_lot_volume_thousands", "hidden_volume_thousands", "trade_volume_for_hidden_thousands",
]
MIDAS_RANKS = ["mcap_rank", "turnover_rank", "volatility_rank", "price_rank"]


def build_midas() -> dict:
    raw = RAW_DIR / "sec_midas_security_exchange"
    partials, exchange_rows = [], 0
    for archive in sorted(raw.glob("individual_security_exchange_*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in sorted(zf.namelist()):
                if not member.lower().endswith(".csv"):
                    continue
                with zf.open(member) as handle:
                    for chunk in pd.read_csv(handle, chunksize=4_000_000, low_memory=False):
                        chunk = chunk.rename(columns=MIDAS_COLUMNS)
                        exchange_rows += len(chunk)
                        grouped = chunk.groupby(["date", "security_type", "ticker"], sort=False)
                        part = grouped[MIDAS_SUMS].sum()
                        part["exchange_count"] = grouped["date"].size()
                        for rank in MIDAS_RANKS:
                            part[rank] = grouped[rank].first()
                        partials.append(part.reset_index())
                print(f"  {archive.name}/{member}: {exchange_rows:,} exchange rows so far", flush=True)
    frame = pd.concat(partials, ignore_index=True)
    grouped = frame.groupby(["date", "security_type", "ticker"], sort=True)
    out = grouped[MIDAS_SUMS + ["exchange_count"]].sum()
    for rank in MIDAS_RANKS:
        out[rank] = grouped[rank].first()
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    out = out.loc[out["date"].le(FREEZE_DATE)]
    columns = ["date", "security_type", "ticker", *MIDAS_RANKS, "exchange_count", *MIDAS_SUMS]
    out = out[columns].sort_values(["date", "ticker", "security_type"]).reset_index(drop=True)
    info = _write(out, DATASETS / "sec_midas_security_exchange" / "daily_security.csv.gz")
    info["exchange_level_rows_read"] = exchange_rows
    return {"daily_security": info}


# --------------------------------------------------------------------------- 13F

def _13f_member(zf: zipfile.ZipFile, name: str) -> str:
    for member in zf.namelist():
        if member.upper().endswith(f"/{name}.TSV") or member.upper() == f"{name}.TSV":
            return member
    raise KeyError(name)


def _read_tsv(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(_13f_member(zf, name)) as handle:
        return pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def _infer_value_scale(table: pd.DataFrame, filed_of: dict) -> dict:
    """Scale factor (1 or 1000) that brings each accession's VALUE to dollars.

    SEC required VALUE in thousands of dollars until filings made on or after
    2023-01-03, but the data sets show filers on both conventions on either
    side of that date, so the filing date alone misclassifies. The unit is
    inferred per accession from its holdings: for plain share rows the implied
    price VALUE / shares is a few dollars to a few hundred dollars when VALUE
    is in dollars and a thousandth of that when VALUE is in thousands. An
    accession whose median implied price is below one dollar is treated as a
    thousands-of-dollars filing. Accessions with fewer than three comparable
    rows fall back to the filing-date rule.
    """
    rows = table.loc[table["SSHPRNAMTTYPE"].eq("SH") & table["PUTCALL"].eq("")
                     & table["value_reported"].gt(0) & table["shares"].gt(0)]
    implied = (rows["value_reported"] / rows["shares"]).groupby(rows["ACCESSION_NUMBER"]).agg(["median", "size"])
    scale = {}
    for accession in table["ACCESSION_NUMBER"].unique():
        if accession in implied.index and implied.loc[accession, "size"] >= 3:
            scale[accession] = 1000.0 if implied.loc[accession, "median"] < 1.0 else 1.0
        else:
            scale[accession] = 1.0 if filed_of.get(accession, "") >= "2023-01-03" else 1000.0
    return scale


def build_13f() -> dict:
    raw = RAW_DIR / "sec_13f"
    filings, partials, unit_of = [], [], {}
    for archive in sorted(raw.glob("*_form13f.zip")):
        with zipfile.ZipFile(archive) as zf:
            submission = _read_tsv(zf, "SUBMISSION")
            cover = _read_tsv(zf, "COVERPAGE")[[
                "ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTTYPE",
                "FILINGMANAGER_NAME", "FILINGMANAGER_STATEORCOUNTRY", "REPORTTYPE",
            ]]
            summary = _read_tsv(zf, "SUMMARYPAGE")[[
                "ACCESSION_NUMBER", "OTHERINCLUDEDMANAGERSCOUNT", "TABLEENTRYTOTAL",
                "TABLEVALUETOTAL", "ISCONFIDENTIALOMITTED",
            ]]
            frame = submission.merge(cover, on="ACCESSION_NUMBER", how="left").merge(
                summary, on="ACCESSION_NUMBER", how="left")
            # 13F data-set dates read 31-MAR-2025; store ISO dates throughout.
            for column in ("FILING_DATE", "PERIODOFREPORT", "REPORTCALENDARORQUARTER"):
                frame[column] = pd.to_datetime(frame[column].str.title(), format="%d-%b-%Y", errors="coerce").dt.strftime("%Y-%m-%d")
            frame["source_archive"] = archive.name
            filings.append(frame)
            original = frame.loc[frame["SUBMISSIONTYPE"].eq("13F-HR")]
            period_of = dict(zip(original["ACCESSION_NUMBER"], original["PERIODOFREPORT"]))
            filed_of = dict(zip(original["ACCESSION_NUMBER"], original["FILING_DATE"]))
            with zf.open(_13f_member(zf, "INFOTABLE")) as handle:
                table = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False,
                                    usecols=["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE",
                                             "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "VOTING_AUTH_SOLE"])
            table = table.loc[table["ACCESSION_NUMBER"].isin(set(period_of))].copy()
            table["period_of_report"] = table["ACCESSION_NUMBER"].map(period_of)
            table["value_reported"] = pd.to_numeric(table["VALUE"], errors="coerce")
            table["shares"] = pd.to_numeric(table["SSHPRNAMT"], errors="coerce")
            scale = _infer_value_scale(table, filed_of)
            unit_of.update({acc: ("thousands_of_dollars" if s == 1000.0 else "dollars") for acc, s in scale.items()})
            table["value"] = table["value_reported"] * table["ACCESSION_NUMBER"].map(scale)
            is_share = table["SSHPRNAMTTYPE"].eq("SH")
            no_option = table["PUTCALL"].eq("")
            table["share_value"] = table["value"].where(is_share & no_option)
            table["share_amount"] = table["shares"].where(is_share & no_option)
            table["put_value"] = table["value"].where(table["PUTCALL"].eq("Put"))
            table["call_value"] = table["value"].where(table["PUTCALL"].eq("Call"))
            table["sole_voting"] = pd.to_numeric(table["VOTING_AUTH_SOLE"], errors="coerce")
            grouped = table.groupby(["period_of_report", "CUSIP"], sort=False)
            part = grouped.agg(
                issuer_name=("NAMEOFISSUER", "first"), title_of_class=("TITLEOFCLASS", "first"),
                holding_rows=("ACCESSION_NUMBER", "size"),
                value_usd_total=("value", "sum"), share_value_usd_total=("share_value", "sum"),
                share_amount_total=("share_amount", "sum"), put_value_usd_total=("put_value", "sum"),
                call_value_usd_total=("call_value", "sum"), sole_voting_shares_total=("sole_voting", "sum"),
            )
            part["_accessions"] = grouped["ACCESSION_NUMBER"].agg(lambda s: set(s))
            partials.append(part.reset_index())
            del table
        print(f"  {archive.name}: {len(frame):,} filings", flush=True)

    filings_frame = pd.concat(filings, ignore_index=True)
    filings_frame.columns = [c.lower() for c in filings_frame.columns]
    filings_frame = filings_frame.loc[filings_frame["filing_date"].le(FREEZE_DATE)]
    filings_frame = filings_frame.drop_duplicates("accession_number").sort_values("accession_number")
    # Unit per accession as inferred from implied prices (see _infer_value_scale);
    # notice-type filings (13F-NT) carry no holdings and are left blank.
    filings_frame["value_reporting_convention"] = filings_frame["accession_number"].map(unit_of)

    holdings = pd.concat(partials, ignore_index=True)
    grouped = holdings.groupby(["period_of_report", "CUSIP"], sort=True)
    out = grouped.agg(
        issuer_name=("issuer_name", "first"), title_of_class=("title_of_class", "first"),
        holding_rows=("holding_rows", "sum"), value_usd_total=("value_usd_total", "sum"),
        share_value_usd_total=("share_value_usd_total", "sum"),
        share_amount_total=("share_amount_total", "sum"),
        put_value_usd_total=("put_value_usd_total", "sum"),
        call_value_usd_total=("call_value_usd_total", "sum"),
        sole_voting_shares_total=("sole_voting_shares_total", "sum"),
    )
    out["filer_count"] = grouped["_accessions"].agg(lambda sets: len(set().union(*sets)))
    out = out.reset_index().rename(columns={"CUSIP": "cusip"})
    # Filers occasionally mistype the report period (values back to 1987 appear);
    # the data sets cover periods from 2018-12-31, so anything outside is a typo.
    out = out.loc[out["period_of_report"].between("2018-12-31", FREEZE_DATE)]
    period_convention = (filings_frame.loc[filings_frame["submissiontype"].eq("13F-HR") & filings_frame["value_reporting_convention"].notna()]
                         .groupby("periodofreport")["value_reporting_convention"]
                         .agg(lambda s: "mixed" if s.nunique() > 1 else s.iloc[0]))
    # Recorded for provenance: "mixed" periods had both thousands- and dollar-unit
    # filings; the *_usd_total columns are already normalised to dollars.
    out["value_reporting_convention"] = out["period_of_report"].map(period_convention)
    return {
        "filings": _write(filings_frame, DATASETS / "sec_13f" / "filings.csv.gz"),
        "holdings_by_cusip": _write(out, DATASETS / "sec_13f" / "holdings_by_cusip.csv.gz"),
    }


# --------------------------------------------------------------------------- Z.1

Z1_MIN_DATE = "2015-01-01"


def _z1_date(token: str) -> str:
    match = re.fullmatch(r"(\d{4}):Q([1-4])", token)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        return f"{year}-{quarter * 3:02d}-{'31' if quarter in (1, 4) else '30'}"
    if re.fullmatch(r"\d{4}", token):
        return f"{token}-12-31"
    raise ValueError(token)


def build_z1() -> dict:
    raw = RAW_DIR / "fed_z1_vintages"
    parts = []
    for archive in sorted(raw.glob("z1_csv_files_*.zip")):
        vintage = archive.stem.rsplit("_", 1)[-1]
        vintage = f"{vintage[:4]}-{vintage[4:6]}-{vintage[6:]}"
        with zipfile.ZipFile(archive) as zf:
            for member in sorted(zf.namelist()):
                name = Path(member).name
                if not member.startswith("csv/") or not name.endswith(".csv") or name.startswith("all_sectors"):
                    continue
                with zf.open(member) as handle:
                    wide = pd.read_csv(handle, dtype=str, keep_default_na=False)
                long = wide.melt(id_vars="date", var_name="series", value_name="value")
                long = long.loc[~long["value"].isin(("ND", ""))]
                long["date"] = long["date"].map(_z1_date)
                long = long.loc[long["date"].ge(Z1_MIN_DATE)]
                long["vintage"] = vintage
                long["table"] = name[:-4]
                long["frequency"] = long["series"].str.rsplit(".", n=1).str[-1]
                parts.append(long)
        print(f"  {archive.name}: {sum(len(p) for p in parts):,} observations so far", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[["vintage", "table", "series", "frequency", "date", "value"]]
    frame = frame.drop_duplicates(["vintage", "table", "series", "date"]).sort_values(
        ["vintage", "table", "series", "date"]).reset_index(drop=True)
    return {"series_observations": _write(frame, DATASETS / "fed_z1_vintages" / "series_observations.csv.gz")}


# --------------------------------------------------------------------------- FTD

def build_ftd() -> dict:
    raw = RAW_DIR / "sec_ftd"
    parts = []
    for archive in sorted(raw.glob("cnsfails*.zip")):
        with zipfile.ZipFile(archive) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as handle:
                text = handle.read().decode("latin-1")
        frame = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
        frame = frame.loc[frame["SETTLEMENT DATE"].str.fullmatch(r"\d{8}")]
        frame["source_file"] = archive.name
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True).rename(columns={
        "SETTLEMENT DATE": "settlement_date", "CUSIP": "cusip", "SYMBOL": "symbol",
        "QUANTITY (FAILS)": "quantity_fails", "DESCRIPTION": "description", "PRICE": "price",
    })
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    frame["quantity_fails"] = pd.to_numeric(frame["quantity_fails"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.loc[frame["settlement_date"].le(FREEZE_DATE)]
    frame = frame.drop_duplicates(["settlement_date", "cusip", "symbol"]).sort_values(
        ["settlement_date", "symbol", "cusip"]).reset_index(drop=True)
    return {"fails_to_deliver": _write(frame, DATASETS / "sec_ftd" / "fails_to_deliver.csv.gz")}


# --------------------------------------------------------------------------- FS&Notes

NUM_KEY = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx", "coreg"]
SUB_COLUMNS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "prevrpt", "detail"]


def build_notes() -> dict:
    raw = RAW_DIR / "sec_notes"
    universe = _universe_ciks()
    subs, nums = [], []
    for archive in sorted(raw.glob("*_notes.zip")):
        with zipfile.ZipFile(archive) as zf:
            with zf.open("sub.tsv") as handle:
                sub = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, usecols=SUB_COLUMNS)
            sub = sub.loc[sub["cik"].str.lstrip("0").isin(universe) & sub["filed"].le(FREEZE_DATE.replace("-", ""))]
            sub["source_archive"] = archive.name
            subs.append(sub)
            keep = set(sub["adsh"])
            with zf.open("num.tsv") as handle:
                for chunk in pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, chunksize=2_000_000):
                    chunk = chunk.loc[chunk["adsh"].isin(keep)]
                    if len(chunk):
                        nums.append(chunk)
        print(f"  {archive.name}: {len(sub):,} submissions, {sum(len(n) for n in nums):,} facts so far", flush=True)
    submissions = pd.concat(subs, ignore_index=True).drop_duplicates("adsh").sort_values("adsh")
    facts = pd.concat(nums, ignore_index=True).drop_duplicates(NUM_KEY)
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.merge(submissions[["adsh", "cik", "name", "form", "filed"]], on="adsh", how="left")
    facts = facts.sort_values(["adsh", "tag", "ddate", "qtrs", "dimh"]).reset_index(drop=True)
    return {
        "submissions": _write(submissions, DATASETS / "sec_notes" / "submissions.csv.gz"),
        "numeric_facts": _write(facts, DATASETS / "sec_notes" / "numeric_facts.csv.gz"),
    }


# --------------------------------------------------------------------------- QCEW

QCEW_AGGREGATION_LEVELS = {
    "50": "state_total", "51": "state_by_ownership", "54": "state_naics_sector",
    "70": "county_total", "71": "county_by_ownership", "74": "county_naics_sector",
}
QCEW_OWNERSHIP = {"0": "total_covered", "5": "private"}
QCEW_COLUMNS = {
    "area_fips": "area_fips", "own_code": "own_code", "industry_code": "industry_code",
    "agglvl_code": "agglvl_code", "year": "year", "disclosure_code": "disclosure_code",
    "annual_avg_estabs": "annual_avg_establishments", "annual_avg_emplvl": "annual_avg_employment",
    "total_annual_wages": "total_annual_wages_usd", "annual_avg_wkly_wage": "annual_avg_weekly_wage_usd",
    "avg_annual_pay": "average_annual_pay_usd", "lq_annual_avg_emplvl": "employment_location_quotient",
    "oty_annual_avg_emplvl_pct_chg": "over_year_employment_change_pct",
    "oty_avg_annual_pay_pct_chg": "over_year_average_annual_pay_change_pct",
    "oty_disclosure_code": "over_year_disclosure_code",
}


def build_qcew() -> dict:
    """County and state annual averages by NAICS sector, 2019-2025.

    The 2024/2025 build kept 212 state x year totals out of the 3.5M-row annual
    singlefiles. This projection keeps the state and county aggregation levels
    (totals, by ownership, and NAICS sector by ownership) for the total-covered
    and private ownership codes -- the grain finance work on local labour
    markets actually uses -- and every year on hand.
    """
    raw = RAW_DIR / "bls_qcew"
    parts = []
    for archive in sorted(raw.glob("*_annual_singlefile.zip")):
        with zipfile.ZipFile(archive) as zf:
            member = next(m for m in zf.namelist() if m.endswith(".csv"))
            with zf.open(member) as handle:
                for chunk in pd.read_csv(handle, dtype=str, keep_default_na=False,
                                         usecols=list(QCEW_COLUMNS), chunksize=1_000_000):
                    chunk = chunk.loc[chunk["agglvl_code"].isin(QCEW_AGGREGATION_LEVELS)
                                      & chunk["own_code"].isin(QCEW_OWNERSHIP)]
                    if len(chunk):
                        chunk = chunk.rename(columns=QCEW_COLUMNS).copy()
                        chunk["source_archive"] = archive.name
                        parts.append(chunk)
        print(f"  {archive.name}: {sum(len(p) for p in parts):,} rows so far", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame["aggregation_level"] = frame["agglvl_code"].map(QCEW_AGGREGATION_LEVELS)
    frame["ownership_scope"] = frame["own_code"].map(QCEW_OWNERSHIP)
    frame["state_fips"] = frame["area_fips"].str.slice(0, 2)
    for column in QCEW_COLUMNS.values():
        if column.endswith(("_usd", "_pct", "_establishments", "_employment", "_quotient")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["year"] = frame["year"].astype(int)
    frame = frame.loc[frame["year"].between(2019, 2025)]
    columns = ["area_fips", "state_fips", "aggregation_level", "ownership_scope", "industry_code", "year",
               *[c for c in QCEW_COLUMNS.values() if c not in {"area_fips", "own_code", "industry_code", "agglvl_code", "year"}],
               "source_archive"]
    frame = frame[columns].drop_duplicates(["area_fips", "aggregation_level", "ownership_scope", "industry_code", "year"])
    frame = frame.sort_values(["year", "area_fips", "aggregation_level", "ownership_scope", "industry_code"]).reset_index(drop=True)
    return {"area_industry_annual": _write(frame, DATASETS / "bls_qcew" / "area_industry_annual.csv.gz")}


# --------------------------------------------------------------------------- HMDA

HMDA_COLUMNS = [
    "activity_year", "lei", "state_code", "county_code", "derived_msa-md",
    "derived_loan_product_type", "derived_dwelling_category", "derived_ethnicity",
    "derived_race", "derived_sex", "action_taken", "purchaser_type", "loan_type",
    "loan_purpose", "lien_status", "business_or_commercial_purpose", "loan_amount",
    "loan_to_value_ratio", "interest_rate", "rate_spread", "property_value", "income",
    "debt_to_income_ratio", "construction_method", "occupancy_type", "total_units",
    "applicant_age", "denial_reason-1", "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
]
HMDA_NUMERIC = ["loan_amount", "loan_to_value_ratio", "interest_rate", "rate_spread",
                "property_value", "income", "tract_minority_population_percent",
                "tract_to_msa_income_percentage"]


def build_hmda() -> dict:
    """The complete public 2024 LAR, every state, one row per record.

    The five-lender cohort excluded geography; the full population keeps state,
    county and MSA so lending can be placed against the deposit, disaster and
    labour-market tables. Columns are the public-LAR fields finance work uses;
    the applicant/co-applicant detail fields (race-2..5, AUS, ages observed)
    stay in the raw files. `institution_name` comes from the HMDA filer list.
    """
    raw = RAW_DIR / "hmda"
    filers = json.loads((raw / "filers_2024.json").read_text())["institutions"]
    names = {f["lei"]: f["name"] for f in filers}
    files = sorted(raw.glob("lar_2024_??.csv"))
    output = DATASETS / "hmda" / "lar_2024.csv.gz"
    if output.exists():
        output.unlink()
    # ~13M records: stream each state's chunks straight into the gzip file
    # instead of holding the whole register as Python strings.
    rows, columns, first = 0, None, True
    for path in files:
        for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, usecols=HMDA_COLUMNS,
                                 chunksize=500_000):
            chunk = chunk[HMDA_COLUMNS]
            chunk.columns = [c.replace("-", "_") for c in chunk.columns]
            chunk.insert(2, "institution_name", chunk["lei"].map(names))
            for column in HMDA_NUMERIC:
                chunk[column] = pd.to_numeric(chunk[column].replace({"NA": "", "Exempt": ""}), errors="coerce")
            chunk = chunk.loc[chunk["activity_year"].eq("2024")]
            chunk.to_csv(output, index=False, header=first, mode="a", compression="gzip")
            rows += len(chunk)
            columns, first = list(chunk.columns), False
        print(f"  {path.name}: {rows:,} rows so far", flush=True)
    print(f"  {output.relative_to(DATASETS)}: {rows:,} rows", flush=True)
    return {"lar_2024": {"file": str(output.relative_to(DATASETS)), "rows": rows, "columns": columns,
                         "state_files": [p.name for p in files]}}


# --------------------------------------------------------------------------- FS&Notes, dimensional

NOTES_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
DIM_COLUMNS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx", "value", "coreg", "dcml"]


def build_notes_dimensional() -> dict:
    """Single-axis dimensional facts from every FS&Notes archive, 2019-2025.

    `sec_notes.numeric_facts` keeps every fact (dimensional or not) for the two
    most recent years; this projection keeps the facts that CompanyFacts does
    not carry at all — values broken down along one XBRL axis (business
    segment, geography, product, class of stock, ...) — for the whole window
    and every 10-K/10-Q of the 500-filer universe. Multi-axis combinations
    (about half of all dimensional rows) are excluded to keep the table inside
    one runtime; the decode table `sec_notes.dimensions` resolves each `dimh`
    hash to its axis and member.
    """
    raw = RAW_DIR / "sec_notes"
    universe = _universe_ciks()
    subs, facts, dims = [], [], []
    for archive in sorted(raw.glob("*_notes.zip")):
        with zipfile.ZipFile(archive) as zf:
            with zf.open("sub.tsv") as handle:
                sub = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, usecols=SUB_COLUMNS)
            sub = sub.loc[sub["cik"].str.lstrip("0").isin(universe) & sub["form"].isin(NOTES_FORMS)
                          & sub["filed"].le(FREEZE_DATE.replace("-", ""))]
            keep = set(sub["adsh"])
            seen = set()
            with zf.open("num.tsv") as handle:
                for chunk in pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, usecols=DIM_COLUMNS, chunksize=2_000_000):
                    chunk = chunk.loc[chunk["adsh"].isin(keep) & chunk["dimh"].ne("0x00000000")]
                    if len(chunk):
                        chunk = chunk.assign(source_archive=archive.name)
                        facts.append(chunk); seen.update(chunk["dimh"].unique())
            with zf.open("dim.tsv") as handle:
                dim = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False)
            dims.append(dim.loc[dim["dimhash"].isin(seen)])
        sub["source_archive"] = archive.name
        subs.append(sub)
        print(f"  {archive.name}: {len(sub):,} filings, {sum(len(f) for f in facts):,} dimensional rows so far", flush=True)
    dimensions = pd.concat(dims, ignore_index=True).drop_duplicates("dimhash")
    dimensions["axis_count"] = dimensions["segments"].str.count("=")
    single = dimensions.loc[dimensions["axis_count"].eq(1)].copy()
    split = single["segments"].str.rstrip(";").str.split("=", n=1, expand=True)
    single["axis"], single["member"] = split[0], split[1]
    frame = pd.concat(facts, ignore_index=True).drop_duplicates(NUM_KEY)
    frame = frame.loc[frame["dimh"].isin(set(single["dimhash"]))].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    submissions = pd.concat(subs, ignore_index=True).drop_duplicates("adsh")
    frame = frame.merge(submissions[["adsh", "cik", "name", "form", "filed"]], on="adsh", how="left")
    frame = frame.merge(single[["dimhash", "axis", "member"]], left_on="dimh", right_on="dimhash", how="left").drop(columns="dimhash")
    frame = frame[["adsh", "cik", "name", "form", "filed", "tag", "version", "ddate", "qtrs", "uom", "dimh", "axis", "member",
                   "iprx", "value", "coreg", "dcml", "source_archive"]]
    frame = frame.sort_values(["adsh", "tag", "ddate", "qtrs", "dimh"]).reset_index(drop=True)
    dimensions = dimensions.rename(columns={"dimhash": "dimh", "segt": "segment_type"})
    dimensions = dimensions.merge(single[["dimhash", "axis", "member"]], left_on="dimh", right_on="dimhash", how="left").drop(columns="dimhash")
    dimensions = dimensions[["dimh", "axis_count", "axis", "member", "segments", "segment_type"]].sort_values("dimh").reset_index(drop=True)
    return {
        "dimensional_facts": _write(frame, DATASETS / "sec_notes" / "dimensional_facts.csv.gz"),
        "dimensions": _write(dimensions, DATASETS / "sec_notes" / "dimensions.csv.gz"),
    }


BUILDERS = {"midas": build_midas, "13f": build_13f, "z1": build_z1, "ftd": build_ftd,
            "notes": build_notes, "notes_dimensional": build_notes_dimensional, "qcew": build_qcew, "hmda": build_hmda}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BUILDERS))
    args = parser.parse_args()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name in args.only or list(BUILDERS):
        print(f"== {name}", flush=True)
        report[name] = {"built_at": datetime.now(timezone.utc).isoformat(), "freeze_date": FREEZE_DATE,
                        "tables": BUILDERS[name]()}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
