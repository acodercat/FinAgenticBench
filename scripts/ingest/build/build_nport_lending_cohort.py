"""Stage every holding of one frozen 2025Q3 N-PORT securities-lending filing.

The accession is a declared benchmark cohort, not a discovered numerical answer.
The raw SEC archive remains the provenance; no lending flags are inferred from
missing rows or from a fund-level balance sheet.
"""

from __future__ import annotations

import argparse
import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


ACCESSION = "0001410368-25-031694"
ARCHIVE = "2025q4_nport.zip"
KEEP = ["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_CUSIP",
        "CURRENCY_VALUE", "ASSET_CAT", "IS_RESTRICTED_SECURITY", "FAIR_VALUE_LEVEL"]
LEND = ["HOLDING_ID", "IS_CASH_COLLATERAL", "CASH_COLLATERAL_AMOUNT",
        "IS_NON_CASH_COLLATERAL", "NON_CASH_COLLATERAL_VALUE", "IS_LOAN_BY_FUND", "LOAN_VALUE"]


def build() -> pd.DataFrame:
    source = RAW_DIR / "sec_nport" / ARCHIVE
    with zipfile.ZipFile(source) as archive:
        pieces = []
        with archive.open("FUND_REPORTED_HOLDING.tsv") as handle:
            for chunk in pd.read_csv(handle, sep="\t", dtype="string", usecols=KEEP,
                                     chunksize=250_000, keep_default_na=False, na_values=[""]):
                part = chunk.loc[chunk.ACCESSION_NUMBER.eq(ACCESSION)]
                if not part.empty:
                    pieces.append(part)
        holdings = pd.concat(pieces, ignore_index=True)
        with archive.open("SECURITIES_LENDING.tsv") as handle:
            lending = pd.read_csv(handle, sep="\t", dtype="string", usecols=LEND,
                                  keep_default_na=False, na_values=[""])
    if holdings.empty or holdings.HOLDING_ID.isna().any() or holdings.HOLDING_ID.duplicated().any():
        raise ValueError("selected accession has missing or duplicate holding ID")
    lending = lending.loc[lending.HOLDING_ID.isin(holdings.HOLDING_ID)]
    if lending.HOLDING_ID.duplicated().any() or len(lending) != len(holdings):
        raise ValueError("securities-lending schedule does not cover holdings one-to-one")
    result = holdings.merge(lending, on="HOLDING_ID", validate="one_to_one")
    result = result.rename(columns={
        "ACCESSION_NUMBER": "accession", "HOLDING_ID": "holding_id",
        "ISSUER_NAME": "issuer_name", "ISSUER_CUSIP": "issuer_cusip",
        "CURRENCY_VALUE": "currency_value_usd", "ASSET_CAT": "asset_category",
        "IS_RESTRICTED_SECURITY": "is_restricted_security",
        "FAIR_VALUE_LEVEL": "fair_value_level", "IS_CASH_COLLATERAL": "is_cash_collateral_reinvestment",
        "CASH_COLLATERAL_AMOUNT": "cash_collateral_reinvestment_value_usd",
        "IS_NON_CASH_COLLATERAL": "is_non_cash_collateral_asset",
        "NON_CASH_COLLATERAL_VALUE": "non_cash_collateral_asset_value_usd",
        "IS_LOAN_BY_FUND": "is_loan_by_fund", "LOAN_VALUE": "loan_value_usd",
    })
    for column in ["currency_value_usd", "cash_collateral_reinvestment_value_usd",
                   "non_cash_collateral_asset_value_usd", "loan_value_usd"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    out = BUILD_DIR / "sec_nport" / "lending_cohort_2025q3.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(result)} holding rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
