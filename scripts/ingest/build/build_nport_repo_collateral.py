"""Stage the complete repo schedule of a frozen N-PORT filing, preserving collateral grain."""

from __future__ import annotations

import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


ARCHIVE = "2025q3_nport.zip"
ACCESSION = "0000810893-25-002583"


def build() -> pd.DataFrame:
    source = RAW_DIR / "sec_nport" / ARCHIVE
    with zipfile.ZipFile(source) as archive:
        def read(name: str, columns: list[str]) -> pd.DataFrame:
            with archive.open(name) as handle:
                return pd.read_csv(handle, sep="\t", dtype="string", usecols=columns,
                                   keep_default_na=False, na_values=[""])

        info = read("FUND_REPORTED_INFO.tsv", ["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "NET_ASSETS"])
        submission = read("SUBMISSION.tsv", ["ACCESSION_NUMBER", "SUB_TYPE", "REPORT_DATE", "FILING_DATE"])
        agreement = read("REPURCHASE_AGREEMENT.tsv", ["HOLDING_ID", "TRANSACTION_TYPE", "REPURCHASE_RATE", "MATURITY_DATE"])
        collateral = read("REPURCHASE_COLLATERAL.tsv", ["HOLDING_ID", "REPURCHASE_COLLATERAL_ID",
            "PRINCIPAL_AMOUNT", "PRINCIPAL_CURRENCY_CODE", "COLLATERAL_AMOUNT",
            "COLLATERAL_CURRENCY_CODE", "INVESTMENT_CAT"])
        ids = set(agreement.HOLDING_ID)
        pieces = []
        with archive.open("FUND_REPORTED_HOLDING.tsv") as handle:
            for chunk in pd.read_csv(handle, sep="\t", dtype="string", usecols=[
                    "ACCESSION_NUMBER", "HOLDING_ID", "BALANCE", "UNIT", "CURRENCY_CODE",
                    "PAYOFF_PROFILE", "CURRENCY_VALUE"], chunksize=250_000,
                    keep_default_na=False, na_values=[""]):
                part = chunk.loc[chunk.ACCESSION_NUMBER.eq(ACCESSION) & chunk.HOLDING_ID.isin(ids)]
                if not part.empty:
                    pieces.append(part)
    if not pieces:
        raise ValueError("selected filing contains no repo holdings")
    holdings = pd.concat(pieces, ignore_index=True)
    if holdings.HOLDING_ID.duplicated().any():
        raise ValueError("duplicate holding ID")
    agreement = agreement.loc[agreement.HOLDING_ID.isin(holdings.HOLDING_ID)]
    collateral = collateral.loc[collateral.HOLDING_ID.isin(holdings.HOLDING_ID)]
    if len(agreement) != len(holdings) or agreement.HOLDING_ID.duplicated().any():
        raise ValueError("repo agreements not one-to-one with holdings")
    if set(collateral.HOLDING_ID) != set(holdings.HOLDING_ID):
        raise ValueError("a repo holding lacks collateral rows")
    if collateral.duplicated(["HOLDING_ID", "REPURCHASE_COLLATERAL_ID"]).any():
        raise ValueError("duplicate collateral key")
    info = info.loc[info.ACCESSION_NUMBER.eq(ACCESSION)]
    submission = submission.loc[submission.ACCESSION_NUMBER.eq(ACCESSION)]
    if len(info) != 1 or len(submission) != 1 or submission.iloc[0].SUB_TYPE != "NPORT-P":
        raise ValueError("selected original filing is not unique")
    result = holdings.merge(agreement, on="HOLDING_ID", validate="one_to_one")
    result = result.merge(collateral, on="HOLDING_ID", validate="one_to_many")
    result = result.merge(info, on="ACCESSION_NUMBER", validate="many_to_one")
    result = result.merge(submission, on="ACCESSION_NUMBER", validate="many_to_one")
    result = result.rename(columns={
        "ACCESSION_NUMBER": "accession", "HOLDING_ID": "holding_id",
        "BALANCE": "balance", "UNIT": "unit", "CURRENCY_CODE": "holding_currency",
        "CURRENCY_VALUE": "holding_value_usd", "PAYOFF_PROFILE": "payoff_profile",
        "TRANSACTION_TYPE": "transaction_type", "REPURCHASE_RATE": "repurchase_rate_pct",
        "MATURITY_DATE": "maturity_date", "REPURCHASE_COLLATERAL_ID": "collateral_id",
        "PRINCIPAL_AMOUNT": "collateral_security_principal",
        "PRINCIPAL_CURRENCY_CODE": "collateral_principal_currency",
        "COLLATERAL_AMOUNT": "collateral_value_usd",
        "COLLATERAL_CURRENCY_CODE": "collateral_currency", "INVESTMENT_CAT": "collateral_category",
        "SERIES_NAME": "series_name", "SERIES_ID": "series_id", "NET_ASSETS": "net_assets_usd",
        "SUB_TYPE": "form", "REPORT_DATE": "report_date", "FILING_DATE": "filing_date",
    })
    for column in ["balance", "holding_value_usd", "repurchase_rate_pct",
                   "collateral_security_principal", "collateral_value_usd", "net_assets_usd"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    for column in ["report_date", "filing_date", "maturity_date"]:
        result[column] = pd.to_datetime(result[column], format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
    if result.empty or result.collateral_value_usd.lt(0).any():
        raise ValueError("empty or negative collateral value")
    out = BUILD_DIR / "sec_nport" / "repo_collateral_2025q2.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(result)} collateral rows, {result.holding_id.nunique()} holdings")
    return result


if __name__ == "__main__":
    build()
