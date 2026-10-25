"""Stage source-native per-accession 13F information-table rows for one amendment chain."""

from __future__ import annotations

import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


CHAIN = {
    "0001214659-25-002354": "01dec2024-28feb2025_form13f.zip",
    "0001214659-25-002443": "01dec2024-28feb2025_form13f.zip",
    "0001214659-25-004240": "01mar2025-31may2025_form13f.zip",
    "0001214659-25-004241": "01mar2025-31may2025_form13f.zip",
}
COLUMNS = ["ACCESSION_NUMBER", "INFOTABLE_SK", "NAMEOFISSUER", "TITLEOFCLASS",
           "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL",
           "INVESTMENTDISCRETION"]


def build() -> pd.DataFrame:
    pieces = []
    for archive_name in sorted(set(CHAIN.values())):
        source = RAW_DIR / "sec_13f" / archive_name
        with zipfile.ZipFile(source) as archive:
            with archive.open("INFOTABLE.tsv") as handle:
                native = pd.read_csv(handle, sep="\t", dtype="string", usecols=COLUMNS)
        chosen = [accession for accession, name in CHAIN.items() if name == archive_name]
        frame = native.loc[native.ACCESSION_NUMBER.isin(chosen)].copy()
        frame["source_archive"] = archive_name
        pieces.append(frame)
    staged = pd.concat(pieces, ignore_index=True)
    if set(staged.ACCESSION_NUMBER) != set(CHAIN):
        raise ValueError("13F amendment chain incomplete")
    if staged.duplicated(["ACCESSION_NUMBER", "INFOTABLE_SK"]).any():
        raise ValueError("duplicate 13F information-table row")
    for column in COLUMNS:
        staged[column] = staged[column].str.strip()
    for column in ["VALUE", "SSHPRNAMT"]:
        staged[column] = pd.to_numeric(staged[column], errors="coerce")
    if staged[["VALUE", "SSHPRNAMT"]].isna().any().any():
        raise ValueError("missing value or quantity in selected 13F chain")
    if staged[["VALUE", "SSHPRNAMT"]].lt(0).any().any():
        raise ValueError("negative value or quantity in selected 13F chain")
    staged = staged.rename(columns={
        "ACCESSION_NUMBER": "accession",
        "INFOTABLE_SK": "source_row_key",
        "NAMEOFISSUER": "issuer_name",
        "TITLEOFCLASS": "title_of_class",
        "CUSIP": "cusip",
        "VALUE": "reported_value_usd",
        "SSHPRNAMT": "reported_quantity",
        "SSHPRNAMTTYPE": "quantity_type",
        "PUTCALL": "option_side",
        "INVESTMENTDISCRETION": "investment_discretion",
    })
    out = BUILD_DIR / "sec_13f" / "amendment_chain_2024q4.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(staged)} per-accession information-table rows")
    return staged


if __name__ == "__main__":
    build()
