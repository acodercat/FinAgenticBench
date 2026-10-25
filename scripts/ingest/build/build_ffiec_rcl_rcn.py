"""Stage FFIEC 031 consolidated 2025-Q3 RC-L and RC-N totals from bulk data.

Only FFIEC 031 filers with an RCFD value for every selected field enter these projections.
Domestic RCON values are not substituted into consolidated rows.
"""

from __future__ import annotations

import argparse
import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


ARCHIVE = "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
TABLES = {
    "bank_unused_commitments_rcl_2025q3.csv.gz": (
        "FFIEC CDR Call Schedule RCL 09302025(1 of 2).txt",
        {
            "RCFD3815": "unused_credit_card_lines_thousand_usd",
            "RCFDJ457": "unused_ci_commitments_thousand_usd",
        },
    ),
    "bank_past_due_rcn_2025q3.csv.gz": (
        "FFIEC CDR Call Schedule RCN 09302025.txt",
        {
            "RCFD1406": "loans_30_89_accruing_thousand_usd",
            "RCFD1407": "loans_90_plus_accruing_thousand_usd",
            "RCFD1403": "loans_nonaccrual_thousand_usd",
        },
    ),
}


def build() -> dict[str, pd.DataFrame]:
    source = RAW_DIR / "ffiec_call_reports" / ARCHIVE
    output = {}
    with zipfile.ZipFile(source) as archive:
        with archive.open("FFIEC CDR Call Bulk POR 09302025.txt") as handle:
            identities = pd.read_csv(handle, sep="\t", dtype="string",
                                     usecols=["IDRSSD", "FDIC Certificate Number", "Financial Institution Filing Type"])
        identities = identities.rename(columns={"FDIC Certificate Number": "fdic_certificate",
                                               "Financial Institution Filing Type": "filing_type"})
        identities["IDRSSD"] = identities["IDRSSD"].str.strip()
        identities["fdic_certificate"] = identities["fdic_certificate"].str.strip()
        if identities.IDRSSD.isna().any() or identities.IDRSSD.duplicated().any():
            raise ValueError("POR: missing or duplicate RSSD ID")
        for filename, (member, fields) in TABLES.items():
            with archive.open(member) as handle:
                raw = pd.read_csv(handle, sep="\t", skiprows=[1], dtype="string",
                                  usecols=["IDRSSD", *fields])
            values = raw[list(fields)].apply(pd.to_numeric, errors="coerce")
            complete = values.notna().all(axis=1)
            staged = pd.DataFrame({
                "bank_rssd_id": raw.loc[complete, "IDRSSD"].str.strip(),
                "report_date": "2025-09-30",
            })
            for source_column, target_column in fields.items():
                staged[target_column] = values.loc[complete, source_column]
            complete_count = len(staged)
            staged = staged.merge(identities, left_on="bank_rssd_id", right_on="IDRSSD",
                                  how="inner", validate="one_to_one").drop(columns="IDRSSD")
            if len(staged) != complete_count:
                raise ValueError(f"{member}: POR certificate join lost complete RC rows")
            if staged.bank_rssd_id.isna().any() or staged.bank_rssd_id.duplicated().any():
                raise ValueError(f"{member}: missing or duplicate RSSD ID")
            if staged.fdic_certificate.isna().any() or staged.fdic_certificate.eq("").any():
                raise ValueError(f"{member}: missing FDIC certificate")
            if not staged.filing_type.eq("031").all():
                raise ValueError(f"{member}: complete RCFD row outside FFIEC 031 population")
            if staged.empty or (staged[list(fields.values())] < 0).any().any():
                raise ValueError(f"{member}: no complete consolidated rows or negative amount")
            staged = staged.reset_index(drop=True)
            destination = BUILD_DIR / "ffiec_call_reports" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})
            output[filename] = staged
            print(f"{destination}: {len(staged)} consolidated bank rows")
    return output


def main() -> None:
    argparse.ArgumentParser().parse_args()
    build()


if __name__ == "__main__":
    main()
