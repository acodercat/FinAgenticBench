"""Stage FFIEC 031 consolidated RC-S sold assets and related recourse exposure."""

from __future__ import annotations

import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


ARCHIVE = "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
FIELDS = {
    "RCFDB790": "residential_sold_principal_thousand_usd",
    "RCFDB796": "other_sold_principal_thousand_usd",
    "RCFDB797": "residential_max_recourse_thousand_usd",
    "RCFDB803": "other_max_recourse_thousand_usd",
}


def build() -> pd.DataFrame:
    source = RAW_DIR / "ffiec_call_reports" / ARCHIVE
    with zipfile.ZipFile(source) as archive:
        with archive.open("FFIEC CDR Call Bulk POR 09302025.txt") as handle:
            identity = pd.read_csv(handle, sep="\t", dtype="string", usecols=[
                "IDRSSD", "FDIC Certificate Number", "Financial Institution Filing Type",
            ])
        with archive.open("FFIEC CDR Call Schedule RCS 09302025.txt") as handle:
            raw = pd.read_csv(handle, sep="\t", skiprows=[1], dtype="string",
                              usecols=["IDRSSD", *FIELDS])
    identity = identity.rename(columns={
        "FDIC Certificate Number": "fdic_certificate",
        "Financial Institution Filing Type": "filing_type",
    })
    for column in identity.columns:
        identity[column] = identity[column].str.strip()
    if identity.IDRSSD.isna().any() or identity.IDRSSD.duplicated().any():
        raise ValueError("POR has missing or duplicate RSSD IDs")
    values = raw[list(FIELDS)].apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    staged = pd.DataFrame({"bank_rssd_id": raw.loc[complete, "IDRSSD"].str.strip(),
                           "report_date": "2025-09-30"})
    for raw_column, named_column in FIELDS.items():
        staged[named_column] = values.loc[complete, raw_column]
    count = len(staged)
    staged = staged.merge(identity, left_on="bank_rssd_id", right_on="IDRSSD",
                          validate="one_to_one").drop(columns="IDRSSD")
    if len(staged) != count or staged.empty:
        raise ValueError("RC-S complete rows lost at identity join")
    if not staged.filing_type.eq("031").all():
        raise ValueError("complete consolidated RC-S outside FFIEC 031")
    if staged.fdic_certificate.isna().any() or staged.fdic_certificate.eq("").any():
        raise ValueError("RC-S bank without FDIC certificate")
    if staged[list(FIELDS.values())].lt(0).any().any():
        raise ValueError("negative RC-S amount")
    out = BUILD_DIR / "ffiec_call_reports" / "bank_sold_recourse_rcs_2025q3.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(staged)} consolidated bank rows")
    return staged


if __name__ == "__main__":
    build()
