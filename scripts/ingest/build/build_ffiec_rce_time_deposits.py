"""Stage 2025Q3 FFIEC 041 domestic time-deposit maturity/repricing items."""

from __future__ import annotations

import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


ARCHIVE = "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
FIELDS = {
    "RCONHK07": "small_within_3m_maturity_or_reset_thousand_usd",
    "RCONHK08": "small_3_to_12m_maturity_or_reset_thousand_usd",
    "RCONHK11": "small_within_1y_maturity_thousand_usd",
    "RCONHK12": "large_within_3m_maturity_or_reset_thousand_usd",
    "RCONHK13": "large_3_to_12m_maturity_or_reset_thousand_usd",
    "RCONK222": "large_within_1y_maturity_thousand_usd",
}


def build() -> pd.DataFrame:
    source = RAW_DIR / "ffiec_call_reports" / ARCHIVE
    with zipfile.ZipFile(source) as archive:
        with archive.open("FFIEC CDR Call Bulk POR 09302025.txt") as handle:
            identity = pd.read_csv(handle, sep="\t", dtype="string", usecols=[
                "IDRSSD", "FDIC Certificate Number", "Financial Institution Filing Type",
            ])
        with archive.open("FFIEC CDR Call Schedule RCE 09302025.txt") as handle:
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
    staged = staged.merge(identity, left_on="bank_rssd_id", right_on="IDRSSD",
                          validate="one_to_one").drop(columns="IDRSSD")
    staged = staged.loc[staged.filing_type.eq("041")].copy()
    if staged.empty or staged.bank_rssd_id.duplicated().any():
        raise ValueError("no unique complete FFIEC 041 RC-E rows")
    if staged.fdic_certificate.isna().any() or staged.fdic_certificate.eq("").any():
        raise ValueError("missing FDIC certificate")
    amounts = staged[list(FIELDS.values())]
    if amounts.lt(0).any().any():
        raise ValueError("negative RC-E amount")
    small_short = amounts.iloc[:, 0] + amounts.iloc[:, 1]
    large_short = amounts.iloc[:, 3] + amounts.iloc[:, 4]
    if (small_short < amounts.iloc[:, 2]).any() or (large_short < amounts.iloc[:, 5]).any():
        raise ValueError("one-year maturity exceeds maturity-or-repricing bucket")
    out = BUILD_DIR / "ffiec_call_reports" / "bank_time_deposits_rce_2025q3.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(staged)} complete FFIEC 041 domestic bank rows")
    return staged


if __name__ == "__main__":
    build()
