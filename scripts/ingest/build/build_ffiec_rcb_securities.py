"""Stage the 2025Q3 RC-B securities totals from the frozen FFIEC bulk file.

RCFD and RCON columns can differ: prefer consolidated RCFD where reported,
otherwise use domestic RCON and disclose that mixed scope. The staging file is
an ex-post 2025Q3 report snapshot, not a claim about publication by quarter end.
"""

from __future__ import annotations

import argparse
import zipfile

import pandas as pd

from core.dataset_layers import BUILD_DIR, RAW_DIR


RAW_NAME = "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
MEMBER = "FFIEC CDR Call Schedule RCB 09302025(1 of 2).txt"
FIELDS = {
    "1754": "htm_amortized_cost_thousand_usd",
    "1771": "htm_fair_value_thousand_usd",
    "1772": "afs_amortized_cost_thousand_usd",
    "1773": "afs_fair_value_thousand_usd",
    "0416": "pledged_securities_book_value_thousand_usd",
}


def build() -> pd.DataFrame:
    source = RAW_DIR / "ffiec_call_reports" / RAW_NAME
    with zipfile.ZipFile(source) as archive, archive.open(MEMBER) as handle:
        frame = pd.read_csv(handle, sep="\t", skiprows=[1], dtype="string", low_memory=False)
    result = pd.DataFrame({"bank_rssd_id": frame["IDRSSD"].str.strip(),
                           "report_date": "2025-09-30"})
    consolidated = pd.to_numeric(frame["RCFD1754"], errors="coerce").notna()
    result["reporting_basis"] = pd.Series(
        "domestic", index=frame.index, dtype="string"
    ).mask(consolidated, "consolidated")
    for suffix, target in FIELDS.items():
        federal = pd.to_numeric(frame[f"RCFD{suffix}"], errors="coerce")
        domestic = pd.to_numeric(frame[f"RCON{suffix}"], errors="coerce")
        result[target] = domestic.mask(consolidated, federal)
    if result.bank_rssd_id.isna().any() or result.bank_rssd_id.duplicated().any():
        raise ValueError("RC-B bank identifiers are missing or duplicated")
    out = BUILD_DIR / "ffiec_call_reports" / "bank_securities_rcb_2025q3.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out}: {len(result)} bank rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
