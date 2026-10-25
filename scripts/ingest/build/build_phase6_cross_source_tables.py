"""Build compact local runtime tables for two cross-source case families.

The FDIC Summary of Deposits raw file is a branch-level annual snapshot.  The
runtime projection retains branch identity, geography and branch deposits but
does not repeat institution-level balance-sheet fields on every branch row;
those balances remain the independent BankFind source in cross-source cases.

FINRA consolidated short-volume files are pipe-delimited daily files with a
one-field trailer after the data rows.  The runtime builder validates and drops
that trailer, normalizes the date, and materializes only April 2025, matching the
frozen SEC MIDAS monthly slice used by the intended integrated case.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
SOD_FIELDS = [
    "report_date",
    "fdic_certificate",
    "bank_rssd_id",
    "bank_name",
    "branch_id",
    "branch_number",
    "branch_name",
    "main_office_indicator",
    "branch_state",
    "branch_county_fips",
    "branch_deposits_thousand_usd",
    "source_file",
]
FINRA_FIELDS = [
    "date",
    "symbol",
    "short_volume_shares",
    "short_exempt_volume_shares",
    "total_volume_shares",
    "market_codes",
    "source_file",
]


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _write_deterministic_gzip_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def build_fdic_sod_branches() -> dict[str, object]:
    root = DATASETS / "fdic_sod"
    source_path = RAW_DIR / "finra_short_volume" / "sod_2025_all.csv"
    rows: list[dict[str, str]] = []
    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "CERT", "RSSDID", "NAMEFULL", "ID", "BRNUM", "NAMEBR", "BKMO",
            "STALPBR", "STCNTYBR", "DEPSUMBR",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"FDIC SOD source is missing fields {sorted(missing)}")
        for source in reader:
            rows.append({
                "report_date": "2025-06-30",
                "fdic_certificate": _clean(source["CERT"]),
                "bank_rssd_id": _clean(source["RSSDID"]),
                "bank_name": _clean(source["NAMEFULL"]),
                "branch_id": _clean(source["ID"]),
                "branch_number": _clean(source["BRNUM"]),
                "branch_name": _clean(source["NAMEBR"]),
                "main_office_indicator": _clean(source["BKMO"]),
                "branch_state": _clean(source["STALPBR"]),
                "branch_county_fips": _clean(source["STCNTYBR"]).zfill(5),
                "branch_deposits_thousand_usd": _clean(source["DEPSUMBR"]),
                "source_file": source_path.name,
            })
    rows.sort(key=lambda row: (row["fdic_certificate"], row["branch_id"]))
    keys = [row["branch_id"] for row in rows]
    if len(rows) != 76120 or len(keys) != len(set(keys)):
        raise ValueError("unexpected FDIC SOD 2025 branch population or duplicate IDs")
    if any(not row["branch_state"] for row in rows):
        raise ValueError("FDIC SOD branch row has a missing branch state")
    output = root / "branches_2025.csv.gz"
    _write_deterministic_gzip_csv(output, SOD_FIELDS, rows)
    return {
        "rows": len(rows),
        "report_date": "2025-06-30",
        "output": str(output.relative_to(DATASETS)),
    }


def build_finra_april_short_volume() -> dict[str, object]:
    root = DATASETS / "finra_short_volume"
    source_paths = sorted((RAW_DIR / "finra_short_volume").glob("CNMSshvol202504*.txt"))
    if len(source_paths) != 21:
        raise ValueError(f"expected 21 April FINRA files, found {len(source_paths)}")
    rows: list[dict[str, str]] = []
    for source_path in source_paths:
        with source_path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.reader(handle, delimiter="|"))
        if raw_rows[0] != [
            "Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"
        ]:
            raise ValueError(f"unexpected FINRA header in {source_path.name}")
        if len(raw_rows[-1]) != 1 or not raw_rows[-1][0].isdigit():
            raise ValueError(f"missing FINRA trailer in {source_path.name}")
        trailer_count = int(raw_rows[-1][0])
        data_rows = raw_rows[1:-1]
        if trailer_count != len(data_rows):
            raise ValueError(f"FINRA trailer count mismatch in {source_path.name}")
        expected_date = source_path.stem.removeprefix("CNMSshvol")
        for source in data_rows:
            if len(source) != 6 or source[0] != expected_date:
                raise ValueError(f"invalid FINRA data row in {source_path.name}")
            rows.append({
                "date": f"{source[0][:4]}-{source[0][4:6]}-{source[0][6:]}",
                "symbol": _clean(source[1]),
                "short_volume_shares": _clean(source[2]),
                "short_exempt_volume_shares": _clean(source[3]),
                "total_volume_shares": _clean(source[4]),
                "market_codes": _clean(source[5]),
                "source_file": source_path.name,
            })
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    keys = [(row["date"], row["symbol"]) for row in rows]
    if len(rows) != 219470 or len(keys) != len(set(keys)):
        raise ValueError("unexpected FINRA April population or duplicate date-symbol rows")
    if any(not row["symbol"] for row in rows):
        raise ValueError("FINRA runtime row has a missing symbol")
    output = root / "april_2025.csv.gz"
    _write_deterministic_gzip_csv(output, FINRA_FIELDS, rows)
    return {
        "rows": len(rows),
        "trading_dates": len({row["date"] for row in rows}),
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "output": str(output.relative_to(DATASETS)),
        "trailer_rule": "one-field trailer equals data-row count",
    }


def main() -> None:
    report = {
        "fdic_sod_branches": build_fdic_sod_branches(),
        "finra_april_short_volume": build_finra_april_short_volume(),
        "rules": [
            "FDIC SOD is a 2025-06-30 branch snapshot; branch deposits are reported in thousand USD.",
            "Institution-level deposit balances repeated in raw SOD rows are intentionally excluded from the branch runtime projection.",
            "FINRA one-field trailers are validated and excluded from the runtime table.",
            "FINRA ShortVolume includes executed short-sale and short-sale-exempt volume; ShortExemptVolume is the separately reported exempt subset and must not be added again.",
            "FINRA daily short volume is publicly disseminated off-exchange transaction volume reported to FINRA facilities, not short interest.",
            "FINRA field layout: https://www.finra.org/sites/default/files/2021-07/DailyShortSaleVolumeFileLayout.pdf",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase6_cross_source_table_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
