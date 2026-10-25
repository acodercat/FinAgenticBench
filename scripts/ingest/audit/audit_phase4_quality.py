"""Audit integrity, completeness, and key semantics of phase-four datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
EXPECTED_COUNTS = {
    "dol_form5500": 63,
    "ffiec_call_reports": 8,
    "fed_fomc_materials": 56,
    "treasury_debt": 14,
    "treasury_mts": 21,
    "treasury_dts": 8,
    "treasury_usfr": 9,
}
TREASURY_DATASETS = {"treasury_debt", "treasury_mts", "treasury_dts", "treasury_usfr"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> tuple[int, list[str], str | None, str | None]:
    count = 0
    minimum = maximum = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for row in reader:
            count += 1
            stamp = row.get("record_date")
            if stamp:
                minimum = stamp if minimum is None or stamp < minimum else minimum
                maximum = stamp if maximum is None or stamp > maximum else maximum
    return count, fields, minimum, maximum


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    summaries: dict[str, dict] = {}

    for dataset, expected_count in EXPECTED_COUNTS.items():
        directory = DATASETS / dataset
        manifest_path = directory / "sources.json"
        if not manifest_path.exists():
            errors.append(f"{dataset}: missing sources.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("raw_files", [])
        if manifest.get("failures"):
            errors.append(f"{dataset}: manifest has {len(manifest['failures'])} failure(s)")
        if len(records) != expected_count:
            errors.append(f"{dataset}: expected {expected_count} files, manifest has {len(records)}")

        manifest_files = {record["file"] for record in records}
        disk_files = {
            str(path.relative_to(directory))
            for path in (RAW_DIR / directory.name).glob("*") if path.is_file()
        }
        extras = sorted(disk_files - manifest_files)
        missing = sorted(manifest_files - disk_files)
        if extras:
            errors.append(f"{dataset}: unmanifested files: {extras[:5]}")
        if missing:
            errors.append(f"{dataset}: missing files: {missing[:5]}")

        total_bytes = 0
        format_counts: Counter[str] = Counter()
        for record in records:
            path = directory / record["file"]
            if not path.exists():
                continue
            total_bytes += path.stat().st_size
            suffix = path.suffix.lower()
            format_counts[suffix] += 1
            if path.stat().st_size != record["bytes"]:
                errors.append(f"{dataset}: byte-size mismatch for {path.name}")
            if _sha256(path) != record["sha256"]:
                errors.append(f"{dataset}: SHA-256 mismatch for {path.name}")
            if suffix == ".zip":
                try:
                    with ZipFile(path) as archive:
                        bad = archive.testzip()
                        if bad:
                            errors.append(f"{dataset}: CRC failure {path.name}:{bad}")
                except Exception as exc:
                    errors.append(f"{dataset}: invalid ZIP {path.name}: {exc}")
            elif suffix == ".pdf" and path.read_bytes()[:5] != b"%PDF-":
                errors.append(f"{dataset}: invalid PDF signature for {path.name}")
            elif suffix == ".xls" and path.read_bytes()[:8] != bytes.fromhex("D0CF11E0A1B11AE1"):
                errors.append(f"{dataset}: invalid legacy XLS signature for {path.name}")
        summaries[dataset] = {
            "files": len(records), "bytes": total_bytes, "formats": dict(format_counts)
        }

    # Form 5500: every table archive must carry its own layout and ACK_ID join key.
    form_root = RAW_DIR / "dol_form5500"
    for path in sorted(form_root.glob("*.zip")):
        with ZipFile(path) as archive:
            names = archive.namelist()
            csv_names = [name for name in names if name.lower().endswith(".csv")]
            layout_names = [name for name in names if name.lower().endswith("_layout.txt")]
            if len(csv_names) != 1 or len(layout_names) != 1:
                errors.append(f"dol_form5500: expected CSV+layout pair in {path.name}")
                continue
            header = archive.open(csv_names[0]).readline().decode("utf-8-sig", errors="replace")
            if "ACK_ID" not in header.split(","):
                errors.append(f"dol_form5500: ACK_ID missing from {path.name}")
    for year in (2023, 2024):
        path = form_root / f"F_5500_{year}_all.zip"
        with ZipFile(path) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            rows = sum(1 for _ in archive.open(csv_name)) - 1
            if rows < 100_000:
                errors.append(f"dol_form5500: unexpectedly few main-form rows for {year}: {rows}")
            summaries["dol_form5500"][f"main_form_{year}_rows"] = rows

    # FFIEC: each quarter must include POR plus core balance-sheet/income schedules.
    ffiec_root = RAW_DIR / "ffiec_call_reports"
    for path in sorted(ffiec_root.glob("*.zip")):
        stamp_match = re.search(r"(\d{8})", path.name)
        stamp = stamp_match.group(1) if stamp_match else ""
        with ZipFile(path) as archive:
            names = archive.namelist()
            required = {
                f"FFIEC CDR Call Bulk POR {stamp}.txt",
                f"FFIEC CDR Call Schedule RC {stamp}.txt",
                f"FFIEC CDR Call Schedule RI {stamp}.txt",
            }
            if not required.issubset(names):
                errors.append(f"ffiec_call_reports: core schedules missing from {path.name}")
                continue
            por_rows = sum(1 for _ in archive.open(f"FFIEC CDR Call Bulk POR {stamp}.txt")) - 1
            if por_rows < 4_000:
                errors.append(f"ffiec_call_reports: too few institutions in {path.name}: {por_rows}")
            summaries["ffiec_call_reports"][f"institutions_{stamp}"] = por_rows

    # Treasury: manifest API metadata is the row-count and pagination authority.
    for dataset in TREASURY_DATASETS:
        root = RAW_DIR / dataset
        metadata = json.loads((root / "api_metadata_and_pages.json").read_text(encoding="utf-8"))
        by_name = {table["name"]: table for table in metadata["tables"]}
        for path in sorted(root.glob("*.csv")):
            match = re.match(r"(.+?)_(?:2020|2024)_through_2025\.csv", path.name)
            if not match or match.group(1) not in by_name:
                errors.append(f"{dataset}: CSV has no API metadata entry: {path.name}")
                continue
            count, fields, minimum, maximum = _csv_rows(path)
            table = by_name[match.group(1)]
            if count != table["row_count"]:
                errors.append(f"{dataset}: row count mismatch {path.name}: {count} != {table['row_count']}")
            if "record_date" not in fields:
                errors.append(f"{dataset}: record_date missing from {path.name}")
            if minimum and minimum < table["start_date"]:
                errors.append(f"{dataset}: pre-window row in {path.name}: {minimum}")
            if maximum and maximum > "2025-12-31":
                errors.append(f"{dataset}: post-cutoff row in {path.name}: {maximum}")

    # Published accounting identity in Debt to the Penny.
    debt_path = RAW_DIR / "treasury_debt" / "debt_to_penny_2024_through_2025.csv"
    with debt_path.open(encoding="utf-8", newline="") as handle:
        identity_anomalies: list[tuple[str, Decimal]] = []
        for line, row in enumerate(csv.DictReader(handle), 2):
            total = Decimal(row["tot_pub_debt_out_amt"])
            parts = Decimal(row["debt_held_public_amt"]) + Decimal(row["intragov_hold_amt"])
            if total != parts:
                identity_anomalies.append((row["record_date"], total - parts))
        expected_source_anomaly = [("2025-08-04", Decimal("-10000000000.00"))]
        if identity_anomalies == expected_source_anomaly:
            warnings.append(
                "treasury_debt: official Debt to the Penny API has one published accounting "
                "identity anomaly on 2025-08-04 (total is $10bn below its two components); "
                "raw value retained without silent correction"
            )
        elif identity_anomalies:
            errors.append(f"treasury_debt: unexpected debt identity anomalies: {identity_anomalies[:5]}")

    residual = sorted(
        str(path.relative_to(DATASETS))
        for dataset in EXPECTED_COUNTS
        for path in (RAW_DIR / dataset).glob("*.part")
    )
    if residual:
        errors.append(f"residual partial files: {residual[:5]}")

    print(json.dumps(
        {"datasets": summaries, "warnings": warnings, "errors": errors},
        indent=2, ensure_ascii=False,
    ))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
