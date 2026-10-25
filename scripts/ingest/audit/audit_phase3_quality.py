"""Audit phase-three inventory, container integrity, and provenance coverage."""

from __future__ import annotations

import json
import sys
from collections import Counter

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
EXPECTED_FILES = {
    "sec_13f": 9,
    "sec_ncen": 9,
    "sec_midas_security_exchange": 8,
    "sec_adv": 39,
    "gleif_golden_copy": 3,
    "fed_scf": 9,
    "fed_z1_vintages": 7,
    "fed_fwtw": 3,
    "ncua_call_reports": 8,
    "fdic_sod": 5,
    "census_qfr": 16,
    "treasury_tic": 10,
    "bea_mne": 1,
    "fema_nfip": 138,
    "fema_disasters": 1,
    "eia_bulk": 6,
    "eia_steo_vintages": 24,
    "cfpb_credit_card_agreements": 8,
    "cfpb_prepaid_agreements": 3,
    "ofac_sanctions": 266,
    "usaspending": 4,
}


def main() -> None:
    errors: list[str] = []
    total_bytes = 0
    total_files = 0
    for dataset, expected in EXPECTED_FILES.items():
        root = DATASETS / dataset
        manifest_path = root / "sources.json"
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("raw_files", [])
        if len(records) != expected:
            errors.append(f"{dataset}: expected {expected} manifested files, found {len(records)}")
        if manifest.get("failures"):
            errors.append(f"{dataset}: {len(manifest['failures'])} recorded download failures")
        notes = " ".join(str(row.get("note", "")) for row in records).lower()
        if dataset in {"sec_adv", "fed_fwtw", "fema_nfip", "cfpb_prepaid_agreements"}:
            if not any(word in notes for word in ("current", "cutoff", "retrospective", "backfill")):
                errors.append(f"{dataset}: missing cutoff/current-snapshot warning")
        for record in records:
            path = root / record["file"]
            total_files += 1
            total_bytes += int(record["bytes"])
            if path.suffix.lower() == ".parquet" and path.is_file():
                with path.open("rb") as handle:
                    start = handle.read(4)
                    handle.seek(-4, 2)
                    end = handle.read(4)
                if start != b"PAR1" or end != b"PAR1":
                    errors.append(f"invalid Parquet signature: {path}")
            if path.suffix.lower() == ".json" and path.is_file():
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid JSON {path}: {exc}")

    # Source-specific completeness checks catch silent API pagination and
    # discovery mistakes that byte/hash validation alone cannot detect.
    for year, expected_rows in ((2024, 76727), (2025, 76120)):
        path = RAW_DIR / "fdic_sod" / f"sod_{year}_all.csv"
        if path.is_file():
            with path.open("rb") as handle:
                rows = sum(1 for _ in handle) - 1
            if rows != expected_rows:
                errors.append(f"FDIC SOD {year}: expected {expected_rows} rows, found {rows}")

    deltas = Counter()
    for path in (RAW_DIR / "ofac_sanctions").glob("*.xml"):
        if path.name.startswith("2024"):
            deltas[2024] += 1
        elif path.name.startswith("2025"):
            deltas[2025] += 1
    if deltas != Counter({2024: 138, 2025: 123}):
        errors.append(f"OFAC delta counts differ from discovery archive: {dict(deltas)}")

    print(
        f"Phase 3 quality audit: {total_files} files, {total_bytes / 2**30:.4f} GiB, "
        f"dataset families={len(EXPECTED_FILES)}, errors={len(errors)}"
    )
    for error in errors:
        print(error)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
