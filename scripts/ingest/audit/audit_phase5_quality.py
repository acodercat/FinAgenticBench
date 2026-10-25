"""Audit integrity, provenance, time scope, and linkage keys for phase five."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
REPORT = PROJECT_ROOT / "metadata" / "phase5_quality_audit.json"
EXPECTED_COUNTS = {
    "bls_qcew": 2,
    "noaa_storm_events": 7,
    "ofr_market_stress": 5,
    "epa_ghgrp": 3,
    "census_international_trade": 4,
    "census_geo_crosswalk": 2,
    "ffiec_y9c": 9,
    "ffiec_nic_structure": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xlsx_sheets(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return [element.attrib["name"] for element in root.iter() if element.tag.endswith("sheet")]


def _check_inventory(errors: list[str], summaries: dict) -> None:
    for dataset, expected in EXPECTED_COUNTS.items():
        root = DATASETS / dataset
        manifest_path = root / "sources.json"
        if not manifest_path.is_file():
            errors.append(f"{dataset}: missing sources.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("raw_files", [])
        if manifest.get("failures"):
            errors.append(f"{dataset}: manifest records failures: {manifest['failures']}")
        if len(records) != expected:
            errors.append(f"{dataset}: expected {expected} files, found {len(records)}")
        manifested = {record["file"] for record in records}
        on_disk = {
            str(path.relative_to(root)) for path in (RAW_DIR / root.name).glob("*") if path.is_file()
        }
        if manifested != on_disk:
            errors.append(
                f"{dataset}: raw/manifest mismatch missing={sorted(manifested-on_disk)} "
                f"extra={sorted(on_disk-manifested)}"
            )
        total_bytes = 0
        formats: Counter[str] = Counter()
        for record in records:
            path = root / record["file"]
            if not path.is_file():
                continue
            total_bytes += path.stat().st_size
            formats[path.suffix.lower()] += 1
            if path.stat().st_size != record["bytes"]:
                errors.append(f"{dataset}: byte mismatch {path.name}")
            if _sha256(path) != record["sha256"]:
                errors.append(f"{dataset}: SHA-256 mismatch {path.name}")
            suffix = path.suffix.lower()
            if suffix in {".zip", ".xlsx"}:
                try:
                    with ZipFile(path) as archive:
                        bad = archive.testzip()
                        if bad:
                            errors.append(f"{dataset}: CRC failure {path.name}:{bad}")
                except Exception as exc:
                    errors.append(f"{dataset}: invalid ZIP/OOXML {path.name}: {exc}")
            elif suffix == ".xlsb":
                signature = path.read_bytes()[:8]
                if signature[:4] == b"PK\x03\x04":
                    try:
                        with ZipFile(path) as archive:
                            bad = archive.testzip()
                            if bad:
                                errors.append(f"{dataset}: CRC failure {path.name}:{bad}")
                    except Exception as exc:
                        errors.append(f"{dataset}: invalid ZIP-based XLSB {path.name}: {exc}")
                elif signature != bytes.fromhex("D0CF11E0A1B11AE1"):
                    errors.append(f"{dataset}: invalid XLSB signature {path.name}")
            elif suffix == ".pdf" and path.read_bytes()[:5] != b"%PDF-":
                errors.append(f"{dataset}: invalid PDF signature {path.name}")
            elif suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"{dataset}: invalid JSON {path.name}: {exc}")
            elif suffix == ".gz":
                try:
                    with gzip.open(path, "rb") as handle:
                        handle.read(1)
                except Exception as exc:
                    errors.append(f"{dataset}: invalid gzip {path.name}: {exc}")
        summaries[dataset] = {
            "files": len(records), "bytes": total_bytes, "formats": dict(formats)
        }


def _audit_qcew(errors: list[str], summaries: dict) -> None:
    result = {}
    required = {"area_fips", "industry_code", "year", "qtr", "annual_avg_emplvl", "total_annual_wages"}
    for year in (2024, 2025):
        path = RAW_DIR / "bls_qcew" / f"{year}_annual_singlefile.zip"
        with ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                errors.append(f"bls_qcew: unexpected members in {path.name}: {names}")
                continue
            with io.TextIOWrapper(archive.open(names[0]), encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not required.issubset(reader.fieldnames or []):
                    errors.append(f"bls_qcew: missing required columns in {path.name}")
                rows = 0
                areas = set()
                years = set()
                quarters = set()
                for row in reader:
                    rows += 1
                    areas.add(row["area_fips"])
                    years.add(row["year"])
                    quarters.add(row["qtr"])
        if years != {str(year)} or quarters != {"A"} or rows < 3_000_000:
            errors.append(
                f"bls_qcew: implausible annual file {year}: rows={rows}, years={years}, qtr={quarters}"
            )
        result[str(year)] = {"rows": rows, "distinct_areas": len(areas)}
    summaries["bls_qcew"]["semantics"] = result


def _audit_noaa(errors: list[str], summaries: dict) -> None:
    result = {}
    for year in (2024, 2025):
        tables: dict[str, tuple[int, set[str]]] = {}
        detail_ids: set[str] = set()
        for kind in ("details", "locations", "fatalities"):
            path = next((RAW_DIR / "noaa_storm_events").glob(f"StormEvents_{kind}-*_d{year}_*.csv.gz"))
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            ids = {row["EVENT_ID"] for row in rows}
            tables[kind] = (len(rows), ids)
            if kind == "details":
                detail_ids = ids
                found_years = {row["YEAR"] for row in rows}
                if found_years != {str(year)}:
                    errors.append(f"noaa_storm_events: wrong YEAR in {path.name}: {found_years}")
        orphan_counts = {}
        for kind in ("locations", "fatalities"):
            orphan = tables[kind][1] - detail_ids
            orphan_counts[kind] = len(orphan)
        result[str(year)] = {kind: value[0] for kind, value in tables.items()}
        result[str(year)]["orphan_event_ids"] = orphan_counts
    summaries["noaa_storm_events"]["semantics"] = result


def _audit_ofr(errors: list[str], summaries: dict) -> None:
    root = RAW_DIR / "ofr_market_stress"
    with (root / "fsi.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fsi_dates = [row["Date"] for row in rows]
    if not fsi_dates or min(fsi_dates) > "2024-01-01" or max(fsi_dates) < "2025-12-31":
        errors.append("ofr_market_stress: FSI does not span the benchmark window")
    result = {"fsi_rows": len(rows), "fsi_min_date": min(fsi_dates), "fsi_max_date": max(fsi_dates)}
    for dataset in ("repo", "mmf"):
        payload = json.loads((root / f"stfm_{dataset}_2024_2025.json").read_text())
        dates = []
        for series in payload["timeseries"].values():
            # ``disclosure_edits`` is metadata about historical publication
            # changes and is intentionally not bounded by observation dates.
            dates.extend(point[0] for point in series["timeseries"].get("aggregation", []))
        if not dates or min(dates) < "2024-01-01" or max(dates) > "2025-12-31":
            errors.append(f"ofr_market_stress: {dataset} API response escaped requested window")
        result[f"{dataset}_series"] = len(payload["timeseries"])
        result[f"{dataset}_min_date"] = min(dates)
        result[f"{dataset}_max_date"] = max(dates)
    summaries["ofr_market_stress"]["semantics"] = result


def _audit_epa_and_census(errors: list[str], summaries: dict) -> None:
    root = RAW_DIR / "epa_ghgrp"
    with ZipFile(root / "2023_data_summary_spreadsheets.zip") as archive:
        annual_names = [name for name in archive.namelist() if re.fullmatch(r"ghgp_data_\d{4}\.xlsx", name)]
        years = sorted(int(Path(name).stem.rsplit("_", 1)[-1]) for name in annual_names)
        supplemental = sorted(name for name in archive.namelist() if name not in annual_names)
    if years[0] != 2010 or years[-1] != 2023 or len(years) != 14:
        errors.append(f"epa_ghgrp: expected annual workbooks 2010-2023, found {years}")
    crosswalk_sheets = _xlsx_sheets(root / "ghgrp_oris_power_plant_crosswalk_12_13_21.xlsx")
    if "ORIS Crosswalk" not in crosswalk_sheets:
        errors.append("epa_ghgrp: ORIS Crosswalk sheet missing")
    summaries["epa_ghgrp"]["semantics"] = {
        "summary_workbook_years": years,
        "supplemental_workbooks": supplemental,
        "power_crosswalk_sheets": crosswalk_sheets,
        "latest_observation_year": 2023,
    }

    trade_root = RAW_DIR / "census_international_trade"
    expected = {
        "country_enduse_exports_2016_2025.xlsx": "EnduseExp",
        "country_enduse_imports_2016_2025.xlsx": "EnduseImp",
        "monthly_principal_enduse_exports.xlsx": "Exhibit 12",
        "monthly_principal_enduse_imports.xlsx": "Exhibit 12A",
    }
    sheets = {name: _xlsx_sheets(trade_root / name) for name in expected}
    for name, sheet in expected.items():
        if sheet not in sheets[name]:
            errors.append(f"census_international_trade: missing sheet {sheet} in {name}")
    summaries["census_international_trade"]["semantics"] = {"sheets": sheets}

    geo_root = RAW_DIR / "census_geo_crosswalk"
    with ZipFile(geo_root / "2024_Gaz_counties_national.zip") as archive:
        with io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding="utf-8-sig", newline="") as handle:
            county_rows = list(csv.DictReader(handle, delimiter="\t"))
    with (geo_root / "tab20_zcta520_county20_natl.txt").open(encoding="utf-8-sig", newline="") as handle:
        relationships = list(csv.DictReader(handle, delimiter="|"))
    mapped = [row for row in relationships if row["GEOID_ZCTA5_20"]]
    if len(county_rows) < 3_200 or len(mapped) < 40_000:
        errors.append("census_geo_crosswalk: implausibly sparse geography tables")
    summaries["census_geo_crosswalk"]["semantics"] = {
        "county_rows": len(county_rows),
        "zcta_county_mapped_rows": len(mapped),
        "distinct_zctas": len({row["GEOID_ZCTA5_20"] for row in mapped}),
    }


def _audit_ffiec(errors: list[str], summaries: dict) -> None:
    root = RAW_DIR / "ffiec_y9c"
    quarters = {}
    for path in sorted(root.glob("BHCF*.ZIP")):
        stamp = path.stem[4:]
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names != [f"BHCF{stamp}.txt"]:
                errors.append(f"ffiec_y9c: unexpected member in {path.name}: {names}")
                continue
            # NIC's legacy caret files contain Windows-1252 institution names.
            with io.TextIOWrapper(archive.open(names[0]), encoding="cp1252", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="^")
                rows = list(reader)
        required = {"RSSD9001", "RSSD9999", "BHCA2170"}
        if not required.issubset(reader.fieldnames or []):
            errors.append(f"ffiec_y9c: required identifiers/total assets absent in {path.name}")
        report_dates = {row["RSSD9999"] for row in rows}
        rssd_ids = [row["RSSD9001"] for row in rows]
        if report_dates != {stamp} or len(rssd_ids) != len(set(rssd_ids)):
            errors.append(f"ffiec_y9c: bad date or duplicate RSSD in {path.name}")
        quarters[stamp] = {"rows": len(rows), "columns": len(reader.fieldnames or [])}
    if len(quarters) != 8:
        errors.append(f"ffiec_y9c: expected 8 quarterly files, found {len(quarters)}")
    summaries["ffiec_y9c"]["semantics"] = quarters

    nic_root = RAW_DIR / "ffiec_nic_structure"
    with ZipFile(nic_root / "attributes_active_current.zip") as archive:
        with io.TextIOWrapper(archive.open("CSV_ATTRIBUTES_ACTIVE.CSV"), encoding="utf-8-sig", newline="") as handle:
            attrs = list(csv.DictReader(handle))
    rssd = [row["#ID_RSSD"] for row in attrs]
    if len(attrs) < 30_000 or len(rssd) != len(set(rssd)):
        errors.append("ffiec_nic_structure: active attributes sparse or duplicate RSSD IDs")
    with ZipFile(nic_root / "relationships_current.zip") as archive:
        with io.TextIOWrapper(archive.open("CSV_RELATIONSHIPS.CSV"), encoding="utf-8-sig", newline="") as handle:
            relationships = list(csv.DictReader(handle))
    current_relationships = sum(row["DT_END"] == "99991231" for row in relationships)
    with ZipFile(nic_root / "transformations_current.zip") as archive:
        member = archive.namelist()[0]
        transformation_dates = [
            element.attrib.get("d_dt_trans", "")[:10]
            for _, element in ET.iterparse(archive.open(member), events=("end",))
            if element.tag.lower().endswith("transformation")
        ]
    if not transformation_dates:
        errors.append("ffiec_nic_structure: no transformation records")
    summaries["ffiec_nic_structure"]["semantics"] = {
        "active_institutions": len(attrs),
        "attributes_with_lei": sum(row["ID_LEI"].strip() not in {"", "0"} for row in attrs),
        "relationship_rows": len(relationships),
        "current_relationship_rows": current_relationships,
        "transformation_rows": len(transformation_dates),
        "transformation_date_min": min(transformation_dates),
        "transformation_date_max": max(transformation_dates),
    }


def main() -> None:
    errors: list[str] = []
    summaries: dict[str, dict] = {}
    _check_inventory(errors, summaries)
    _audit_qcew(errors, summaries)
    _audit_noaa(errors, summaries)
    _audit_ofr(errors, summaries)
    _audit_epa_and_census(errors, summaries)
    _audit_ffiec(errors, summaries)
    report = {
        "audit_date": "2026-08-13",
        "benchmark_cutoff": "2025-12-31",
        "dataset_families": len(EXPECTED_COUNTS),
        "raw_files": sum(summary["files"] for summary in summaries.values()),
        "raw_bytes": sum(summary["bytes"] for summary in summaries.values()),
        "datasets": summaries,
        "warnings": [
            "2025 annual QCEW and Census trade totals, 2025Q4 Y-9C, and current NOAA/NIC artifacts are retrospective or revised after the benchmark cutoff.",
            "EPA GHGRP's latest cutoff-compatible observation year is 2023 and must be treated as lagged structural evidence.",
            "Current NIC structures support identifier discovery and linkage, not 2025-12-31 point-in-time ownership claims.",
            "The official NOAA 2025 extraction contains one fatality EVENT_ID (1229449) absent from the same extraction's details table; raw data are retained and the row must be excluded from event-detail joins or handled explicitly.",
        ],
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
