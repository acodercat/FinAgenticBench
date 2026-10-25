"""Build lossless, reviewable geography and institution linkage tables.

The outputs are derived only from the official phase-five raw files.  They do
not use fuzzy name matching: geographic joins use Census GEOIDs and financial
entity joins use NIC RSSD, FDIC, OCC, NCUA, ABA, CUSIP, and LEI identifiers.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _identifier(value: str | None) -> str:
    value = _clean(value)
    return "" if not value or set(value) == {"0"} else value


def _write_gzip_csv(path: Path, fields: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def build_geography() -> dict[str, int]:
    root = DATASETS / "census_geo_crosswalk"
    out = root / "normalized"

    gazetteer = RAW_DIR / "ffiec_nic_structure" / "2024_Gaz_counties_national.zip"
    with ZipFile(gazetteer) as archive:
        member = archive.namelist()[0]
        with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as handle:
            county_rows = [
                {key.strip(): _clean(value) for key, value in row.items()}
                for row in csv.DictReader(handle, delimiter="\t")
            ]
    county_fields = [
        "USPS", "GEOID", "ANSICODE", "NAME", "ALAND", "AWATER",
        "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG",
    ]
    county_count = _write_gzip_csv(out / "county_fips_2024.csv.gz", county_fields, county_rows)

    relationship = RAW_DIR / "ffiec_nic_structure" / "tab20_zcta520_county20_natl.txt"
    with relationship.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = [
            {key: _clean(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="|")
        ]
    # Census includes county-only residual rows with no ZCTA.  They are not
    # mappings, so exclude them explicitly instead of treating blank as a ZIP.
    mapped = [row for row in all_rows if row["GEOID_ZCTA5_20"]]
    relationship_fields = [
        "GEOID_ZCTA5_20", "NAMELSAD_ZCTA5_20", "GEOID_COUNTY_20",
        "NAMELSAD_COUNTY_20", "AREALAND_PART", "AREAWATER_PART",
    ]
    relation_count = _write_gzip_csv(
        out / "zcta_county_relationship_2020.csv.gz", relationship_fields, mapped
    )

    # A clearly labelled convenience mapping: select the county with the
    # greatest overlapping land area (water is only a tie-breaker).
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mapped:
        groups[row["GEOID_ZCTA5_20"]].append(row)
    primary = []
    for zcta, rows in sorted(groups.items()):
        chosen = max(
            rows,
            key=lambda row: (int(row["AREALAND_PART"] or 0), int(row["AREAWATER_PART"] or 0)),
        )
        primary.append({
            "zcta5": zcta,
            "zcta_name": chosen["NAMELSAD_ZCTA5_20"],
            "county_geoid": chosen["GEOID_COUNTY_20"],
            "county_name": chosen["NAMELSAD_COUNTY_20"],
            "selection_rule": "maximum overlapping land area; water area tie-breaker",
        })
    primary_count = _write_gzip_csv(
        out / "zcta_primary_county_2020.csv.gz",
        ["zcta5", "zcta_name", "county_geoid", "county_name", "selection_rule"],
        primary,
    )
    return {
        "county_fips_rows": county_count,
        "zcta_county_relationship_rows": relation_count,
        "zcta_primary_county_rows": primary_count,
    }


def build_financial_entities() -> dict[str, int]:
    root = DATASETS / "ffiec_nic_structure"
    out = root / "normalized"
    attributes_zip = RAW_DIR / "ffiec_nic_structure" / "attributes_active_current.zip"
    with ZipFile(attributes_zip) as archive:
        with io.TextIOWrapper(
            archive.open("CSV_ATTRIBUTES_ACTIVE.CSV"), encoding="utf-8-sig", newline=""
        ) as handle:
            attributes = list(csv.DictReader(handle))

    by_rssd: dict[str, dict[str, str]] = {}
    identifier_rows = []
    for row in attributes:
        rssd = _identifier(row["#ID_RSSD"])
        clean = {
            "rssd_id": rssd,
            "legal_name": _clean(row["NM_LGL"]),
            "entity_type": _clean(row["ENTITY_TYPE"]),
            "primary_activity_naics": _clean(row["ACT_PRIM_CD"]),
            "city": _clean(row["CITY"]),
            "state": _clean(row["STATE_ABBR_NM"]),
            "county_code": _identifier(row["COUNTY_CD"]),
            "zip_code": _identifier(row["ZIP_CD"]),
            "fdic_certificate": _identifier(row["ID_FDIC_CERT"]),
            "occ_id": _identifier(row["ID_OCC"]),
            "ncua_id": _identifier(row["ID_NCUA"]),
            "aba_primary": _identifier(row["ID_ABA_PRIM"]),
            "cusip_id": _identifier(row["ID_CUSIP"]),
            "lei": _identifier(row["ID_LEI"]),
            "tax_id": _identifier(row["ID_TAX"]),
            "primary_federal_regulator": _clean(row["PRIM_FED_REG"]),
            "source_snapshot": "FFIEC NIC current active attributes observed 2026-08-13",
        }
        by_rssd[rssd] = clean
        identifier_rows.append(clean)
    identifier_fields = list(identifier_rows[0])
    identifier_count = _write_gzip_csv(
        out / "institution_identifiers_current.csv.gz", identifier_fields, identifier_rows
    )

    relationships_zip = RAW_DIR / "ffiec_nic_structure" / "relationships_current.zip"
    with ZipFile(relationships_zip) as archive:
        with io.TextIOWrapper(
            archive.open("CSV_RELATIONSHIPS.CSV"), encoding="utf-8-sig", newline=""
        ) as handle:
            relationships = list(csv.DictReader(handle))
    current = []
    for row in relationships:
        if _clean(row["DT_END"]) != "99991231":
            continue
        parent_id = _identifier(row["#ID_RSSD_PARENT"])
        child_id = _identifier(row["ID_RSSD_OFFSPRING"])
        parent = by_rssd.get(parent_id, {})
        child = by_rssd.get(child_id, {})
        current.append({
            "parent_rssd_id": parent_id,
            "parent_legal_name": parent.get("legal_name", ""),
            "offspring_rssd_id": child_id,
            "offspring_legal_name": child.get("legal_name", ""),
            "relationship_start": _clean(row["DT_RELN_EST"]),
            "relationship_level": _clean(row["RELN_LVL"]),
            "control_indicator": _clean(row["CTRL_IND"]),
            "equity_percent": _clean(row["PCT_EQUITY"]),
            "regulatory_relationship_indicator": _clean(row["REG_IND"]),
            "source_snapshot": "FFIEC NIC current relationships observed 2026-08-13",
        })
    relationship_fields = list(current[0])
    current_relationship_count = _write_gzip_csv(
        out / "ownership_relationships_current.csv.gz", relationship_fields, current
    )
    return {
        "active_institution_rows": identifier_count,
        "current_relationship_rows": current_relationship_count,
    }


def main() -> None:
    report = {
        "geography": build_geography(),
        "financial_entities": build_financial_entities(),
        "rules": [
            "No fuzzy entity-name joins are introduced.",
            "Zero-valued NIC identifier placeholders are normalized to blank.",
            "Current NIC tables are discovery/crosswalk support, not a 2025 point-in-time snapshot.",
            "The primary ZCTA county is a derived maximum-overlap convenience field; use the full relationship table when allocation matters.",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase5_crosswalk_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
