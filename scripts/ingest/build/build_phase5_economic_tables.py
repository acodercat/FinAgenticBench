"""Build reviewed economic tables from official phase-five local archives.

The raw BLS QCEW annual single-file archives contain many ownership, industry,
geography and aggregation levels.  This builder materializes only state-area,
all-industry annual totals for total covered and private ownership.  It keeps the
source codes and does not impute suppressed values or attach state names by fuzzy
matching; geographic labels remain a separate Census crosswalk responsibility.

The OFR FSI raw file is a current revised history extending beyond the benchmark
cutoff. This builder exposes a compact daily table only through 2025-12-31 and
preserves the composite, category and regional contribution columns verbatim.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
YEARS = (2024, 2025)
OWNERSHIP_SCOPES = {"0": "total_covered", "5": "private"}
QCEW_FIELDS = [
    "area_fips",
    "state_fips",
    "ownership_scope",
    "year",
    "annual_avg_establishments",
    "annual_avg_employment",
    "total_annual_wages_usd",
    "annual_avg_weekly_wage_usd",
    "average_annual_pay_usd",
    "over_year_employment_change_pct",
    "over_year_average_annual_pay_change_pct",
    "disclosure_code",
    "over_year_disclosure_code",
    "source_archive",
]
OFR_FSI_FIELDS = [
    "date",
    "ofr_fsi",
    "credit",
    "equity_valuation",
    "safe_assets",
    "funding",
    "volatility",
    "united_states",
    "other_advanced_economies",
    "emerging_markets",
    "source_file",
]
OFR_SOURCE_FIELDS = {
    "Date": "date",
    "OFR FSI": "ofr_fsi",
    "Credit": "credit",
    "Equity valuation": "equity_valuation",
    "Safe assets": "safe_assets",
    "Funding": "funding",
    "Volatility": "volatility",
    "United States": "united_states",
    "Other advanced economies": "other_advanced_economies",
    "Emerging markets": "emerging_markets",
}


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


def build_qcew_state_annual_totals() -> dict[str, object]:
    root = DATASETS / "bls_qcew"
    rows: list[dict[str, str]] = []
    for year in YEARS:
        archive_path = RAW_DIR / "ofr_market_stress" / f"{year}_annual_singlefile.zip"
        with ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"expected one QCEW CSV in {archive_path.name}")
            with io.TextIOWrapper(
                archive.open(members[0]), encoding="utf-8-sig", newline=""
            ) as handle:
                for source in csv.DictReader(handle):
                    area = _clean(source["area_fips"])
                    ownership = _clean(source["own_code"])
                    if not re.fullmatch(r"\d{2}000", area):
                        continue
                    if ownership not in OWNERSHIP_SCOPES:
                        continue
                    if _clean(source["industry_code"]) != "10":
                        continue
                    if _clean(source["qtr"]) != "A":
                        continue
                    rows.append({
                        "area_fips": area,
                        "state_fips": area[:2],
                        "ownership_scope": OWNERSHIP_SCOPES[ownership],
                        "year": _clean(source["year"]),
                        "annual_avg_establishments": _clean(source["annual_avg_estabs"]),
                        "annual_avg_employment": _clean(source["annual_avg_emplvl"]),
                        "total_annual_wages_usd": _clean(source["total_annual_wages"]),
                        "annual_avg_weekly_wage_usd": _clean(source["annual_avg_wkly_wage"]),
                        "average_annual_pay_usd": _clean(source["avg_annual_pay"]),
                        "over_year_employment_change_pct": _clean(source["oty_annual_avg_emplvl_pct_chg"]),
                        "over_year_average_annual_pay_change_pct": _clean(source["oty_avg_annual_pay_pct_chg"]),
                        "disclosure_code": _clean(source["disclosure_code"]),
                        "over_year_disclosure_code": _clean(source["oty_disclosure_code"]),
                        "source_archive": archive_path.name,
                    })
    rows.sort(key=lambda row: (row["year"], row["area_fips"], row["ownership_scope"]))
    keys = {(row["area_fips"], row["ownership_scope"], row["year"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate QCEW state-year-ownership rows")
    expected = len(YEARS) * len(OWNERSHIP_SCOPES) * 53
    if len(rows) != expected:
        raise ValueError(f"expected {expected} QCEW state total rows, found {len(rows)}")
    output = root / "state_annual_totals.csv.gz"
    _write_deterministic_gzip_csv(output, QCEW_FIELDS, rows)
    return {
        "rows": len(rows),
        "years": list(YEARS),
        "ownership_scopes": sorted(OWNERSHIP_SCOPES.values()),
        "output": str(output.relative_to(DATASETS)),
    }


def build_ofr_financial_stress_index() -> dict[str, object]:
    root = DATASETS / "ofr_market_stress"
    source_path = RAW_DIR / "ofr_market_stress" / "fsi.csv"
    rows: list[dict[str, str]] = []
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(OFR_SOURCE_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"OFR FSI source is missing fields {sorted(missing)}")
        for source in reader:
            date = _clean(source["Date"])
            if date > "2025-12-31":
                continue
            row = {
                target: _clean(source[original])
                for original, target in OFR_SOURCE_FIELDS.items()
            }
            row["source_file"] = source_path.name
            if any(not row[field] for field in OFR_FSI_FIELDS[:-1]):
                raise ValueError(f"OFR FSI row has a missing value on {date}")
            rows.append(row)
    rows.sort(key=lambda row: row["date"])
    dates = [row["date"] for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate dates in cutoff-safe OFR FSI table")
    if len(rows) != 6575 or dates[0] != "2000-01-03" or dates[-1] != "2025-12-31":
        raise ValueError("unexpected cutoff-safe OFR FSI coverage")
    output = root / "financial_stress_index.csv.gz"
    _write_deterministic_gzip_csv(output, OFR_FSI_FIELDS, rows)
    return {
        "rows": len(rows),
        "start_date": dates[0],
        "end_date": dates[-1],
        "output": str(output.relative_to(DATASETS)),
        "vintage": "current revised history observed 2026-08-13",
    }


def main() -> None:
    report = {
        "bls_qcew_state_annual_totals": build_qcew_state_annual_totals(),
        "ofr_financial_stress_index": build_ofr_financial_stress_index(),
        "rules": [
            "Only five-digit state aggregate areas ending 000 are retained.",
            "Only annual all-industry rows for total covered and private ownership are retained.",
            "Suppressed fields remain blank; no values are imputed.",
            "State labels are not attached here; use an explicit Census geography crosswalk.",
            "The frozen 2025 annual QCEW input is a preliminary retrospective vintage; BLS finalizes annual averages with the subsequent year's preliminary first-quarter release.",
            "OFR FSI observations after 2025-12-31 are excluded; retained history is the current revised snapshot observed on 2026-08-13, not point-in-time vintages.",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase5_economic_table_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
