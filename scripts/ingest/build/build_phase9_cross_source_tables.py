"""Build governed inputs for the ninth cross-source batch.

The EIA tables preserve fixed STEO publication vintages separately from the
current revised monthly and daily petroleum-series snapshots.  The FEMA projection retains every
2024-2025 declaration-area row through the benchmark cutoff; downstream cases
must still select declaration type, incident window and distinct county grain.
"""

from __future__ import annotations

import csv
from datetime import datetime
import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
MONTHS = {
    name: number
    for number, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def _publication_date(text: object) -> str:
    value = str(text).replace(",", "")
    _, date_text = value.split(" ", maxsplit=1)
    return datetime.strptime(date_text, "%B %d %Y").date().isoformat()


def build_steo_wti_vintages() -> dict[str, object]:
    root = DATASETS / "eia_steo_vintages"
    rows: list[dict[str, object]] = []
    source_paths = sorted((RAW_DIR / "fema_disasters").glob("*24_base.xlsx")) + sorted(
        (RAW_DIR / "fema_disasters").glob("*25_base.xlsx")
    )
    if len(source_paths) != 24:
        raise ValueError(f"expected 24 STEO vintage workbooks, found {len(source_paths)}")
    for source_path in source_paths:
        worksheet = load_workbook(
            source_path, read_only=True, data_only=True
        )["2tab"]
        publication_date = _publication_date(worksheet.cell(4, 1).value)
        year_row = [cell.value for cell in worksheet[3]]
        month_row = [cell.value for cell in worksheet[4]]
        current_year: int | None = None
        target_columns: dict[str, int] = {}
        for index, (year_value, month_value) in enumerate(
            zip(year_row, month_row), start=1
        ):
            if isinstance(year_value, (int, float)) and 2000 <= int(year_value) <= 2100:
                current_year = int(year_value)
            month = MONTHS.get(str(month_value).strip()[:3].title()) if month_value else None
            if current_year in {2024, 2025} and month:
                target_columns[f"{current_year}-{month:02d}"] = index
        if len(target_columns) != 24:
            raise ValueError(
                f"expected 24 target months in {source_path.name}, found {len(target_columns)}"
            )
        matches = [
            row for row in worksheet.iter_rows(min_row=5) if row[0].value == "WTIPUUS"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one WTIPUUS row in {source_path.name}, found {len(matches)}"
            )
        series_row = matches[0]
        publication_month = publication_date[:7]
        for target_month, column in sorted(target_columns.items()):
            value = series_row[column - 1].value
            if value is None:
                raise ValueError(
                    f"missing WTIPUUS value for {target_month} in {source_path.name}"
                )
            rows.append(
                {
                    "publication_date": publication_date,
                    "publication_month": publication_month,
                    "target_month": target_month,
                    "value_status_in_vintage": (
                        "forecast" if target_month >= publication_month else "history"
                    ),
                    "series_code": "WTIPUUS",
                    "series_name": "West Texas Intermediate Spot Average",
                    "wti_spot_average_usd_per_barrel": float(value),
                    "source_workbook": source_path.name,
                    "source_sheet": "2tab",
                }
            )
    rows.sort(key=lambda row: (str(row["publication_date"]), str(row["target_month"])))
    keys = [(row["publication_date"], row["target_month"]) for row in rows]
    if len(rows) != 576 or len(keys) != len(set(keys)):
        raise ValueError("unexpected STEO WTI vintage panel")
    output = root / "wti_monthly_vintages.csv.gz"
    _write(output, list(rows[0]), rows)
    return {
        "rows": len(rows),
        "publication_vintages": len({row["publication_date"] for row in rows}),
        "target_months": len({row["target_month"] for row in rows}),
        "output": str(output.relative_to(DATASETS)),
    }


def build_pet_wti_monthly_actuals() -> dict[str, object]:
    root = DATASETS / "eia_bulk"
    source_path = RAW_DIR / "fema_disasters" / "PET.zip"
    selected: dict[str, object] | None = None
    with ZipFile(source_path) as archive:
        with archive.open("PET.txt") as handle:
            for raw in handle:
                if b'"series_id":"PET.RWTC.M"' not in raw:
                    continue
                candidate = json.loads(raw)
                if selected is not None:
                    raise ValueError("duplicate PET.RWTC.M series")
                selected = candidate
    if selected is None:
        raise ValueError("PET.RWTC.M was not found in PET.zip")
    rows = [
        {
            "observation_month": f"{period[:4]}-{period[4:]}",
            "series_id": selected["series_id"],
            "series_name": selected["name"],
            "wti_spot_price_usd_per_barrel": value,
            "snapshot_last_updated": str(selected["last_updated"]),
            "source_archive": source_path.name,
            "source_member": "PET.txt",
        }
        for period, value in selected["data"]
        if period <= "202512" and value is not None
    ]
    rows.sort(key=lambda row: str(row["observation_month"]))
    if (
        len(rows) != 480
        or rows[0]["observation_month"] != "1986-01"
        or rows[-1]["observation_month"] != "2025-12"
    ):
        raise ValueError("unexpected cutoff-safe PET.RWTC.M coverage")
    output = root / "wti_monthly_spot_prices.csv.gz"
    _write(output, list(rows[0]), rows)
    return {
        "rows": len(rows),
        "start_month": rows[0]["observation_month"],
        "end_month": rows[-1]["observation_month"],
        "series_id": "PET.RWTC.M",
        "output": str(output.relative_to(DATASETS)),
        "vintage": "current revised bulk snapshot downloaded 2026-08-12",
    }


def build_pet_wti_daily_actuals() -> dict[str, object]:
    root = DATASETS / "eia_bulk"
    source_path = RAW_DIR / "fema_disasters" / "PET.zip"
    selected: dict[str, object] | None = None
    with ZipFile(source_path) as archive:
        with archive.open("PET.txt") as handle:
            for raw in handle:
                if b'"series_id":"PET.RWTC.D"' not in raw:
                    continue
                candidate = json.loads(raw)
                if selected is not None:
                    raise ValueError("duplicate PET.RWTC.D series")
                selected = candidate
    if selected is None:
        raise ValueError("PET.RWTC.D was not found in PET.zip")
    rows = [
        {
            "observation_date": f"{period[:4]}-{period[4:6]}-{period[6:]}",
            "series_id": selected["series_id"],
            "series_name": selected["name"],
            "wti_spot_price_usd_per_barrel": value,
            "snapshot_last_updated": str(selected["last_updated"]),
            "source_archive": source_path.name,
            "source_member": "PET.txt",
        }
        for period, value in selected["data"]
        if period <= "20251231" and value is not None
    ]
    rows.sort(key=lambda row: str(row["observation_date"]))
    if (
        len(rows) != 10091
        or rows[0]["observation_date"] != "1986-01-02"
        or rows[-1]["observation_date"] != "2025-12-31"
    ):
        raise ValueError("unexpected cutoff-safe PET.RWTC.D coverage")
    output = root / "wti_daily_spot_prices.csv.gz"
    _write(output, list(rows[0]), rows)
    return {
        "rows": len(rows),
        "start_date": rows[0]["observation_date"],
        "end_date": rows[-1]["observation_date"],
        "series_id": "PET.RWTC.D",
        "output": str(output.relative_to(DATASETS)),
        "vintage": "current revised bulk snapshot downloaded 2026-08-12",
    }


def build_fema_declaration_areas() -> dict[str, object]:
    root = DATASETS / "fema_disasters"
    source_path = RAW_DIR / "fema_disasters" / "DisasterDeclarationsSummaries.parquet"
    source = pd.read_parquet(source_path)
    declaration_dates = pd.to_datetime(source["declarationDate"], errors="raise")
    selected = source.loc[
        declaration_dates.between("2024-01-01", "2025-12-31")
    ].copy()
    fields = [
        "id",
        "disasterNumber",
        "state",
        "declarationType",
        "declarationDate",
        "incidentType",
        "incidentBeginDate",
        "incidentEndDate",
        "tribalRequest",
        "fipsStateCode",
        "fipsCountyCode",
        "designatedArea",
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
        "lastRefresh",
    ]
    selected = selected[fields].rename(
        columns={
            "id": "declaration_area_id",
            "disasterNumber": "disaster_number",
            "state": "state_usps",
            "declarationType": "declaration_type",
            "declarationDate": "declaration_date",
            "incidentType": "incident_type",
            "incidentBeginDate": "incident_begin_date",
            "incidentEndDate": "incident_end_date",
            "tribalRequest": "tribal_request",
            "fipsStateCode": "fips_state_code",
            "fipsCountyCode": "fips_county_code",
            "designatedArea": "designated_area",
            "ihProgramDeclared": "individuals_households_program_declared",
            "iaProgramDeclared": "individual_assistance_declared",
            "paProgramDeclared": "public_assistance_declared",
            "hmProgramDeclared": "hazard_mitigation_declared",
            "lastRefresh": "snapshot_last_refresh",
        }
    )
    selected["county_fips"] = (
        selected["fips_state_code"].astype(str).str.zfill(2)
        + selected["fips_county_code"].astype(str).str.zfill(3)
    )
    for column in ("declaration_date", "incident_begin_date", "incident_end_date"):
        selected[column] = pd.to_datetime(selected[column]).dt.strftime("%Y-%m-%d")
    selected["snapshot_last_refresh"] = pd.to_datetime(
        selected["snapshot_last_refresh"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    selected["source_file"] = source_path.name
    selected = selected.sort_values(
        ["declaration_date", "disaster_number", "declaration_area_id"]
    )
    if len(selected) != 3567 or selected["declaration_area_id"].duplicated().any():
        raise ValueError("unexpected FEMA 2024-2025 declaration-area population")
    output = root / "declaration_areas_2024_2025.csv.gz"
    rows = selected.astype(object).where(selected.notna(), "").to_dict("records")
    _write(output, list(selected.columns), rows)
    return {
        "rows": len(selected),
        "disaster_numbers": int(selected["disaster_number"].nunique()),
        "start_declaration_date": selected["declaration_date"].min(),
        "end_declaration_date": selected["declaration_date"].max(),
        "output": str(output.relative_to(DATASETS)),
        "vintage": "current OpenFEMA snapshot downloaded 2026-08-12",
    }


def main() -> None:
    report = {
        "eia_steo_wti_vintages": build_steo_wti_vintages(),
        "eia_pet_wti_monthly_actuals": build_pet_wti_monthly_actuals(),
        "eia_pet_wti_daily_actuals": build_pet_wti_daily_actuals(),
        "fema_declaration_areas": build_fema_declaration_areas(),
        "rules": [
            "STEO rows retain both publication date and target month; later-vintage history is never substituted for an earlier forecast.",
            "PET.RWTC.M is a current revised monthly spot-price snapshot filtered to observations through 2025-12.",
            "PET.RWTC.D is a current revised daily spot-price snapshot filtered to observations through 2025-12-31; non-observation dates are not synthesized.",
            "FEMA rows preserve declaration-area grain and repeated counties across disaster numbers; cases must explicitly choose declaration type, time window and deduplication grain.",
            "FEMA declaration status is administrative evidence, not a dollar-loss or causal-impact measure.",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase9_cross_source_table_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
