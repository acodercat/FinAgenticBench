"""Build governed balance-sheet inputs for the tenth cross-source case batch.

The projections retain full quarterly reporter populations and source-native
legal-entity identifiers.  They do not pre-link banks to holding companies or
precompute any case cohort, lineage, ratio, ranking or answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=GZIP)


def _report_date(stamp: str) -> str:
    return pd.to_datetime(stamp, format="%m%d%Y").strftime("%Y-%m-%d")


def build_call_report_balance_sheets() -> dict[str, object]:
    root = DATASETS / "ffiec_call_reports"
    parts: list[pd.DataFrame] = []
    source_archives: list[str] = []
    for source_path in sorted((RAW_DIR / "ffiec_y9c").glob("*.zip")):
        match = re.search(r"(\d{8})", source_path.name)
        if match is None:
            raise ValueError(f"missing report-date stamp in {source_path.name}")
        stamp = match.group(1)
        if stamp == "12312025":
            # The 2025-Q4 archive was first published after the benchmark cutoff.
            continue
        with ZipFile(source_path) as archive:
            por = pd.read_csv(
                archive.open(f"FFIEC CDR Call Bulk POR {stamp}.txt"),
                sep="\t",
                dtype=str,
                low_memory=False,
            )
            balance = pd.read_csv(
                archive.open(f"FFIEC CDR Call Schedule RC {stamp}.txt"),
                sep="\t",
                skiprows=[1],
                dtype=str,
                low_memory=False,
            )
        selected = balance[
            [
                "IDRSSD",
                "RCFD2170",
                "RCON2170",
                "RCFD2948",
                "RCON2948",
                "RCFD3210",
                "RCON3210",
            ]
        ].copy()
        joined = por[
            [
                "IDRSSD",
                "FDIC Certificate Number",
                "Financial Institution Name",
                "Financial Institution State",
                "Financial Institution Filing Type",
                "Last Date/Time Submission Updated On",
            ]
        ].merge(selected, on="IDRSSD", how="inner", validate="one_to_one")
        joined.insert(1, "report_date", _report_date(stamp))
        joined["source_archive"] = source_path.name
        parts.append(joined)
        source_archives.append(source_path.name)
    result = pd.concat(parts, ignore_index=True).rename(
        columns={
            "IDRSSD": "bank_rssd_id",
            "FDIC Certificate Number": "fdic_certificate",
            "Financial Institution Name": "bank_legal_name",
            "Financial Institution State": "bank_state",
            "Financial Institution Filing Type": "filing_type",
            "Last Date/Time Submission Updated On": "last_submission_update",
            "RCFD2170": "consolidated_total_assets_thousand_usd",
            "RCON2170": "domestic_office_total_assets_thousand_usd",
            "RCFD2948": "consolidated_total_liabilities_thousand_usd",
            "RCON2948": "domestic_office_total_liabilities_thousand_usd",
            "RCFD3210": "consolidated_total_equity_thousand_usd",
            "RCON3210": "domestic_office_total_equity_thousand_usd",
        }
    )
    identifier_columns = ["bank_rssd_id", "fdic_certificate", "filing_type"]
    for column in identifier_columns:
        result[column] = result[column].astype("string")
    numeric_columns = [column for column in result if column.endswith("_thousand_usd")]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values(["report_date", "bank_rssd_id"]).reset_index(drop=True)
    if result.duplicated(["report_date", "bank_rssd_id"]).any():
        raise ValueError("duplicate Call Report balance-sheet key")
    if len(source_archives) != 7 or result.report_date.nunique() != 7:
        raise ValueError("unexpected Call Report quarterly coverage")
    output = root / "bank_balance_sheets.csv.gz"
    _write(result, output)
    return {
        "rows": len(result),
        "report_dates": sorted(result.report_date.unique().tolist()),
        "reporters_2025q3": int(result.loc[result.report_date.eq("2025-09-30"), "bank_rssd_id"].nunique()),
        "output": str(output.relative_to(DATASETS)),
        "source_archives": source_archives,
    }


def build_y9c_balance_sheets() -> dict[str, object]:
    root = DATASETS / "ffiec_y9c"
    parts: list[pd.DataFrame] = []
    source_archives: list[str] = []
    for source_path in sorted((RAW_DIR / "ffiec_y9c").glob("BHCF*.ZIP")):
        stamp = source_path.stem[4:]
        if stamp == "20251231":
            # The 2025-Q4 archive was first published after the benchmark cutoff.
            continue
        with ZipFile(source_path) as archive:
            member = f"BHCF{stamp}.txt"
            source = pd.read_csv(
                archive.open(member), sep="^", dtype=str, encoding="cp1252", low_memory=False
            )
        selected = source[
            [
                "RSSD9001",
                "RSSD9999",
                "RSSD9017",
                "BHCK2170",
                "BHCK2948",
                "BHCK3210",
            ]
        ].rename(
            columns={
                "RSSD9001": "reporter_rssd_id",
                "RSSD9999": "report_date",
                "RSSD9017": "reporter_legal_name",
                "BHCK2170": "consolidated_total_assets_thousand_usd",
                "BHCK2948": "consolidated_total_liabilities_thousand_usd",
                "BHCK3210": "consolidated_total_equity_thousand_usd",
            }
        )
        selected["report_date"] = pd.to_datetime(
            selected["report_date"], format="%Y%m%d"
        ).dt.strftime("%Y-%m-%d")
        selected["source_archive"] = source_path.name
        selected["source_member"] = member
        selected = selected.loc[
            selected.consolidated_total_assets_thousand_usd.notna()
            & selected.consolidated_total_assets_thousand_usd.ne("")
        ].copy()
        for column in (
            "consolidated_total_assets_thousand_usd",
            "consolidated_total_liabilities_thousand_usd",
            "consolidated_total_equity_thousand_usd",
        ):
            selected[column] = pd.to_numeric(selected[column], errors="raise")
        parts.append(selected)
        source_archives.append(source_path.name)
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["report_date", "reporter_rssd_id"]
    ).reset_index(drop=True)
    if result.duplicated(["report_date", "reporter_rssd_id"]).any():
        raise ValueError("duplicate Y-9C balance-sheet key")
    if len(source_archives) != 7 or result.report_date.nunique() != 7:
        raise ValueError("unexpected Y-9C quarterly coverage")
    output = root / "holding_company_balance_sheets.csv.gz"
    _write(result, output)
    return {
        "rows": len(result),
        "report_dates": sorted(result.report_date.unique().tolist()),
        "reporters_2025q3": int(result.loc[result.report_date.eq("2025-09-30"), "reporter_rssd_id"].nunique()),
        "output": str(output.relative_to(DATASETS)),
        "source_archives": source_archives,
    }


def main() -> None:
    report = {
        "ffiec_call_report_balance_sheets": build_call_report_balance_sheets(),
        "ffiec_y9c_balance_sheets": build_y9c_balance_sheets(),
        "rules": [
            "The projections retain all reporters through 2025-Q3; the post-cutoff-published 2025-Q4 archives are excluded.",
            "Call Report RCFD fields are consolidated-bank measures; RCON fields are domestic-office measures and remain separate.",
            "Y-9C BHCK fields are holding-company consolidated measures and are not additive to subsidiary-bank Call Report values.",
            "No bank-parent relationship or case cohort is precomputed; legal-entity linkage remains in the separate NIC relationship graph.",
            "The current NIC graph and historical financial archives support retrospective analysis, not a strict 2025-Q3 as-known ownership claim.",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase10_cross_source_table_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
