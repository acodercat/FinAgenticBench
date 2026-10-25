"""Build deterministic runtime tables for the Phase 3 tasks.

Raw source archives remain unchanged.  The generated gzip tables retain the
population and row grain required to expose each diagnostic path; they are not
precomputed answer tables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> tuple[Path, int]:
    frame.to_csv(path, index=False, compression=GZIP)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}: {len(frame):,} rows, {path.stat().st_size:,} bytes")
    return path, len(frame)


def build_ncua() -> tuple[Path, int]:
    source = RAW_DIR / "ncua_call_reports" / "call-report-data-2025-09.zip"
    with ZipFile(source) as archive:
        institutions = pd.read_csv(
            archive.open("FOICU.txt"),
            usecols=["CU_NUMBER", "CYCLE_DATE", "CU_NAME", "STATE"],
        )
        financials = pd.read_csv(
            archive.open("FS220.txt"),
            usecols=["CU_NUMBER", "ACCT_041B", "ACCT_025B"],
            low_memory=False,
        )
    result = institutions.merge(financials, on="CU_NUMBER", validate="one_to_one")
    result = result.rename(columns={
        "CU_NUMBER": "credit_union_number",
        "CYCLE_DATE": "report_date",
        "CU_NAME": "credit_union_name",
        "STATE": "state",
        "ACCT_041B": "delinquent_loans_two_plus_months_usd",
        "ACCT_025B": "total_loans_and_leases_usd",
    })
    result["report_date"] = pd.to_datetime(result["report_date"]).dt.strftime("%Y-%m-%d")
    for column in ("delinquent_loans_two_plus_months_usd", "total_loans_and_leases_usd"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    assert len(result) == result["credit_union_number"].nunique()
    assert result["report_date"].dropna().unique().tolist() == ["2025-09-30"]
    return _write(result, DATASETS / "ncua_call_reports" / "institutions_2025q3.csv.gz")


def build_midas() -> tuple[Path, int]:
    source = (
        RAW_DIR / "sec_midas_security_exchange"
        / "individual_security_exchange_2025_q2.zip"
    )
    with ZipFile(source) as archive:
        result = pd.read_csv(
            archive.open("apr25.csv"),
            usecols=[
                "Date", "Security", "Ticker", "Exchange", "TradeVol('000)",
                "OddLotVol('000)",
            ],
        )
    result = result.rename(columns={
        "Date": "date",
        "Security": "security_type",
        "Ticker": "ticker",
        "Exchange": "exchange",
        "TradeVol('000)": "trade_volume_thousands",
        "OddLotVol('000)": "odd_lot_volume_thousands",
    })
    result["date"] = pd.to_datetime(result["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    assert result["date"].str.startswith("2025-04-").all()
    assert result["exchange"].nunique() == 15
    return _write(result, DATASETS / "sec_midas_security_exchange" / "april_2025.csv.gz")


def build_scf() -> tuple[Path, int]:
    source = RAW_DIR / "fed_scf" / "scfp2022excel.zip"
    with ZipFile(source) as archive:
        result = pd.read_csv(
            archive.open("SCFP2022.csv"),
            usecols=["YY1", "Y1", "WGT", "AGE", "NETWORTH"],
        )
    result = result.rename(columns={
        "YY1": "household_id",
        "Y1": "record_id",
        "WGT": "survey_weight",
        "AGE": "reference_person_age",
        "NETWORTH": "net_worth_usd",
    })
    result["implicate"] = result["record_id"] % 10
    result = result[[
        "household_id", "record_id", "implicate", "survey_weight",
        "reference_person_age", "net_worth_usd",
    ]]
    assert len(result) == 22_975
    assert result.groupby("household_id")["implicate"].nunique().eq(5).all()
    return _write(result, DATASETS / "fed_scf" / "summary_2022.csv.gz")


def _raw_sha(dataset: str, filename: str) -> str:
    manifest = json.loads((DATASETS / dataset / "sources.json").read_text(encoding="utf-8"))
    record = next(row for row in manifest["raw_files"] if row["file"] == f"raw/{filename}")
    return record["sha256"]


def main() -> None:
    built = {
        "ncua_call_reports/institutions_2025q3.csv.gz": (
            build_ncua(), "ncua_call_reports", "call-report-data-2025-09.zip"
        ),
        "sec_midas_security_exchange/april_2025.csv.gz": (
            build_midas(), "sec_midas_security_exchange",
            "individual_security_exchange_2025_q2.zip",
        ),
        "fed_scf/summary_2022.csv.gz": (
            build_scf(), "fed_scf", "scfp2022excel.zip"
        ),
    }
    files = {}
    for relative, ((path, rows), dataset, raw_file) in built.items():
        files[relative] = {
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source_dataset": dataset,
            "source_file": f"raw/{raw_file}",
            "source_sha256": _raw_sha(dataset, raw_file),
        }
    manifest = {
        "schema_version": 1,
        "benchmark_cutoff": "2025-12-31",
        "builder": "scripts.ingest.build.build_phase3_runtime",
        "files": files,
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase3_runtime_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
