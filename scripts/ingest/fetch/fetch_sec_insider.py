"""Download and normalize SEC 2025 Q3-Q4 Forms 3/4/5 data for our cohort."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, PROJECT_ROOT


OUT_DIR = DATA_DIR.parent / "sec_insider"
RAW_DIR = PROJECT_ROOT / "datasets_raw" / "sec_insider"
QUARTERS = {
    "2025q3": "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2025q3_form345.zip",
    "2025q4": "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2025q4_form345.zip",
}
USER_AGENT = os.environ.get(
    "FINBENCH_SEC_USER_AGENT",
    "finagenticbench/0.1 academic benchmark (contact: repository owner)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(session: requests.Session, url: str, path: Path) -> None:
    response = session.get(url, timeout=180)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    time.sleep(0.15)


def _read_tables(path: Path, quarter: str) -> dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith((".tsv", ".txt")):
                continue
            key = Path(member).stem.lower()
            with archive.open(member) as handle:
                frame = pd.read_csv(
                    io.BytesIO(handle.read()), sep="\t", dtype=str,
                    keep_default_na=False, low_memory=False,
                )
            frame.columns = [column.lower() for column in frame.columns]
            frame["source_quarter"] = quarter
            tables[key] = frame
    return tables


def main() -> None:
    companies = pd.read_csv(DATA_DIR / "companies.csv", dtype={"cik": str})
    target_ciks = set(companies["cik"].astype(str).str.lstrip("0"))
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
    })

    raw_paths = {}
    quarterly_tables: dict[str, list[pd.DataFrame]] = {}
    for quarter, url in QUARTERS.items():
        path = RAW_DIR / f"{quarter}_form345.zip"
        _download(session, url, path)
        raw_paths[quarter] = path
        for name, frame in _read_tables(path, quarter).items():
            quarterly_tables.setdefault(name, []).append(frame)

    all_tables = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in quarterly_tables.items()
    }
    submission_key = next(
        name for name, frame in all_tables.items()
        if "issuercik" in frame.columns and "accession_number" in frame.columns
    )
    submissions = all_tables[submission_key]
    issuer_ciks = submissions["issuercik"].astype(str).str.lstrip("0")
    selected_submissions = submissions.loc[issuer_ciks.isin(target_ciks)].copy()
    selected_accessions = set(selected_submissions["accession_number"])

    normalized = {}
    for name, frame in all_tables.items():
        if "accession_number" not in frame.columns:
            continue
        selected = frame.loc[frame["accession_number"].isin(selected_accessions)].copy()
        selected = selected.drop_duplicates().sort_values(
            [column for column in ("accession_number", "source_quarter") if column in selected]
        )
        output = OUT_DIR / f"{name}.csv.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output, index=False, compression="gzip")
        normalized[name] = {
            "file": output.name,
            "rows": len(selected),
            "columns": list(selected.columns),
            "bytes": output.stat().st_size,
        }

    manifest = {
        "dataset": "SEC Insider Transactions Forms 3/4/5",
        "quarters": list(QUARTERS),
        "cohort_tickers": companies["ticker"].tolist(),
        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_documentation": "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
        "raw_files": {
            quarter: {
                "file": str(path.relative_to(OUT_DIR)),
                "url": QUARTERS[quarter],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for quarter, path in raw_paths.items()
        },
        "normalized_tables": normalized,
    }
    (OUT_DIR / "sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_size = sum(path.stat().st_size for path in raw_paths.values())
    normalized_size = sum(item["bytes"] for item in normalized.values())
    print(
        f"submissions={len(selected_submissions):,} tables={len(normalized)} "
        f"rows={sum(item['rows'] for item in normalized.values()):,}"
    )
    print(f"raw={raw_size / 2**20:.2f} MiB normalized={normalized_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
