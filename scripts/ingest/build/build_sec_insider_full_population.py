"""Rebuild the SEC insider tables from every quarterly data set, 2019–2025.

The cohort build kept two quarters (2025 Q3–Q4) for 20 issuers. This build
reads all 28 Insider Transactions Data Sets in `sec_insider/raw/` and keeps
every Form 3/4/5 filed by an issuer in the deep-tier universe — the 500
largest filers written to `sec/companies.csv` by `build_sec_full_population`.
The scope rule is the same one that bounds `sec.company_facts`, so the two
tables describe the same companies; all issuers would be roughly ten million
rows across seven tables, more than one runtime can hold.

Table shapes are unchanged: the seven TSV members of each quarterly ZIP,
lower-cased columns, `source_quarter` appended, duplicates dropped, sorted by
accession. `OWNER_SIGNATURE` stays out of the runtime as before.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
OUT_DIR = DATASETS / "sec_insider"
RAW_DIR = RAW_DIR / "sec_insider"
UNIVERSE = DATASETS / "sec" / "companies.csv"
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "sec_insider_full_population_build.json"
FREEZE_DATE = "2025-12-31"
RUNTIME_TABLES = (
    "submission", "reportingowner", "nonderiv_trans", "nonderiv_holding",
    "deriv_trans", "deriv_holding", "footnotes",
)


def _quarters() -> list[Path]:
    paths = sorted(RAW_DIR.glob("*_form345.zip"))
    return [p for p in paths if "2019q1" <= p.name[:6] <= "2025q4"]


def _read_tables(path: Path) -> dict[str, pd.DataFrame]:
    quarter = path.name[:6]
    tables = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".tsv"):
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
    universe = pd.read_csv(UNIVERSE, dtype={"cik": str})
    universe_ciks = set(universe["cik"].astype(str).str.lstrip("0"))
    quarters = _quarters()
    if len(quarters) != 28:
        raise ValueError(f"expected 28 quarterly archives 2019q1..2025q4, found {len(quarters)}")

    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in RUNTIME_TABLES}
    per_quarter = {}
    for path in quarters:
        tables = _read_tables(path)
        submissions = tables["submission"]
        issuer = submissions["issuercik"].astype(str).str.lstrip("0")
        filed = submissions["filing_date"].str.slice(0, 10)
        keep = issuer.isin(universe_ciks) & (pd.to_datetime(filed, errors="coerce") <= FREEZE_DATE)
        accessions = set(submissions.loc[keep, "accession_number"])
        per_quarter[path.name[:6]] = {"submissions_total": len(submissions), "submissions_kept": len(accessions)}
        for name in RUNTIME_TABLES:
            frame = tables[name]
            collected[name].append(frame.loc[frame["accession_number"].isin(accessions)])
        print(f"{path.name[:6]}: {len(accessions):,} of {len(submissions):,} submissions in universe", flush=True)

    normalized = {}
    for name in RUNTIME_TABLES:
        frame = pd.concat(collected[name], ignore_index=True).drop_duplicates()
        frame = frame.sort_values(
            [column for column in ("accession_number", "source_quarter") if column in frame]
        )
        output = OUT_DIR / f"{name}.csv.gz"
        frame.to_csv(output, index=False, compression="gzip")
        normalized[name] = {"file": output.name, "rows": len(frame), "columns": list(frame.columns)}
        print(f"  {name}: {len(frame):,} rows", flush=True)

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "freeze_date": FREEZE_DATE,
        "quarters": [p.name[:6] for p in quarters],
        "scope_rule": "Forms 3/4/5 whose issuer CIK is in the deep-tier universe (sec/companies.csv)",
        "universe_size": len(universe_ciks),
        "per_quarter": per_quarter,
        "normalized_tables": normalized,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["rows"] for k, v in normalized.items()}))


if __name__ == "__main__":
    main()
