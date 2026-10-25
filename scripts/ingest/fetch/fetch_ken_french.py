"""Download and normalize frozen Kenneth French daily benchmark tables."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from config import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "datasets_build" / "ken_french"
RAW_DIR_ROOT = PROJECT_ROOT / "datasets_raw" / "ken_french"
RAW_DIR = RAW_DIR_ROOT
FREEZE_DATE = date(2025, 12, 31)
SOURCES = {
    "daily_factors": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip",
        "raw_name": "F-F_Research_Data_Factors_daily_CSV.zip",
        "output_name": "daily_factors.csv.gz",
        "columns": ["date", "mkt_rf_pct", "smb_pct", "hml_pct", "rf_pct"],
    },
    "daily_industry_returns": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "10_Industry_Portfolios_daily_CSV.zip",
        "raw_name": "10_Industry_Portfolios_daily_CSV.zip",
        "output_name": "daily_industry_returns.csv.gz",
        "columns": [
            "date", "nodur_pct", "durbl_pct", "manuf_pct", "enrgy_pct",
            "hitec_pct", "telcm_pct", "shops_pct", "hlth_pct", "utils_pct",
            "other_pct",
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def _first_daily_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the first contiguous YYYYMMDD table and ignore later text/tables."""
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"expected one CSV in {path}, found {names}")
        text = archive.read(names[0]).decode("utf-8-sig")

    rows = []
    started = False
    for line in text.splitlines():
        if re.match(r"^\d{8},", line):
            started = True
            rows.append(line)
        elif started:
            break
    if not rows:
        raise ValueError(f"no daily rows found in {path}")

    frame = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=columns)
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    frame = frame.loc[frame["date"].dt.date <= FREEZE_DATE].copy()
    numeric = columns[1:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame = frame.loc[~frame[numeric].isin([-99.99, -999.0]).any(axis=1)]
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"dates are not unique and increasing in {path}")
    return frame


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    normalized = {}
    raw_files = {}
    for name, spec in SOURCES.items():
        raw_path = RAW_DIR / spec["raw_name"]
        _download(spec["url"], raw_path)
        frame = _first_daily_table(raw_path, spec["columns"])
        output_path = DATA_DIR / spec["output_name"]
        frame.to_csv(output_path, index=False, compression="gzip")
        normalized[spec["output_name"]] = {
            "rows": len(frame),
            "minimum_date": frame["date"].min(),
            "maximum_date": frame["date"].max(),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
        raw_files[spec["raw_name"]] = {
            "url": spec["url"],
            "bytes": raw_path.stat().st_size,
            "sha256": _sha256(raw_path),
        }
        print(f"{name}: rows={len(frame):,} through {frame['date'].max()}")

    manifest = {
        "dataset": "Kenneth French daily research factors and 10 industry portfolios",
        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "freeze_date": FREEZE_DATE.isoformat(),
        "units": "daily simple returns in percent, not decimal fractions",
        "industry_weighting": "value weighted",
        "source_vintage": (
            "Retrospective library snapshot based on the 202606 CRSP database, "
            "retaining observations through the 2025-12-31 observation cutoff."
        ),
        "source_documentation": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "raw_files": raw_files,
        "normalized_files": normalized,
    }
    (DATA_DIR / "sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
