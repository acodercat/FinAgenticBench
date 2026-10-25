"""Download and normalize the frozen twenty-company SEC EDGAR benchmark dataset.

The SEC's current ticker map is useful discovery metadata, but it is not a
point-in-time entity map.  The benchmark therefore pins the CIKs that were the
public-company registrants at the freeze date and records any later ticker-map
drift instead of silently following it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import DATA_DIR, PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "datasets_raw" / "sec"
FREEZE_DATE = date(2025, 12, 31)
MIN_PERIOD_END = date(2021, 1, 1)
TARGET_COMPANIES = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "AMZN": "technology", "GOOGL": "technology",
    "WMT": "consumer", "COST": "consumer", "HD": "consumer",
    "KO": "consumer", "NKE": "consumer",
    "CAT": "industrial", "GE": "industrial", "UPS": "industrial",
    "XOM": "energy", "CVX": "energy",
    "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
    "JPM": "banking", "BAC": "banking",
}
# SEC registrant identities for the configured tickers as of FREEZE_DATE.  CIKs
# are durable EDGAR identifiers, whereas a ticker can be reassigned to a legal
# successor after the benchmark cutoff (XOM did so in July 2026).
FROZEN_COMPANY_IDENTITIES = {
    "AAPL": (320193, "Apple Inc.", "Nasdaq"),
    "MSFT": (789019, "MICROSOFT CORP", "Nasdaq"),
    "NVDA": (1045810, "NVIDIA CORP", "Nasdaq"),
    "AMZN": (1018724, "AMAZON COM INC", "Nasdaq"),
    "GOOGL": (1652044, "Alphabet Inc.", "Nasdaq"),
    "WMT": (104169, "Walmart Inc.", "Nasdaq"),
    "COST": (909832, "COSTCO WHOLESALE CORP /NEW", "Nasdaq"),
    "HD": (354950, "HOME DEPOT, INC.", "NYSE"),
    "KO": (21344, "COCA COLA CO", "NYSE"),
    "NKE": (320187, "NIKE, Inc.", "NYSE"),
    "CAT": (18230, "CATERPILLAR INC", "NYSE"),
    "GE": (40545, "GENERAL ELECTRIC CO", "NYSE"),
    "UPS": (1090727, "UNITED PARCEL SERVICE INC", "NYSE"),
    "XOM": (34088, "EXXON MOBIL CORP", "NYSE"),
    "CVX": (93410, "CHEVRON CORP", "NYSE"),
    "JNJ": (200406, "JOHNSON & JOHNSON", "NYSE"),
    "PFE": (78003, "PFIZER INC", "NYSE"),
    "UNH": (731766, "UNITEDHEALTH GROUP INC", "NYSE"),
    "JPM": (19617, "JPMORGAN CHASE & CO", "NYSE"),
    "BAC": (70858, "BANK OF AMERICA CORP /DE/", "NYSE"),
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


def _download(session: requests.Session, url: str, path: Path) -> dict:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.15)
    return payload


def _date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _resolve_companies(payload: dict) -> tuple[dict[str, dict], list[dict]]:
    """Resolve the cohort using frozen CIKs and audit the current ticker map."""
    fields = payload["fields"]
    available = {row[fields.index("ticker")]: dict(zip(fields, row)) for row in payload["data"]}
    missing = sorted(set(TARGET_COMPANIES) - set(available))
    if missing:
        raise ValueError(f"SEC ticker mapping is missing configured tickers: {missing}")
    if set(TARGET_COMPANIES) != set(FROZEN_COMPANY_IDENTITIES):
        raise ValueError("Every target ticker must have exactly one frozen identity")

    companies = {}
    mapping_drift = []
    for ticker, sector in TARGET_COMPANIES.items():
        cik, name, exchange = FROZEN_COMPANY_IDENTITIES[ticker]
        current = available[ticker]
        companies[ticker] = {
            "cik": cik, "name": name, "sector": sector, "exchange": exchange,
        }
        if int(current["cik"]) != cik:
            mapping_drift.append({
                "ticker": ticker,
                "freeze_date_cik": cik,
                "freeze_date_name": name,
                "current_mapping_cik": int(current["cik"]),
                "current_mapping_name": current["name"],
                "reason": "current SEC ticker mapping differs from the frozen registrant",
            })
    return companies, mapping_drift


def _history_files(payload: dict) -> list[dict]:
    """Return submission archive descriptors that can intersect our window."""
    selected = []
    for descriptor in payload.get("filings", {}).get("files", []):
        filing_from = _date(descriptor.get("filingFrom"))
        filing_to = _date(descriptor.get("filingTo"))
        if filing_from is None or filing_to is None:
            raise ValueError(f"Invalid SEC submission archive descriptor: {descriptor}")
        if filing_to >= MIN_PERIOD_END and filing_from <= FREEZE_DATE:
            selected.append(descriptor)
    return selected


def _facts(ticker: str, company: dict, payload: dict) -> list[dict]:
    rows = []
    for taxonomy, concepts in payload.get("facts", {}).items():
        for concept, definition in concepts.items():
            for unit, observations in definition.get("units", {}).items():
                for obs in observations:
                    end = _date(obs.get("end"))
                    filed = _date(obs.get("filed"))
                    if end is None or end < MIN_PERIOD_END:
                        continue
                    if filed is None or filed > FREEZE_DATE:
                        continue
                    if obs.get("form") not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
                        continue
                    value = obs.get("val")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    rows.append({
                        "cik": company["cik"], "ticker": ticker,
                        "company_name": company["name"], "taxonomy": taxonomy,
                        "concept": concept, "label": definition.get("label"),
                        "description": definition.get("description"), "unit": unit,
                        "value": value, "start_date": obs.get("start"),
                        "end_date": obs.get("end"), "filed_date": obs.get("filed"),
                        "form": obs.get("form"), "fiscal_year": obs.get("fy"),
                        "fiscal_period": obs.get("fp"), "frame": obs.get("frame"),
                        "accession": obs.get("accn"),
                    })
    return rows


def _filings(
    ticker: str,
    company: dict,
    payload: dict,
    *,
    fiscal_year_end: str | None,
) -> list[dict]:
    # The primary submissions response nests rows under filings.recent; archive
    # files are directly columnar at their top level.
    recent = payload.get("filings", {}).get("recent", payload)
    rows = []
    for index in range(len(recent.get("accessionNumber", []))):
        item = {key: values[index] for key, values in recent.items()}
        filed = _date(item.get("filingDate"))
        reported = _date(item.get("reportDate"))
        form = item.get("form")
        if form not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
            continue
        if filed is None or filed > FREEZE_DATE:
            continue
        if reported is None or reported < MIN_PERIOD_END:
            continue
        accession = item["accessionNumber"]
        document = item.get("primaryDocument")
        rows.append({
            "cik": company["cik"], "ticker": ticker,
            "company_name": company["name"], "accession": accession,
            "form": form, "filing_date": item.get("filingDate"),
            "report_date": item.get("reportDate"),
            "acceptance_datetime": item.get("acceptanceDateTime"),
            "fiscal_year_end": fiscal_year_end,
            "primary_document": document, "is_amendment": form.endswith("/A"),
            "source_url": "https://www.sec.gov/Archives/edgar/data/"
            f"{company['cik']}/{accession.replace('-', '')}/{document}",
        })
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    session = requests.Session()
    session.headers.update(headers)
    source_files: dict[Path, str] = {}
    fact_rows, filing_rows = [], []

    ticker_path = RAW_DIR / "company_tickers_exchange.json"
    tickers = requests.Session()
    tickers.headers.update(headers)
    ticker_url = "https://www.sec.gov/files/company_tickers_exchange.json"
    ticker_payload = _download(tickers, ticker_url, ticker_path)
    source_files[ticker_path] = ticker_url
    companies, mapping_drift = _resolve_companies(ticker_payload)

    for ticker, company in companies.items():
        cik = f"{company['cik']:010d}"
        company_dir = RAW_DIR / ticker
        submissions_path = company_dir / "submissions.json"
        facts_path = company_dir / "companyfacts.json"
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        submissions = _download(session, submissions_url, submissions_path)
        companyfacts = _download(session, facts_url, facts_path)
        source_files[submissions_path] = submissions_url
        source_files[facts_path] = facts_url
        fiscal_year_end = submissions.get("fiscalYearEnd")
        filing_rows.extend(
            _filings(
                ticker, company, submissions, fiscal_year_end=fiscal_year_end
            )
        )
        for descriptor in _history_files(submissions):
            history_name = descriptor["name"]
            history_path = company_dir / "submissions_history" / history_name
            history_url = f"https://data.sec.gov/submissions/{history_name}"
            history = _download(session, history_url, history_path)
            source_files[history_path] = history_url
            filing_rows.extend(
                _filings(
                    ticker, company, history, fiscal_year_end=fiscal_year_end
                )
            )
        fact_rows.extend(_facts(ticker, company, companyfacts))

    facts_df = pd.DataFrame(fact_rows).sort_values(
        ["ticker", "filed_date", "concept", "end_date", "accession"]
    )
    filings_df = (
        pd.DataFrame(filing_rows)
        .drop_duplicates(subset=["ticker", "accession"], keep="last")
        .sort_values(["ticker", "filing_date", "accession"])
    )
    companies_df = pd.DataFrame(
        [{"ticker": ticker, **company} for ticker, company in companies.items()]
    )
    facts_df.to_csv(DATA_DIR / "facts.csv.gz", index=False, compression="gzip")
    filings_df.to_csv(DATA_DIR / "filings.csv", index=False)
    companies_df.to_csv(DATA_DIR / "companies.csv", index=False)

    manifest = {
        "dataset": "SEC EDGAR 20-company benchmark cohort",
        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "freeze_date": FREEZE_DATE.isoformat(),
        "minimum_period_end": MIN_PERIOD_END.isoformat(),
        "companies": companies,
        "entity_resolution": {
            "method": "CIKs pinned to the public-company registrants at the benchmark freeze date",
            "current_ticker_mapping_drift": mapping_drift,
        },
        "source_documentation": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        "normalized_files": {
            "facts.csv.gz": {"rows": len(facts_df)},
            "filings.csv": {"rows": len(filings_df)},
            "companies.csv": {"rows": len(companies_df)},
        },
        "raw_files": {
            str(path.relative_to(DATA_DIR)): {
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path, url in source_files.items()
        },
    }
    (DATA_DIR / "sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_size = sum(path.stat().st_size for path in source_files)
    normalized_size = sum(
        (DATA_DIR / name).stat().st_size
        for name in ("facts.csv.gz", "filings.csv", "companies.csv")
    )
    print(f"companies={len(companies)} facts={len(facts_df):,} filings={len(filings_df):,}")
    print(f"raw={raw_size / 2**20:.2f} MiB normalized={normalized_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
