"""Fetch the phase-one and phase-two FinAgenticBench expansion data.

The collector deliberately freezes bounded, benchmark-useful snapshots instead of
mirroring whole web sites.  Downloads are idempotent, streamed to ``.part`` files,
validated for non-empty content (and ZIP integrity where applicable), and recorded
with SHA-256 hashes in a manifest inside each dataset directory.

Usage::

    uv run python -m scripts.ingest.fetch.fetch_expansion --stage all --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin, urlparse
from zipfile import BadZipFile, ZipFile

import requests

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
FREEZE_DATE = "2025-12-31"
USER_AGENT = os.environ.get(
    "FINBENCH_DOWNLOAD_USER_AGENT",
    "finagenticbench/0.1 academic benchmark (contact: repository owner)",
)
SEC_REQUEST_INTERVAL_SECONDS = 1 / 8
_SEC_RATE_LOCK = Lock()
_sec_next_request_at = 0.0


@dataclass(frozen=True)
class Download:
    dataset: str
    url: str
    filename: str | None = None
    stage: int = 1
    note: str = ""

    @property
    def name(self) -> str:
        if self.filename:
            return self.filename
        name = Path(urlparse(self.url).path).name
        return name or "download.bin"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return session


def _respect_source_rate_limit(url: str) -> None:
    """Keep SEC request starts below its published fair-access ceiling."""
    global _sec_next_request_at
    host = (urlparse(url).hostname or "").lower()
    if host != "sec.gov" and not host.endswith(".sec.gov"):
        return
    with _SEC_RATE_LOCK:
        now = time.monotonic()
        if now < _sec_next_request_at:
            time.sleep(_sec_next_request_at - now)
            now = time.monotonic()
        _sec_next_request_at = now + SEC_REQUEST_INTERVAL_SECONDS


def _get_links(page_url: str) -> list[tuple[str, str]]:
    _respect_source_rate_limit(page_url)
    response = _session().get(page_url, timeout=120)
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.text)
    return [(urljoin(response.url, href), text) for href, text in parser.links]


def _static_downloads() -> list[Download]:
    items: list[Download] = []

    # Phase 1: detailed filing notes, macro vintages, regulatory banking,
    # stress tests, money-market funds, registered funds, and mortgage lending.
    notes_root = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets"
    for period in [*(f"2024q{q}" for q in range(1, 5)), "2025q1", "2025q2"]:
        items.append(Download("sec_notes", f"{notes_root}/{period}_notes.zip", stage=1))
    for month in range(7, 13):
        period = f"2025_{month:02d}"
        items.append(Download("sec_notes", f"{notes_root}/{period}_notes.zip", stage=1))

    rtdsm_root = (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "real-time-data/data-files/xlsx"
    )
    for series in [
        "routput", "noutput", "pcpi", "pcpix", "pconx", "employ_level",
        "employ_pct_chg", "hstarts",
    ]:
        items.append(
            Download(
                "rtdsm",
                f"{rtdsm_root}/{series}_first_second_third.xlsx?sc_lang=en",
                f"{series}_first_second_third.xlsx",
                1,
                "Philadelphia Fed first/second/third release values",
            )
        )

    fed_root = "https://www.federalreserve.gov/supervisionreg/files"
    fed_files = [
        "public_results_DFAST_2025.csv", "DFAST_data_dictionary.pdf",
        "2025-Table_1A_Historic_Domestic.csv", "2025-Table_1B_Historic_International.csv",
        "2025-Table_2A_Supervisory_Baseline_Domestic.csv",
        "2025-Table_2B_Supervisory_Baseline_International.csv",
        "2025-Table_3A_Supervisory_Severely_Adverse_Domestic.csv",
        "2025-Table_3B_Supervisory_Severely_Adverse_International.csv",
        "ccar-2025-stress-test-severely-adverse-market-shocks.xlsx",
        "ccar-2025-exploratory-market-shocks.xlsx",
        "2025-Exploratory_Macro_Conditions_Domestic.csv",
        "2025-Exploratory_Macro_Conditions_International.csv",
        "corporate-high-risk-2025.csv", "corporate-low-risk-2025.csv",
        "corporate-typical-risk-2025.csv", "cre-high-risk-2025.csv",
        "cre-low-risk-2025.csv", "cre-typical-risk-2025.csv",
        "usfirstlien-high-risk-2025.csv", "usfirstlien-low-risk-2025.csv",
        "usfirstlien-typical-risk-2025.csv", "cards-high-risk-2025.csv",
        "cards-low-risk-2025.csv", "cards-typical-risk-2025.csv",
    ]
    items.extend(Download("fed_stress_tests", f"{fed_root}/{name}", stage=1) for name in fed_files)

    nmfp = [
        "20250110-20250207", "20250210-20250307", "20250310-20250407",
        "20250408-20250507", "20250508-20250606", "20250609-20250709",
        "20250710-20250807", "20250808-20250908", "20250909-20251007",
        "20251008-20251107", "20251110-20251205", "20251208-20260108",
    ]
    nmfp_root = "https://www.sec.gov/files/dera/data/form-n-mfp-data-sets"
    items.extend(Download("sec_nmfp", f"{nmfp_root}/{p}_nmfp.zip", stage=1) for p in nmfp)

    nport_root = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"
    items.extend(
        Download("sec_nport", f"{nport_root}/{p}_nport.zip", stage=1)
        for p in ("2025q3", "2025q4")
    )

    # Phase 2: rates, fixed income, private credit/capital formation, consumer
    # finance, housing, international finance, and market microstructure.
    treasury_rate = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&"
        "field_tdr_date_value={year}&page&_format=csv"
    )
    for year in range(2021, 2026):
        items.append(Download("treasury_rates", treasury_rate.format(year=year), f"yield_curve_{year}.csv", 2))
    items.append(
        Download(
            "nyfed_rates",
            "https://markets.newyorkfed.org/api/rates/all/search.csv?startDate=01/01/2021&endDate=12/31/2025&type=rate",
            "reference_rates_2021_2025.csv", 2,
        )
    )
    auctions_url = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
        "accounting/od/auctions_query?filter=record_date:lte:2025-12-31&"
        "page%5Bsize%5D=10000&page%5Bnumber%5D={page}&format=csv"
    )
    for page in (1, 2):
        items.append(
            Download(
                "treasury_auctions",
                auctions_url.format(page=page),
                f"treasury_auctions_through_2025_page_{page}.csv",
                2,
                "FiscalData paginated snapshot filtered by record_date <= 2025-12-31",
            )
        )

    bdc_root = (
        "https://www.sec.gov/files/datastandardsinnovation/data/"
        "business-development-company-bdc-data-sets"
    )
    items.extend(
        Download("sec_bdc", f"{bdc_root}/{year}q{quarter}_bdc.zip", stage=2)
        for year in (2024, 2025) for quarter in range(1, 5)
    )

    sba_root = "https://data.sba.gov/sites/default/files/uploaded_resources"
    sba_files = [
        "7a_504_foia_data_dictionary.xlsx", "FOIA_504_FY1991_FY2009_asof_260630.csv",
        "FOIA_504_FY2010_Present_asof_260630.csv", "FOIA_7a_FY1991_FY1999_asof_260630.csv",
        "FOIA_7a_FY2000_FY2009_asof_260630.csv", "FOIA_7a_FY2010_FY2019_asof_260630.csv",
        "FOIA_7a_FY2020_Present_asof_260630.csv",
    ]
    items.extend(Download("sba_7a_504", f"{sba_root}/{name}", stage=2) for name in sba_files)

    items.append(
        Download(
            "cfpb_complaints", "https://files.consumerfinance.gov/ccdb/complaints.csv.zip",
            "complaints.csv.zip", 2,
            "Public complaint data; narratives are consumer allegations, not verified facts",
        )
    )

    sec_d_root = "https://www.sec.gov/files/structureddata/data/form-d-data-sets"
    sec_c_root = "https://www.sec.gov/files/dera/data/crowdfunding-offerings-data-sets"
    for year in (2024, 2025):
        for quarter in range(1, 5):
            period = f"{year}q{quarter}"
            items.append(Download("sec_form_d", f"{sec_d_root}/{period}_d.zip", stage=2))
            items.append(Download("sec_form_c", f"{sec_c_root}/{period}_cf.zip", stage=2))

    fhfa_files = {
        "hpi_master.csv": "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv",
        "hpi_dictionary.xlsx": "https://www.fhfa.gov/document/d/hpi/hpi_dictionary.xlsx",
        "hpi_specifications.xlsx": "https://www.fhfa.gov/document/d/hpi/hpi_specifications.xlsx",
        "hpi_at_tract.csv": "https://www.fhfa.gov/hpi/download/annual/hpi_at_tract.csv",
    }
    items.extend(Download("fhfa_hpi", url, name, 2) for name, url in fhfa_files.items())
    items.append(
        Download(
            "fhfa_uad",
            "https://www.fhfa.gov/document/d/uad-al/ent_uad_puf_annual_csv_v2_1.zip",
            "enterprise_uad_puf_annual_csv_v2_1.zip", 2,
        )
    )

    bis_root = "https://data.bis.org/static/bulk"
    bis_topics = [
        "WS_LBS_D_PUB", "WS_CBS_PUB", "WS_NA_SEC_DSS", "WS_TC",
        "WS_CREDIT_GAP", "WS_DSR", "WS_GLI", "WS_XTD_DERIV",
        "WS_OTC_DERIV2", "WS_XRU", "WS_EER", "WS_CBPOL",
    ]
    items.extend(
        Download("bis", f"{bis_root}/{topic}_csv_flat.zip", stage=2)
        for topic in bis_topics
    )

    cftc_root = "https://www.cftc.gov/files/dea/history"
    for year in (2024, 2025):
        for prefix in ("fut_disagg_txt", "com_disagg_txt", "fut_fin_txt", "com_fin_txt"):
            items.append(Download("cftc_cot", f"{cftc_root}/{prefix}_{year}.zip", stage=2))

    return items


def _discover_ftd() -> list[Download]:
    page = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
    result = []
    for url, text in _get_links(page):
        if url.lower().endswith(".zip") and re.search(r"\b(2024|2025)\b", text):
            result.append(Download("sec_ftd", url, stage=2, note=text))
    return result


def _finra_daily() -> list[Download]:
    result = []
    current = date(2024, 1, 1)
    end = date(2025, 12, 31)
    while current <= end:
        if current.weekday() < 5:
            stamp = current.strftime("%Y%m%d")
            result.append(
                Download(
                    "finra_short_volume",
                    f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt",
                    f"CNMSshvol{stamp}.txt", 2,
                )
            )
        current = date.fromordinal(current.toordinal() + 1)
    return result


def _download_one(item: Download) -> tuple[Download, dict]:
    raw_dir = RAW_DIR / item.dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / item.name
    part = path.with_name(path.name + ".part")
    if path.exists() and path.stat().st_size > 0:
        if path.suffix.lower() == ".zip":
            try:
                with ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise BadZipFile("member CRC failure")
            except BadZipFile:
                path.unlink()
            else:
                return item, _file_record(path, item, "existing")
        elif path.suffix.lower() == ".parquet":
            with path.open("rb") as handle:
                start = handle.read(4)
                handle.seek(-4, 2)
                end = handle.read(4)
            if start != b"PAR1" or end != b"PAR1":
                path.unlink()
            else:
                return item, _file_record(path, item, "existing")
        else:
            return item, _file_record(path, item, "existing")

    session = _session()
    last_error: Exception | None = None
    # Large public archives occasionally reset long-lived connections. Six
    # attempts, combined with byte-range continuation when the origin supports
    # it, avoids treating a transient CDN reset as a missing dataset.
    for attempt in range(6):
        try:
            existing_bytes = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
            _respect_source_rate_limit(item.url)
            with session.get(
                item.url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True
            ) as response:
                if response.status_code == 404 and item.dataset == "finra_short_volume":
                    response.raise_for_status()
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                # A compliant range response can be appended.  If the origin ignored
                # Range and returned 200, restart this one file without touching any
                # already completed downloads.
                mode = "ab" if existing_bytes and response.status_code == 206 else "wb"
                with part.open(mode) as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                if part.stat().st_size == 0:
                    raise ValueError("empty response")
                if item.name.lower().endswith(".zip"):
                    with ZipFile(part) as archive:
                        bad = archive.testzip()
                        if bad is not None:
                            raise BadZipFile(f"CRC failure in {bad}")
                elif item.name.lower().endswith(".parquet"):
                    with part.open("rb") as handle:
                        start = handle.read(4)
                        handle.seek(-4, 2)
                        end = handle.read(4)
                    if start != b"PAR1" or end != b"PAR1":
                        raise ValueError("incomplete or invalid Parquet file")
                elif "text/html" in content_type.lower():
                    head = part.read_bytes()[:300].lower()
                    if b"<html" in head or b"<!doctype" in head:
                        raise ValueError("received HTML instead of dataset")
                part.replace(path)
                return item, _file_record(path, item, "downloaded", content_type)
        except Exception as exc:  # retry transient public-data host failures
            last_error = exc
            if (
                item.dataset == "finra_short_volume"
                and isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code in {403, 404}
            ):
                # FINRA's CDN uses both statuses for dates with no consolidated
                # file, such as market holidays. Adjacent trading dates remain
                # accessible, so retrying a missing date cannot improve coverage.
                raise
            if isinstance(exc, (BadZipFile, ValueError)):
                part.unlink(missing_ok=True)
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 416:
                # A stale/full partial is not resumable; restart it once.
                part.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _file_record(path: Path, item: Download, status: str, content_type: str = "") -> dict:
    return {
        "file": str(path.relative_to(DATASETS / item.dataset)),
        "url": item.url,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "status": status,
        "content_type": content_type,
        "note": item.note,
    }


def _write_manifest(dataset: str, records: list[dict], failures: list[dict]) -> None:
    directory = DATASETS / dataset
    prior_path = directory / "sources.json"
    prior = {}
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
    merged = {
        record["file"]: record
        for record in prior.get("raw_files", [])
        if isinstance(record, dict)
        and isinstance(record.get("file"), str)
        and (directory / record["file"]).is_file()
    }
    merged.update({record["file"]: record for record in records})
    manifest = {
        "dataset": dataset,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_freeze_date": FREEZE_DATE,
        "raw_snapshot_notice": (
            "Current/bulk source files can contain observations after the benchmark "
            "freeze date; downstream normalization must filter by the relevant "
            "observation, report, or filing date before authoring cases."
        ),
        "raw_files": sorted(merged.values(), key=lambda value: value["file"]),
        "failures": failures,
    }
    prior_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fetch_hmda() -> tuple[list[dict], list[dict]]:
    dataset = "hmda"
    raw_dir = RAW_DIR / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    filers_url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/filers?years=2024"
    response = _session().get(filers_url, timeout=120)
    response.raise_for_status()
    filers_path = raw_dir / "filers_2024.json"
    filers_path.write_bytes(response.content)
    institutions = response.json()["institutions"]
    wanted = [
        "JPMorgan Chase Bank, National Association",
        "Bank of America, National Association",
        "Wells Fargo Bank, National Association",
        "U.S. Bank National Association",
        "Citibank, National Association",
    ]
    by_name = {row["name"]: row for row in institutions}
    downloads = []
    for name in wanted:
        row = by_name[name]
        lei = row["lei"]
        downloads.append(
            Download(
                dataset,
                f"https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2024&leis={lei}",
                f"2024_{lei}.csv", 1, f"{name}; reported row count={row.get('count')}",
            )
        )
    records = [{
        "file": "raw/filers_2024.json", "url": filers_url,
        "bytes": filers_path.stat().st_size, "sha256": _sha256(filers_path),
        "status": "downloaded", "content_type": "application/json",
        "note": "Full 2024 HMDA filer list used to resolve the fixed lender cohort",
    }]
    failures = []
    for item in downloads:
        try:
            _, record = _download_one(item)
            records.append(record)
            print(f"OK   {dataset:24s} {item.name:48s} {record['bytes'] / 2**20:9.2f} MiB", flush=True)
        except Exception as exc:
            failures.append({"url": item.url, "file": item.name, "error": repr(exc)})
            print(f"FAIL {dataset:24s} {item.name}: {exc}", flush=True)
    return records, failures


def _fetch_fdic() -> tuple[list[dict], list[dict]]:
    dataset = "fdic_bankfind"
    fields = [
        "REPDTE", "CERT", "NAME", "ASSET", "DEP", "DEPDOM", "EQ", "NETINC",
        "LNLSNET", "LNATRES", "LNCON", "LNCI", "LNRE", "NCLNLS", "NTLNLS",
        "ROA", "ROE", "NIMY", "INTINC", "EINTEXP", "NONII", "NONIX", "CHBAL",
        "SC", "FREPO", "RBCT1J", "RBCT2", "RBCRWAJ", "RBCRWAK", "RBC1AAJ",
    ]
    base = "https://api.fdic.gov/banks/financials"
    top_params = {
        "filters": "REPDTE:20251231", "fields": "REPDTE,CERT,NAME,ASSET",
        "sort_by": "ASSET", "sort_order": "DESC", "limit": "12", "format": "csv",
    }
    session = _session()
    top = session.get(base, params=top_params, timeout=120)
    top.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(top.text)))
    certs = sorted({row["CERT"] for row in rows})
    params = {
        "filters": f"REPDTE:[20220101 TO 20251231] AND CERT:({' OR '.join(certs)})",
        "fields": ",".join(fields), "limit": "10000", "format": "csv", "download": "true",
    }
    url = requests.Request("GET", base, params=params).prepare().url
    item = Download(dataset, url, "top12_banks_financials_2022_2025.csv", 1, f"CERT cohort: {','.join(certs)}")
    try:
        _, record = _download_one(item)
        return [record], []
    except Exception as exc:
        return [], [{"url": url, "file": item.name, "error": repr(exc)}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = {1, 2} if args.stage == "all" else {int(args.stage)}

    downloads = [item for item in _static_downloads() if item.stage in selected]
    if 2 in selected:
        downloads.extend(_discover_ftd())
        downloads.extend(_finra_daily())

    records: dict[str, list[dict]] = {}
    failures: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, item): item for item in downloads}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, record = future.result()
                records.setdefault(item.dataset, []).append(record)
                print(f"OK   {item.dataset:24s} {item.name:48s} {record['bytes'] / 2**20:9.2f} MiB", flush=True)
            except requests.HTTPError as exc:
                # FINRA has no files on market holidays; retain these as expected gaps.
                expected = (
                    item.dataset == "finra_short_volume"
                    and exc.response is not None
                    and exc.response.status_code in {403, 404}
                )
                failure = {"url": item.url, "file": item.name, "error": repr(exc), "expected_market_holiday": expected}
                failures.setdefault(item.dataset, []).append(failure)
                label = "SKIP" if expected else "FAIL"
                print(f"{label} {item.dataset:24s} {item.name}: {exc}", flush=True)
            except Exception as exc:
                failures.setdefault(item.dataset, []).append({"url": item.url, "file": item.name, "error": repr(exc)})
                print(f"FAIL {item.dataset:24s} {item.name}: {exc}", flush=True)

    if 1 in selected:
        for dataset, fetcher in (("hmda", _fetch_hmda), ("fdic_bankfind", _fetch_fdic)):
            extra_records, extra_failures = fetcher()
            records.setdefault(dataset, []).extend(extra_records)
            failures.setdefault(dataset, []).extend(extra_failures)

    for dataset in sorted(set(records) | set(failures)):
        _write_manifest(dataset, records.get(dataset, []), failures.get(dataset, []))

    unexpected = [
        failure for values in failures.values() for failure in values
        if not failure.get("expected_market_holiday")
    ]
    total_bytes = sum(record["bytes"] for values in records.values() for record in values)
    print(f"\nDownloaded/verified {sum(map(len, records.values()))} files, {total_bytes / 2**30:.2f} GiB")
    print(f"Unexpected failures: {len(unexpected)}")
    if unexpected:
        sys.exit(1)


if __name__ == "__main__":
    main()
