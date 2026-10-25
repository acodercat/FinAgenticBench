"""Download the fourth, capability-focused FinAgenticBench data expansion.

This bounded phase fills four structural gaps rather than adding more of the
same market/company observations:

* DOL Form 5500 plan filings and schedules (2023--2024, including amendments)
* FFIEC all-bank Call Report schedules (2024Q1--2025Q4)
* Treasury fiscal/debt tables with explicit cutoff filters and fixed reports
* FOMC statements, minutes, projections, and press-conference transcripts

All files are source-native.  Downloads are idempotent, checksummed in each
dataset manifest, and parallelized without exceeding source-specific limits.

Usage::

    uv run python -m scripts.ingest.fetch.fetch_phase4 --workers 10
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import BadZipFile, ZipFile


from core.dataset_layers import RAW_DIR
from scripts.ingest.fetch.fetch_expansion import (
    FREEZE_DATE,
    Download,
    _download_one,
    _file_record,
    _session,
    _write_manifest,
)


FORM_5500_TABLES = (
    "F_5500", "F_5500_SF", "F_5500_SF_PART7",
    "F_SCH_A", "F_SCH_A_PART1",
    "F_SCH_C", "F_SCH_C_PART1_ITEM1", "F_SCH_C_PART1_ITEM2",
    "F_SCH_C_PART1_ITEM2_CODES", "F_SCH_C_PART1_ITEM3",
    "F_SCH_C_PART1_ITEM3_CODES", "F_SCH_C_PART2",
    "F_SCH_C_PART2_CODES", "F_SCH_C_PART3",
    "F_SCH_D", "F_SCH_D_PART1", "F_SCH_D_PART2",
    "F_SCH_G", "F_SCH_G_PART1", "F_SCH_G_PART2", "F_SCH_G_PART3",
    "F_SCH_H", "F_SCH_H_PART1", "F_SCH_I", "F_SCH_R",
    "F_SCH_MB", "F_SCH_MEP", "F_SCH_MEP_PART2", "F_SCH_SB", "F_SCH_DCG",
)

FOMC_DATES = {
    2024: ("20240131", "20240320", "20240501", "20240612", "20240731",
           "20240918", "20241107", "20241218"),
    2025: ("20250129", "20250319", "20250507", "20250618", "20250730",
           "20250917", "20251029", "20251210"),
}

SEP_DATES = {
    "20240320", "20240612", "20240918", "20241218",
    "20250319", "20250618", "20250917", "20251210",
}

FISCAL_ENDPOINTS = {
    "treasury_debt": (
        ("debt_to_penny", "v2/accounting/od/debt_to_penny", "2024-01-01"),
        ("interest_expense", "v2/accounting/od/interest_expense", "2024-01-01"),
        ("mspd_table_1", "v1/debt/mspd/mspd_table_1", "2024-01-01"),
        ("mspd_table_2", "v1/debt/mspd/mspd_table_2", "2024-01-01"),
        ("mspd_table_3", "v1/debt/mspd/mspd_table_3", "2024-01-01"),
        ("mspd_table_3_market", "v1/debt/mspd/mspd_table_3_market", "2024-01-01"),
        ("mspd_table_3_nonmarket", "v1/debt/mspd/mspd_table_3_nonmarket", "2024-01-01"),
        ("mspd_table_4", "v1/debt/mspd/mspd_table_4", "2024-01-01"),
        ("mspd_table_5", "v1/debt/mspd/mspd_table_5", "2024-01-01"),
    ),
    "treasury_mts": tuple(
        (name, f"v1/accounting/mts/{name}", "2024-01-01")
        for name in (
            "mts_table_1", "mts_table_2", "mts_table_3", "mts_table_4",
            "mts_table_5", "mts_table_6", "mts_table_6a", "mts_table_6b",
            "mts_table_6c", "mts_table_6d", "mts_table_6e", "mts_table_7",
            "mts_table_8", "mts_table_9", "mts_receipts_outlays_deficit_surplus",
            "mts_distributed_offsetting_receipts",
            "mts_table_9_outlays_functions_subfunctions", "mts_table_5m",
        )
    ),
    "treasury_dts": tuple(
        (name, f"v1/accounting/dts/{name}", "2024-01-01")
        for name in (
            "operating_cash_balance", "deposits_withdrawals_operating_cash",
            "public_debt_transactions", "adjustment_public_debt_transactions_cash_basis",
            "debt_subject_to_limit", "inter_agency_tax_transfers",
            "income_tax_refunds_issued",
        )
    ),
    "treasury_usfr": (
        ("statement_net_cost", "v2/accounting/od/statement_net_cost", "2020-01-01"),
        ("net_position", "v1/accounting/od/net_position", "2020-01-01"),
        ("reconciliations", "v1/accounting/od/reconciliations", "2020-01-01"),
        ("cash_balance", "v1/accounting/od/cash_balance", "2020-01-01"),
        ("balance_sheets", "v2/accounting/od/balance_sheets", "2020-01-01"),
        ("long_term_projections", "v1/accounting/od/long_term_projections", "2020-01-01"),
        ("social_insurance", "v1/accounting/od/social_insurance", "2020-01-01"),
        ("insurance_amounts", "v1/accounting/od/insurance_amounts", "2020-01-01"),
    ),
}


def _static_downloads() -> list[Download]:
    items: list[Download] = []

    # "All" retains superseded submissions, unlike "Latest".  That overlap is
    # intentional: ACK_ID/filing timestamps allow an as-of latest filing to be
    # reconstructed and amendments to be evaluated without look-ahead.
    for year in (2023, 2024):
        root = f"https://www.askebsa.dol.gov/FOIA%20Files/{year}/All"
        for table in FORM_5500_TABLES:
            items.append(Download(
                "dol_form5500", f"{root}/{table}_{year}_all.zip",
                note=(f"Form year {year}, all submissions including amendments; select the "
                      f"latest filing received by {FREEZE_DATE} for strict as-of use"),
            ))
    items.extend([
        Download(
            "dol_form5500",
            "https://www.dol.gov/sites/dolgov/files/EBSA/pdf_files/form-5500-datasets-guide.pdf",
            note="Official guide to EFAST2 Form 5500 dataset structure and identifiers",
        ),
        Download(
            "dol_form5500",
            "https://www.dol.gov/sites/dolgov/files/EBSA/employers-and-advisers/plan-administration-and-compliance/reporting-and-filing/form-5500/2023-instructions.pdf",
            note="Official 2023 form and schedule instructions",
        ),
        Download(
            "dol_form5500",
            "https://www.dol.gov/sites/dolgov/files/EBSA/employers-and-advisers/plan-administration-and-compliance/reporting-and-filing/form-5500/2024-instructions.pdf",
            note="Official 2024 form and schedule instructions",
        ),
    ])

    # Fixed FOMC artifacts avoid mirroring site chrome. Statements include the
    # implementation note attachment; minutes, projections, and press Q&A have
    # different information sets and release times.
    for dates in FOMC_DATES.values():
        for stamp in dates:
            items.extend([
                Download(
                    "fed_fomc_materials",
                    f"https://www.federalreserve.gov/monetarypolicy/files/monetary{stamp}a1.pdf",
                    f"fomc_statement_and_implementation_{stamp}.pdf",
                    note="Fixed policy statement plus implementation-note attachment",
                ),
                Download(
                    "fed_fomc_materials",
                    f"https://www.federalreserve.gov/monetarypolicy/files/fomcminutes{stamp}.pdf",
                    f"fomc_minutes_{stamp}.pdf",
                    note="Minutes released after the meeting; do not treat as decision-day information",
                ),
                Download(
                    "fed_fomc_materials",
                    f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{stamp}.pdf",
                    f"fomc_press_conference_{stamp}.pdf",
                    note="Chair press-conference transcript; distinguish answers from Committee decisions",
                ),
            ])
            if stamp in SEP_DATES:
                items.append(Download(
                    "fed_fomc_materials",
                    f"https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl{stamp}.pdf",
                    f"fomc_sep_{stamp}.pdf",
                    note="Summary of Economic Projections; participant projections are not a Committee forecast",
                ))

    # A small set of immutable Treasury report artifacts anchors the current,
    # revision-prone API tables to publications that existed before the cutoff.
    for stamp in ("202409", "202509"):
        items.extend([
            Download(
                "treasury_mts",
                f"https://fiscaldata.treasury.gov/static-data/published-reports/mts/MonthlyTreasuryStatement_{stamp}.pdf",
                note="Fixed fiscal-year-end Monthly Treasury Statement publication",
            ),
            Download(
                "treasury_debt",
                f"https://fiscaldata.treasury.gov/static-data/published-reports/mspd-entire/MonthlyStatementPublicDebt_Entire_{stamp}.pdf",
                note="Fixed fiscal-year-end Monthly Statement of the Public Debt publication",
            ),
            Download(
                "treasury_debt",
                f"https://fiscaldata.treasury.gov/static-data/published-reports/mspd-entire/MonthlyStatementPublicDebt_Entire_{stamp}.xls",
                note="Source spreadsheet corresponding to the fixed MSPD publication",
            ),
        ])
    return items


def _hidden_fields(text: str) -> dict[str, str]:
    pattern = r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"'
    return {html.unescape(name): html.unescape(value) for name, value in re.findall(pattern, text)}


def _ffiec_download(report_date: str) -> dict:
    """Submit the official ASP.NET bulk form for one Call Report quarter."""
    dataset = "ffiec_call_reports"
    landing = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    stamp = report_date.replace("/", "")
    filename = f"FFIEC_CDR_Call_Bulk_All_Schedules_{stamp}.zip"
    raw = RAW_DIR / dataset
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / filename
    item = Download(
        dataset, landing, filename,
        note=(f"Official all-commercial-bank, all-schedules single-period bulk export for {report_date}; "
              "the FFIEC service can revise historical quarters in place"
              + ("; 2025Q4 was published after the benchmark cutoff and is retrospective only"
                 if report_date == "12/31/2025" else "")),
    )
    if path.exists() and path.stat().st_size:
        with ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise BadZipFile(f"CRC failure in existing archive member {bad}")
        return _file_record(path, item, "existing")

    last_error: Exception | None = None
    for attempt in range(5):
        part = path.with_name(path.name + ".part")
        part.unlink(missing_ok=True)
        try:
            session = _session()
            response = session.get(landing, timeout=120)
            response.raise_for_status()

            data = _hidden_fields(response.text)
            data.update({
                "__EVENTTARGET": "ctl00$MainContentHolder$ListBox1",
                "ctl00$MainContentHolder$ListBox1": "ReportingSeriesSinglePeriod",
            })
            response = session.post(landing, data=data, timeout=120)
            response.raise_for_status()
            date_block = re.search(
                r'<select[^>]+id="DatesDropDownList"[^>]*>(.*?)</select>', response.text, re.S
            )
            if not date_block:
                raise ValueError("FFIEC report-date selector was not returned")
            choices = {
                html.unescape(label): value
                for value, label in re.findall(
                    r'<option value="([^"]*)">(.*?)</option>', date_block.group(1)
                )
            }
            if report_date not in choices:
                raise ValueError(f"FFIEC report date is unavailable: {report_date}")

            selected = choices[report_date]
            data = _hidden_fields(response.text)
            data.update({
                "__EVENTTARGET": "ctl00$MainContentHolder$TSVRadioButton",
                "ctl00$MainContentHolder$ListBox1": "ReportingSeriesSinglePeriod",
                "ctl00$MainContentHolder$DatesDropDownList": selected,
                "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
            })
            response = session.post(landing, data=data, timeout=120)
            response.raise_for_status()

            data = _hidden_fields(response.text)
            data.update({
                "ctl00$MainContentHolder$ListBox1": "ReportingSeriesSinglePeriod",
                "ctl00$MainContentHolder$DatesDropDownList": selected,
                "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
                "ctl00$MainContentHolder$TabStrip1$Download_0": "Download",
            })
            with session.post(landing, data=data, stream=True, timeout=(30, 300)) as download:
                download.raise_for_status()
                if "application/octet-stream" not in download.headers.get("Content-Type", ""):
                    raise ValueError("FFIEC form did not return a bulk archive")
                with part.open("wb") as handle:
                    for block in download.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            with ZipFile(part) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise BadZipFile(f"CRC failure in {bad}")
            part.replace(path)
            return _file_record(path, item, "downloaded", "application/zip")
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _write_content(dataset: str, filename: str, content: bytes, url: str, note: str,
                   content_type: str) -> dict:
    raw = RAW_DIR / dataset
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / filename
    path.write_bytes(content)
    return _file_record(path, Download(dataset, url, filename, note=note), "downloaded", content_type)


def _fetch_fiscal_table(dataset: str, name: str, endpoint: str, start_date: str) -> tuple[dict, dict]:
    """Fetch and combine every cutoff-filtered FiscalData CSV page."""
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    url = f"{base}/{endpoint}"
    params = {
        "filter": f"record_date:gte:{start_date},record_date:lte:{FREEZE_DATE}",
        "sort": "record_date",
        "page[size]": 10000,
    }
    session = _session()
    probe = session.get(url, params={**params, "page[size]": 1}, timeout=120)
    probe.raise_for_status()
    payload = probe.json()
    meta = payload.get("meta", {})
    total = int(meta.get("total-count", 0))
    if total <= 0:
        raise ValueError(f"FiscalData returned no rows for {endpoint}")

    rows: list[str] = []
    header: str | None = None
    page_urls: list[str] = []
    for page in range(1, (total + 9999) // 10000 + 1):
        response = session.get(
            url, params={**params, "page[number]": page, "format": "csv"}, timeout=300
        )
        response.raise_for_status()
        lines = response.text.splitlines()
        if not lines:
            raise ValueError(f"empty FiscalData CSV page {page} for {endpoint}")
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise ValueError(f"FiscalData header changed between pages for {endpoint}")
        rows.extend(lines[1:])
        page_urls.append(response.url)
    if len(rows) != total:
        raise ValueError(f"FiscalData row count mismatch for {endpoint}: {len(rows)} != {total}")

    csv_content = ((header or "") + "\n" + "\n".join(rows) + "\n").encode("utf-8")
    record = _write_content(
        dataset, f"{name}_{start_date[:4]}_through_2025.csv", csv_content, page_urls[0],
        f"All {total:,} API rows with record_date {start_date} through {FREEZE_DATE}; "
        f"combined from {len(page_urls)} official page(s)", "text/csv",
    )
    metadata = {
        "name": name, "endpoint": endpoint, "start_date": start_date,
        "end_date": FREEZE_DATE, "row_count": total, "page_urls": page_urls,
        "api_meta": meta,
    }
    return record, metadata


def _fetch_fiscal_tables(workers: int) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    records: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, list[dict]] = defaultdict(list)
    metadata: dict[str, list[dict]] = defaultdict(list)
    jobs = [
        (dataset, name, endpoint, start)
        for dataset, endpoints in FISCAL_ENDPOINTS.items()
        for name, endpoint, start in endpoints
    ]
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        future_map = {
            pool.submit(_fetch_fiscal_table, *job): job for job in jobs
        }
        for future in as_completed(future_map):
            dataset, name, endpoint, _ = future_map[future]
            try:
                record, table_meta = future.result()
                records[dataset].append(record)
                metadata[dataset].append(table_meta)
                print(f"OK   {dataset:28s} {Path(record['file']).name:52s} "
                      f"{record['bytes'] / 2**20:10.2f} MiB", flush=True)
            except Exception as exc:
                failures[dataset].append({"url": endpoint, "file": f"{name}.csv", "error": repr(exc)})
                print(f"FAIL {dataset:28s} {name}: {exc}", flush=True)

    for dataset, tables in metadata.items():
        content = json.dumps(
            {"observed_at_download": True, "tables": sorted(tables, key=lambda x: x["name"])},
            indent=2, ensure_ascii=False,
        ).encode("utf-8")
        records[dataset].append(_write_content(
            dataset, "api_metadata_and_pages.json", content,
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/",
            "Official API metadata plus exact paginated request URLs used for this snapshot",
            "application/json",
        ))
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    records: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, list[dict]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, item): item for item in _static_downloads()}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, record = future.result()
                records[item.dataset].append(record)
                print(f"OK   {item.dataset:28s} {item.name:52s} "
                      f"{record['bytes'] / 2**20:10.2f} MiB", flush=True)
            except Exception as exc:
                failures[item.dataset].append({"url": item.url, "file": item.name, "error": repr(exc)})
                print(f"FAIL {item.dataset:28s} {item.name}: {exc}", flush=True)

    quarters = (
        "03/31/2024", "06/30/2024", "09/30/2024", "12/31/2024",
        "03/31/2025", "06/30/2025", "09/30/2025", "12/31/2025",
    )
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        futures = {pool.submit(_ffiec_download, date): date for date in quarters}
        for future in as_completed(futures):
            date = futures[future]
            try:
                record = future.result()
                records["ffiec_call_reports"].append(record)
                print(f"OK   {'ffiec_call_reports':28s} {Path(record['file']).name:52s} "
                      f"{record['bytes'] / 2**20:10.2f} MiB", flush=True)
            except Exception as exc:
                failures["ffiec_call_reports"].append({
                    "url": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx",
                    "file": date, "error": repr(exc),
                })
                print(f"FAIL {'ffiec_call_reports':28s} {date}: {exc}", flush=True)

    fiscal_records, fiscal_failures = _fetch_fiscal_tables(args.workers)
    for dataset, values in fiscal_records.items():
        records[dataset].extend(values)
    for dataset, values in fiscal_failures.items():
        failures[dataset].extend(values)

    for dataset in sorted(set(records) | set(failures)):
        _write_manifest(dataset, records[dataset], failures[dataset])

    total_files = sum(map(len, records.values()))
    total_bytes = sum(record["bytes"] for values in records.values() for record in values)
    unexpected = sum(map(len, failures.values()))
    print(f"\nDownloaded/verified {total_files} files, {total_bytes / 2**30:.4f} GiB")
    print(f"Unexpected failures: {unexpected}")
    if unexpected:
        sys.exit(1)


if __name__ == "__main__":
    main()
