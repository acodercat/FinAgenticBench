"""Download high-linkage sources for integrated FinAgenticBench cases.

This phase adds bridge datasets rather than more isolated financial tables:

* FR Y-9 holding-company financials and NIC organization structure
* NOAA Storm Events for physical-risk event intensity
* BLS QCEW for county/industry employment and wage denominators
* OFR financial-stress, repo, and money-market-fund series
* EPA GHGRP facility emissions, parents, and EIA power-plant crosswalk
* Census international trade and geographic crosswalks

Downloads are bounded, parallel, idempotent, checksummed, and source-native.
Some official products are current revised snapshots or were released after the
benchmark cutoff; their per-file notes state that they are retrospective only.

Usage::

    uv run python -m scripts.ingest.fetch.fetch_phase5 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import BadZipFile, ZipFile

from curl_cffi import requests as curl_requests

from core.dataset_layers import RAW_DIR
from scripts.ingest.fetch.fetch_expansion import (
    Download,
    _download_one,
    _file_record,
    _write_manifest,
)


def _downloads() -> list[Download]:
    items: list[Download] = []

    # BLS annual single files retain quarterly measures while avoiding four
    # highly overlapping quarterly archives per year.
    for year in (2024, 2025):
        items.append(Download(
            "bls_qcew",
            f"https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip",
            note=(
                f"Official QCEW county/industry annual single file for {year}; includes quarterly fields"
                + ("; published after the benchmark cutoff and retrospective only; the frozen download is a preliminary annual vintage because BLS finalizes annual averages with the subsequent year's preliminary first-quarter data, scheduled for 2026-08-28 as of retrieval (https://www.bls.gov/cew/release-calendar.htm)"
                   if year == 2025 else "")
            ),
        ))

    noaa_root = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
    for year in (2024, 2025):
        for kind in ("details", "locations", "fatalities"):
            name = f"StormEvents_{kind}-ftp_v1.0_d{year}_c20260728.csv.gz"
            items.append(Download(
                "noaa_storm_events", f"{noaa_root}/{name}", name,
                note=(f"Official NOAA Storm Events {kind} table for event year {year}; "
                      "current revised extraction created 2026-07-28, retrospective only for strict as-of tasks"),
            ))
    items.append(Download(
        "noaa_storm_events",
        f"{noaa_root}/Storm-Data-Export-Format.pdf",
        note="Official Storm Events bulk-data field documentation",
    ))

    items.extend([
        Download(
            "ofr_market_stress",
            "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv",
            note="Official daily OFR Financial Stress Index; current revised history, filter dates for benchmark use",
        ),
        Download(
            "ofr_market_stress",
            "https://www.financialresearch.gov/financial-stress-index/files/FSI_Revision_History_2023-06-27.xlsx",
            note="Official OFR FSI revision history",
        ),
        Download(
            "ofr_market_stress",
            "https://data.financialresearch.gov/v1/metadata/mnemonics/",
            "stfm_mnemonics.json",
            note="Complete official STFM mnemonic inventory observed at download time",
        ),
        Download(
            "ofr_market_stress",
            "https://data.financialresearch.gov/v1/series/dataset?dataset=repo&start_date=2024-01-01&end_date=2025-12-31",
            "stfm_repo_2024_2025.json",
            note="Official OFR repo dataset filtered to the common observation window; current revised API snapshot",
        ),
        Download(
            "ofr_market_stress",
            "https://data.financialresearch.gov/v1/series/dataset?dataset=mmf&start_date=2024-01-01&end_date=2025-12-31",
            "stfm_mmf_2024_2025.json",
            note="Official OFR money-market-fund dataset filtered to the common observation window; current revised API snapshot",
        ),
    ])

    items.extend([
        Download(
            "epa_ghgrp",
            "https://www.epa.gov/system/files/other-files/2024-10/2023_data_summary_spreadsheets.zip",
            note="Latest GHGRP reporting-year summary publicly available by the cutoff (2023 data, released 2024); lagged structural evidence",
        ),
        Download(
            "epa_ghgrp",
            "https://www.epa.gov/system/files/other-files/2024-10/ghgp_data_parent_company.xlsb",
            note="Official facility-to-reported-parent ownership table for the 2023 GHGRP release",
        ),
        Download(
            "epa_ghgrp",
            "https://www.epa.gov/system/files/documents/2022-04/ghgrp_oris_power_plant_crosswalk_12_13_21.xlsx",
            note="Official GHGRP facility-to-EIA/ORIS power-plant identifier crosswalk; static supporting artifact",
        ),
    ])

    census_files = [
        ("https://www.census.gov/foreign-trade/statistics/product/enduse/exports/enduse_exports.xlsx",
         "country_enduse_exports_2016_2025.xlsx",
         "Country by five-digit end-use annual exports, 2016-2025; 2025 annual total is retrospective after cutoff"),
        ("https://www.census.gov/foreign-trade/statistics/product/enduse/imports/enduse_imports.xlsx",
         "country_enduse_imports_2016_2025.xlsx",
         "Country by five-digit end-use annual imports, 2016-2025; 2025 annual total is retrospective after cutoff"),
        ("https://www.census.gov/foreign-trade/statistics/historical/NSAEXP.xlsx",
         "monthly_principal_enduse_exports.xlsx",
         "Official monthly principal end-use exports history; filter to 2024-2025"),
        ("https://www.census.gov/foreign-trade/statistics/historical/NSAIMP.xlsx",
         "monthly_principal_enduse_imports.xlsx",
         "Official monthly principal end-use imports history; filter to 2024-2025"),
    ]
    items.extend(Download("census_international_trade", url, name, note=note)
                 for url, name, note in census_files)

    items.extend([
        Download(
            "census_geo_crosswalk",
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip",
            note="Official 2024 Census county Gazetteer with GEOID/FIPS and coordinates",
        ),
        Download(
            "census_geo_crosswalk",
            "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt",
            note="Official 2020 ZCTA-to-county relationship file; static geography bridge, not an economic observation",
        ),
    ])
    return items


def _ffiec_downloads() -> list[Download]:
    items = []
    root = "https://www.ffiec.gov/npw/FinancialReport"
    for year in (2024, 2025):
        for stamp in ("0331", "0630", "0930", "1231"):
            name = f"BHCF{year}{stamp}.ZIP"
            items.append(Download(
                "ffiec_y9c",
                f"{root}/ReturnBHCFZipFiles?zipfilename={name}",
                name,
                note=(f"Official FR Y-9C/Y-9LP/Y-9SP holding-company data as of {year}-{stamp[:2]}-{stamp[2:]}; "
                      "NIC historical files can be revised in place"
                      + ("; 2025Q4 was released after the cutoff and is retrospective only"
                         if name == "BHCF20251231.ZIP" else "")),
            ))
    items.extend([
        Download(
            "ffiec_y9c",
            "https://www.ffiec.gov/npw/StaticData/DataDownload/Financial_Download_Dictionary.xlsx",
            "Financial_Download_Dictionary.xlsx",
            note="Official BHCF/RSSD variable dictionary",
        ),
        Download(
            "ffiec_nic_structure", f"{root}/ReturnAttributesActiveZipFileCSV",
            "attributes_active_current.zip",
            note="Current NIC institution attributes observed in 2026; discovery/crosswalk support only, not a 2025-12-31 snapshot",
        ),
        Download(
            "ffiec_nic_structure", f"{root}/ReturnRelationshipsZipFileCSV",
            "relationships_current.zip",
            note="Current NIC parent/subsidiary relationships observed in 2026; discovery/crosswalk support only",
        ),
        Download(
            "ffiec_nic_structure", f"{root}/ReturnTransformationZipFile",
            "transformations_current.zip",
            note="NIC merger and transformation history from the current extraction; event dates may be filtered but artifact is not a frozen 2025 snapshot",
        ),
    ])
    return items


def _download_ffiec(item: Download) -> tuple[Download, dict]:
    """Use a browser TLS fingerprint for FFIEC's Cloudflare-protected binary endpoints."""
    raw = RAW_DIR / item.dataset
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / item.name
    if path.exists() and path.stat().st_size:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise BadZipFile(f"CRC failure in {bad}")
        return item, _file_record(path, item, "existing")

    response = None
    for attempt in range(6):
        response = curl_requests.get(
            item.url,
            impersonate="chrome",
            headers={"Referer": "https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload"},
            timeout=300,
        )
        if response.status_code == 200:
            break
        if response.status_code not in {403, 429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(min(2 ** attempt, 30))
    assert response is not None
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        raise ValueError(f"FFIEC returned HTML instead of {item.name}")
    part = path.with_name(path.name + ".part")
    part.write_bytes(response.content)
    if part.stat().st_size == 0:
        raise ValueError(f"empty FFIEC response for {item.name}")
    if path.suffix.lower() == ".zip":
        with ZipFile(part) as archive:
            bad = archive.testzip()
            if bad:
                raise BadZipFile(f"CRC failure in {bad}")
    part.replace(path)
    return item, _file_record(path, item, "downloaded", content_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = [(item, _download_one) for item in _downloads()]
    records: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, list[dict]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(fetcher, item): item for item, fetcher in jobs}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                _, record = future.result()
                records[item.dataset].append(record)
                print(f"OK   {item.dataset:28s} {item.name:52s} {record['bytes']/2**20:8.2f} MiB", flush=True)
            except Exception as exc:
                failures[item.dataset].append({
                    "url": item.url, "file": item.name, "error": repr(exc),
                })
                print(f"FAIL {item.dataset:28s} {item.name}: {exc}", flush=True)

    # FFIEC's public endpoint is protected against bursts. Fetch these small
    # files serially and leave a short gap between request starts.
    for item in _ffiec_downloads():
        try:
            _, record = _download_ffiec(item)
            records[item.dataset].append(record)
            print(f"OK   {item.dataset:28s} {item.name:52s} {record['bytes']/2**20:8.2f} MiB", flush=True)
        except Exception as exc:
            failures[item.dataset].append({
                "url": item.url, "file": item.name, "error": repr(exc),
            })
            print(f"FAIL {item.dataset:28s} {item.name}: {exc}", flush=True)
        time.sleep(1.0)

    for dataset in sorted(set(records) | set(failures)):
        _write_manifest(dataset, records[dataset], failures[dataset])

    total = sum(record["bytes"] for values in records.values() for record in values)
    failed = sum(map(len, failures.values()))
    print(f"Downloaded/verified {sum(map(len, records.values()))} files, {total/2**30:.3f} GiB")
    print(f"Unexpected failures: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
