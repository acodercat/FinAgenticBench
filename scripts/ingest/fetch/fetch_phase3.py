"""Download the third FinAgenticBench public-data expansion snapshot.

This phase adds institution holdings, investment funds and advisers, market
microstructure, entity relationships, household finance, macro vintages,
credit unions, bank branches, cross-border finance, energy, disasters,
consumer agreements, government spending, and sanctions data.

The command is idempotent and uses the validated/resumable download machinery
from :mod:`scripts.ingest.fetch.fetch_expansion`. Source-native files remain in ``raw``;
current bulk products are deliberately not presented as knowledge-cutoff-safe.

Usage::

    uv run python -m scripts.ingest.fetch.fetch_phase3 --workers 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


from core.dataset_layers import RAW_DIR
from scripts.ingest.fetch.fetch_expansion import (
    FREEZE_DATE,
    Download,
    _download_one,
    _file_record,
    _session,
    _write_manifest,
)


def _static_downloads() -> list[Download]:
    items: list[Download] = []

    # SEC institutional holdings. Windows are filing windows; the Dec 2025 to
    # Feb 2026 product is excluded because it crosses the knowledge cutoff.
    root = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets"
    for window in (
        "01jan2024-29feb2024", "01mar2024-31may2024",
        "01jun2024-31aug2024", "01sep2024-30nov2024",
        "01dec2024-28feb2025", "01mar2025-31may2025",
        "01jun2025-31aug2025", "01sep2025-30nov2025",
    ):
        items.append(Download("sec_13f", f"{root}/{window}_form13f.zip"))
    items.append(Download("sec_13f", "https://www.sec.gov/files/form_13f_readme.pdf"))

    root = "https://www.sec.gov/files/dera/data/form-n-cen-data-sets"
    for period in ("2024q1_ncen_0", "2024q2_ncen_0", "2024q3_ncen_0", "2024q4_ncen_0",
                   "2025q1_ncen_0", "2025q2_ncen_0", "2025q3_ncen", "2025q4_ncen"):
        note = ("2025Q4 bulk artifact was published after the cutoff; retrospective only"
                if period.startswith("2025q4") else "")
        items.append(Download("sec_ncen", f"{root}/{period}.zip", note=note))
    items.append(Download("sec_ncen", "https://www.sec.gov/files/ncen_readme.pdf"))

    for year in (2024, 2025):
        for quarter in range(1, 5):
            slug = ("metrics-individual-security-and-exchange" if (year, quarter) == (2024, 1)
                    else "metrics-individual-security-exchange")
            items.append(Download(
                "sec_midas_security_exchange",
                f"https://www.sec.gov/files/opa/data/market-structure/{slug}/"
                f"individual_security_exchange_{year}_q{quarter}.zip",
                note=("2025Q4 quarterly artifact was published after the cutoff; retrospective only"
                      if (year, quarter) == (2025, 4) else ""),
            ))

    # The two 2024 ADV products are cumulative historical exports with a fixed
    # 2024-12-31 endpoint.  2025 structural packages are discovered separately.
    adv_root = "https://www.sec.gov/files"
    items.extend([
        Download("sec_adv", f"{adv_root}/adv-filing-data-20111105-20241231-part1.zip",
                 note="Cumulative ADV Part 1 history through 2024-12-31"),
        Download("sec_adv", f"{adv_root}/adv-filing-data-20111105-20241231-part2.zip",
                 note="Cumulative ADV Part 1 continuation through 2024-12-31"),
    ])

    # GLEIF end-of-cutoff Golden Copy: entity level 1, relationships, and
    # relationship reporting exceptions. These ZIP-compressed CSVs are CC0.
    gleif = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes"
    for product in ("lei2", "rr", "repex"):
        items.append(Download(
            "gleif_golden_copy", f"{gleif}/{product}/20251231-1600.csv",
            f"{product}_20251231_1600.csv.zip",
            note="GLEIF Golden Copy at 2025-12-31 16:00 UTC; CC0",
        ))

    # Federal Reserve household finance and distributional accounts.
    fed_files = {
        "scfp2022excel.zip": "SCF summary extract with five implicates per family",
        "scf2022ascii.zip": "SCF full public data with five implicates per family",
        "scf2022rw1s.zip": "SCF replicate weights (Stata)",
        "codebk2022.txt": "SCF codebook",
        "bulletin.macro.txt": "SCF Bulletin table macro source",
        "Standard_Error_Documentation.pdf": "SCF standard-error methodology",
        "2022_scf_changes.txt": "SCF variable changes",
        "2022_showcards.pdf": "SCF survey showcards",
        "scfoutline.2022.pdf": "SCF questionnaire outline",
    }
    for name, note in fed_files.items():
        items.append(Download("fed_scf", f"https://www.federalreserve.gov/econres/files/{name}", note=note))

    for release in ("20240307", "20240607", "20240912", "20241212",
                    "20250313", "20250612", "20250911"):
        items.append(Download(
            "fed_z1_vintages",
            f"https://www.federalreserve.gov/releases/z1/{release}/z1_csv_files.zip",
            f"z1_csv_files_{release}.zip",
            note=f"Z.1 publication vintage released {release}",
        ))
    for name in ("fwtw_data.csv", "fwtw_data_dictionary.txt", "fwtw_templates.xlsx"):
        items.append(Download(
            "fed_fwtw", f"https://www.federalreserve.gov/releases/efa/{name}",
            note="Current revised FWTW product; filter observations and do not use for strict as-of cases",
        ))

    # Credit-union quarterly call reports and national bank branch snapshots.
    for year in (2024, 2025):
        for month in (3, 6, 9, 12):
            items.append(Download(
                "ncua_call_reports",
                f"https://ncua.gov/files/publications/analysis/call-report-data-{year}-{month:02d}.zip",
                note=("NCUA may revise an archive in place; hash and observed time are authoritative"
                      + ("; 2025Q4 was published after the cutoff and is retrospective only"
                         if (year, month) == (2025, 12) else "")),
            ))
    items.append(Download(
        "fdic_sod", "https://api.fdic.gov/banks/docs//sod_variables_definitions.csv",
        "sod_variables_definitions.csv",
    ))

    # Census quarterly corporate financial statements, full and summary.
    for year in (2024, 2025):
        for quarter in range(1, 5):
            for level in ("f", "s"):
                name = f"qfr{str(year)[2:]}q{quarter}{level}.xlsx"
                note = "full tables" if level == "f" else "summary tables"
                if (year, quarter) == (2025, 4):
                    note += "; published 2026-03-23, retrospective only"
                items.append(Download(
                    "census_qfr", f"https://www2.census.gov/econ/qfr/xls/{name}",
                    note=note,
                ))

    # Treasury International Capital tables and portfolio survey packages.
    # Stable Documents URLs redirect to Treasury's current object store.
    # www.treasury.gov currently redirects more reliably than home.treasury.gov
    # to the same official ticdata.treasury.gov objects.
    tic_docs = "https://www.treasury.gov/resource-center/data-chart-center/tic/Documents"
    for name in ("slt1d_globl.csv", "slt3d_globl.csv", "slt2d_history.csv",
                 "slt1f_globl.csv", "slt2f_history.csv", "totalticliabs_hist.csv", "mfhhis01.csv",
                 "cpis_treas_202406_slt.zip", "cpis_treas_202412_slt.zip",
                 "feds_note_nationalitydata_to2024.zip"):
        items.append(Download(
            "treasury_tic", f"{tic_docs}/{name}",
            note="TIC generally classifies by residence; distinguish holdings from flows",
        ))

    items.append(Download(
        "bea_mne", "https://www.bea.gov/sites/default/files/2025-08/usdia2023p-CombinedP.zip",
        note="2023 preliminary US direct investment abroad tables, released before cutoff",
    ))

    # OpenFEMA full public-use snapshots. Current snapshots require date filters
    # during normalization; original high-volume policy data is kept as Parquet.
    items.append(Download(
        "fema_disasters", "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries.parquet",
        note="Current full snapshot; normalize declarationDate through cutoff",
    ))
    # OpenFEMA's unfiltered NFIP responses are chunked and can terminate without
    # an HTTP error. Smaller official OData partitions plus footer validation
    # prevent a truncated response from being mistaken for a valid Parquet file.
    nfip_jurisdictions = (
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "AS", "GU", "MP", "PR", "VI",
    )
    for year in (2024, 2025):
        claims_filter = quote(f"yearOfLoss eq {year}")
        items.append(Download(
            "fema_nfip", f"https://www.fema.gov/api/open/v3/NfipClaims.parquet?$filter={claims_filter}",
            f"NfipClaims_{year}.parquet",
            note="Official yearOfLoss partition from current revised OpenFEMA snapshot",
        ))
        for jurisdiction in nfip_jurisdictions:
            if jurisdiction == "TX":
                continue
            policy_filter = quote(
                f"policyEffectiveDate ge '{year}-01-01' and policyEffectiveDate le '{year}-12-31' "
                f"and propertyState eq '{jurisdiction}'"
            )
            items.append(Download(
                "fema_nfip",
                f"https://www.fema.gov/api/open/v3/NfipPolicies.parquet?$filter={policy_filter}",
                f"NfipPolicies_{year}_{jurisdiction}.parquet",
                note="Official policyEffectiveDate/year and propertyState partition from current revised snapshot",
            ))
        for month in range(1, 13):
            next_year = year + (month == 12)
            next_month = 1 if month == 12 else month + 1
            tx_filter = quote(
                f"policyEffectiveDate ge '{year}-{month:02d}-01' and "
                f"policyEffectiveDate lt '{next_year}-{next_month:02d}-01' and propertyState eq 'TX'"
            )
            items.append(Download(
                "fema_nfip",
                f"https://www.fema.gov/api/open/v3/NfipPolicies.parquet?$filter={tx_filter}",
                f"NfipPolicies_{year}_TX_{month:02d}.parquet",
                note="Official Texas monthly partition from current revised OpenFEMA snapshot",
            ))
        missing_state_filter = quote(
            f"policyEffectiveDate ge '{year}-01-01' and policyEffectiveDate le '{year}-12-31' "
            "and propertyState eq null"
        )
        items.append(Download(
            "fema_nfip",
            f"https://www.fema.gov/api/open/v3/NfipPolicies.parquet?$filter={missing_state_filter}",
            f"NfipPolicies_{year}_MISSING_STATE.parquet",
            note="Official partition retaining rows with null propertyState",
        ))

    # EIA bulk series and real historical forecast vintages.
    eia_bulk = "https://www.eia.gov/opendata/bulk"
    for name in ("EBA.zip", "ELEC.zip", "NG.zip", "PET.zip", "STEO.zip"):
        items.append(Download(
            "eia_bulk", f"{eia_bulk}/{name}",
            note="Current revised bulk snapshot; observation dates must be cutoff-filtered",
        ))
    items.append(Download("eia_bulk", f"{eia_bulk}/manifest.txt"))
    months = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    for year in (2024, 2025):
        for month in months:
            name = f"{month}{str(year)[2:]}_base.xlsx"
            items.append(Download(
                "eia_steo_vintages", f"https://www.eia.gov/outlooks/steo/archives/{name}",
                note="Immutable STEO publication vintage",
            ))

    # Quarterly agreement snapshots overlap at the file level, but each vintage
    # enables distinct high-value temporal cases (new/withdrawn agreements and
    # changes in APR, fees, and issuer terms), so retain every 2024--2025 quarter.
    for year in (2024, 2025):
        for quarter in range(1, 5):
            name = f"{year}_Q{quarter}.zip"
            items.append(Download(
                "cfpb_credit_card_agreements",
                f"https://files.consumerfinance.gov/a/assets/bulk_agreements/{name}",
                note=("Fixed quarterly issuer agreement archive for temporal terms/change cases; "
                      "publication availability must be checked separately from agreement dates"),
            ))
    prepaid = "https://files.consumerfinance.gov/a/assets/prepaid-agreements"
    for name in ("bulk.zip", "prepaid_metadata_all_agreements.csv", "prepaid_metadata_recent_agreements.csv"):
        items.append(Download(
            "cfpb_prepaid_agreements", f"{prepaid}/{name}",
            note="Current agreement snapshot; apply agreement and withdrawal cutoff dates",
        ))

    return items


def _adv_2025_downloads() -> tuple[list[Download], dict]:
    """Discover only structured 2025 ADV files from SEC's official metadata."""
    url = "https://reports.adviserinfo.sec.gov/reports/foia/reports_metadata.json"
    response = _session().get(url, timeout=120)
    response.raise_for_status()
    metadata = response.json()
    items: list[Download] = []
    for key in ("advFilingData", "advW", "advFirmCRS"):
        group = metadata.get(key, {})
        year = group.get("2025") or group.get(2025)
        if not year:
            continue
        files = year.get("files", []) if isinstance(year, dict) else []
        for entry in files:
            name = entry.get("fileName") or entry.get("filename")
            if not name:
                continue
            file_url = f"https://reports.adviserinfo.sec.gov/reports/foia/{key}/2025/{name}"
            items.append(Download(
                "sec_adv", file_url, f"2025_{key}_{name}",
                note=("2025 period structured ADV export; current metadata indicates a post-cutoff "
                      "upload/backfill, so it is retrospective rather than strict as-of evidence"),
            ))
    return items, {"url": url, "content": response.content, "metadata": metadata}


def _ofac_downloads() -> tuple[list[Download], list[dict]]:
    """Discover the official 2024--2025 OFAC delta archive.

    OFAC does not retain historical full-list snapshots.  These files therefore
    support designation/change-event cases; the current full exports below are
    explicitly unsuitable as a 2025-12-31 state snapshot.
    """
    root = "https://sanctionslistservice.ofac.treas.gov/api"
    session = _session()
    items = [
        Download("ofac_sanctions", f"{root}/PublicationPreview/exports/SDN_ADVANCED.XML",
                 "SDN_ADVANCED_current.xml",
                 note="Current full export observed after cutoff; not a 2025-12-31 snapshot"),
        Download("ofac_sanctions", f"{root}/PublicationPreview/exports/SDN.XML",
                 "SDN_current.xml",
                 note="Current full export observed after cutoff; not a 2025-12-31 snapshot"),
        Download("ofac_sanctions", f"{root}/PublicationPreview/exports/CONSOLIDATED.XML",
                 "CONSOLIDATED_current.xml",
                 note="Current full export observed after cutoff; not a 2025-12-31 snapshot"),
    ]
    discoveries: list[dict] = []
    for year in (2024, 2025):
        endpoint = f"{root}/PublicationPreview/GetDeltaFileArchive?year={year}"
        response = session.post(endpoint, json=None, timeout=120)
        response.raise_for_status()
        payload = response.json()
        discoveries.append({"year": year, "url": endpoint, "content": response.content})
        entries = payload if isinstance(payload, list) else payload.get("files", payload.get("data", []))
        for entry in entries:
            link = (entry.get("downloadLink") or entry.get("downloadlink") or
                    entry.get("fileName") or entry.get("filename"))
            if not link:
                continue
            filename = Path(link).name
            url = f"{root}/download/delta?filename={quote(link, safe='')}"
            items.append(Download(
                "ofac_sanctions", url, filename,
                note="Official OFAC daily delta; use for change events, not fuzzy name-hit adjudication",
            ))
    return items, discoveries


def _write_bytes(dataset: str, filename: str, content: bytes, url: str, note: str) -> dict:
    raw = RAW_DIR / dataset
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / filename
    path.write_bytes(content)
    item = Download(dataset, url, filename, note=note)
    return _file_record(path, item, "downloaded", "application/json")


def _fetch_fdic_sod() -> tuple[list[dict], list[dict]]:
    """Fetch all SOD rows; the API silently defaults to only 100 otherwise."""
    dataset = "fdic_sod"
    records, failures = [], []
    session = _session()
    for year in (2024, 2025):
        rows: list[str] = []
        header: str | None = None
        urls: list[str] = []
        try:
            for offset in range(0, 80000, 10000):
                params = {
                    "filters": f"YEAR:{year}", "format": "csv", "download": "true",
                    "limit": 10000, "offset": offset,
                }
                response = session.get("https://api.fdic.gov/banks/sod", params=params, timeout=300)
                response.raise_for_status()
                lines = response.text.splitlines()
                urls.append(response.url)
                if not lines:
                    break
                if header is None:
                    header = lines[0]
                page_rows = lines[1:]
                rows.extend(page_rows)
                if len(page_rows) < 10000:
                    break
            content = ((header or "") + "\n" + "\n".join(rows) + "\n").encode("utf-8")
            records.append(_write_bytes(
                dataset, f"sod_{year}_all.csv", content, urls[0],
                f"All {len(rows):,} rows combined from FDIC API pages; page URLs recorded in sidecar",
            ))
            sidecar = json.dumps({"year": year, "row_count": len(rows), "page_urls": urls}, indent=2).encode()
            records.append(_write_bytes(dataset, f"sod_{year}_pages.json", sidecar, urls[0], "Pagination provenance"))
        except Exception as exc:
            failures.append({"url": "https://api.fdic.gov/banks/sod", "file": f"sod_{year}_all.csv", "error": repr(exc)})
    return records, failures


def _usaspending_body() -> dict:
    return {
        "filters": {
            "time_period": [{"start_date": "2021-01-01", "end_date": FREEZE_DATE}],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "limit": 100,
    }


def _fetch_usaspending() -> tuple[list[dict], list[dict]]:
    """Download bounded nationwide award aggregates with reproducible pages."""
    dataset = "usaspending"
    session = _session()
    records, failures = [], []
    base = "https://api.usaspending.gov/api/v2/search/spending_by_category"
    for category in ("recipient", "awarding_agency", "naics", "state_territory"):
        endpoint = f"{base}/{category}/"
        pages = []
        try:
            for page in range(1, 11):
                body = _usaspending_body()
                body["page"] = page
                response = session.post(endpoint, json=body, timeout=300)
                response.raise_for_status()
                payload = response.json()
                pages.append(payload)
                page_meta = payload.get("page_metadata", {})
                if not page_meta.get("hasNext", page_meta.get("has_next_page", False)):
                    break
            content = json.dumps(
                {"request": _usaspending_body(), "pages": pages}, indent=2, ensure_ascii=False
            ).encode("utf-8")
            records.append(_write_bytes(
                dataset, f"contracts_2021_2025_by_{category}.json", content, endpoint,
                "Nationwide contract aggregate, at most top 1,000 categories; API request embedded",
            ))
        except Exception as exc:
            failures.append({"url": endpoint, "file": f"contracts_2021_2025_by_{category}.json", "error": repr(exc)})
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    downloads = _static_downloads()
    seed_records: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, list[dict]] = defaultdict(list)
    try:
        adv_items, metadata = _adv_2025_downloads()
        downloads.extend(adv_items)
        seed_records["sec_adv"].append(_write_bytes(
            "sec_adv", "reports_metadata_observed.json", metadata["content"], metadata["url"],
            "Discovery metadata observed at download time; file availability may be retrospective",
        ))
    except Exception as exc:
        failures["sec_adv"].append({
            "url": "https://reports.adviserinfo.sec.gov/reports/foia/reports_metadata.json",
            "file": "reports_metadata_observed.json", "error": repr(exc),
        })

    try:
        ofac_items, discoveries = _ofac_downloads()
        downloads.extend(ofac_items)
        for discovery in discoveries:
            seed_records["ofac_sanctions"].append(_write_bytes(
                "ofac_sanctions", f"delta_archive_{discovery['year']}_observed.json",
                discovery["content"], discovery["url"],
                "Official delta archive discovery response observed at download time",
            ))
    except Exception as exc:
        failures["ofac_sanctions"].append({
            "url": "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/GetDeltaFileArchive",
            "file": "delta_archive_discovery", "error": repr(exc),
        })

    records: dict[str, list[dict]] = defaultdict(list, seed_records)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, item): item for item in downloads}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, record = future.result()
                records[item.dataset].append(record)
                print(f"OK   {item.dataset:28s} {item.name:52s} {record['bytes'] / 2**20:10.2f} MiB", flush=True)
            except Exception as exc:
                failures[item.dataset].append({"url": item.url, "file": item.name, "error": repr(exc)})
                print(f"FAIL {item.dataset:28s} {item.name}: {exc}", flush=True)

    for fetcher in (_fetch_fdic_sod, _fetch_usaspending):
        extra_records, extra_failures = fetcher()
        for record in extra_records:
            # file paths are raw/foo; identify the source directory from caller output.
            dataset = "fdic_sod" if "sod_" in Path(record["file"]).name else "usaspending"
            records[dataset].append(record)
        for failure in extra_failures:
            dataset = "fdic_sod" if "fdic" in failure["url"] else "usaspending"
            failures[dataset].append(failure)

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
