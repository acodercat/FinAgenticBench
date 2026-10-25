"""Replace cohort and single-period slices with full populations, 2019–2025.

Acquisition record for the data-quality audit of 2026-08-26
(`docs/data_quality_audit.md`). Every source below existed as a hand-picked
cohort or a single month/quarter; this module fetches the complete population
for the 2019–2025 window into the same `raw/` directories, recording provenance
through the shared `sources.json` writer. Nothing is deleted here: governed
tables are rebuilt from these files in a separate step, and the slices are
retired only after every dependent case has been re-verified on the full data.

The 2026-08-27 additions cover the sources the first pass deferred: FS&Notes
2019–2023, QCEW annual singlefiles 2019–2023 and the full 2024 HMDA LAR pulled
state by state through the data-browser API (the S3 bulk file returns 403).
Still deferred: Fed DFAST result tables per year.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import json

from scripts.ingest.fetch.fetch_expansion import (
    Download, _download_one, _write_manifest,
)


SEC = "https://www.sec.gov"
YEARS = range(2019, 2026)
QUARTERS = [(y, q) for y in YEARS for q in (1, 2, 3, 4)]
FDIC_FIELDS = [
    "REPDTE", "CERT", "NAME", "ASSET", "DEP", "DEPDOM", "EQ", "NETINC",
    "LNLSNET", "LNATRES", "LNCON", "LNCI", "LNRE", "NCLNLS", "NTLNLS",
    "ROA", "ROE", "NIMY", "INTINC", "EINTEXP", "NONII", "NONIX", "CHBAL",
    "SC", "FREPO", "RBCT1J", "RBCT2", "RBCRWAJ", "RBCRWAK", "RBC1AAJ",
]


def _quarter_end(year: int, quarter: int) -> str:
    return f"{year}{'0331 0630 0930 1231'.split()[quarter - 1]}"


def _sec_bulk() -> list[Download]:
    return [Download(
        "sec", f"{SEC}/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "companyfacts.zip", 1,
        "Full CompanyFacts bulk archive, every CIK; same JSON shape as the "
        "20-company cohort it replaces. Facts filed after the freeze date are "
        "dropped at build time.",
    )]


def _sec_insider() -> list[Download]:
    return [Download(
        "sec_insider",
        f"{SEC}/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip",
        None, 1, "Insider Transactions Data Set, all filers",
    ) for y, q in QUARTERS]


def _sec_nport() -> list[Download]:
    return [Download(
        "sec_nport", f"{SEC}/files/dera/data/form-n-port-data-sets/{y}q{q}_nport.zip",
        None, 1, "Form N-PORT Data Set, all filers",
    ) for y, q in QUARTERS if (y, q) >= (2019, 4)]


def _sec_13f() -> list[Download]:
    windows = [("01dec", "28feb", -1, 0), ("01mar", "31may", 0, 0),
               ("01jun", "31aug", 0, 0), ("01sep", "30nov", 0, 0)]
    items = []
    for y in YEARS:
        for start, end, dy_start, dy_end in windows:
            ys, ye = y + dy_start, y + dy_end
            if start == "01dec" and y == 2019:
                continue
            leap = "29feb" if ye % 4 == 0 else "28feb"
            end = leap if end == "28feb" else end
            items.append(Download(
                "sec_13f",
                f"{SEC}/files/structureddata/data/form-13f-data-sets/{start}{ys}-{end}{ye}_form13f.zip",
                None, 1, "Form 13F Data Set, all filers",
            ))
    return items


def _finra_daily() -> list[Download]:
    items, day = [], date(2024, 1, 1)
    while day <= date(2025, 12, 31):
        if day.weekday() < 5:
            stamp = day.strftime("%Y%m%d")
            items.append(Download(
                "finra_short_volume",
                f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt",
                None, 1, "Consolidated NMS daily short-sale volume",
            ))
        day += timedelta(days=1)
    return items


def _treasury_curve() -> list[Download]:
    base = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
    return [Download(
        "treasury_rates",
        f"{base}/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv",
        f"daily_treasury_yield_curve_{y}.csv", 1, "Daily par yield curve, full year",
    ) for y in YEARS]


def _cftc() -> list[Download]:
    return [Download(
        "cftc_cot", f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip",
        None, 1, "Traders in Financial Futures, full year",
    ) for y in YEARS]


def _bis() -> list[Download]:
    return [Download("bis", f"https://data.bis.org/static/bulk/{code}_csv_flat.zip", None, 1, note)
            for code, note in (
                ("WS_EER", "Effective exchange rates, full history"),
                ("WS_GLI", "Global liquidity indicators, full history"),
                ("WS_OTC_DERIV2", "OTC derivatives statistics, full history"),
            )]


def _fdic_financials() -> list[Download]:
    base = "https://api.fdic.gov/banks/financials"
    items = []
    for y, q in QUARTERS:
        repdte = _quarter_end(y, q)
        url = (f"{base}?filters=REPDTE:{repdte}&fields={','.join(FDIC_FIELDS)}"
               f"&limit=10000&format=csv&download=true")
        items.append(Download("fdic_bankfind", url, f"all_banks_financials_{repdte}.csv", 1,
                              "Every FDIC-insured institution reporting at this date"))
    return items


def _fdic_sod() -> list[Download]:
    base = "https://banks.data.fdic.gov/api/sod"
    items = []
    for y in YEARS:
        for offset in range(0, 90000, 10000):
            url = (f"{base}?filters=YEAR:{y}&limit=10000&offset={offset}"
                   f"&format=csv&download=true")
            items.append(Download("fdic_sod", url, f"sod_{y}_offset{offset:05d}.csv", 1,
                                  "Summary of Deposits, all branches (paged)"))
    return items


def _ncua() -> list[Download]:
    return [Download(
        "ncua_call_reports",
        f"https://ncua.gov/files/publications/analysis/call-report-data-{y}-{m:02d}.zip",
        None, 1, "NCUA quarterly call report bulk",
    ) for y in YEARS for m in (3, 6, 9, 12) if (y, m) <= (2025, 9)]


US_STATES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR"
).split()


def _sec_notes() -> list[Download]:
    return [Download(
        "sec_notes",
        f"{SEC}/files/dera/data/financial-statement-notes-data-sets/{y}q{q}_notes.zip",
        None, 1, "Financial Statement and Notes Data Set, all filers",
    ) for y, q in QUARTERS if y <= 2023]


def _bls_qcew() -> list[Download]:
    return [Download(
        "bls_qcew", f"https://data.bls.gov/cew/data/files/{y}/csv/{y}_annual_singlefile.zip",
        None, 1, "QCEW annual averages, every area x industry x ownership",
    ) for y in YEARS if y <= 2023]


def _hmda() -> list[Download]:
    return [Download(
        "hmda", f"https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2024&states={state}",
        f"lar_2024_{state}.csv", 1, "Full 2024 LAR for one state via the HMDA data browser API",
    ) for state in US_STATES]


NOAA_DETAIL_FILES = {
    2019: "StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz",
    2020: "StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz",
    2021: "StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz",
    2022: "StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz",
    2023: "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz",
}


def _noaa() -> list[Download]:
    root = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
    return [Download("noaa_storm_events", f"{root}/{name}", None, 1,
                     f"Storm Events details, {year} (directory listing of 2026-08-27)")
            for year, name in NOAA_DETAIL_FILES.items()]


def _dfast() -> list[Download]:
    return [Download(
        "fed_stress_tests",
        f"https://www.federalreserve.gov/supervisionreg/files/public_results_DFAST_{y}.csv",
        None, 1, "DFAST public results CSV; the Board publishes this file from the 2023 exercise onward",
    ) for y in range(2023, 2025)]


def _sec_quarterly(dataset: str, path: str, suffix: str, note: str, start=(2019, 1)) -> list[Download]:
    return [Download(dataset, f"{SEC}/files/{path}/{y}q{q}_{suffix}.zip", None, 1, note)
            for y, q in QUARTERS if (y, q) >= start and y <= 2023]


def _form_d() -> list[Download]:
    return _sec_quarterly("sec_form_d", "structureddata/data/form-d-data-sets", "d", "Form D data set, all filers")


def _form_c() -> list[Download]:
    return _sec_quarterly("sec_form_c", "dera/data/crowdfunding-offerings-data-sets", "cf", "Crowdfunding offerings data set")


def _bdc() -> list[Download]:
    return _sec_quarterly("sec_bdc", "datastandardsinnovation/data/business-development-company-bdc-data-sets", "bdc",
                          "BDC data set, all filers")


def _y9c() -> list[Download]:
    return [Download(
        "ffiec_y9c", f"https://www.ffiec.gov/npw/FinancialReport/ReturnBHCFZipFiles?zipfilename=BHCF{_quarter_end(y, q)}.ZIP",
        f"BHCF{_quarter_end(y, q)}.ZIP", 1, "FR Y-9C bulk, one quarter",
    ) for y, q in QUARTERS if y <= 2023]


def _nfip() -> list[Download]:
    return [Download(
        "fema_nfip", f"https://www.fema.gov/api/open/v3/NfipClaims.parquet?$filter=yearOfLoss%20eq%20{y}",
        f"NfipClaims_{y}.parquet", 1, "OpenFEMA NFIP claims, one year of loss",
    ) for y in YEARS if y <= 2023]


def _nmfp() -> list[Download]:
    # Quarterly archives through 2024; SEC switched to monthly filing-window
    # archives (YYYYMMDD-YYYYMMDD_nmfp.zip) in 2025, which are already on hand.
    return [Download("sec_nmfp", f"{SEC}/files/dera/data/form-n-mfp-data-sets/{y}q{q}_nmfp.zip", None, 1,
                     "Form N-MFP data set, all filers") for y, q in QUARTERS if y <= 2024]


GROUPS = {
    "sec_notes": _sec_notes, "bls_qcew": _bls_qcew, "hmda": _hmda, "sec_nmfp": _nmfp,
    "sec_form_d": _form_d, "sec_form_c": _form_c, "sec_bdc": _bdc, "ffiec_y9c": _y9c, "fema_nfip": _nfip,
    "noaa_storm_events": _noaa, "fed_stress_tests": _dfast,
    "sec": _sec_bulk, "sec_insider": _sec_insider, "sec_nport": _sec_nport,
    "sec_13f": _sec_13f, "finra_short_volume": _finra_daily,
    "treasury_rates": _treasury_curve, "cftc_cot": _cftc, "bis": _bis,
    "fdic_bankfind": _fdic_financials, "fdic_sod": _fdic_sod,
    "ncua_call_reports": _ncua,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(GROUPS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = args.only or list(GROUPS)
    summary = {}
    for dataset in selected:
        items = GROUPS[dataset]()
        records, failures = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_download_one, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    _, record = future.result()
                    records.append(record)
                except Exception as exc:
                    failures.append({"url": item.url, "file": item.name, "error": repr(exc)[:300]})
        _write_manifest(dataset, records, failures)
        summary[dataset] = {"files": len(records), "failures": len(failures)}
        print(f"{dataset}: {len(records)} files, {len(failures)} failures", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
