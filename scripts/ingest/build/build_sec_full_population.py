"""Build the full-population SEC tables from the bulk EDGAR archives.

Replaces the 20-company cohort behind `sec.company_facts`, `sec.filings` and
`sec.companies` (see `docs/data_quality_audit.md`). Inputs are the two bulk
archives SEC publishes for every CIK — `companyfacts.zip` (all XBRL facts) and
`submissions.zip` (all filing metadata) — plus SEC's ticker/exchange map.

Two tiers, both with objective membership rules and no case-specific logic:

* **Deep tier** — `sec.company_facts`: every numeric 10-K/10-Q fact, all
  concepts, for the UNIVERSE_SIZE largest 10-K filers ranked by their latest
  fiscal-year `us-gaap:Assets` (USD) reported on or before the freeze date.
  Same columns as the cohort table it replaces, so existing loaders and cases
  are unchanged; the 20 cohort companies all rank inside the top 350.
* **Population tier** — `sec.financial_statements`: one row per 10-K/10-Q
  accession for EVERY XBRL filer, carrying STATEMENT_CONCEPTS as columns.
  Values are read from the filing's own facts with a fixed rule: an instant
  concept takes the observation whose `end` equals the filing's report date;
  a duration concept yields `<concept>_ytd` (the observation ending at the
  report date with the earliest start — fiscal year on a 10-K, fiscal
  year-to-date on a 10-Q) and `<concept>_q` (the observation ending at the
  report date spanning 80–100 days — a single quarter, absent on most 10-Ks).
  Nothing is imputed: a concept the filer did not tag is left empty.

`sec.filings` lists 10-K/10-Q/20-F/40-F family filings, 2019–2025, for every
CIK present in `companyfacts.zip`; `sec.companies` describes the deep-tier
universe with its rank, ranking basis, ticker, exchange and SIC code.

Facts filed after the freeze date, and periods ending before WINDOW_START, are
excluded at build time as the cohort builder did.
"""

from __future__ import annotations

import csv
import gzip
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
SEC = DATASETS / "sec"
RAW = RAW_DIR / "sec"
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "sec_full_population_build.json"

# Tickers as of the freeze date where SEC's current mapping has since moved.
# XOM now resolves to ExxonMobil Holdings Corp (CIK 2115436), the post-2025
# holding company; the frozen registrant with the 2019-2025 filings is CIK 34088.
FREEZE_DATE_TICKERS = {34088: "XOM"}
FREEZE_DATE = "2025-12-31"
WINDOW_START = "2019-01-01"
UNIVERSE_SIZE = 500
FACT_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
FILING_FORMS = FACT_FORMS | {"20-F", "20-F/A", "40-F", "40-F/A"}

FACT_COLUMNS = [
    "cik", "ticker", "company_name", "taxonomy", "concept", "label", "description",
    "unit", "value", "start_date", "end_date", "filed_date", "form", "fiscal_year",
    "fiscal_period", "frame", "accession",
]
FILING_COLUMNS = [
    "cik", "ticker", "company_name", "accession", "form", "filing_date", "report_date",
    "acceptance_datetime", "fiscal_year_end", "primary_document", "is_amendment",
    "source_url",
]

# (taxonomy, concept, unit, kind). kind: "instant" or "duration".
STATEMENT_CONCEPTS = [
    ("us-gaap", "Assets", "USD", "instant"),
    ("us-gaap", "AssetsCurrent", "USD", "instant"),
    ("us-gaap", "Liabilities", "USD", "instant"),
    ("us-gaap", "LiabilitiesCurrent", "USD", "instant"),
    ("us-gaap", "StockholdersEquity", "USD", "instant"),
    ("us-gaap", "LiabilitiesAndStockholdersEquity", "USD", "instant"),
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", "instant"),
    ("us-gaap", "AccountsReceivableNetCurrent", "USD", "instant"),
    ("us-gaap", "InventoryNet", "USD", "instant"),
    ("us-gaap", "PropertyPlantAndEquipmentNet", "USD", "instant"),
    ("us-gaap", "Goodwill", "USD", "instant"),
    ("us-gaap", "LongTermDebt", "USD", "instant"),
    ("us-gaap", "LongTermDebtNoncurrent", "USD", "instant"),
    ("us-gaap", "OperatingLeaseLiability", "USD", "instant"),
    ("us-gaap", "RetainedEarningsAccumulatedDeficit", "USD", "instant"),
    ("us-gaap", "CommonStockSharesOutstanding", "shares", "instant"),
    ("dei", "EntityCommonStockSharesOutstanding", "shares", "instant"),
    ("dei", "EntityPublicFloat", "USD", "instant"),
    ("us-gaap", "Revenues", "USD", "duration"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", "duration"),
    ("us-gaap", "CostOfRevenue", "USD", "duration"),
    ("us-gaap", "GrossProfit", "USD", "duration"),
    ("us-gaap", "ResearchAndDevelopmentExpense", "USD", "duration"),
    ("us-gaap", "SellingGeneralAndAdministrativeExpense", "USD", "duration"),
    ("us-gaap", "OperatingIncomeLoss", "USD", "duration"),
    ("us-gaap", "InterestExpense", "USD", "duration"),
    ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "USD", "duration"),
    ("us-gaap", "IncomeTaxExpenseBenefit", "USD", "duration"),
    ("us-gaap", "NetIncomeLoss", "USD", "duration"),
    ("us-gaap", "ComprehensiveIncomeNetOfTax", "USD", "duration"),
    ("us-gaap", "EarningsPerShareBasic", "USD/shares", "duration"),
    ("us-gaap", "EarningsPerShareDiluted", "USD/shares", "duration"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares", "duration"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", "duration"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD", "duration"),
    ("us-gaap", "NetCashProvidedByUsedInInvestingActivities", "USD", "duration"),
    ("us-gaap", "NetCashProvidedByUsedInFinancingActivities", "USD", "duration"),
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD", "duration"),
    ("us-gaap", "DepreciationDepletionAndAmortization", "USD", "duration"),
    ("us-gaap", "ShareBasedCompensation", "USD", "duration"),
    ("us-gaap", "PaymentsOfDividends", "USD", "duration"),
    ("us-gaap", "PaymentsForRepurchaseOfCommonStock", "USD", "duration"),
]
STATEMENT_HEAD = [
    "cik", "ticker", "company_name", "accession", "form", "fiscal_year",
    "fiscal_period", "report_date", "filed_date", "sic",
]


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _days(start: str, end: str) -> int:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days


def _in_window(obs: dict) -> bool:
    filed, end = obs.get("filed") or "", obs.get("end") or ""
    return (
        obs.get("form") in FACT_FORMS and _numeric(obs.get("val"))
        and filed <= FREEZE_DATE and end >= WINDOW_START
    )


def _cik(payload: dict) -> int | None:
    """The archive mixes int and str for `cik`; everything downstream keys on int."""
    value = payload.get("cik")
    return int(value) if value not in (None, "") else None


def _iter_facts(payload: dict):
    for taxonomy, concepts in (payload.get("facts") or {}).items():
        for concept, definition in concepts.items():
            for unit, observations in (definition.get("units") or {}).items():
                for obs in observations:
                    yield taxonomy, concept, definition, unit, obs


# --------------------------------------------------------------------------- pass 1

def rank_universe(facts_zip: zipfile.ZipFile, members: list[str]) -> dict:
    """Latest fiscal-year total assets per CIK, from 10-K facts inside the window."""
    latest = {}
    for member in members:
        payload = json.loads(facts_zip.read(member))
        cik = _cik(payload)
        assets = (payload.get("facts") or {}).get("us-gaap", {}).get("Assets", {})
        best = None
        for obs in (assets.get("units") or {}).get("USD", []):
            if not _in_window(obs) or not obs.get("form", "").startswith("10-K"):
                continue
            if obs.get("fp") != "FY":
                continue
            key = (obs["end"], obs["filed"])
            if best is None or key > best[0]:
                best = (key, obs["val"])
        if best is not None and best[1] > 0:
            latest[cik] = {"assets_period_end": best[0][0], "assets_filed": best[0][1], "total_assets_usd": best[1]}
    return latest


# --------------------------------------------------------------------------- submissions

def read_submissions(cik_set: set[int]) -> tuple[dict, list[dict], dict]:
    """Company identity, in-window filings, and accession → report date, per CIK."""
    identities, filings, report_dates = {}, [], {}
    with zipfile.ZipFile(RAW / "submissions.zip") as archive:
        for cik in sorted(cik_set):
            name = f"CIK{cik:010d}.json"
            try:
                payload = json.loads(archive.read(name))
            except KeyError:
                continue
            tickers = payload.get("tickers") or []
            exchanges = payload.get("exchanges") or []
            identities[cik] = {
                "cik": cik, "name": payload.get("name"),
                "ticker": FREEZE_DATE_TICKERS.get(cik, tickers[0] if tickers else None),
                "exchange": exchanges[0] if exchanges else None,
                "sic": payload.get("sic"), "sic_description": payload.get("sicDescription"),
                "fiscal_year_end": payload.get("fiscalYearEnd"),
                "state_of_incorporation": payload.get("stateOfIncorporation"),
            }
            recent = (payload.get("filings") or {}).get("recent") or {}
            for index in range(len(recent.get("accessionNumber", []))):
                item = {key: values[index] for key, values in recent.items()}
                accession = item["accessionNumber"]
                report_dates[accession] = item.get("reportDate") or ""
                form, filed = item.get("form"), item.get("filingDate") or ""
                if form not in FILING_FORMS or not (WINDOW_START <= filed <= FREEZE_DATE):
                    continue
                document = item.get("primaryDocument") or ""
                filings.append({
                    "cik": cik, "ticker": identities[cik]["ticker"],
                    "company_name": identities[cik]["name"], "accession": accession,
                    "form": form, "filing_date": filed, "report_date": item.get("reportDate"),
                    "acceptance_datetime": item.get("acceptanceDateTime"),
                    "fiscal_year_end": identities[cik]["fiscal_year_end"],
                    "primary_document": document, "is_amendment": form.endswith("/A"),
                    "source_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}",
                })
    return identities, filings, report_dates


# --------------------------------------------------------------------------- pass 2

def facts_rows(cik: int, identity: dict, payload: dict):
    for taxonomy, concept, definition, unit, obs in _iter_facts(payload):
        if not _in_window(obs):
            continue
        yield [
            cik, identity["ticker"], identity["name"], taxonomy, concept,
            definition.get("label"), definition.get("description"), unit, obs["val"],
            obs.get("start"), obs.get("end"), obs.get("filed"), obs.get("form"),
            obs.get("fy"), obs.get("fp"), obs.get("frame"), obs.get("accn"),
        ]


def statement_rows(cik: int, identity: dict, payload: dict, report_dates: dict):
    """One row per 10-K/10-Q accession, values chosen by the documented rule."""
    wanted = {(t, c): (u, k) for t, c, u, k in STATEMENT_CONCEPTS}
    by_accession: dict[str, dict] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for taxonomy, concept, definition, unit, obs in _iter_facts(payload):
        spec = wanted.get((taxonomy, concept))
        if spec is None or unit != spec[0] or not _in_window(obs):
            continue
        accession = obs.get("accn")
        if not accession:
            continue
        meta.setdefault(accession, {"form": obs.get("form"), "fy": obs.get("fy"), "fp": obs.get("fp"), "filed": obs.get("filed")})
        report = report_dates.get(accession) or ""
        if obs.get("end") != report:
            continue
        cell = by_accession[accession]
        if spec[1] == "instant":
            if obs.get("start") is None:
                cell.setdefault(concept, obs["val"])
            continue
        start = obs.get("start")
        if start is None:
            continue
        span = _days(start, report)
        ytd = cell.get(f"{concept}_ytd")
        if ytd is None or start < ytd[0]:
            cell[f"{concept}_ytd"] = (start, obs["val"])
        if 80 <= span <= 100:
            cell.setdefault(f"{concept}_q", (start, obs["val"]))
    for accession, cell in by_accession.items():
        info = meta[accession]
        row = {
            "cik": cik, "ticker": identity["ticker"], "company_name": identity["name"],
            "accession": accession, "form": info["form"], "fiscal_year": info["fy"],
            "fiscal_period": info["fp"], "report_date": report_dates.get(accession),
            "filed_date": info["filed"], "sic": identity["sic"],
        }
        for _, concept, _, kind in STATEMENT_CONCEPTS:
            if kind == "instant":
                row[concept] = cell.get(concept)
            else:
                for suffix in ("_ytd", "_q"):
                    value = cell.get(f"{concept}{suffix}")
                    row[f"{concept}{suffix}"] = value[1] if value else None
        yield row


def statement_columns() -> list[str]:
    columns = list(STATEMENT_HEAD)
    for _, concept, _, kind in STATEMENT_CONCEPTS:
        columns.extend([concept] if kind == "instant" else [f"{concept}_ytd", f"{concept}_q"])
    return columns


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    facts_zip = zipfile.ZipFile(RAW / "companyfacts.zip")
    members = sorted(m for m in facts_zip.namelist() if m.endswith(".json"))
    print(f"companyfacts members: {len(members)}", flush=True)

    ranked = rank_universe(facts_zip, members)
    order = sorted(ranked.items(), key=lambda kv: (-kv[1]["total_assets_usd"], kv[0]))
    universe = {cik: dict(info, rank=index + 1) for index, (cik, info) in enumerate(order[:UNIVERSE_SIZE])}
    print(f"10-K filers ranked: {len(ranked)}; universe: {len(universe)}; "
          f"smallest total assets in universe: {order[UNIVERSE_SIZE - 1][1]['total_assets_usd']:,.0f}", flush=True)

    all_ciks = {_cik(json.loads(facts_zip.read(m))) for m in members}
    all_ciks.discard(None)
    identities, filings, report_dates = read_submissions(all_ciks)
    print(f"submissions read: {len(identities)} identities, {len(filings)} in-window filings", flush=True)

    facts_path = SEC / "facts.csv.gz"
    statements_path = SEC / "financial_statements.csv.gz"
    fact_count = statement_count = 0
    with gzip.open(facts_path, "wt", newline="", compresslevel=6) as facts_handle, \
            gzip.open(statements_path, "wt", newline="", compresslevel=6) as statements_handle:
        facts_writer = csv.writer(facts_handle)
        facts_writer.writerow(FACT_COLUMNS)
        statement_writer = csv.DictWriter(statements_handle, fieldnames=statement_columns())
        statement_writer.writeheader()
        for index, member in enumerate(members):
            payload = json.loads(facts_zip.read(member))
            cik = _cik(payload)
            identity = identities.get(cik)
            if identity is None:
                continue
            for row in statement_rows(cik, identity, payload, report_dates):
                statement_writer.writerow(row)
                statement_count += 1
            if cik in universe:
                for row in facts_rows(cik, identity, payload):
                    facts_writer.writerow(row)
                    fact_count += 1
            if index % 2000 == 0:
                print(f"  {index}/{len(members)} facts={fact_count:,} statements={statement_count:,}", flush=True)

    with (SEC / "companies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "ticker", "cik", "name", "exchange", "sic", "sic_description",
            "state_of_incorporation", "fiscal_year_end", "universe_rank",
            "total_assets_usd", "assets_period_end", "assets_filed",
        ])
        writer.writeheader()
        for cik, info in sorted(universe.items(), key=lambda kv: kv[1]["rank"]):
            identity = identities.get(cik, {"cik": cik})
            writer.writerow({
                "ticker": identity.get("ticker"), "cik": cik, "name": identity.get("name"),
                "exchange": identity.get("exchange"), "sic": identity.get("sic"),
                "sic_description": identity.get("sic_description"),
                "state_of_incorporation": identity.get("state_of_incorporation"),
                "fiscal_year_end": identity.get("fiscal_year_end"),
                "universe_rank": info["rank"], "total_assets_usd": info["total_assets_usd"],
                "assets_period_end": info["assets_period_end"], "assets_filed": info["assets_filed"],
            })
    with (SEC / "filings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILING_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(filings, key=lambda r: (r["cik"], r["filing_date"], r["accession"])))

    report = {
        "built_at": started, "freeze_date": FREEZE_DATE, "window_start": WINDOW_START,
        "universe_rule": f"top {UNIVERSE_SIZE} 10-K filers by latest fiscal-year us-gaap:Assets (USD) filed on or before the freeze date",
        "universe_min_total_assets_usd": order[UNIVERSE_SIZE - 1][1]["total_assets_usd"],
        "ranked_10k_filers": len(ranked), "xbrl_ciks": len(all_ciks),
        "rows": {"company_facts": fact_count, "financial_statements": statement_count,
                 "filings": len(filings), "companies": len(universe)},
        "statement_concepts": [f"{t}:{c}" for t, c, _, _ in STATEMENT_CONCEPTS],
        "selection_rule": {
            "instant": "observation whose end equals the filing's report date",
            "ytd": "duration observation ending at the report date with the earliest start",
            "q": "duration observation ending at the report date spanning 80-100 days",
        },
        "inputs": ["sec/raw/companyfacts.zip", "sec/raw/submissions.zip"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
