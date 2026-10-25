"""Rebuild `ffiec_call_reports.bank_capital` with the domestic/consolidated fallback.

The phase-4 build read only the `RCFA*` / `RCFW*` items of Schedule RC-R Part I.
Those are the consolidated (foreign-office) reporting variants, filed by the
few dozen banks with foreign offices; every other bank files the domestic
`RCOA*` variant. The result was a capital table whose core columns were
98-100 % missing (found in the 2026-08-27 quality audit).

This build coalesces consolidated-first, domestic-second for every item and
records which basis each row came from:

| column | consolidated | domestic | meaning |
|---|---|---|---|
| standardized_risk_weighted_assets_thousand_usd | RCFAA223 | RCOAA223 | standardized-approach RWA |
| advanced_risk_weighted_assets_thousand_usd | RCFWA223 | — | advanced-approaches RWA (advanced banks only) |
| common_equity_tier1_capital_thousand_usd | RCFAP859 | RCOAP859, then RCFWP859 | CET1 capital (advanced-approaches banks file it under RCFW only) |
| reported_standardized_cet1_ratio_pct | RCFAP793 | RCOAP793 | CET1 ratio, standardized |
| reported_advanced_cet1_ratio_pct | RCFWP793 | — | CET1 ratio, advanced approaches |
| reported_tier1_leverage_ratio_pct | RCFA7204 | RCOA7204 | tier-1 leverage ratio |
| reported_supplementary_leverage_ratio_pct | RCFAH036 | RCOAH036 | supplementary leverage ratio |

The earlier table mapped `common_equity_tier1_capital_thousand_usd` to the
advanced-approaches item (RCFWP859); the standardized item is the one every
bank files and is what the CET1-ratio columns are computed from, so the
column now carries P859 on the same basis as the ratio.

`ffiec_call_reports.bank_balance_sheets` keeps its consolidated (RCFD) and
domestic (RCON) columns; this build adds coalesced `total_*_thousand_usd`
columns beside them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT
from scripts.ingest.build.build_phase4_runtime import _ffiec_table, _ratio_strings


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
REPORT = PROJECT_ROOT / "metadata" / "build_reports" / "ffiec_capital_full_build.json"
CUTOFF = pd.Timestamp("2025-12-31")
ITEMS = {
    # output column: (consolidated code, domestic code, is_ratio)
    "standardized_risk_weighted_assets_thousand_usd": ("RCFAA223", "RCOAA223", False),
    "advanced_risk_weighted_assets_thousand_usd": ("RCFWA223", None, False),
    # Advanced-approaches banks file CET1 capital under the RCFW column and leave
    # RCFA blank, so the capital item falls through RCFA -> RCOA -> RCFW.
    "common_equity_tier1_capital_thousand_usd": ("RCFAP859", ("RCOAP859", "RCFWP859"), False),
    "reported_standardized_cet1_ratio_pct": ("RCFAP793", "RCOAP793", True),
    "reported_advanced_cet1_ratio_pct": ("RCFWP793", None, True),
    "reported_tier1_leverage_ratio_pct": ("RCFA7204", "RCOA7204", True),
    "reported_supplementary_leverage_ratio_pct": ("RCFAH036", "RCOAH036", True),
}


def _write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, compression="gzip")
    print(f"  {path.relative_to(DATASETS)}: {len(frame):,} rows", flush=True)
    return {"file": str(path.relative_to(DATASETS)), "rows": len(frame), "columns": list(frame.columns)}


def _coalesce(frame: pd.DataFrame, consolidated: str, domestic, ratio: bool) -> pd.Series:
    parse = _ratio_strings if ratio else (lambda s: pd.to_numeric(s.astype("string"), errors="coerce"))
    values = parse(frame[consolidated]) if consolidated in frame else pd.Series(pd.NA, index=frame.index, dtype="Float64")
    fallbacks = domestic if isinstance(domestic, tuple) else ((domestic,) if domestic else ())
    for code in fallbacks:
        if code in frame:
            values = values.where(values.notna(), parse(frame[code]))
    return values


def build_capital() -> dict:
    root = RAW_DIR / "ffiec_call_reports"
    parts = []
    for path in sorted(root.glob("*.zip")):
        stamp = re.search(r"(\d{8})", path.name).group(1)
        report_date = pd.to_datetime(stamp, format="%m%d%Y")
        if report_date > CUTOFF or stamp == "12312025":
            continue  # the 2025-Q4 archive was first published after the cutoff
        with ZipFile(path) as archive:
            por = _ffiec_table(archive, f"FFIEC CDR Call Bulk POR {stamp}.txt", has_label_row=False, dtype=str)
            capital = _ffiec_table(archive, f"FFIEC CDR Call Schedule RCRI {stamp}.txt", has_label_row=True,
                                   dtype={"IDRSSD": "string"})
        keep = pd.DataFrame({"IDRSSD": capital["IDRSSD"]})
        for column, (consolidated, domestic, ratio) in ITEMS.items():
            keep[column] = _coalesce(capital, consolidated, domestic, ratio)
        # Banks that elect the community bank leverage ratio framework (RCOALE74 /
        # RCFALE74 = 1) do not report risk-weighted assets or a CET1 ratio; this
        # flag explains the ~38% of banks whose RWA columns are legitimately empty.
        cblr = pd.Series(pd.NA, index=capital.index, dtype="string")
        for code in ("RCFALE74", "RCOALE74"):
            if code in capital:
                cblr = cblr.where(cblr.notna(), capital[code].astype("string").str.strip().replace({"": pd.NA}))
        keep["cblr_election"] = cblr.map({"1": True, "true": True, "0": False, "false": False}).astype("boolean")
        consolidated_rwa = pd.to_numeric(capital["RCFAA223"].astype("string"), errors="coerce") if "RCFAA223" in capital else pd.Series(pd.NA, index=capital.index)
        keep["capital_reporting_basis"] = pd.Series("domestic", index=capital.index).where(consolidated_rwa.isna(), "consolidated")
        keep.loc[keep["standardized_risk_weighted_assets_thousand_usd"].isna(), "capital_reporting_basis"] = pd.NA
        joined = por[[
            "IDRSSD", "Financial Institution Name", "Financial Institution State",
            "Financial Institution Filing Type", "Last Date/Time Submission Updated On",
        ]].merge(keep, on="IDRSSD", how="inner", validate="one_to_one")
        joined.insert(1, "report_date", report_date.strftime("%Y-%m-%d"))
        parts.append(joined)
        print(f"  {stamp}: {len(joined):,} banks, RWA reported for {joined['standardized_risk_weighted_assets_thousand_usd'].notna().mean():.1%}", flush=True)
    result = pd.concat(parts, ignore_index=True).rename(columns={
        "IDRSSD": "bank_id", "Financial Institution Name": "bank_name",
        "Financial Institution State": "bank_state", "Financial Institution Filing Type": "filing_type",
        "Last Date/Time Submission Updated On": "last_submission_update",
    })
    result = result.sort_values(["report_date", "bank_id"]).reset_index(drop=True)
    info = _write(result, DATASETS / "ffiec_call_reports" / "bank_capital.csv.gz")
    info["standardized_rwa_coverage"] = float(result["standardized_risk_weighted_assets_thousand_usd"].notna().mean())
    return info


def add_balance_totals() -> dict:
    path = DATASETS / "ffiec_call_reports" / "bank_balance_sheets.csv.gz"
    # The reviewed staging copy is not distributed with every checkout; the
    # governed runtime copy carries the same bytes and is the fallback input.
    source = path if path.exists() else DATASETS / "ffiec_call_reports" / "runtime" / "bank_balance_sheets.csv.gz"
    balance = pd.read_csv(source, dtype={"bank_rssd_id": str, "fdic_certificate": str}, low_memory=False)
    for item in ("assets", "liabilities", "equity"):
        consolidated = balance[f"consolidated_total_{item}_thousand_usd"]
        domestic = balance[f"domestic_office_total_{item}_thousand_usd"]
        balance[f"total_{item}_thousand_usd"] = consolidated.where(consolidated.notna(), domestic)
    balance["balance_reporting_basis"] = balance["consolidated_total_assets_thousand_usd"].notna().map(
        {True: "consolidated", False: "domestic"})
    balance.loc[balance["total_assets_thousand_usd"].isna(), "balance_reporting_basis"] = pd.NA
    info = _write(balance, path)
    info["total_assets_coverage"] = float(balance["total_assets_thousand_usd"].notna().mean())
    return info


def main() -> None:
    report = {"built_at": datetime.now(timezone.utc).isoformat(),
              "bank_capital": build_capital(), "bank_balance_sheets": add_balance_totals()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
