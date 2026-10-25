"""Build deterministic runtime tables for the Phase 4 sources.

The raw downloads remain immutable.  These tables retain the row grain and
competing accounting or regulatory measures needed by the cases; they do not
precompute case answers.  Filing-received dates and report dates are preserved
so cases can make their own point-in-time and period selections.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
CUTOFF = pd.Timestamp("2025-12-31")
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=GZIP)
    print(
        f"wrote {path.relative_to(PROJECT_ROOT)}: "
        f"{len(frame):,} rows, {path.stat().st_size:,} bytes"
    )
    return path, len(frame)


def _zip_csv(
    path: Path, columns: list[str], *, dtype: dict[str, str] | None = None
) -> pd.DataFrame:
    with ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.lower().endswith(".csv"))
        return pd.read_csv(
            archive.open(name), usecols=columns, dtype=dtype, low_memory=False
        )


def _ffiec_table(
    archive: ZipFile,
    name: str,
    *,
    has_label_row: bool,
    dtype: dict[str, str] | type | None = None,
) -> pd.DataFrame:
    """Read an FFIEC bulk member without discarding a POR data row.

    Schedule files have a second human-readable label row; the POR identity
    file does not. POR is read as text so fixed-width codes such as ``051``
    survive normalization.
    """
    return pd.read_csv(
        archive.open(name),
        sep="\t",
        skiprows=[1] if has_label_row else None,
        dtype=dtype,
        low_memory=False,
    )


def _numeric_strings(series: pd.Series) -> pd.Series:
    """Parse FiscalData's source-native apostrophe-prefixed negative values."""
    return pd.to_numeric(series.astype("string").str.removeprefix("'"), errors="coerce")


def _ratio_strings(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.removesuffix("%"), errors="coerce"
    )


def build_form5500() -> list[tuple[Path, int]]:
    root = RAW_DIR / "dol_form5500"
    main_columns = [
        "ACK_ID", "TYPE_PLAN_ENTITY_CD", "PLAN_NAME", "SPONSOR_DFE_NAME",
        "SPONS_DFE_EIN", "SPONS_DFE_PN", "FORM_TAX_PRD", "INITIAL_FILING_IND",
        "AMENDED_IND", "FILING_STATUS", "DATE_RECEIVED", "TOT_PARTCP_BOY_CNT",
        "TOT_ACTIVE_PARTCP_CNT", "PARTCP_ACCOUNT_BAL_CNT",
    ]
    main = _zip_csv(
        root / "F_5500_2024_all.zip",
        main_columns,
        dtype={
            "ACK_ID": "string",
            "SPONS_DFE_EIN": "string",
            "SPONS_DFE_PN": "string",
        },
    )
    main["ACK_ID"] = main["ACK_ID"].str.strip()
    main["SPONS_DFE_EIN"] = main["SPONS_DFE_EIN"].str.strip().str.zfill(9)
    main["SPONS_DFE_PN"] = main["SPONS_DFE_PN"].str.strip().str.zfill(3)
    main["DATE_RECEIVED"] = pd.to_datetime(main["DATE_RECEIVED"], errors="coerce")
    main = main.loc[main["DATE_RECEIVED"].le(CUTOFF)].copy()

    financial_columns = [
        "ACK_ID", "NET_ASSETS_BOY_AMT", "NET_ASSETS_EOY_AMT", "TOT_INCOME_AMT",
        "TOT_EXPENSES_AMT", "NET_INCOME_AMT", "TOT_TRANSFERS_TO_AMT",
        "TOT_TRANSFERS_FROM_AMT",
    ]
    financials = _zip_csv(root / "F_SCH_H_2024_all.zip", financial_columns)
    plan_financials = main.merge(
        financials, on="ACK_ID", how="inner", validate="one_to_one"
    )
    plan_financials = plan_financials.rename(columns={
        "ACK_ID": "filing_id",
        "TYPE_PLAN_ENTITY_CD": "plan_entity_code",
        "PLAN_NAME": "plan_name",
        "SPONSOR_DFE_NAME": "sponsor_name",
        "SPONS_DFE_EIN": "sponsor_ein",
        "SPONS_DFE_PN": "plan_number",
        "FORM_TAX_PRD": "plan_year_end",
        "INITIAL_FILING_IND": "initial_filing_indicator",
        "AMENDED_IND": "amended_indicator",
        "FILING_STATUS": "filing_status",
        "DATE_RECEIVED": "date_received",
        "TOT_PARTCP_BOY_CNT": "participants_beginning_count",
        "TOT_ACTIVE_PARTCP_CNT": "active_participants_end_count",
        "PARTCP_ACCOUNT_BAL_CNT": "participants_with_account_balance_count",
        "NET_ASSETS_BOY_AMT": "net_assets_beginning_usd",
        "NET_ASSETS_EOY_AMT": "net_assets_ending_usd",
        "TOT_INCOME_AMT": "total_income_usd",
        "TOT_EXPENSES_AMT": "total_expenses_usd",
        "NET_INCOME_AMT": "net_income_usd",
        "TOT_TRANSFERS_TO_AMT": "transfers_to_plan_usd",
        "TOT_TRANSFERS_FROM_AMT": "transfers_from_plan_usd",
    })
    plan_financials["date_received"] = plan_financials["date_received"].dt.strftime(
        "%Y-%m-%d"
    )

    sb_columns = [
        "ACK_ID", "SB_PLAN_TYPE_CODE", "SB_VALUE_DATE", "SB_CURR_VALUE_AST_01_AMT",
        "SB_ACTRL_VALUE_AST_AMT", "SB_TOT_FNDNG_TGT_AMT", "SB_PLAN_AT_RISK_IND",
        "SB_TGT_DISREGARD_ASSUMP_AMT", "SB_CARRYOVER_BOY_TOT_AMT",
        "SB_PRE_FNDNG_BOY_TOT_AMT", "SB_FNDNG_TGT_PRCNT",
        "SB_ADJ_FNDNG_TGT_PRCNT", "SB_PR_YR_FNDNG_PRCNT", "SB_TOT_PARTCP_CNT",
    ]
    schedule_sb = _zip_csv(root / "F_SCH_SB_2024_all.zip", sb_columns)
    pension = main.merge(schedule_sb, on="ACK_ID", how="inner", validate="one_to_one")
    pension = pension.rename(columns={
        "ACK_ID": "filing_id",
        "PLAN_NAME": "plan_name",
        "SPONSOR_DFE_NAME": "sponsor_name",
        "SPONS_DFE_EIN": "sponsor_ein",
        "SPONS_DFE_PN": "plan_number",
        "FORM_TAX_PRD": "plan_year_end",
        "INITIAL_FILING_IND": "initial_filing_indicator",
        "AMENDED_IND": "amended_indicator",
        "FILING_STATUS": "filing_status",
        "DATE_RECEIVED": "date_received",
        "SB_PLAN_TYPE_CODE": "sb_plan_type_code",
        "SB_VALUE_DATE": "valuation_date",
        "SB_CURR_VALUE_AST_01_AMT": "current_value_assets_usd",
        "SB_ACTRL_VALUE_AST_AMT": "actuarial_value_assets_usd",
        "SB_TOT_FNDNG_TGT_AMT": "total_funding_target_usd",
        "SB_PLAN_AT_RISK_IND": "at_risk_indicator",
        "SB_TGT_DISREGARD_ASSUMP_AMT": "funding_target_without_at_risk_assumptions_usd",
        "SB_CARRYOVER_BOY_TOT_AMT": "funding_standard_carryover_balance_usd",
        "SB_PRE_FNDNG_BOY_TOT_AMT": "prefunding_balance_usd",
        "SB_FNDNG_TGT_PRCNT": "reported_ftap_pct",
        "SB_ADJ_FNDNG_TGT_PRCNT": "reported_adjusted_ftap_pct",
        "SB_PR_YR_FNDNG_PRCNT": "prior_year_ftap_pct",
        "SB_TOT_PARTCP_CNT": "schedule_sb_participant_count",
    })
    pension["date_received"] = pension["date_received"].dt.strftime("%Y-%m-%d")

    return [
        _write(
            plan_financials,
            DATASETS / "dol_form5500" / "plan_financials_2024.csv.gz",
        ),
        _write(
            pension,
            DATASETS / "dol_form5500" / "schedule_sb_2024.csv.gz",
        ),
    ]


def build_ffiec() -> tuple[Path, int]:
    root = RAW_DIR / "ffiec_call_reports"
    parts: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.zip")):
        stamp_match = re.search(r"(\d{8})", path.name)
        if stamp_match is None:
            raise ValueError(f"FFIEC archive lacks report-date stamp: {path.name}")
        stamp = stamp_match.group(1)
        report_date = pd.to_datetime(stamp, format="%m%d%Y")
        if report_date > CUTOFF or stamp == "12312025":
            # The 2025-Q4 archive was first published after the benchmark cutoff.
            continue
        with ZipFile(path) as archive:
            por = _ffiec_table(
                archive,
                f"FFIEC CDR Call Bulk POR {stamp}.txt",
                has_label_row=False,
                dtype=str,
            )
            capital = _ffiec_table(
                archive,
                f"FFIEC CDR Call Schedule RCRI {stamp}.txt",
                has_label_row=True,
                dtype={"IDRSSD": "string"},
            )
        keep = capital[[
            "IDRSSD", "RCFAA223", "RCFWA223", "RCFWP859", "RCFAP793",
            "RCFWP793", "RCFA7204", "RCFAH036",
        ]].copy()
        keep["RCFAP793"] = _ratio_strings(keep["RCFAP793"])
        keep["RCFWP793"] = _ratio_strings(keep["RCFWP793"])
        keep["RCFA7204"] = _ratio_strings(keep["RCFA7204"])
        keep["RCFAH036"] = _ratio_strings(keep["RCFAH036"])
        joined = por[[
            "IDRSSD", "Financial Institution Name", "Financial Institution State",
            "Financial Institution Filing Type", "Last Date/Time Submission Updated On",
        ]].merge(keep, on="IDRSSD", how="inner", validate="one_to_one")
        joined.insert(1, "report_date", report_date.strftime("%Y-%m-%d"))
        parts.append(joined)
    result = pd.concat(parts, ignore_index=True).rename(columns={
        "IDRSSD": "bank_id",
        "Financial Institution Name": "bank_name",
        "Financial Institution State": "bank_state",
        "Financial Institution Filing Type": "filing_type",
        "Last Date/Time Submission Updated On": "last_submission_update",
        "RCFAA223": "standardized_risk_weighted_assets_thousand_usd",
        "RCFWA223": "advanced_risk_weighted_assets_thousand_usd",
        "RCFWP859": "common_equity_tier1_capital_thousand_usd",
        "RCFAP793": "reported_standardized_cet1_ratio_pct",
        "RCFWP793": "reported_advanced_cet1_ratio_pct",
        "RCFA7204": "reported_tier1_leverage_ratio_pct",
        "RCFAH036": "reported_supplementary_leverage_ratio_pct",
    })
    return _write(result, DATASETS / "ffiec_call_reports" / "bank_capital.csv.gz")


def build_treasury() -> list[tuple[Path, int]]:
    debt_root = RAW_DIR / "treasury_debt"
    debt = pd.read_csv(
        debt_root / "debt_to_penny_2024_through_2025.csv", dtype={"record_date": str}
    )
    debt = debt[[
        "record_date", "debt_held_public_amt", "intragov_hold_amt",
        "tot_pub_debt_out_amt",
    ]].rename(columns={
        "debt_held_public_amt": "debt_held_by_public_usd",
        "intragov_hold_amt": "intragovernmental_holdings_usd",
        "tot_pub_debt_out_amt": "total_public_debt_outstanding_usd",
    })

    market = pd.read_csv(
        debt_root / "mspd_table_3_market_2024_through_2025.csv", low_memory=False
    )
    market = market[[
        "record_date", "security_class1_desc", "security_class2_desc", "series_cd",
        "interest_rate_pct", "yield_pct", "issue_date", "maturity_date", "issued_amt",
        "inflation_adj_amt", "redeemed_amt", "outstanding_amt", "src_line_nbr",
    ]].copy()
    for column in [
        "interest_rate_pct", "yield_pct", "issued_amt", "inflation_adj_amt",
        "redeemed_amt", "outstanding_amt",
    ]:
        market[column] = _numeric_strings(market[column])
    market = market.rename(columns={
        "security_class1_desc": "security_class",
        "security_class2_desc": "security_identifier_or_total_label",
        "series_cd": "series_code",
        "issued_amt": "issued_million_usd",
        "inflation_adj_amt": "inflation_adjustment_million_usd",
        "redeemed_amt": "redeemed_million_usd",
        "outstanding_amt": "outstanding_million_usd",
        "src_line_nbr": "source_line_number",
    })

    usfr_root = RAW_DIR / "treasury_usfr"
    reconciliations = pd.read_csv(
        usfr_root / "reconciliations_2020_through_2025.csv", low_memory=False
    )
    reconciliations["position_billion_usd"] = _numeric_strings(
        reconciliations["position_bil_amt"]
    )
    reconciliations = reconciliations[[
        "record_date", "stmt_fiscal_year", "restmt_flag", "account_desc",
        "component_desc", "line_item_desc", "position_billion_usd", "src_line_nbr",
    ]].rename(columns={
        "stmt_fiscal_year": "statement_fiscal_year",
        "restmt_flag": "restatement_flag",
        "account_desc": "account_description",
        "component_desc": "component_description",
        "line_item_desc": "line_item_description",
        "src_line_nbr": "source_line_number",
    })

    return [
        _write(debt, DATASETS / "treasury_debt" / "debt_to_penny.csv.gz"),
        _write(
            market,
            DATASETS / "treasury_debt" / "marketable_securities.csv.gz",
        ),
        _write(
            reconciliations,
            DATASETS / "treasury_usfr" / "reconciliations.csv.gz",
        ),
    ]


def _raw_record(dataset: str, filename: str) -> dict:
    manifest = json.loads(
        (DATASETS / dataset / "sources.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in manifest["raw_files"] if item["file"] == f"raw/{filename}"
    )
    return {
        "dataset": dataset,
        "file": record["file"],
        "sha256": record["sha256"],
    }


def main() -> None:
    outputs: list[tuple[Path, int]] = []
    outputs.extend(build_form5500())
    outputs.append(build_ffiec())
    outputs.extend(build_treasury())

    sources = [
        _raw_record("dol_form5500", "F_5500_2024_all.zip"),
        _raw_record("dol_form5500", "F_SCH_H_2024_all.zip"),
        _raw_record("dol_form5500", "F_SCH_SB_2024_all.zip"),
        *[
            _raw_record(
                "ffiec_call_reports",
                f"FFIEC_CDR_Call_Bulk_All_Schedules_{stamp}.zip",
            )
            for stamp in (
                "03312024", "06302024", "09302024", "12312024",
                "03312025", "06302025", "09302025",
            )
        ],
        _raw_record("treasury_debt", "debt_to_penny_2024_through_2025.csv"),
        _raw_record("treasury_debt", "mspd_table_3_market_2024_through_2025.csv"),
        _raw_record("treasury_usfr", "reconciliations_2020_through_2025.csv"),
    ]
    records = {}
    for path, rows in outputs:
        relative = str(path.relative_to(DATASETS))
        records[relative] = {
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": 1,
        "benchmark_cutoff": "2025-12-31",
        "builder": "python -m scripts.ingest.build.build_phase4_runtime",
        "source_files": sources,
        "files": records,
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase4_runtime_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
