import csv
import importlib
import io
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from config import DATASETS_DIR, PROJECT_ROOT
from core.data import load_expansion_table, runtime_variables
from core.dataset_layers import RAW_DIR


def _expected_outputs(module):
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def _claims(spec):
    """Every claim in the case, in turn order: a case may spread them over turns."""
    return [claim for turn in spec["conversations"][0]["turns"] for claim in turn["claims"]]


def test_phase10_build_report_and_governed_balance_sheet_tables():
    report = json.loads(
        (PROJECT_ROOT / "metadata/build_reports/phase10_cross_source_table_build.json").read_text()
    )
    call = report["ffiec_call_report_balance_sheets"]
    y9c = report["ffiec_y9c_balance_sheets"]
    assert (call["rows"], call["reporters_2025q3"]) == (31763, 4435)
    assert (y9c["rows"], y9c["reporters_2025q3"]) == (2687, 382)
    assert call["report_dates"] == y9c["report_dates"]
    assert call["report_dates"][-1] == "2025-09-30"

    banks = load_expansion_table("ffiec_call_reports_balance")
    parents = load_expansion_table("ffiec_y9c")
    assert not banks.duplicated(["report_date", "bank_rssd_id"]).any()
    assert not parents.duplicated(["report_date", "reporter_rssd_id"]).any()
    assert banks.consolidated_total_assets_thousand_usd.notna().any()
    assert parents.consolidated_total_assets_thousand_usd.notna().all()


def test_phase10_runtime_exposes_complete_local_source_tables():
    variables = runtime_variables(
        {
            "ffiec_call_reports_balance",
            "ffiec_nic_structure",
            "ffiec_y9c",
            "ofr_market_stress",
            "ken_french_10_industry_portfolios",
        }
    )
    names = [variable.name for variable in variables]
    # Exposure is what the declaration guarantees; the sequence is registry order.
    assert sorted(names) == sorted([
        "ff_industry_daily_df",
        "ofr_financial_stress_index_df",
        "ffiec_bank_balance_sheets_df",
        "ffiec_y9c_balance_sheets_df",
        "ffiec_nic_institutions_df",
        "ffiec_nic_ownership_df",
    ])
    descriptions = {variable.name: variable.description for variable in variables}
    assert "current revised history" in descriptions[
        "ofr_financial_stress_index_df"
    ]
    assert "202606 CRSP database" in descriptions["ff_industry_daily_df"]
    assert "consolidated RCFD" in descriptions["ffiec_bank_balance_sheets_df"]
    assert "must not be added" in descriptions["ffiec_y9c_balance_sheets_df"]
    assert "Relationship level 1 is direct" in descriptions[
        "ffiec_nic_ownership_df"
    ]


def test_bank_y9c_case_anchor_values_and_validator():
    module = importlib.import_module(
        "evals.calculation_reconciliation.bank_y9c_asset_scope_lineage"
    )
    truth = module.ground_truth()
    assert truth[:10] == (
        4,
        "JPMORGAN CHASE & CO.", 1,
        "BANK OF AMERICA CORPORATION", 3,
        "CITIGROUP INC.", 2,
        "WELLS FARGO & COMPANY", 2,
        8,
    )
    assert tuple(round(value, 3) for value in truth[10:12]) == (
        10075.815,
        12669.373,
    )
    assert round(truth[12], 4) == 79.5289
    assert truth[13:16] == (
        "WELLS FARGO BANK, NATIONAL ASSOCIATION",
        "WELLS FARGO & COMPANY",
        2,
    )
    assert round(truth[16], 4) == 85.6580
    assert truth[17:20] == ("CITIBANK, N.A.", "CITIGROUP INC.", 2)
    assert round(truth[20], 4) == 69.7902
    assert truth[21:] == (2, "CITIBANK, N.A.", "2026-06-30")
    expected = _expected_outputs(module)
    assert module.validate(
        {
            **expected,
            "bank_of_america_matched_parent_legal_name": " bank of america corporation ",
            "highest_asset_share_bank_legal_name": " wells fargo bank, national association ",
            "lowest_asset_share_parent_legal_name": "citigroup inc.",
        }
    ).success
    assert not module.validate(
        {**expected, "total_nic_relationship_edge_count": 4}
    ).success


def test_ofr_industry_case_anchor_values_and_validator():
    module = importlib.import_module(
        "evals.temporal_path.ofr_stress_jump_industry_dispersion"
    )
    truth = module.ground_truth()
    assert truth[:3] == (253, "2020-03-06", "2020-03-09")
    assert tuple(round(value, 3) for value in truth[3:5]) == (1.680, 5.134)
    assert truth[5] == "Volatility"
    assert round(truth[6], 3) == 2.178
    assert truth[7:11] == ("Enrgy", -19.68, "Shops", -5.05)
    assert truth[11] == 10
    assert round(truth[12], 3) == -8.531
    expected = _expected_outputs(module)
    assert module.validate(
        {
            **expected,
            "largest_category_contribution_increase_name": " volatility ",
            "worst_event_date_industry_label": " enrgy ",
        }
    ).success
    assert not module.validate(
        {**expected, "event_date_negative_industry_count": 9}
    ).success


def test_phase10_registration_source_necessity_and_claim_modes():
    registry = json.loads((PROJECT_ROOT / "benchmarks.json").read_text())
    paths = registry["cases"]
    assert paths["bank_y9c_asset_scope_lineage"] == (
        "evals/calculation_reconciliation/bank_y9c_asset_scope_lineage.json"
    )
    assert paths["ofr_stress_jump_industry_dispersion"] == (
        "evals/temporal_path/ofr_stress_jump_industry_dispersion.json"
    )

    bank = json.loads(
        (PROJECT_ROOT / "evals/calculation_reconciliation/bank_y9c_asset_scope_lineage.json").read_text()
    )
    stress = json.loads(
        (PROJECT_ROOT / "evals/temporal_path/ofr_stress_jump_industry_dispersion.json").read_text()
    )
    assert bank["data_sources"] == [
        "ffiec_call_reports_balance",
        "ffiec_nic_structure",
        "ffiec_y9c",
    ]
    assert stress["data_sources"] == [
        "ofr_market_stress",
        "ken_french_10_industry_portfolios",
    ]
    bank_modes = {claim["id"]: claim["mode"] for claim in _claims(bank)}
    assert bank_modes["multi_hop_lineage_meaning"] == "required"
    assert bank_modes["consolidated_scope_nonadditivity"] == "required"
    assert bank_modes["submission_vintage_meaning"] == "required"
    assert bank_modes["ratio_overclaim_boundary"] == "conditional"
    assert bank_modes["lineage_shortcut_boundary"] == "conditional"
    assert bank_modes["ownership_vintage_boundary"] == "conditional"
    stress_modes = {claim["id"]: claim["mode"] for claim in _claims(stress)}
    assert stress_modes["fsi_decomposition_scope"] == "required"
    assert stress_modes["same_day_causality_boundary"] == "required"
    assert stress_modes["decomposition_double_count_boundary"] == "conditional"
    assert stress_modes["industry_mean_overclaim_boundary"] == "conditional"
    assert stress_modes["stress_vintage_boundary"] == "conditional"
    assert stress_modes["industry_vintage_boundary"] == "conditional"

    ken_french_manifest = json.loads(
        (DATASETS_DIR / "ken_french/sources.json").read_text()
    )
    source_vintage = ken_french_manifest.get("source_vintage")
    if source_vintage is not None:
        assert "observation cutoff" in source_vintage
        assert "202606 CRSP database" in source_vintage
    else:
        # The runtime-only release omits the text field, but carries the same
        # frozen 202606 industry archive fingerprint as the older full release.
        assert ken_french_manifest["freeze_date"] == "2025-12-31"
        assert ken_french_manifest["raw_files"]["10_Industry_Portfolios_daily_CSV.zip"]["sha256"] == (
            "c4c9b7c3969e5477630e9e312f61a19bb74c0f8d159612ef3adf33380eeb301c"
        )




def _raw_bank_y9c_outputs():
    bank_ids = {"852218", "480228", "476810", "451965"}
    call_zip = RAW_DIR / (
        "ffiec_call_reports/FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
    )
    with ZipFile(call_zip) as archive:
        schedule = pd.read_csv(
            archive.open("FFIEC CDR Call Schedule RC 09302025.txt"),
            sep="\t",
            dtype=str,
            low_memory=False,
        )
        por = pd.read_csv(
            archive.open("FFIEC CDR Call Bulk POR 09302025.txt"),
            sep="\t",
            dtype=str,
            low_memory=False,
        )
    banks = schedule.loc[schedule.IDRSSD.isin(bank_ids), ["IDRSSD", "RCFD2170"]].merge(
        por[[
            "IDRSSD",
            "Financial Institution Name",
            "Last Date/Time Submission Updated On",
        ]],
        on="IDRSSD",
        validate="one_to_one",
    ).rename(
        columns={
            "Financial Institution Name": "bank_name",
            "Last Date/Time Submission Updated On": "submission_update",
        }
    )

    y9c_zip = RAW_DIR / "ffiec_y9c/BHCF20250930.ZIP"
    with ZipFile(y9c_zip) as archive:
        parents = pd.read_csv(
            archive.open("BHCF20250930.txt"),
            sep="^",
            dtype=str,
            encoding="cp1252",
            low_memory=False,
        )
    parents = parents.loc[
        parents.BHCK2170.notna(), ["RSSD9001", "RSSD9017", "BHCK2170"]
    ].set_index("RSSD9001")

    relationship_zip = (
        RAW_DIR / "ffiec_nic_structure/relationships_current.zip"
    )
    with ZipFile(relationship_zip) as archive:
        relationships = pd.read_csv(
            archive.open("CSV_RELATIONSHIPS.CSV"), dtype=str, low_memory=False
        )
    relationships = relationships.loc[
        relationships.DT_END.eq("99991231")
        & relationships.RELN_LVL.eq("1")
        & relationships.CTRL_IND.eq("1")
        & relationships.REG_IND.eq("1")
    ]
    parent_by_child = relationships.groupby("ID_RSSD_OFFSPRING")[
        "#ID_RSSD_PARENT"
    ].agg(lambda values: tuple(sorted(set(values))))

    matched = []
    for bank in banks.itertuples(index=False):
        current = bank.IDRSSD
        edges = 0
        while current not in parents.index:
            candidates = parent_by_child[current]
            assert len(candidates) == 1
            current = candidates[0]
            edges += 1
            assert edges <= 10
        parent = parents.loc[current]
        matched.append(
            {
                "bank": bank.bank_name,
                "bank_rssd_id": bank.IDRSSD,
                "parent": parent.RSSD9017,
                "edges": edges,
                "bank_assets": float(bank.RCFD2170),
                "parent_assets": float(parent.BHCK2170),
                "submission_update_date": bank.submission_update[:10],
            }
        )
    result = pd.DataFrame(matched)
    result["share"] = result.bank_assets / result.parent_assets * 100
    highest = result.sort_values(["share", "bank"], ascending=[False, True]).iloc[0]
    lowest = result.sort_values(["share", "bank"], ascending=[True, True]).iloc[0]
    by_rssd = result.set_index("bank_rssd_id")
    latest_updated = result.sort_values(
        ["submission_update_date", "bank"], ascending=[False, True]
    ).iloc[0]
    bank_total = result.bank_assets.sum()
    parent_total = result.parent_assets.sum()
    return (
        len(result),
        by_rssd.loc["852218", "parent"], int(by_rssd.loc["852218", "edges"]),
        by_rssd.loc["480228", "parent"], int(by_rssd.loc["480228", "edges"]),
        by_rssd.loc["476810", "parent"], int(by_rssd.loc["476810", "edges"]),
        by_rssd.loc["451965", "parent"], int(by_rssd.loc["451965", "edges"]),
        int(result.edges.sum()),
        bank_total / 1e6,
        parent_total / 1e6,
        bank_total / parent_total * 100,
        highest.bank,
        highest.parent,
        int(highest.edges),
        highest.share,
        lowest.bank,
        lowest.parent,
        int(lowest.edges),
        lowest.share,
        highest.share - lowest.share,
        int(result.submission_update_date.gt("2025-12-31").sum()),
        latest_updated.bank,
        latest_updated.submission_update_date,
    )


@pytest.mark.raw_data
def test_independent_raw_recomputation_matches_bank_y9c_case():
    independent = _raw_bank_y9c_outputs()
    module = importlib.import_module(
        "evals.calculation_reconciliation.bank_y9c_asset_scope_lineage"
    )
    truth = module.ground_truth()
    for left, right in zip(independent, truth):
        if isinstance(left, str):
            assert " ".join(left.split()).casefold() == " ".join(right.split()).casefold()
        else:
            assert abs(left - right) < 1e-12


def test_call_report_submission_metadata_closes_local_bankfind_substitute():
    banks = load_expansion_table("ffiec_call_reports_balance").loc[
        lambda frame: frame.report_date.eq("2025-09-30")
        & frame.bank_rssd_id.isin({"852218", "480228", "476810", "451965"})
    ]
    bankfind = load_expansion_table("fdic_bankfind").loc[
        lambda frame: frame.REPDTE.eq("2025-09-30")
    ]
    alternate = banks[["fdic_certificate", "consolidated_total_assets_thousand_usd"]].merge(
        bankfind[["CERT", "ASSET"]],
        left_on="fdic_certificate",
        right_on="CERT",
        validate="one_to_one",
    )
    assert (
        alternate.consolidated_total_assets_thousand_usd.astype(float)
        == alternate.ASSET.astype(float)
    ).all()
    assert "last_submission_update" not in bankfind.columns
    assert banks.last_submission_update.str.slice(0, 10).gt("2025-12-31").sum() == 2


def _raw_ken_french_value_weighted_rows():
    path = RAW_DIR / "ken_french/10_Industry_Portfolios_daily_CSV.zip"
    with ZipFile(path) as archive:
        text = archive.read("10_Industry_Portfolios_Daily.csv").decode("cp1252")
    lines = text.splitlines()
    header_index = lines.index(
        ",NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth,Utils,Other"
    )
    rows = []
    for row in csv.reader(io.StringIO("\n".join(lines[header_index + 1 :]))):
        if not row or not row[0].strip().isdigit():
            break
        rows.append([row[0].strip(), *[float(value) for value in row[1:]]])
    return pd.DataFrame(
        rows,
        columns=[
            "date", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
            "Shops", "Hlth", "Utils", "Other",
        ],
    ).assign(date=lambda frame: pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"))


@pytest.mark.raw_data
def test_independent_raw_recomputation_matches_ofr_industry_case():
    stress = pd.read_csv(RAW_DIR / "ofr_market_stress/fsi.csv")
    stress = stress.rename(
        columns={
            "Date": "date",
            "OFR FSI": "fsi",
            "Equity valuation": "Equity valuation",
            "Safe assets": "Safe assets",
        }
    )
    panel = stress.merge(_raw_ken_french_value_weighted_rows(), on="date")
    panel = panel.loc[panel.date.between("2020-01-01", "2020-12-31")].sort_values(
        "date"
    ).reset_index(drop=True)
    panel["change"] = panel.fsi.diff()
    end_index = int(panel.index[panel.change.eq(panel.change.max())][0])
    start, end = panel.iloc[end_index - 1], panel.iloc[end_index]
    categories = ["Credit", "Equity valuation", "Safe assets", "Funding", "Volatility"]
    category_changes = {name: float(end[name] - start[name]) for name in categories}
    category = sorted(category_changes, key=lambda name: (-category_changes[name], name))[0]
    industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
    returns = {name: float(end[name]) for name in industries}
    worst = sorted(returns, key=lambda name: (returns[name], name))[0]
    best = sorted(returns, key=lambda name: (-returns[name], name))[0]
    values = list(returns.values())
    independent = (
        len(panel), start.date, end.date, float(start.fsi), float(end.fsi),
        category, category_changes[category], worst,
        returns[worst], best, returns[best], sum(value < 0 for value in values),
        sum(values) / len(values),
    )
    module = importlib.import_module(
        "evals.temporal_path.ofr_stress_jump_industry_dispersion"
    )
    truth = module.ground_truth()
    assert independent[:3] == truth[:3]
    for left, right in zip(independent[3:], truth[3:]):
        if isinstance(left, str):
            assert left == right
        else:
            assert abs(left - right) < 1e-12
