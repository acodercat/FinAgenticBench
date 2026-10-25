import importlib
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table, runtime_variables
from core.dataset_layers import RAW_DIR, TABLES


def _expected_outputs(module):
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def _claims(spec):
    """Every claim in the case, in turn order: a case may spread them over turns."""
    return [claim for turn in spec["conversations"][0]["turns"] for claim in turn["claims"]]


def test_phase12_runtime_exposes_only_requested_local_sources():
    variables = runtime_variables(
        {
            "fed_stress_tests",
            "ffiec_y9c",
            "bis_effective_exchange_rates",
        }
    )
    # Exposure is what the declaration guarantees; the sequence is registry order.
    assert sorted(variable.name for variable in variables) == sorted([
        "fed_stress_results_df",
        "ffiec_y9c_balance_sheets_df",
        "bis_effective_exchange_rates_df",
    ])
    descriptions = {variable.name: variable.description for variable in variables}
    assert "amount fields are billion USD" in descriptions["fed_stress_results_df"]
    assert "conditional supervisory results" in descriptions["fed_stress_results_df"]
    assert "liabilities and equity in thousand USD" in descriptions[
        "ffiec_y9c_balance_sheets_df"
    ]
    assert "2020=100" in descriptions["bis_effective_exchange_rates_df"]
    assert "not bilaterally against the U.S. dollar" in descriptions[
        "bis_effective_exchange_rates_df"
    ]


def test_stress_equity_case_anchor_values_and_validator():
    module = importlib.import_module(
        "evals.comparison_ranking.stress_loss_equity_absorption_leaders"
    )
    truth = module.ground_truth()
    assert truth[:3] == pytest.approx((22, 1854.898991, 471.9))
    assert tuple(round(value, 4) for value in truth[3:5]) == (
        25.4407,
        26.3632,
    )
    assert truth[5:7] == ("Capital One Financial Corporation", "2277860")
    assert tuple(round(value, 4) for value in truth[7:12]) == (
        60.7834,
        53.7,
        88.3465,
        13.5,
        9.2,
    )
    assert truth[12:14] == ("American Express Company", "1275216")
    assert tuple(round(value, 4) for value in truth[14:]) == (
        30.264,
        27.2,
        89.8758,
        10.5,
        9.4,
    )
    expected = _expected_outputs(module)
    assert module.validate(
        {
            **expected,
            "loan_loss_to_equity_leader_name": " capital one FINANCIAL corporation ",
            "loan_loss_to_equity_leader_rssd_id": 2277860,
            "provision_to_equity_leader_name": "AMERICAN   EXPRESS COMPANY",
        }
    ).success
    assert not module.validate(
        {**expected, "cohort_loan_loss_to_equity_ratio_pct": truth[4]}
    ).success



def test_phase12_registration_capabilities_and_claim_contracts():
    registry = json.loads((PROJECT_ROOT / "benchmarks.json").read_text())
    paths = registry["cases"]
    stress_path = "evals/comparison_ranking/stress_loss_equity_absorption_leaders.json"
    assert paths["stress_loss_equity_absorption_leaders"] == stress_path
    stress = json.loads((PROJECT_ROOT / stress_path).read_text())
    assert stress["data_sources"] == ["fed_stress_tests", "ffiec_y9c"]
    assert {claim["id"]: claim["mode"] for claim in _claims(stress)} == {
        "cohort_equity_weighting_meaning": "required",
        "loss_provision_leader_divergence": "required",
        "cet1_nonrollforward_meaning": "required",
        "loss_capacity_overclaim_boundary": "conditional",
        "scenario_forecast_boundary": "conditional",
        "y9c_vintage_boundary": "conditional",
    }
    capabilities = json.loads(
        (PROJECT_ROOT / "metadata/capabilities.json").read_text()
    )["cases"]
    assert capabilities[stress["name"]] == {
        "primary": "ratio_denominator",
        "secondary": [
            "cross_table_join",
            "aggregation_weighting",
            "comparative_ranking",
        ],
    }
    for spec in (stress,):
        module = importlib.import_module(spec["module"])
        assert len(module.variables) == len(module.ground_truth())
        assert len({variable.name for variable in module.variables}) == len(
            module.variables
        )


@pytest.mark.raw_data
def test_independent_raw_recomputation_matches_stress_equity_case():
    stress = pd.read_csv(
        RAW_DIR / "fed_stress_tests/public_results_DFAST_2025.csv"
    )
    stress = stress.loc[
        stress.exercise_name.eq("2025 Stress Test")
        & stress.scenario_name.eq("Supervisory Severely Adverse")
        & stress.id_rssd.notna()
    ].copy()
    stress["rssd"] = stress.id_rssd.astype(int).astype(str)
    with ZipFile(RAW_DIR / "ffiec_y9c/BHCF20241231.ZIP") as archive:
        y9c = pd.read_csv(
            archive.open("BHCF20241231.txt"),
            sep="^",
            dtype=str,
            encoding="cp1252",
            low_memory=False,
        )
    y9c = y9c.loc[
        y9c.BHCK3210.notna() & y9c.BHCK3210.ne(""),
        ["RSSD9001", "BHCK3210"],
    ].copy()
    y9c["equity_billions"] = pd.to_numeric(y9c.BHCK3210) / 1e6
    panel = stress.merge(
        y9c,
        left_on="rssd",
        right_on="RSSD9001",
        validate="one_to_one",
    )
    panel["loss_ratio"] = panel.loss_total_loan_amt / panel.equity_billions * 100
    panel["provision_ratio"] = panel.provision_amt / panel.equity_billions * 100
    loss = panel.sort_values(
        ["loss_ratio", "disclosure_legal_name"], ascending=[False, True]
    ).iloc[0]
    provision = panel.sort_values(
        ["provision_ratio", "disclosure_legal_name"], ascending=[False, True]
    ).iloc[0]
    equity_total = panel.equity_billions.sum()
    loss_total = panel.loss_total_loan_amt.sum()
    ratio = loss_total / equity_total * 100
    simple = panel.loss_ratio.mean()
    independent = (
        len(panel),
        equity_total,
        loss_total,
        ratio,
        simple,
        loss.disclosure_legal_name,
        loss.rssd,
        loss.equity_billions,
        loss.loss_total_loan_amt,
        loss.loss_ratio,
        loss.common_equity_tier1_actual_rat,
        loss.common_equity_tier1_min_rat,
        provision.disclosure_legal_name,
        provision.rssd,
        provision.equity_billions,
        provision.provision_amt,
        provision.provision_ratio,
        provision.common_equity_tier1_actual_rat,
        provision.common_equity_tier1_min_rat,
    )
    module = importlib.import_module(
        "evals.comparison_ranking.stress_loss_equity_absorption_leaders"
    )
    assert module.validate(
        dict(zip([variable.name for variable in module.variables], independent))
    ).success



@pytest.mark.raw_data
def test_phase12_local_table_keys_text_and_bis_unit_provenance():
    stress = load_expansion_table("fed_stress_tests").loc[
        lambda frame: frame.exercise_name.eq("2025 Stress Test")
        & frame.scenario_name.eq("Supervisory Severely Adverse")
        & frame.id_rssd.notna()
    ]
    y9c = load_expansion_table("ffiec_y9c")
    bis = load_expansion_table("bis_effective_exchange_rates")
    assert len(stress) == 22 and not stress.id_rssd.duplicated().any()
    assert not y9c.duplicated(["report_date", "reporter_rssd_id"]).any()
    assert not bis.duplicated(
        [
            "EER_TYPE:Type",
            "EER_BASKET:Basket",
            "REF_AREA:Reference area",
            "TIME_PERIOD:Time period or range",
        ]
    ).any()
    for frame in (stress, y9c, bis):
        text = frame.select_dtypes(include=["object", "string"])
        assert not text.apply(
            lambda column: column.astype("string").str.contains(
                "\ufffd|\\x00", regex=True, na=False
            ).any()
        ).any()
    # BIS carries the index base on metadata rows in the raw flat file, while its
    # observation rows leave UNIT_MEASURE blank. The column is registered as empty
    # at source, so clean_table asserts it holds nothing and drops it: the unit is
    # not readable from the runtime table at all, and its provenance is the raw
    # metadata row checked below.
    definition = next(
        table for table in TABLES if table.table_id == "bis.effective_exchange_rates"
    )
    assert definition.empty_at_source_columns == ("UNIT_MEASURE:Unit of measure",)
    assert "UNIT_MEASURE:Unit of measure" not in bis.columns
    with ZipFile(RAW_DIR / "bis/WS_EER_csv_flat.zip") as archive:
        metadata = pd.read_csv(
            archive.open("WS_EER_csv_flat.csv"),
            usecols=["UNIT_MEASURE:Unit of measure"],
            low_memory=False,
        )
    assert "882: Index, 2020 = 100" in set(metadata.iloc[:, 0].dropna())


