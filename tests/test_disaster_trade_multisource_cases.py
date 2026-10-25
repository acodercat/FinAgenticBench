import importlib
import json
import re

import pandas as pd
import pytest

from config import PROJECT_ROOT
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


def test_phase11_build_report_and_governed_tables():
    report = json.loads(
        (PROJECT_ROOT / "metadata/build_reports/phase11_cross_source_table_build.json").read_text()
    )
    assert report["noaa_storm_event_details"] == {
        "rows": 69801,
        "state_dc_rows": 66387,
        "reported_property_damage_rows": 54755,
        "output": "noaa_storm_events/storm_event_details_2024.csv.gz",
        "source_file": "StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz",
    }
    assert report["fema_nfip_claims"]["rows"] == 102045
    assert report["fema_nfip_claims"]["represented_policy_count"] == 208437
    assert report["fema_nfip_claims"]["max_policy_count_per_record"] == 618
    assert report["fema_nfip_claims"]["signed_negative_net_payment_cells"] == 42
    assert report["bea_affiliate_goods_trade"] == {
        "rows": 76,
        "source_marker_d_cells": 111,
        "source_marker_star_cells": 7,
        "output": "bea_mne/affiliate_goods_trade_by_geography_2023.csv.gz",
        "source_member": "Part-II-H1-H14.xls",
    }

    storms = load_expansion_table("noaa_storm_events")
    # The claims table now spans 2019-2025; the phase-11 population checks
    # were measured on the 2024 year of loss and are asserted on that subset.
    claims = load_expansion_table("fema_nfip")
    claims = claims.loc[claims.date_of_loss.astype(str).str.startswith("2024")]
    assert storms.event_id.is_unique
    assert claims.claim_record_id.is_unique
    assert claims.policy_count.max() == 618
    assert claims[[
        "net_building_payment_usd", "net_contents_payment_usd", "net_icc_payment_usd"
    ]].lt(0).sum().sum() == 42


def test_phase11_runtime_exposes_only_requested_local_sources():
    variables = runtime_variables(
        {"noaa_storm_events", "fema_nfip"}
    )
    # Exposure is what the declaration guarantees; the sequence is registry order.
    assert sorted(variable.name for variable in variables) == sorted([
        "noaa_storm_event_details_df",
        "nfip_claims_df",
    ])
    descriptions = {variable.name: variable.description for variable in variables}
    assert "Missing damage is distinct from reported zero" in descriptions[
        "noaa_storm_event_details_df"
    ]
    assert "row count is not represented policy count" in descriptions[
        "nfip_claims_df"
    ]


def test_storm_nfip_case_anchor_values_and_validator():
    module = importlib.import_module(
        "evals.risk_screening.storm_damage_nfip_claim_grain_bridge"
    )
    truth = module.ground_truth()
    assert truth[:5] == ("FLORIDA", "FL", "2024-09", 264, 261)
    assert round(truth[5], 6) == 5.107544
    assert truth[6:10] == (55067, 117036, "Hurricane Helene", 115932)
    assert round(truth[10], 4) == 99.0567
    assert tuple(round(value, 6) for value in truth[11:13]) == (
        6.491534,
        28.885395,
    )
    assert round(truth[13], 4) == 22.4734
    expected = _expected_outputs(module)
    assert module.validate({
        **expected,
        "noaa_damage_leader_state_name": " florida ",
        "noaa_damage_leader_state_usps": "fl",
        "nfip_dominant_named_flood_event": " HURRICANE   HELENE ",
    }).success
    assert not module.validate({
        **expected, "nfip_represented_policy_count": 55067
    }).success



def test_phase11_registration_output_lineage_and_claim_contracts():
    registry = json.loads((PROJECT_ROOT / "benchmarks.json").read_text())
    paths = registry["cases"]
    assert paths["storm_damage_nfip_claim_grain_bridge"] == (
        "evals/risk_screening/storm_damage_nfip_claim_grain_bridge.json"
    )
    storm = json.loads((
        PROJECT_ROOT
        / "evals/risk_screening/storm_damage_nfip_claim_grain_bridge.json"
    ).read_text())
    assert storm["data_sources"] == ["noaa_storm_events", "fema_nfip"]
    storm_modes = {claim["id"]: claim["mode"] for claim in _claims(storm)}
    assert storm_modes == {
        "unreported_damage_handling": "required",
        "claim_record_policy_grain": "required",
        "damage_payment_nonreconciliation": "required",
        "flood_event_label_scope": "required",
        "coverage_ratio_aggregation_meaning": "required",
        "payment_coverage_overclaim_boundary": "conditional",
        "storm_causality_boundary": "conditional",
        "retrospective_vintage_boundary": "conditional",
    }
    # Mechanical output lineage. Repository-wide substitute-source necessity is
    # a semantic Gate-B check and is recorded separately, not proven by imports.
    output_lineage = {
        storm["name"]: {
            **{
                name: {"noaa_storm_events"}
                for name in {
                    "noaa_damage_leader_state_name", "noaa_damage_leader_state_usps",
                    "noaa_damage_leader_month", "noaa_selected_event_row_count",
                    "noaa_reported_property_damage_row_count",
                    "noaa_property_damage_usd_billions",
                }
            },
            **{
                name: {"noaa_storm_events", "fema_nfip"}
                for name in {
                    "nfip_claim_record_count", "nfip_represented_policy_count",
                    "nfip_dominant_named_flood_event",
                    "nfip_dominant_event_policy_count",
                    "nfip_dominant_event_policy_share_pct",
                    "nfip_total_net_claim_payment_usd_billions",
                    "nfip_total_reported_coverage_usd_billions",
                    "nfip_net_payment_to_coverage_pct",
                }
            },
        },
    }
    for spec in (storm,):
        module = importlib.import_module(spec["module"])
        declared = {variable.name for variable in module.variables}
        assert set(output_lineage[spec["name"]]) == declared
        assert all(
            sources <= set(spec["data_sources"])
            for sources in output_lineage[spec["name"]].values()
        )
        assert set().union(*output_lineage[spec["name"]].values()) == set(
            spec["data_sources"]
        )


def _raw_damage_usd(value):
    if pd.isna(value) or not str(value).strip():
        return float("nan")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMB]?)", str(value).strip())
    assert match is not None
    return float(match.group(1)) * {
        "": 1.0, "K": 1e3, "M": 1e6, "B": 1e9
    }[match.group(2)]


@pytest.mark.raw_data
def test_independent_raw_recomputation_matches_storm_nfip_case():
    storm_path = next((
        RAW_DIR / "noaa_storm_events"
    ).glob("StormEvents_details-ftp_v1.0_d2024_*.csv.gz"))
    storms = pd.read_csv(storm_path, low_memory=False, dtype={"STATE_FIPS": str})
    states = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY",
    }
    storms["state_fips"] = storms.STATE_FIPS.str.zfill(2)
    storms = storms.loc[storms.state_fips.isin(states)].copy()
    storms["state_usps"] = storms.state_fips.map(states)
    storms["month"] = pd.to_datetime(
        storms.BEGIN_DATE_TIME, format="%d-%b-%y %H:%M:%S"
    ).dt.strftime("%Y-%m")
    storms["property_damage_usd"] = storms.DAMAGE_PROPERTY.map(_raw_damage_usd)
    grouped = storms.groupby(["STATE", "state_usps", "month"], as_index=False).agg(
        damage=("property_damage_usd", "sum")
    )
    leader = grouped.sort_values(
        ["damage", "STATE", "month"], ascending=[False, True, True]
    ).iloc[0]
    selected_storms = storms.loc[
        storms.state_usps.eq(leader.state_usps) & storms.month.eq(leader.month)
    ]

    claims = pd.read_parquet(
        RAW_DIR / "fema_nfip/NfipClaims_2024.parquet"
    )
    claims["month"] = pd.to_datetime(claims.dateOfLoss).dt.strftime("%Y-%m")
    claims = claims.loc[
        claims.state.eq(leader.state_usps) & claims.month.eq(leader.month)
    ]
    events = claims.loc[claims.floodEvent.notna()].groupby(
        "floodEvent", as_index=False
    ).policyCount.sum().sort_values(
        ["policyCount", "floodEvent"], ascending=[False, True]
    )
    dominant = events.iloc[0]
    net = claims[[
        "netBuildingPaymentAmount", "netContentsPaymentAmount", "netIccPaymentAmount"
    ]].sum().sum()
    coverage = claims[[
        "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage"
    ]].sum().sum()
    independent = (
        leader.STATE,
        leader.state_usps,
        leader.month,
        len(selected_storms),
        int(selected_storms.property_damage_usd.notna().sum()),
        selected_storms.property_damage_usd.sum() / 1e9,
        len(claims),
        int(claims.policyCount.sum()),
        dominant.floodEvent,
        int(dominant.policyCount),
        dominant.policyCount / claims.policyCount.sum() * 100,
        net / 1e9,
        coverage / 1e9,
        net / coverage * 100,
    )
    module = importlib.import_module(
        "evals.risk_screening.storm_damage_nfip_claim_grain_bridge"
    )
    assert module.validate(dict(zip(
        [variable.name for variable in module.variables], independent
    ))).success



