import gzip
import json

from config import DATASETS_DIR, PROJECT_ROOT
from core.dataset_layers import load_runtime_table


EXPECTED_FILES = {
    "bls_qcew": 7,  # 2019-2025 annual singlefiles (2026-08-27 full-population pull)
    "noaa_storm_events": 12,  # 2019-2025 details files (2026-08-27 pull)
    "ofr_market_stress": 5,
    "census_geo_crosswalk": 2,
    "ffiec_y9c": 9,
    "ffiec_nic_structure": 3,
}


def test_phase5_quality_report_is_clean_and_complete():
    report = json.loads(
        (PROJECT_ROOT / "metadata" / "phase5_quality_audit.json").read_text()
    )
    assert report["benchmark_cutoff"] == "2025-12-31"
    assert report["dataset_families"] == 8
    assert report["raw_files"] == 35
    assert report["errors"] == []
    # epa_ghgrp was removed on 2026-08-27; the phase-5 report is the acquisition record.
    assert set(EXPECTED_FILES) <= set(report["datasets"])


def test_phase5_manifests_have_no_download_failures():
    for dataset, expected in EXPECTED_FILES.items():
        manifest = json.loads(
            (DATASETS_DIR / dataset / "sources.json").read_text()
        )
        assert manifest["benchmark_freeze_date"] == "2025-12-31"
        assert len(manifest["raw_files"]) == expected
        assert manifest["failures"] == []


def test_phase5_crosswalk_outputs_have_expected_cardinality():
    build = json.loads(
        (PROJECT_ROOT / "metadata/build_reports/phase5_crosswalk_build.json").read_text()
    )
    assert build["geography"] == {
        "county_fips_rows": 3222,
        "zcta_county_relationship_rows": 46960,
        "zcta_primary_county_rows": 33791,
    }
    assert build["financial_entities"] == {
        "active_institution_rows": 61914,
        "current_relationship_rows": 35922,
    }
    outputs = [
        DATASETS_DIR / "census_geo_crosswalk/runtime/county_fips_2024.csv.gz",
        DATASETS_DIR / "census_geo_crosswalk/runtime/zcta_county_relationship_2020.csv.gz",
        DATASETS_DIR / "census_geo_crosswalk/runtime/zcta_primary_county_2020.csv.gz",
        DATASETS_DIR / "ffiec_nic_structure/runtime/institution_identifiers_current.csv.gz",
        DATASETS_DIR / "ffiec_nic_structure/runtime/ownership_relationships_current.csv.gz",
    ]
    for path in outputs:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert next(handle).strip()
            assert next(handle).strip()


def test_phase5_economic_table_has_fixed_width_state_keys_and_expected_grain():
    build = json.loads(
        (PROJECT_ROOT / "metadata/build_reports/phase5_economic_table_build.json").read_text()
    )
    assert build["bls_qcew_state_annual_totals"] == {
        "rows": 212,
        "years": [2024, 2025],
        "ownership_scopes": ["private", "total_covered"],
        "output": "bls_qcew/state_annual_totals.csv.gz",
    }
    # The state-total slice was retired for the 2019-2025 area x industry table;
    # the phase-5 report above is kept as the acquisition record.
    qcew = load_runtime_table("bls_qcew.area_industry_annual")
    assert sorted(qcew.year.unique()) == list(range(2019, 2026))
    assert qcew.area_fips.str.fullmatch(r"\d{5}").all()
    assert qcew.state_fips.str.fullmatch(r"\d{2}").all()
    assert not qcew.duplicated(["area_fips", "aggregation_level", "ownership_scope", "industry_code", "year"]).any()
    state_totals = qcew.loc[qcew.aggregation_level.eq("state_total")]
    assert state_totals.groupby(["year", "ownership_scope"]).size().eq(53).all()

    assert build["ofr_financial_stress_index"] == {
        "rows": 6575,
        "start_date": "2000-01-03",
        "end_date": "2025-12-31",
        "output": "ofr_market_stress/financial_stress_index.csv.gz",
        "vintage": "current revised history observed 2026-08-13",
    }
    fsi = load_runtime_table("ofr_market_stress.financial_stress_index")
    assert len(fsi) == 6575
    assert fsi.date.is_unique
    assert fsi.date.min() == "2000-01-03"
    assert fsi.date.max() == "2025-12-31"
    assert not fsi.drop(columns=["source_file"]).isna().any().any()


def test_census_county_identifiers_preserve_leading_zero_state_fips():
    counties = load_runtime_table("census_geo_crosswalk.counties_2024")
    assert counties.iloc[0].GEOID == "01001"
    assert counties.GEOID.str.fullmatch(r"\d{5}").all()
    mapping = counties.assign(state_fips=counties.GEOID.str[:2])[
        ["USPS", "state_fips"]
    ].drop_duplicates()
    assert len(mapping) == 52
    assert not mapping.USPS.duplicated().any()


def test_fed_stress_rssd_identifiers_do_not_acquire_float_suffixes():
    stress = load_runtime_table("fed_stress_tests.results")
    identifiers = stress.id_rssd.dropna()
    assert identifiers.str.fullmatch(r"\d+").all()
    assert "1039502" in set(identifiers)


def test_nic_relationship_level_is_preserved_and_governs_immediate_parent():
    relationships = load_runtime_table(
        "ffiec_nic_structure.ownership_relationships_current"
    )
    assert {1, 2}.issubset(
        set(relationships["relationship_level"].dropna().astype(int))
    )
    jpm = relationships.loc[
        relationships.offspring_rssd_id.eq("852218")
        & relationships.relationship_level.eq(1)
        & relationships.control_indicator.eq(1)
        & relationships.regulatory_relationship_indicator.eq(1)
    ]
    assert len(jpm) == 1
    assert jpm.iloc[0].parent_rssd_id == "1039502"

    catalog = json.loads(
        (DATASETS_DIR / "catalog.json").read_text()
    )
    key = catalog["tables"][
        "ffiec_nic_structure.ownership_relationships_current"
    ]["primary_key"]
    assert key["columns"] == [
        "parent_rssd_id",
        "offspring_rssd_id",
        "relationship_start",
        "relationship_level",
    ]
    assert key["is_unique"]
