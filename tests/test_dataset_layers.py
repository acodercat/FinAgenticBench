import json

import pytest

from config import DATASETS_DIR
from core.dataset_assets import (
    load_asset_catalog,
    load_tabular_asset,
    validate_asset_layers,
)
from core.dataset_layers import TABLES, load_runtime_table, validate_layers


def test_governed_dataset_catalog_is_complete_and_valid():
    catalog = json.loads((DATASETS_DIR / "catalog.json").read_text())
    assert catalog["schema_version"] == 1
    assert catalog["benchmark_cutoff"] == "2025-12-31"
    assert catalog["table_count"] == len(TABLES) == len(catalog["tables"])
    assert not validate_layers(verify_hashes=False)
    for table in catalog["tables"].values():
        assert "/raw/" not in f"/{table['layers']['runtime']['path']}"
        assert table["missing_value_policy"].startswith("empty CSV field")
        assert table["duplicate_policy"]["rule"].startswith("preserve source rows")
        assert table["revision_policy"]
        assert table["columns"]


def test_runtime_loader_applies_identifier_and_date_contracts():
    filings = load_runtime_table("sec.filings")
    assert str(filings["cik"].dtype) == "string"
    assert str(filings["accession"].dtype) == "string"
    assert filings["filing_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert filings["filing_date"].max() <= "2025-12-31"


def test_missing_values_remain_distinct_from_zero():
    transactions = load_runtime_table("sec_insider.nonderivative_transactions")
    assert transactions["trans_pricepershare"].isna().any()
    assert transactions["trans_pricepershare"].notna().any()
    assert transactions["trans_shares"].notna().any()


def test_every_dataset_source_has_complete_runtime_discovery():
    datasets = DATASETS_DIR
    source_dirs = sorted(path for path in datasets.iterdir() if path.is_dir())
    catalog = json.loads((datasets / "catalog.json").read_text())
    assert len(source_dirs) == catalog["source_count"] == 44
    assert {path.name for path in source_dirs} == set(catalog["sources"])
    assert catalog["asset_count"] == sum(
        source["raw_asset_count"] for source in catalog["sources"].values()
    )
    assert not validate_asset_layers()
    for source in source_dirs:
        assert (source / "runtime/_assets.csv.gz").is_file()
        assert (source / "schema.json").is_file()
        assets = load_asset_catalog(source.name)
        assert len(assets) == catalog["sources"][source.name]["raw_asset_count"]
        assert assets["asset_id"].is_unique


def test_preexisting_crosswalks_are_present_in_runtime():
    datasets = DATASETS_DIR
    expected = {
        "census_geo_crosswalk": {
            "county_fips_2024.csv.gz",
            "zcta_county_relationship_2020.csv.gz",
            "zcta_primary_county_2020.csv.gz",
        },
        "ffiec_nic_structure": {
            "institution_identifiers_current.csv.gz",
            "ownership_relationships_current.csv.gz",
        },
    }
    for source, names in expected.items():
        assert names <= {path.name for path in (datasets / source / "runtime").iterdir()}


@pytest.mark.raw_data
def test_special_formats_have_strict_on_demand_readers():
    cases = [
        ("fema_disasters", ".parquet"),
        ("sec_ftd", "<none>"),
    ]
    for source, suffix in cases:
        assets = load_asset_catalog(source)
        asset = assets.loc[assets["format"].eq(suffix)].iloc[0]
        frame = load_tabular_asset(source, asset["asset_id"], nrows=3)
        assert not frame.empty
        assert len(frame.columns) >= 1


def test_layer_issues_are_explicit_and_machine_readable():
    report = json.loads(
        (DATASETS_DIR / "layer_issues.json").read_text()
    )
    catalog = json.loads((DATASETS_DIR / "catalog.json").read_text())
    assert report["source_count"] == catalog["source_count"] == 44
    assert report["asset_count"] == catalog["asset_count"]
    assert report["issue_count"] == len(report["issues"])
    assert any(
        issue["source_id"] == "finra_short_volume"
        and issue["code"] == "source_manifest_declared_failures"
        for issue in report["issues"]
    )
