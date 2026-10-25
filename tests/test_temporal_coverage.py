import json

from config import DATASETS_DIR, PROJECT_ROOT


def test_every_dataset_has_exactly_one_temporal_classification():
    metadata = json.loads(
        (PROJECT_ROOT / "metadata" / "dataset_temporal_coverage.json").read_text()
    )
    classified = [
        dataset
        for group in metadata["coverage_classes"].values()
        for dataset in group["datasets"]
    ]
    manifested = {
        path.parent.name for path in DATASETS_DIR.glob("*/sources.json")
    }
    assert len(classified) == len(set(classified))
    assert set(classified) == manifested
    assert len(classified) == 44


def test_temporal_alignment_summary():
    metadata = json.loads(
        (PROJECT_ROOT / "metadata" / "dataset_temporal_coverage.json").read_text()
    )
    classes = metadata["coverage_classes"]
    overlapping = (
        classes["direct_2024_2025"]["datasets"]
        + classes["long_history_overlapping_2024_2025"]["datasets"]
    )
    assert metadata["benchmark_cutoff"] == "2025-12-31"
    assert metadata["canonical_cross_source_window"] == {
        "start": "2024-01-01",
        "end": "2025-12-31",
    }
    assert len(overlapping) == 41
    assert set(classes["lagged_official_annual"]["datasets"]) == {
        "fed_scf"
    }
    assert set(classes["reference_crosswalk"]["datasets"]) == {
        "census_geo_crosswalk", "ffiec_nic_structure"
    }
