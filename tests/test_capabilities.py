from collections import Counter

from core.capabilities import load_capability_index, validate_capability_index
from config import BENCHMARKS_JSON
from core.evaluator import load_specs
from scripts.capability_coverage import CAPABILITY_REPORT_PATH, render_capability_report


def test_capability_index_is_complete_and_valid():
    specs = load_specs(BENCHMARKS_JSON)
    index = load_capability_index()
    assert validate_capability_index(index, specs) == []
    assert set(index["cases"]) == {spec["name"] for spec in specs}


def test_capabilities_have_multiple_cases_and_primary_coverage():
    index = load_capability_index()
    primary = Counter(item["primary"] for item in index["cases"].values())
    all_tags = Counter(
        capability
        for item in index["cases"].values()
        for capability in (item["primary"], *item["secondary"])
    )
    assert set(primary) == set(index["families"])
    assert all(count >= 2 for count in all_tags.values())


def test_generated_capability_report_is_current():
    assert CAPABILITY_REPORT_PATH.read_text() == render_capability_report()
