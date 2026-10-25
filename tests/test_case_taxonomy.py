from collections import Counter
from pathlib import Path

from config import BENCHMARKS_JSON, EVALS_DIR
from core.evaluator import load_specs
from core.taxonomy import load_case_taxonomy, validate_case_taxonomy


def test_case_taxonomy_is_complete_and_valid():
    specs = load_specs(BENCHMARKS_JSON)
    taxonomy = load_case_taxonomy()
    assert validate_case_taxonomy(taxonomy, specs) == []
    assert set(taxonomy["cases"]) == {spec["name"] for spec in specs}


def test_physical_layout_has_only_task_family_and_case_id():
    specs = load_specs(BENCHMARKS_JSON)
    taxonomy = load_case_taxonomy()
    family_counts = Counter(spec["task_family"] for spec in specs)
    assert set(family_counts) == set(taxonomy["task_families"])
    for spec in specs:
        path = Path(spec["_path"])
        assert path.parent.parent == EVALS_DIR
        assert path.parent.name == spec["task_family"]
        assert path.stem == spec["name"]


