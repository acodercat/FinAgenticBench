import importlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from config import BENCHMARKS_JSON, EVALS_DIR
from core.evaluator import load_specs
from scripts.audit_case_design import claim_query_overlap, result_anchors
from scripts.catalog import CATALOG_PATH, render_catalog


def test_all_case_ground_truth_and_validators():
    specs = load_specs(BENCHMARKS_JSON)
    assert specs
    for spec in specs:
        module = importlib.import_module(spec["module"])
        expected = module.ground_truth()
        outputs = {variable.name: value for variable, value in zip(module.variables, expected)}
        verdict = module.validate(outputs)
        assert verdict.success, f"{spec['name']}: {verdict.message}"


def test_jpm_parent_name_accepts_both_declared_source_casings():
    module = importlib.import_module(
        "evals.comparison_ranking.jpm_standardized_advanced_cet1"
    )
    expected = module.ground_truth()
    names = [variable.name for variable in module.variables]
    outputs = dict(zip(names, expected))
    for legal_name in ("JPMORGAN CHASE & CO.", "JPMorgan Chase & Co."):
        candidate = {**outputs, "parent_legal_name": legal_name}
        assert module.validate(candidate).success
    invalid = {**outputs, "parent_legal_name": "JPMORGAN CHASE BANK, N.A."}
    assert not module.validate(invalid).success


def test_registry_covers_every_case_json_exactly_once():
    specs = load_specs(BENCHMARKS_JSON)
    registered = {Path(spec["_path"]).resolve() for spec in specs}
    case_json = {
        path.resolve()
        for path in EVALS_DIR.rglob("*.json")
        if path.name != "registry.json"
    }
    assert registered == case_json


def test_generated_case_catalog_is_current():
    assert CATALOG_PATH.read_text() == render_catalog()


def test_case_metadata_and_category_layout():
    specs = load_specs(BENCHMARKS_JSON)
    required = {
        "name", "module", "task_family", "data_sources", "difficulty",
        "case_type", "conversations", "judge_reference",
    }
    for spec in specs:
        assert required <= spec.keys()
        assert spec["difficulty"] in {"easy", "medium", "hard"}
        assert spec["case_type"] in {"baseline", "trap"}
        assert spec["data_sources"]
        turns = [
            turn
            for conversation in spec["conversations"]
            for turn in conversation["turns"]
        ]
        assert turns and all(turn.get("query") for turn in turns)
        claims = [claim for turn in turns for claim in turn.get("claims", [])]
        claim_ids = [claim["id"] for claim in claims]
        assert claim_ids and len(claim_ids) == len(set(claim_ids))
        assert all(claim["description"] for claim in claims)
        assert all(claim.get("mode") in {"required", "optional", "conditional"}
                   for claim in claims)
        assert any(claim["mode"] == "required" for claim in claims)
        assert "requirements" not in spec
        assert "category" not in spec and "subject" not in spec
        assert spec["source_mode"] in {"single", "multi"}
        assert spec["financial_domains"]
        for claim in claims:
            description = claim["description"]
            if claim["mode"] == "conditional":
                assert description.startswith("If "), (
                    f"{spec['name']}:{claim['id']} must name its trigger"
                )
            if claim["mode"] == "required":
                lowered = description.lower()
                assert not lowered.startswith("the response does not"), (
                    f"{spec['name']}:{claim['id']} is a negative required claim"
                )
                assert " does not " not in lowered, (
                    f"{spec['name']}:{claim['id']} mixes a boundary into a required claim"
                )
        assert spec["judge_reference"].get("facts")
        path = Path(spec["_path"])
        assert path.parent.name == spec["task_family"]
        assert path.stem == spec["name"]
        module_parts = spec["module"].split(".")
        assert module_parts[-2] == spec["task_family"]
        assert module_parts[-1] == spec["name"]

        module = importlib.import_module(spec["module"])
        assert all(variable.description.startswith("Store ") for variable in module.variables)
        for turn in turns:
            for variable in module.variables:
                assert not re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
                    turn["query"],
                ), f"{spec['name']}: query leaks output variable {variable.name}"
            assert not re.search(
                r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", turn["query"]
            ), f"{spec['name']}: query appears to expose a schema/code identifier"


def test_release_uses_quality_first_difficulty_policy():
    registry = json.loads((BENCHMARKS_JSON).read_text())
    assert registry["schema_version"] == 2
    assert registry["difficulty_policy"] == "quality_first_no_quota"
    specs = load_specs(BENCHMARKS_JSON)
    assert set(Counter(spec["difficulty"] for spec in specs)) == {
        "easy", "medium", "hard",
    }


SELF_DECLARED_BASELINE = re.compile(
    r"\bis\s+a\s+(?:hard\s+|clean\s+|pure\s+)?baseline\b", re.IGNORECASE
)


def test_query_pinned_alternatives_are_not_labelled_as_live_traps():
    """A docstring that calls the case a baseline contradicts ``case_type: trap``.

    The standard says the query pinning every alternative leaves no live wrong
    path, so the case is a baseline.  Authors record that conclusion in the
    docstring; this asserts the label agrees, rather than listing the instances
    known when the test was written.
    """
    mislabelled = []
    for spec in load_specs(BENCHMARKS_JSON):
        module = importlib.import_module(spec["module"])
        if not SELF_DECLARED_BASELINE.search(module.__doc__ or ""):
            continue
        if spec["case_type"] != "baseline":
            mislabelled.append((spec["name"], spec["case_type"]))
    assert not mislabelled, (
        "docstring declares a baseline while case_type says otherwise: "
        f"{mislabelled}"
    )


def test_high_overlap_trap_claims_have_evidence_or_review_waiver():
    for spec in load_specs(BENCHMARKS_JSON):
        if spec["case_type"] != "trap":
            continue
        for conversation in spec["conversations"]:
            for turn in conversation["turns"]:
                query_anchors = result_anchors(turn["query"])
                for claim in turn.get("claims", []):
                    if claim["mode"] != "required":
                        continue
                    overlap = claim_query_overlap(
                        turn["query"], claim["description"]
                    )
                    novel_anchors = (
                        result_anchors(claim["description"]) - query_anchors
                    )
                    if overlap >= 0.80 and not novel_anchors:
                        assert claim.get("overlap_waiver", "").strip(), (
                            f"{spec['name']}:{claim['id']} needs semantic overlap review"
                        )


def test_traps_have_measured_wrong_paths():
    chevron_amendment = importlib.import_module(
        "evals.risk_screening.chevron_q1_2022_amendment_no_numeric_change"
    )
    assert chevron_amendment.ground_truth()[2:] == (54.373, 54.373, 330, 0)
    facts = chevron_amendment.load_facts()
    original = facts.loc[
        facts["accession"].eq(chevron_amendment.ORIGINAL)
    ]
    assert original.groupby(chevron_amendment.KEYS).ngroups == 195
    assert original.groupby(chevron_amendment.KEYS, dropna=False).ngroups == 330


def test_jnj_point_in_time_recast_ground_truth():
    jnj_recast = importlib.import_module(
        "evals.temporal_path.jnj_fy2022_revenue_point_in_time_recast"
    )
    jnj_truth = jnj_recast.ground_truth()
    assert jnj_truth[:3] == (3, 3, 2)
    assert jnj_truth[3:6] == (
        "0000200406-23-000016", "2023-02-16T21:01:53", pytest.approx(94.943)
    )
    assert jnj_truth[6:] == (
        "0000200406-24-000013", "2024-02-16T21:12:22", pytest.approx(79.990),
    )


def test_every_trap_docstring_records_a_measured_wrong_path():
    specs = load_specs(BENCHMARKS_JSON)
    for spec in specs:
        if spec["case_type"] != "trap":
            continue
        module = importlib.import_module(spec["module"])
        docstring = module.__doc__ or ""
        assert any(character.isdigit() for character in docstring), spec["name"]
        assert "rounding-boundary tolerance" in docstring, spec["name"]
