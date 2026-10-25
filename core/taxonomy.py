"""Load and validate FinAgenticBench task-family and financial-domain metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import BENCHMARKS_JSON, CASE_TAXONOMY_PATH
from core.evaluator import load_specs


TAXONOMY_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def load_case_taxonomy(path: Path = CASE_TAXONOMY_PATH) -> dict:
    return json.loads(path.read_text())


def validate_case_taxonomy(
    taxonomy: dict,
    specs: list[dict] | None = None,
) -> list[str]:
    """Return human-readable taxonomy errors; an empty list means valid."""
    if specs is None:
        specs = load_specs(BENCHMARKS_JSON)

    errors: list[str] = []
    if taxonomy.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    families = taxonomy.get("task_families")
    domains = taxonomy.get("financial_domains")
    assignments = taxonomy.get("cases")
    for field, value in (
        ("task_families", families),
        ("financial_domains", domains),
        ("cases", assignments),
    ):
        if not isinstance(value, dict) or not value:
            errors.append(f"{field} must be a non-empty object")
    if errors:
        return errors

    for group_name, group in (
        ("task family", families),
        ("financial domain", domains),
    ):
        for item_id, item in group.items():
            if not TAXONOMY_ID.fullmatch(item_id):
                errors.append(f"invalid {group_name} id: {item_id!r}")
            if not isinstance(item, dict):
                errors.append(f"{item_id}: definition must be an object")
                continue
            for field in ("label", "definition"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    errors.append(f"{item_id}: {field} must be a non-empty string")

    by_name = {spec["name"]: spec for spec in specs}
    for name in sorted(set(by_name) - set(assignments)):
        errors.append(f"missing financial-domain assignment: {name}")
    for name in sorted(set(assignments) - set(by_name)):
        errors.append(f"assignment is not an active case: {name}")

    used_families: set[str] = set()
    used_domains: set[str] = set()
    for name, spec in by_name.items():
        family = spec.get("task_family")
        if family not in families:
            errors.append(f"{name}: unknown task family {family!r}")
        else:
            used_families.add(family)
        item = assignments.get(name)
        if not isinstance(item, dict):
            errors.append(f"{name}: taxonomy assignment must be an object")
            continue
        case_domains = item.get("financial_domains")
        if not isinstance(case_domains, list) or not case_domains:
            errors.append(f"{name}: financial_domains must be a non-empty list")
            continue
        if len(case_domains) != len(set(case_domains)):
            errors.append(f"{name}: duplicate financial domain")
        for domain in case_domains:
            if domain not in domains:
                errors.append(f"{name}: unknown financial domain {domain!r}")
            else:
                used_domains.add(domain)

    for family in sorted(set(families) - used_families):
        errors.append(f"task family has no active case: {family}")
    for domain in sorted(set(domains) - used_domains):
        errors.append(f"financial domain has no active case: {domain}")
    return errors
