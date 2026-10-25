"""Load and validate the orthogonal capability index.

The case directory taxonomy describes the task a user sees.  This module loads a
separate, deliberately small reasoning-capability taxonomy used for sampling and
coverage reporting.  Keeping it central avoids duplicating metadata across every
case specification.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from config import BENCHMARKS_JSON, CAPABILITIES_PATH
from core.evaluator import load_specs


FAMILY_ID = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_SECONDARY_CAPABILITIES = 3


def load_capability_index(path: Path = CAPABILITIES_PATH) -> dict:
    """Return the capability taxonomy and case assignments from ``path``."""
    return json.loads(path.read_text())


def validate_capability_index(
    index: dict,
    specs: list[dict] | None = None,
) -> list[str]:
    """Return human-readable errors; an empty list means the index is valid."""
    if specs is None:
        specs = load_specs(BENCHMARKS_JSON)

    errors: list[str] = []
    if index.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    families = index.get("families")
    assignments = index.get("cases")
    if not isinstance(families, dict) or not families:
        return errors + ["families must be a non-empty object"]
    if not isinstance(assignments, dict):
        return errors + ["cases must be an object"]

    family_ids = set(families)
    for family_id, family in families.items():
        if not FAMILY_ID.fullmatch(family_id):
            errors.append(f"invalid capability id: {family_id!r}")
        if not isinstance(family, dict):
            errors.append(f"{family_id}: definition must be an object")
            continue
        for field in ("label", "definition"):
            if not isinstance(family.get(field), str) or not family[field].strip():
                errors.append(f"{family_id}: {field} must be a non-empty string")

    spec_names = {spec["name"] for spec in specs}
    assigned_names = set(assignments)
    for name in sorted(spec_names - assigned_names):
        errors.append(f"missing capability assignment: {name}")
    for name in sorted(assigned_names - spec_names):
        errors.append(f"assignment is not an active case: {name}")

    tagged_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for name, assignment in assignments.items():
        if not isinstance(assignment, dict):
            errors.append(f"{name}: assignment must be an object")
            continue
        primary = assignment.get("primary")
        secondary = assignment.get("secondary")
        if primary not in family_ids:
            errors.append(f"{name}: unknown primary capability {primary!r}")
        else:
            primary_counts[primary] += 1
            tagged_counts[primary] += 1
        if not isinstance(secondary, list):
            errors.append(f"{name}: secondary must be a list")
            continue
        if len(secondary) > MAX_SECONDARY_CAPABILITIES:
            errors.append(
                f"{name}: at most {MAX_SECONDARY_CAPABILITIES} secondary capabilities"
            )
        if len(secondary) != len(set(secondary)):
            errors.append(f"{name}: duplicate secondary capability")
        if primary in secondary:
            errors.append(f"{name}: primary capability repeated as secondary")
        for family_id in secondary:
            if family_id not in family_ids:
                errors.append(f"{name}: unknown secondary capability {family_id!r}")
            else:
                tagged_counts[family_id] += 1

    for family_id in sorted(family_ids):
        if not primary_counts[family_id]:
            errors.append(f"{family_id}: no case has this primary capability")
        if tagged_counts[family_id] < 2:
            errors.append(f"{family_id}: fewer than two independently tagged cases")
    return errors


def capability_names_for_case(index: dict, case_name: str) -> tuple[str, ...]:
    """Return the primary capability followed by secondary capabilities."""
    assignment = index["cases"][case_name]
    return (assignment["primary"], *assignment["secondary"])
