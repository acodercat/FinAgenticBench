"""A pinned combination must not have a second defensible value.

When a case asks for two figures at a stated precision and then for a quantity
derived from them, a model that combines the two figures it was just told to
report is doing something reasonable.  If ``ground_truth()`` instead combines
the unrounded operands, the two paths can disagree in the last place the
validator checks, and the model is marked wrong for a convention the query never
stated.  That is a uniqueness defect (P1), not a model error, so the repair is to
say which operands to use -- never to widen the tolerance.

This test finds the operand pair that reproduces each output from raw values,
recomputes it from the operands rounded to their own declared precision, and
requires the description to pin the convention wherever the two disagree.

Candidates are chosen by the VALUES, not by the wording of the description.  An
earlier version gated on descriptions matching
``minus|difference|spread|gap|margin|residual``, and searched subtraction only.
A 2026-09-07 review measured what that hid: thirteen unpinned outputs whose
descriptions say only "Store the ... present value" or "Store the ... stock or
signed stock change", and a further seven that are additive identities such as
``total debt = held by the public + intragovernmental``.  Neither relaxation is
worth having alone -- dropping the wording gate alone finds thirteen, adding the
addition search alone finds none, and both together find twenty.
"""

from __future__ import annotations

import importlib
import itertools
import re

from config import BENCHMARKS_JSON
from core.evaluator import load_specs

DECIMALS = re.compile(r"round(?:ed)? to (\d+) decimal", re.IGNORECASE)
PINNED = re.compile(r"unrounded", re.IGNORECASE)
# An identity either holds on the raw values or it is not the pair being looked
# for, so the search for operands is exact rather than tolerant. The tolerance
# below applies only to the question this test asks: whether the rounded path
# still lands on the pinned value.
EXACT = 1e-9


def _declared_decimals(description: str) -> int | None:
    found = DECIMALS.search(description or "")
    return int(found.group(1)) if found else None


def _unpinned_combination(numeric, described, name):
    """The first rounded-operand path that misses this output's pinned value."""
    places = _declared_decimals(described[name])
    tolerance = 0.6 * 10**-places
    pinned = round(numeric[name], places)
    for left, right in itertools.permutations(numeric, 2):
        if name in (left, right):
            continue
        first = round(numeric[left], _declared_decimals(described[left]))
        second = round(numeric[right], _declared_decimals(described[right]))
        for symbol, raw, rounded in (
            ("-", numeric[left] - numeric[right], first - second),
            ("+", numeric[left] + numeric[right], first + second),
        ):
            if abs(raw - numeric[name]) > EXACT:
                continue
            if abs(round(rounded, places) - pinned) <= tolerance:
                continue
            return (
                f"{name} pins {pinned} but combining the rounded "
                f"{left} {symbol} {right} gives {round(rounded, places)}"
            )
    return None


def test_derived_outputs_pin_their_intermediate_rounding():
    unpinned = []
    for spec in load_specs(BENCHMARKS_JSON):
        module = importlib.import_module(spec["module"])
        values = dict(zip([v.name for v in module.variables], module.ground_truth()))
        described = {v.name: (v.description or "") for v in module.variables}
        numeric = {
            name: float(value)
            for name, value in values.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and _declared_decimals(described[name]) is not None
        }
        for name in numeric:
            if PINNED.search(described[name]):
                continue
            found = _unpinned_combination(numeric, described, name)
            if found:
                unpinned.append(f"{spec['name']}:{found}")
    assert not unpinned, "outputs with an unstated rounding convention:\n" + "\n".join(unpinned)
