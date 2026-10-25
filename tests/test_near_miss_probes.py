"""What a validator must reject is the answer that is nearly right.

`scripts/verdict_baseline.py` proves each validator is not a no-op by feeding it
a value at least a million away from the truth.  Nothing that far out was ever
going to be accepted, so a tolerance that had quietly widened to admit a real
analyst error would pass that probe unnoticed.  A validator must never accept a
known naive-path answer, a different financial convention, a unit error or a
scope error as a numerical equivalent, and the shared `numeric_equal`
tolerance is never widened to admit one.

Four probes per numeric output, all mechanical, so they need no hand-authored
answer key and cover every case rather than the three traps:

``rounded``      the truth rounded the way its own Variable asks for it MUST
                 PASS.  This is what a model actually stores, and a tolerance
                 tightened around an unrounded figure rejects the case's own
                 answer.
``last_place``   the truth plus one unit of the last declared decimal MUST
                 FAIL.  The shared tolerance is 0.6 of that unit, so accepting
                 this means the output is not scored at the precision its own
                 description claims.
``unit_scale``   the value times 1000 MUST FAIL.  Billions read as millions is
                 the unit error this suite's outputs are most exposed to.
``sign``         the negated value MUST FAIL.  A sign convention is pinned by
                 P1, and a validator comparing magnitudes erases it.

A probe that a case legitimately survives -- an output whose financial meaning
makes one of these indistinguishable from the truth -- is registered in
EXCEPTIONS with the reason, the same way `verdict_baseline` registers its own.
"""

from __future__ import annotations

import importlib
import math
import re
from copy import deepcopy

import pytest

from config import BENCHMARKS_JSON
from core.evaluator import load_specs

DECIMALS = re.compile(r"round(?:ed)? to (\d+) decimal", re.IGNORECASE)

# probe id -> why this case survives it. Keep the reason financial, not
# "the test was inconvenient".
EXCEPTIONS: dict[str, str] = {}


def _declared_decimals(description: str) -> int | None:
    found = DECIMALS.search(description or "")
    return int(found.group(1)) if found else None


def _probes(value: float, places: int):
    """(name, candidate, must_pass) for one numeric output.

    Every miss is measured from the UNROUNDED truth, which is what
    ``numeric_equal`` compares against. A first version stepped away from
    ``round(value, places)`` instead and produced 133 false alarms in one run:
    when the truth sits near the midpoint of its last place -- -18.24535 pinned
    as -18.2454 -- one step from the ROUNDED figure lands 0.00005 from the
    truth, comfortably inside a 0.0006 band, and the validator was right to
    accept it. One step from the truth itself is always exactly 10^-places
    away, which the 0.6 x 10^-places band must always refuse.

    Sign and scale are skipped when the pinned figure is zero, for the same
    reason: a solver residual pinned at 0.0 has no sign to invert and no scale
    to mistake, and testing its raw 3.7e-16 asks a question the output does not
    pose. That guard reads the rounded figure; reading the raw value let 44
    such probes through.
    """
    step = 10.0**-places
    pinned = round(value, places)
    yield "rounded", pinned, True
    yield "last_place", value + step, False
    if pinned != 0:
        yield "unit_scale", value * 1000.0, False
        yield "sign", -value, False


def _cases():
    for spec in load_specs(BENCHMARKS_JSON):
        yield pytest.param(spec, id=spec["name"])


@pytest.mark.parametrize("spec", list(_cases()))
def test_validator_separates_a_near_miss_from_the_answer(spec):
    module = importlib.import_module(spec["module"])
    truth = dict(zip([v.name for v in module.variables], module.ground_truth()))
    described = {v.name: (v.description or "") for v in module.variables}
    wrong = []
    for name, value in truth.items():
        places = _declared_decimals(described[name])
        if places is None or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        for probe, candidate, must_pass in _probes(float(value), places):
            probe_id = f"{spec['name']}:{name}:{probe}"
            if probe_id in EXCEPTIONS:
                continue
            outputs = deepcopy(truth)
            outputs[name] = candidate
            accepted = bool(module.validate(outputs).success)
            if accepted is must_pass:
                continue
            wrong.append(
                f"{probe_id}: {'rejected' if must_pass else 'accepted'} "
                f"{candidate!r} against a pinned {round(float(value), places)!r} "
                f"at {places} decimals"
            )
    assert not wrong, "\n".join(wrong)
