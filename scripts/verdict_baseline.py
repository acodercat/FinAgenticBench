"""Run deterministic correct/wrong probes through every real case validator."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from config import BENCHMARKS_JSON, PROJECT_ROOT
from core.evaluator import load_specs


BASELINE_PATH = PROJECT_ROOT / "metadata" / "verdict_baseline.json"


def mutate(value):
    """Produce an unmistakably different value while preserving broad shape."""
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return 123456789.0
        changed = number + max(1_000_000.0, abs(number) * 10 + 1)
        return int(changed) if isinstance(value, (int, np.integer)) else changed
    if isinstance(value, str):
        return "__FINBENCH_INVALID_VALUE__"
    if isinstance(value, list):
        if not value:
            return ["__FINBENCH_INVALID_VALUE__"]
        changed = deepcopy(value)
        changed[0] = mutate(changed[0])
        return changed
    if isinstance(value, tuple):
        if not value:
            return ("__FINBENCH_INVALID_VALUE__",)
        changed = list(deepcopy(value))
        changed[0] = mutate(changed[0])
        return tuple(changed)
    if isinstance(value, dict):
        if not value:
            return {"__invalid__": "__FINBENCH_INVALID_VALUE__"}
        changed = deepcopy(value)
        first = sorted(changed, key=str)[0]
        changed[first] = mutate(changed[first])
        return changed
    return "__FINBENCH_INVALID_VALUE__"


def check_baseline(path: Path = BASELINE_PATH) -> dict:
    baseline = json.loads(path.read_text())
    if baseline.get("schema") != "finbench-validator-verdict-baseline-v1":
        raise ValueError(f"unsupported verdict baseline schema in {path}")
    exceptions = baseline.get("exceptions", {})
    failures = []
    case_count = probe_count = 0
    for spec in load_specs(BENCHMARKS_JSON):
        case_count += 1
        module = importlib.import_module(spec["module"])
        truth = module.ground_truth()
        if len(module.variables) != len(truth):
            failures.append({"case": spec["name"], "probe": "shape_mismatch"})
            continue
        outputs = {
            variable.name: value
            for variable, value in zip(module.variables, truth)
        }
        correct = module.validate(deepcopy(outputs))
        probe_count += 1
        if not correct.success:
            failures.append({
                "case": spec["name"], "probe": "ground_truth",
                "message": correct.message,
            })
        for variable in module.variables:
            probe_count += 1
            candidate = deepcopy(outputs)
            candidate[variable.name] = mutate(candidate[variable.name])
            verdict = module.validate(candidate)
            probe_id = f"{spec['name']}:{variable.name}"
            expected_success = bool(exceptions.get(probe_id, {}).get("success", False))
            if bool(verdict.success) != expected_success:
                failures.append({
                    "case": spec["name"], "probe": variable.name,
                    "expected_success": expected_success,
                    "actual_success": bool(verdict.success),
                    "message": verdict.message,
                })
    return {
        "case_count": case_count,
        "probe_count": probe_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()
    report = check_baseline(args.baseline)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
