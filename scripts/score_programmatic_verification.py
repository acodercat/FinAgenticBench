"""Replay stored runtime outputs through the current case validators.

The model trajectory is immutable. Programmatic verification is a named,
replaceable channel so a validator fix does not require another model call.
A changed query is never replayed against the new validator because the stored
answer addressed a different task, and neither is a changed output contract: the
Variable descriptions carry this suite's precision, unit, date format and closed
token vocabularies, so editing one of those changes what a correct answer looks
like just as surely as editing the question.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from dataclasses import replace
import inspect
import json
from pathlib import Path

from config import BENCHMARKS_JSON, PROJECT_ROOT
from core import failure
from core.evaluator import load_specs, output_contract_fingerprint
from core.types import conversations_of
from core.results import (
    atomic_write_json, channel_path, iter_result_files, load_result,
    resolve_experiment, result_is_complete, utc_now,
)


SCHEMA = "finbench-programmatic-verification-v3"
SUPPORTED_SCHEMAS = {"finbench-programmatic-verification-v2", SCHEMA}
# Why a run can carry no verdict. Every one of these is a fact about the stored
# trajectory or the case, never a statement about the model's answer, so none of
# them may be reported as a failed verification.
REFUSALS = (
    "incomplete_result",   # the run never finished, or hit an infrastructure error
    "no_snapshot",         # the trajectory stored no outputs to replay
    "query_changed",       # the stored answer addressed a different question
    "contract_changed",    # the outputs were stored under a different storage contract
    "lossy_snapshot",      # an output reached disk only as a repr
    "stale",               # the case now declares outputs this run never held
    "no_validator",        # the case module exposes no validate()
    "raised",              # the validator itself crashed
)


def _validator_fingerprint(module) -> str:
    from core import validation

    paths = [
        Path(inspect.getsourcefile(item) or "")
        for item in (module, validation)
    ]
    if any(not path.is_file() for path in paths):
        return "unavailable"
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _project_variable_results(module, outputs: dict, verdict) -> dict[str, dict]:
    """Diagnose registered outputs through the case's own validator."""
    if isinstance(getattr(verdict, "variable_results", None), dict):
        return dict(verdict.variable_results)
    variables = list(module.variables)
    expected_values = tuple(module.ground_truth())
    if len(variables) != len(expected_values):
        raise ValueError("case variables and ground truth have different lengths")
    expected = {
        variable.name: value for variable, value in zip(variables, expected_values)
    }
    results = {}
    for variable in variables:
        actual = outputs.get(variable.name)
        if actual is None:
            results[variable.name] = {
                "is_set": False, "correct": False, "message": "not set",
                "method": "validator_projection",
            }
            continue
        candidate = dict(expected)
        candidate[variable.name] = actual
        projected = module.validate(candidate)
        results[variable.name] = {
            "is_set": True,
            "correct": bool(projected.success),
            "message": projected.message,
            "method": "validator_projection",
        }
    return results


def _verdict_record(verdict, stop_reason: str | None, variable_results: dict) -> dict:
    if verdict.success:
        failure_type = None
    elif stop_reason in failure.REASONS:
        # Why the outputs are missing, which "variables_not_set" would hide.
        failure_type = stop_reason
    elif verdict.variables_not_set:
        failure_type = "variables_not_set"
    elif stop_reason and stop_reason != "completed":
        failure_type = stop_reason
    else:
        failure_type = "wrong_value"
    total = len(variable_results)
    set_count = sum(bool(item.get("is_set")) for item in variable_results.values())
    correct = sum(bool(item.get("correct")) for item in variable_results.values())
    return {
        "success": bool(verdict.success),
        "failure_type": failure_type,
        "validation_message": verdict.message,
        "variables_not_set": bool(verdict.variables_not_set),
        "variable_verification": {
            "total": total,
            "set": set_count,
            "correct": correct,
            "set_rate": set_count / total if total else None,
            "accuracy": correct / total if total else None,
            "accuracy_among_set": correct / set_count if set_count else None,
            "interaction_failure": bool(
                not verdict.success and not verdict.variables_not_set
                and total > 0 and correct == total
            ),
            "items": variable_results,
        },
    }


def _score_turn(
    module, spec_turn, stored: dict, carried_outputs: dict, refuse
) -> dict:
    """Verify one stored turn, or return the status that refuses it."""
    # A changed question is checked first: the stored answer addressed a
    # different task, so nothing about its outputs is worth reporting. The run
    # records the query the agent was actually shown, which is what the current
    # case text has to match.
    asked = stored.get("query")
    if asked and asked != spec_turn.query:
        return refuse("query_changed")
    # The same argument one surface along. The Variable description carries this
    # suite's precision, unit, date format and closed token vocabularies, so an
    # answer stored under an earlier description is an answer to a different
    # contract, and scoring it against today's validator attributes the change
    # to the model. Runs written before the fingerprint existed carry no value
    # here and are scored as they always were.
    contract = stored.get("output_contract")
    if contract:
        current = output_contract_fingerprint(module, spec_turn.stores or [])
        if contract != current:
            return refuse("contract_changed", stored=contract, current=current)
    outputs = stored.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return refuse("no_snapshot")
    lossy = stored.get("outputs_unserializable") or []
    if lossy:
        return refuse("lossy_snapshot", unserializable=sorted(lossy))

    validators = getattr(module, "validators", None)
    if not isinstance(validators, dict) or spec_turn.validator not in validators:
        return refuse("no_validator", wanted=spec_turn.validator)

    declared = list(spec_turn.stores or outputs)
    missing = [name for name in declared if name not in outputs]
    if missing:
        return refuse("stale", missing=missing)
    # Resolve the scope before judging, for the same reason the evaluator does.
    spec_turn = replace(spec_turn, stores=declared)

    runtime = _StoredRuntime({**carried_outputs, **outputs})
    repair = stored.get("protocol_repair")
    repair_verdict = repair_variable_results = None
    try:
        verdict = validators[spec_turn.validator]("", runtime, spec_turn)
        variable_results = _project_variable_results(module, outputs, verdict)
        if isinstance(repair, dict) and isinstance(repair.get("outputs"), dict):
            repair_verdict = validators[spec_turn.validator](
                "", _StoredRuntime({**carried_outputs, **repair["outputs"]}), spec_turn
            )
            repair_variable_results = _project_variable_results(
                module, repair["outputs"], repair_verdict
            )
    except Exception as error:
        # A broken validator is the thing under test here, and one of them must
        # not take the rest of the experiment down with it. Per-variable
        # projection runs the validator too, so it shares this guard.
        return refuse("raised", error_type=type(error).__name__, error=str(error)[:300])

    record = {
        "status": "scored",
        "programmatic_verification": _verdict_record(
            verdict, stored.get("stop_reason"), variable_results
        ),
    }
    if repair_verdict is not None:
        record["protocol_repair_verification"] = _verdict_record(
            repair_verdict, repair.get("stop_reason"), repair_variable_results
        )
    return record


class _StoredRuntime:
    """Serves a turn's stored outputs through the runtime's read interface.

    Post-hoc verification replays what the agent left behind, so the validator
    must see the same names it saw live without a live runtime existing.
    """

    def __init__(self, outputs: dict):
        self._outputs = outputs

    def retrieve(self, name: str):
        return self._outputs[name]


def score_result_file(
    result_path: Path,
    current_spec: dict,
    *,
    force: bool = False,
) -> str:
    output = channel_path(result_path, "programmatic_verification")
    if output.exists() and not force:
        prior = json.loads(output.read_text())
        if prior.get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError(f"{output}: unsupported programmatic-verification schema")
        return "already_scored"

    saved = load_result(result_path)
    payload = {
        "schema": SCHEMA,
        "scored_at": utc_now(),
        "case": current_spec["name"],
        "run_file": result_path.name,
    }

    def refuse(status: str, **details) -> str:
        if status not in REFUSALS:
            raise ValueError(f"unknown verification refusal {status!r}")
        return {"status": status, **details}

    def finish(status: str) -> str:
        payload["status"] = status
        atomic_write_json(output, payload)
        return status

    if not result_is_complete(saved):
        payload["status"] = "incomplete_result"
        atomic_write_json(output, payload)
        return "incomplete_result"

    current = {conversation.id: conversation.turns
               for conversation in conversations_of(current_spec)}
    stored_shape = {c["id"]: len(c["turns"]) for c in saved["conversations"]}
    if {name: len(turns) for name, turns in current.items()} != stored_shape:
        return finish("turn_count_changed")

    module = importlib.import_module(current_spec["module"])
    payload["validator"] = {
        "module": current_spec["module"],
        "fingerprint": _validator_fingerprint(module),
    }
    conversations, statuses = [], []
    for conversation in saved["conversations"]:
        turns = []
        carried_outputs = {}
        for index, stored in enumerate(conversation["turns"]):
            record = _score_turn(
                module, current[conversation["id"]][index], stored,
                carried_outputs, refuse,
            )
            turns.append({"turn": index + 1, **record})
            statuses.append(record["status"])
            if isinstance(stored.get("outputs"), dict):
                carried_outputs.update(stored["outputs"])
        conversations.append({"id": conversation["id"], "turns": turns})
    payload["conversations"] = conversations
    scored = [item for item in statuses if item == "scored"]
    payload["scoring"] = {
        "turns": len(statuses), "scored": len(scored),
        **{status: statuses.count(status) for status in sorted(set(statuses))},
    }
    return finish("scored" if len(scored) == len(statuses) else "partially_scored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--case", nargs="*", dest="case_names")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment = resolve_experiment(args.experiment)
    specs = {spec["name"]: spec for spec in load_specs(BENCHMARKS_JSON)}
    requested = set(args.case_names) if args.case_names else None
    counts: dict[str, int] = {}
    for case_name, path in iter_result_files(experiment, requested):
        if case_name not in specs:
            raise ValueError(f"result references unknown active case {case_name!r}")
        outcome = score_result_file(path, specs[case_name], force=args.force)
        counts[outcome] = counts.get(outcome, 0) + 1
        print(f"{case_name}: {outcome}", flush=True)

    from scripts.run_stats import write_all_reports
    write_all_reports(experiment)
    print(f"SUMMARY {counts}")


if __name__ == "__main__":
    main()
