import argparse
import asyncio
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path

from config import BENCHMARKS_JSON, EXPERIMENTS_DIR, MODELS_TOML
from core.errors import BenchmarkSpecificationError, BillingError, UnansweredError
from core.evaluator import HARNESSES, INJECTION_MODES, evaluate_case, load_specs, write_result
from core.models import load_model, load_tool_gateway
from core.results import (
    RUNS_DIR, atomic_write_json, channel_path, experiment_summary, iter_result_files,
    latest_result_path, load_result, new_run_id, result_is_complete, result_path,
    sanitize_component, utc_now,
)
from core.transcripts import transcript_path, transcript_reference, write_transcript


def experiment_name(model_id: str, started_at: datetime | None = None) -> str:
    """Return a convenient default study id; custom study ids are also allowed."""
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_component(model_id)}_{timestamp}"


def _public_model(cfg) -> dict:
    if hasattr(cfg, "public_fingerprint"):
        return cfg.public_fingerprint()
    return {"name": cfg.name, "model_id": cfg.api_model, "base_url": cfg.base_url}


async def _evaluate_and_write(
    model,
    cfg,
    spec: dict,
    experiment_dir: Path,
    max_protocol_nudges: int,
    run_id: str = "run",
    arm: str = "strict",
    score_inline: bool = False,
    total_step_budget: int = 30,
    injection: str = "lazy",
    *,
    harness: str,
    accept_unanswered: bool = False,
):
    result = await evaluate_case(
        model, spec, max_protocol_nudges=max_protocol_nudges,
        total_step_budget=total_step_budget, injection=injection,
        accept_unanswered=accept_unanswered, harness=harness,
    )
    output = result_path(experiment_dir, spec["name"], cfg.api_model, run_id)
    # One transcript per conversation, written before the run.json that cites
    # it, so a crash can only leave an unreferenced transcript.
    for conversation in result.conversations:
        if conversation.messages:
            write_transcript(
                transcript_path(output, conversation.id), conversation.messages
            )
            conversation.transcript = transcript_reference(output, conversation.id)
    write_result(
        output,
        result,
        _public_model(cfg),
        {
            "run_id": run_id,
            "study_id": experiment_dir.name,
            "arm": arm,
            "max_protocol_nudges": max_protocol_nudges,
            "total_step_budget": total_step_budget,
            "injection": injection,
            "harness": harness,
            "pv_scored_inline": score_inline,
        },
    )
    if score_inline:
        from scripts.score_programmatic_verification import score_result_file
        try:
            score_result_file(output, spec, force=True)
        except Exception as error:
            # The trajectory is complete and must not trigger another paid model
            # call merely because post-hoc validation infrastructure failed.
            atomic_write_json(channel_path(output, "programmatic_verification"), {
                "schema": "finbench-programmatic-verification-v2",
                "scored_at": utc_now(),
                "case": spec["name"],
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return result


def _write_error(
    experiment_dir: Path,
    cfg,
    spec: dict,
    error: Exception,
    attempts: int,
    run_id: str = "run",
    arm: str = "strict",
) -> None:
    path = result_path(experiment_dir, spec["name"], cfg.api_model, run_id).with_name(
        "error.json"
    )
    payload = {
        "schema": "finbench-infrastructure-error-v1",
        "written_at": utc_now(),
        "model": _public_model(cfg),
        "case": spec["name"],
        "run_id": run_id,
        "arm": arm,
        "attempts": attempts,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    atomic_write_json(path, payload)


def _resume_directory(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = EXPERIMENTS_DIR / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise ValueError("experiment must identify a directory under experiments/")
    if not (resolved / RUNS_DIR).is_dir():
        raise FileNotFoundError(f"no runs to resume under {resolved}")
    return resolved


def _check_harness(experiment_dir: Path, harness: str) -> None:
    """Refuse to add one harness's runs to a study that holds another's.

    Resume decides what is done by case and model alone, so a study continued
    under the other harness would silently mix the two. Runs written before the
    tool harness existed record none and were CaveAgent runs.
    """
    if not experiment_dir.exists():
        return
    for _, path in iter_result_files(experiment_dir):
        try:
            stored = load_result(path).get("run", {}).get("harness", "code")
        except (OSError, ValueError, json.JSONDecodeError):
            continue  # unreadable runs are re-run, as _pending_specs treats them
        if stored != harness:
            raise ValueError(
                f"{experiment_dir.name} holds {stored}-harness runs ({path.parent}); "
                f"start a separate study for --harness {harness}"
            )


def _pending_specs(
    experiment_dir: Path,
    cfg,
    specs: list[dict],
) -> tuple[list[dict], int]:
    pending, skipped = [], 0
    for spec in specs:
        path = latest_result_path(experiment_dir, spec["name"], cfg.api_model)
        try:
            complete = bool(path and result_is_complete(load_result(path)))
        except (OSError, ValueError, json.JSONDecodeError):
            complete = False
        if complete:
            skipped += 1
        else:
            pending.append(spec)
    return pending, skipped


def completion_state(summary: dict, expected: int) -> tuple[str, dict]:
    """Return completion for one invocation; PV may intentionally be deferred."""
    enriched = dict(summary)
    enriched["expected"] = expected
    enriched["remaining"] = max(0, expected - enriched["completed"])
    complete = (
        enriched["completed"] == expected
        and enriched["infrastructure_errors"] == 0
    )
    return ("complete" if complete else "incomplete"), enriched


# An outage is retried once. A run the model ended without an answer (an empty
# answer, output still cut after every recovery, a prompt past the window) is the
# model's own doing, so the case is asked again up to this many times and the last
# try records the run as it ended, scored as a failure (core/failure.py).
UNANSWERED_RETRIES = 3


@dataclass(frozen=True)
class _Attempt:
    spec: dict
    outage_retries: int = 0
    unanswered_retries: int = 0

    @property
    def number(self) -> int:
        return 1 + self.outage_retries + self.unanswered_retries

    @property
    def accepts_unanswered(self) -> bool:
        return self.unanswered_retries >= UNANSWERED_RETRIES

    def retry_after(self, error: Exception) -> "_Attempt | None":
        if isinstance(error, UnansweredError):
            return replace(self, unanswered_retries=self.unanswered_retries + 1)
        if self.outage_retries == 0:
            return replace(self, outage_retries=1)
        return None


async def run_cases(
    model,
    cfg,
    specs: list[dict],
    experiment_dir: Path,
    concurrency: int,
    max_protocol_nudges: int = 0,
    run_id: str = "run",
    arm: str = "strict",
    score_inline: bool = False,
    total_step_budget: int = 30,
    injection: str = "lazy",
    *,
    harness: str,
) -> dict:
    """Run a replenishing worker pool with conservative error adaptation."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    initial_concurrency = concurrency
    target_concurrency = concurrency
    pending = deque(_Attempt(spec) for spec in specs)
    running = {}
    completed = passed = passed_after_repair = infrastructure_errors = 0
    unanswered = failed_after_retry = repair_attempted = repair_successes = 0
    consecutive_successes = 0

    while pending or running:
        while pending and len(running) < target_concurrency:
            attempt = pending.popleft()
            spec = attempt.spec
            task = asyncio.create_task(
                _evaluate_and_write(
                    model, cfg, spec, experiment_dir, max_protocol_nudges,
                    run_id, arm, score_inline,
                    total_step_budget, injection,
                    accept_unanswered=attempt.accepts_unanswered,
                    harness=harness,
                )
            )
            running[task] = attempt
            print(
                f"START {spec['name']} attempt={attempt.number} active={len(running)} "
                f"target={target_concurrency} pending={len(pending)}",
                flush=True,
            )

        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            attempt = running.pop(task)
            spec = attempt.spec
            try:
                result = task.result()
            except (BenchmarkSpecificationError, BillingError):
                # A defect in the case reproduces on every attempt, and an
                # endpoint out of credit fails every case until it is topped up.
                # Retrying only repeats the failure, so stop and let the
                # traceback say what must be fixed.
                for pending_task in running:
                    pending_task.cancel()
                raise
            except Exception as error:
                old_target = target_concurrency
                if isinstance(error, UnansweredError):
                    # The model's behaviour, not the provider's load.
                    unanswered += 1
                else:
                    infrastructure_errors += 1
                    consecutive_successes = 0
                    target_concurrency = max(1, target_concurrency // 2)
                print(
                    f"ERROR {spec['name']} attempt={attempt.number} "
                    f"type={type(error).__name__} concurrency={old_target}->{target_concurrency} "
                    f"message={error}", flush=True,
                )
                retry = attempt.retry_after(error)
                if retry is not None:
                    pending.appendleft(retry)
                    print(f"REQUEUE {spec['name']} attempt={retry.number}", flush=True)
                else:
                    completed += 1
                    failed_after_retry += 1
                    _write_error(
                        experiment_dir, cfg, spec, error, attempts=attempt.number,
                        run_id=run_id, arm=arm,
                    )
                continue

            completed += 1
            # A case passes when every one of its turns does.
            turns = result.turns
            case_passed = bool(turns) and all(turn.success for turn in turns)
            passed += int(case_passed)
            repair = next(
                (turn.protocol_repair for turn in turns if turn.protocol_repair), None
            )
            repaired_success = bool(repair and repair["success"])
            repair_attempted += int(bool(repair and repair["attempted"]))
            repair_successes += int(repaired_success)
            passed_after_repair += int(case_passed or repaired_success)
            consecutive_successes += 1
            cost = result.cost()
            failure_types = [
                turn.failure_type for turn in turns if turn.failure_type
            ] or None
            outputs = {
                f"{turn.turn}": turn.outputs for turn in turns
            } if len(turns) > 1 else (turns[0].outputs if turns else {})
            repair_text = ""
            if repair:
                repair_text = (
                    f" repair={'PASS' if repair['success'] else 'FAIL'}"
                    f" nudges={repair['protocol_nudges']}"
                    f" aggregate_steps={repair['steps']}"
                    f" aggregate_tokens={repair['total_tokens']}"
                )
            print(
                f"DONE {spec['name']} {'PASS' if case_passed else 'FAIL'} "
                f"completed={completed}/{len(specs)} active={len(running)} "
                f"steps={cost['steps']} tokens={cost['total_tokens']} "
                f"failure_type={failure_types} outputs={outputs}"
                f"{repair_text}", flush=True,
            )
            if target_concurrency < initial_concurrency and consecutive_successes >= 5:
                target_concurrency += 1
                consecutive_successes = 0
                print(
                    f"RECOVER concurrency={target_concurrency}/{initial_concurrency}",
                    flush=True,
                )

    return {
        "passed": passed,
        "passed_after_repair": passed_after_repair,
        "repair_attempted": repair_attempted,
        "repair_successes": repair_successes,
        "completed": completed,
        "total": len(specs),
        "infrastructure_errors": infrastructure_errors,
        "unanswered": unanswered,
        "failed_after_retry": failed_after_retry,
        "final_concurrency": target_concurrency,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", nargs="*", dest="case_names")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--step-budget", type=int, default=30,
        help="Maximum model/runtime steps per turn; default: 30.",
    )
    parser.add_argument("--protocol-nudges", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--injection", choices=INJECTION_MODES, default="lazy",
        help="How declared tables reach the agent: lazy (one `datasets` handle that "
             "loads on request; default) or eager (every table pre-loaded as a DataFrame).",
    )
    parser.add_argument(
        "--harness", choices=HARNESSES, default="tool",
        help="How the agent executes code: tool (native `python` tool calls; default) or "
             "code (fenced blocks in its text, CaveAgent). A study holds one harness.",
    )
    parser.add_argument(
        "--experiment",
        help="Study id under experiments/. Existing studies accept another model/run.",
    )
    parser.add_argument("--arm", help="Analysis label stored with this invocation.")
    parser.add_argument("--no-skip", action="store_true", help="Create a repeat run.")
    parser.add_argument(
        "--score-inline", action="store_true",
        help="Write PV immediately; otherwise replay later with score_programmatic_verification.",
    )
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        raise ValueError("concurrency must be between 1 and 32")
    if args.step_budget < 1:
        raise ValueError("step budget must be at least 1")

    load = load_model if args.harness == "code" else load_tool_gateway
    model, cfg = load(MODELS_TOML, args.model)
    all_specs = load_specs(BENCHMARKS_JSON)
    specs = all_specs
    if args.case_names:
        requested = set(args.case_names)
        specs = [spec for spec in specs if spec["name"] in requested]
        missing = requested - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")

    requested_study = args.experiment
    if requested_study:
        candidate = Path(requested_study)
        if not candidate.is_absolute():
            candidate = EXPERIMENTS_DIR / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(EXPERIMENTS_DIR.resolve()):
            raise ValueError("experiment must identify a directory under experiments/")
        if candidate.exists():
            experiment_dir = _resume_directory(requested_study)
        else:
            experiment_dir = candidate
    else:
        experiment_dir = EXPERIMENTS_DIR / experiment_name(cfg.api_model)
        if experiment_dir.exists():
            raise FileExistsError(experiment_dir)

    _check_harness(experiment_dir, args.harness)
    if not args.no_skip:
        specs, skipped = _pending_specs(experiment_dir, cfg, specs)
        print(f"SKIP completed={skipped} pending={len(specs)}", flush=True)
    if not specs:
        print(f"EXPERIMENT {experiment_dir}\nNOOP all requested cases already complete")
        return

    run_id = new_run_id()
    arm = args.arm or (
        "strict" if args.protocol_nudges == 0 else f"protocol-nudges-{args.protocol_nudges}"
    )
    print(f"EXPERIMENT {experiment_dir} RUN {run_id} MODEL {cfg.api_model}", flush=True)
    try:
        summary = await run_cases(
            model, cfg, specs, experiment_dir, concurrency=args.concurrency,
            max_protocol_nudges=args.protocol_nudges, run_id=run_id, arm=arm,
            score_inline=args.score_inline, total_step_budget=args.step_budget,
            injection=args.injection, harness=args.harness,
        )
    except BaseException:
        invocation_summary = experiment_summary(
            experiment_dir, model_id=cfg.api_model, run_id=run_id
        )
        raise

    invocation_summary = experiment_summary(
        experiment_dir, model_id=cfg.api_model, run_id=run_id
    )
    status, persisted = completion_state(invocation_summary, len(specs))
    print(
        f"SUMMARY {summary['passed']}/{summary['total']} inline-validator passed "
        f"completed={summary['completed']} infrastructure_errors="
        f"{summary['infrastructure_errors']} unanswered={summary['unanswered']} "
        f"pv_persisted={invocation_summary['pv_scored']} "
        f"final_concurrency={summary['final_concurrency']} failed_after_retry="
        f"{summary['failed_after_retry']} passed_after_repair="
        f"{summary['passed_after_repair']}/{summary['total']} repair_attempted="
        f"{summary['repair_attempted']} repair_successes={summary['repair_successes']}",
        flush=True,
    )
    if summary["completed"] != summary["total"] or summary["failed_after_retry"]:
        raise RuntimeError("one or more cases did not complete after retry")


if __name__ == "__main__":
    asyncio.run(main())
