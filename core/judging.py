"""Shared judge-panel selection, concurrency and health accounting."""

from __future__ import annotations

from collections import Counter
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, TypeVar

from core.llm import LLMClient
from core.results import atomic_write_json
from core.models import ModelConfig, ModelRegistry


DEFAULT_JUDGE_PANEL = ("gpt55", "qwen-max", "deepseek-pro")

# Named once so the two judged channels cannot drift apart on what a quorum is.
MIN_PANEL = 2

# Which recorded settings make a stored score comparable to a new one. A change
# to any of them means the prior samples were produced under a different
# protocol.
RECONCILE_KEYS = (
    "criteria_fingerprint", "requested_panel", "exclude_same_lab",
    "params",
)

T = TypeVar("T")


@dataclass
class JudgeCallTracker:
    calls: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    prompt_tokens: Counter = field(default_factory=Counter)
    completion_tokens: Counter = field(default_factory=Counter)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(
        self, name: str, success: bool, usage: dict[str, int] | None = None
    ) -> None:
        with self._lock:
            self.calls[name] += 1
            if not success:
                self.failures[name] += 1
            if usage:
                self.prompt_tokens[name] += int(usage.get("prompt_tokens", 0))
                self.completion_tokens[name] += int(usage.get("completion_tokens", 0))

    def summary(self) -> dict[str, dict[str, int | bool]]:
        with self._lock:
            calls = dict(self.calls)
            failures = dict(self.failures)
            prompt_tokens = dict(self.prompt_tokens)
            completion_tokens = dict(self.completion_tokens)
        return {
            name: {
                "calls": calls[name],
                "successes": calls[name] - failures.get(name, 0),
                "failures": failures.get(name, 0),
                "prompt_tokens": prompt_tokens.get(name, 0),
                "completion_tokens": completion_tokens.get(name, 0),
                "total_tokens": (
                    prompt_tokens.get(name, 0) + completion_tokens.get(name, 0)
                ),
                "systematic_failure": (
                    calls[name] >= 3 and calls[name] == failures.get(name, 0)
                ),
            }
            for name in sorted(calls)
        }


def rater_pool(
    agent_lab: str | None,
    candidates: list[ModelConfig],
    *,
    exclude_same_lab: bool = False,
) -> list[ModelConfig]:
    if not exclude_same_lab:
        return list(candidates)
    if not agent_lab:
        raise ValueError("tested model has no lab; cannot exclude same-lab judges")
    unlabeled = [judge.name for judge in candidates if not judge.lab]
    if unlabeled:
        raise ValueError(f"judges lack lab metadata: {unlabeled}")
    selected = [judge for judge in candidates if judge.lab != agent_lab]
    if len(selected) < 2:
        raise ValueError("same-lab exclusion leaves fewer than two judges")
    return selected


def infer_tested_lab(saved_result: dict, registry: ModelRegistry) -> str | None:
    model = saved_result.get("model", {})
    if model.get("lab"):
        return model["lab"]
    if model.get("name"):
        try:
            return registry.get(model["name"]).lab
        except ValueError:
            pass
    config = registry.find_by_api_model(model.get("model_id", ""))
    return config.lab if config else None


def call_json(judge: ModelConfig, prompt: str) -> dict:
    result, completion = LLMClient(judge).complete_json_with_usage(prompt)
    result["judge_id"] = judge.name
    result["_usage"] = {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.prompt_tokens + completion.completion_tokens,
    }
    return result


@dataclass
class PanelOutcome:
    """What the panel actually returned, and whether that is enough to score.

    Reported rather than raised, so a sample lost to a judge outage is a
    recorded, countable absence that a later rerun re-attempts — not an
    exception that leaves no trace of how many judges answered.
    """

    successes: list[tuple[ModelConfig, Any]]
    failures: list[dict]
    requested: int
    minimum: int

    @property
    def responded(self) -> int:
        return len(self.successes)

    @property
    def quorum_met(self) -> bool:
        return self.responded >= self.minimum

    @property
    def complete(self) -> bool:
        return self.responded >= self.requested

    def skip_reason(self) -> str | None:
        """Why this sample cannot be scored, or None when it can."""
        if not self.quorum_met:
            return (
                f"judges_below_quorum: {self.responded}/{self.requested} responded, "
                f"minimum={self.minimum}"
            )
        return None


def evaluate_panel(
    judges: list[ModelConfig],
    function: Callable[[ModelConfig], T],
    *,
    tracker: JudgeCallTracker,
    minimum: int = MIN_PANEL,
) -> PanelOutcome:
    def guarded(judge: ModelConfig):
        try:
            return function(judge), None
        except Exception as error:
            return None, {
                "judge": judge.name,
                "error_type": type(error).__name__,
                "error": str(error)[:240],
            }

    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, len(judges))) as pool:
        for judge, (value, failure) in zip(judges, pool.map(guarded, judges)):
            usage = value.get("_usage") if failure is None and isinstance(value, dict) else None
            tracker.record(judge.name, failure is None, usage)
            if failure is None:
                successes.append((judge, value))
            else:
                failures.append(failure)
    return PanelOutcome(
        successes=successes, failures=failures,
        requested=len(judges), minimum=minimum,
    )


BLOCK_KEY = "judging"


def load_channel(path: Path, schema: str) -> tuple[dict, dict]:
    """Read a channel file as (entries by sample id, process block).

    A corrupt file raises rather than reading as absent: treating it as missing
    would re-run the whole panel and overwrite evidence that is still there.
    """
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported {schema.rsplit('-', 1)[0]} schema")
    entries = {
        turn["sample_id"]: {k: v for k, v in turn.items() if k != "turn"}
        for conversation in payload.get("conversations", [])
        for turn in conversation.get("turns", [])
        if turn.get("sample_id")
    }
    return entries, payload.get(BLOCK_KEY, {})


def write_channel(
    path: Path, *, schema: str, case: str, conversations: list, block: dict,
    entries: dict,
) -> None:
    """Lay a channel out like the run it scored, every turn present.

    A turn this channel never reached is written as `not_scored` rather than
    omitted, so a missing verdict and an unscored one cannot look the same. The
    header counters are derived from the turns written below and appended last,
    so they cannot disagree with the body.
    """
    laid_out, statuses = [], []
    for conversation in conversations:
        turns = []
        for index, identifier in enumerate(conversation["sample_ids"]):
            entry = entries.get(identifier, {"status": "not_scored"})
            turns.append({"turn": index + 1, "sample_id": identifier, **entry})
            statuses.append(entry.get("status", "not_scored"))
        laid_out.append({"id": conversation["id"], "turns": turns})
    atomic_write_json(path, {
        "schema": schema,
        "case": case,
        BLOCK_KEY: {
            **block,
            "turns": len(statuses),
            **{status: statuses.count(status) for status in sorted(set(statuses))},
        },
        "conversations": laid_out,
    })


def reconcile_prior_block(
    path: Path, prior_block: dict, current: dict, *, rescore: bool
) -> bool:
    """Whether stored entries may be resumed under the current settings.

    Same settings, resume. Changed settings with `rescore`, discard them.
    Changed settings without it, refuse: keeping old scores while advertising
    new settings would make a sensitivity run report the headline run's numbers.
    """
    changed = [
        key for key in RECONCILE_KEYS
        if key in current and prior_block.get(key) != current[key]
    ]
    if not changed:
        return True
    if not rescore:
        raise ValueError(
            f"{path}: scoring settings changed since these samples were judged "
            f"({', '.join(changed)}). Re-run with --rescore to recompute them, "
            f"restore the previous settings, or score into a new experiment."
        )
    return False
