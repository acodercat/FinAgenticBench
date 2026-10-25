"""Why a run ended without an answer, and whose failure that was.

A run that ends with ``MODEL_ERROR`` failed on the infrastructure's account or
on the model's, and the two call for opposite responses. A provider or
transport failure -- an outage, a timeout, a stalled stream, the provider's
filter withholding or withdrawing a reply -- says nothing about the model: the
evaluator raises ``InfrastructureError`` and the runner retries it. A reply the
model gave that cannot serve as one is the model's own doing:

- ``empty_answer``: a step ends cleanly with no text and no code. Thinking
  models do this after a few reasoning tokens; measured on 2026-10-05, GPT-6
  Luna at medium effort on 2 of 4 first steps of one case, DeepSeek V4 Flash
  with thinking on in 6 of its first 120 cases.
- ``output_limit``: consecutive replies keep hitting the output cap through
  every recovery the agent allows. DeepSeek V4 Flash in thinking mode does this
  on some cases every time, reasoning for ~60k tokens without ever calling the
  tool.
- ``prompt_too_long``: the history grew past the model's context window.

For these the evaluator raises ``UnansweredError``, and the runner asks again
only so many times before recording the run as it ended. An endpoint out of
credit is the infrastructure's, but no retry helps until it is topped up, so
the run stops (``BillingError``).

Each agent reports the cause as a :class:`Failure`: the tool harness's
:class:`core.tool_agent.ToolAgent` records its own, and
:class:`core.code_agent.CodeAgent` keeps the exception CaveAgent logs. The kinds
include one for an output cut past recovery (``output_limit``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cave_agent.models.errors import (
    PromptTooLongError,
    ProviderBillingError,
    ProviderError,
    StreamStalledError,
)


@dataclass(frozen=True)
class Failure:
    kind: str           # empty_answer, output_limit, prompt_too_long, content_filter, provider, ...
    model_fault: bool   # the model's own failure (scored), not the infrastructure's (retried)
    detail: str


# The model's failures. The first two texts are cave-agent's own
# (cave_agent/agent.py), which names these cases only in the message of the
# `ProviderError` it logs.
EMPTY_ANSWER = Failure("empty_answer", True, "The model completed without returning an answer.")
OUTPUT_LIMIT = Failure("output_limit", True,
                       "Model output was truncated and could not be recovered")
PROMPT_TOO_LONG = Failure("prompt_too_long", True, "The prompt exceeds the context window.")
# The infrastructure's. The provider's filter: with nothing to show for it, or
# withdrawing what it had sent.
FILTERED = Failure("content_filter", False,
                   "The provider filtered this response and returned no content.")
WITHDRAWN = Failure("content_filter", False,
                    "The provider filtered this response after partially streaming it.")
BILLING = Failure("billing", False, "The endpoint is out of credit.")
# Stored as a turn's stop reason when the runner accepts the run as it ended.
REASONS = (EMPTY_ANSWER.kind, OUTPUT_LIMIT.kind, PROMPT_TOO_LONG.kind)
_MODEL_FAILURES = ((EMPTY_ANSWER.detail, EMPTY_ANSWER), (OUTPUT_LIMIT.detail, OUTPUT_LIMIT))
# A gateway's content check blocking an output: a provider rule misfiring on the
# analysis, not a measure of the model.
_CONTENT_CHECK = ("filtered this response", "inappropriate content")


def classify(error: BaseException | None) -> Failure:
    """The failure a model-layer error (cave-agent's typed `ModelError`s) stands for."""
    detail = "no cause recorded" if error is None else f"{type(error).__name__}: {error}"
    if isinstance(error, PromptTooLongError):
        return replace(PROMPT_TOO_LONG, detail=detail)
    if isinstance(error, StreamStalledError):
        return Failure("stream_stalled", False, detail)
    if isinstance(error, ProviderBillingError):
        return replace(BILLING, detail=detail)
    if isinstance(error, ProviderError):
        text = str(error)
        if any(phrase in text for phrase in _CONTENT_CHECK):
            return replace(FILTERED, detail=detail)
        for phrase, failure in _MODEL_FAILURES:
            if phrase in text:
                return replace(failure, detail=detail)
        return Failure("provider", False, detail)
    return Failure("unknown" if error is None else type(error).__name__, False, detail)
