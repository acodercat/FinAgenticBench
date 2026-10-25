"""The code harness's agent reports why CaveAgent ended a run, through cave-agent's own
code paths: what it logs is what the run's failure becomes."""

import asyncio
import logging
from types import SimpleNamespace as NS

import pytest
from cave_agent import CaveAgent
from cave_agent.agent import MAX_OUTPUT_RECOVERIES, StopReason
from cave_agent.models.errors import ProviderError

from core.code_agent import CodeAgent
from core.failure import EMPTY_ANSWER, OUTPUT_LIMIT
from core.models import ModelConfig
from core.responses_model import ResponsesModel
from core.stream_repair import FenceRepairingModel

_USAGE = NS(input_tokens=10, output_tokens=14, total_tokens=24)


def _final(status="completed", reason=None):
    kind = "response.completed" if status == "completed" else "response.incomplete"
    return NS(type=kind, response=NS(status=status, usage=_USAGE,
                                     incomplete_details=NS(reason=reason) if reason else None))


def _message():
    return NS(type="response.output_item.added", item=NS(type="message"))


def _text(delta):
    return NS(type="response.output_text.delta", delta=delta, output_index=0, content_index=0)


def _responses_agent(*replies) -> tuple[CodeAgent, list]:
    """A CodeAgent over CaveAgent over the Responses model, whose endpoint streams *replies*
    (lists of events) in turn, the last one again once they run out."""
    cfg = ModelConfig(name="luna", api_model="gpt-6-luna", api_key="k", max_tokens=1000,
                      temperature=1.0, reasoning_effort="medium")
    model = ResponsesModel("gpt-6-luna", "k", "http://localhost:1/v1", cfg.request_timeout(),
                           cfg.responses_options())
    sent = []

    async def create(**params):
        sent.append(params)
        events = replies[min(len(sent), len(replies)) - 1]

        async def stream():
            for event in events:
                yield event

        return stream()

    model.client = NS(responses=NS(create=create))
    return CodeAgent(CaveAgent(model=model)), sent


def test_cave_agents_empty_answer_reaches_the_code_agent_as_its_failure():
    agent, _ = _responses_agent([_message(), _final()])
    result = asyncio.run(agent.run("question"))
    assert result.stop_reason is StopReason.MODEL_ERROR and agent.failure.model_fault
    assert agent.failure.kind == EMPTY_ANSWER.kind


def test_output_still_cut_after_every_recovery_is_the_output_limit():
    agent, sent = _responses_agent(
        [_message(), _text("still reasoning"), _final("incomplete", "max_output_tokens")])
    result = asyncio.run(agent.run("question"))
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert agent.failure.kind == OUTPUT_LIMIT.kind and len(sent) == MAX_OUTPUT_RECOVERIES + 1


def test_a_refusal_is_the_answer_and_no_failure():
    refusal = NS(type="response.refusal.delta", delta="I can't help with that.")
    agent, _ = _responses_agent([_message(), refusal, _final()])
    result = asyncio.run(agent.run("question"))
    assert result.stop_reason is StopReason.COMPLETED and agent.failure is None


class _Chunks:
    """A Chat Completions stream of *chunks*, as LiteLLM yields it."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _chunk(content, finish=None):
    delta = NS(content=content, reasoning_content=None, refusal=None)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=None)


def test_a_whitespace_reply_through_the_fence_repair_is_an_empty_answer(monkeypatch):
    async def open_stream(self):
        return _Chunks([_chunk("  \n"), _chunk("", "stop")])

    monkeypatch.setattr("cave_agent.models.litellm._LiteLLMStreamResponse._open_stream",
                        open_stream)
    agent = CodeAgent(CaveAgent(model=FenceRepairingModel(model_id="m", api_key="k")))
    result = asyncio.run(agent.run("question"))
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert agent.failure.kind == EMPTY_ANSWER.kind


class _Logging:
    """A CaveAgent stand-in that logs *error* after *before* seconds and ends its run
    with a model error *after* seconds later."""

    def __init__(self, error, before, after):
        self.error, self.before, self.after = error, before, after

    async def run(self, query):
        await asyncio.sleep(self.before)
        try:
            raise self.error
        except ProviderError:
            logging.getLogger("cave_agent.agent").exception("Model error — ending run")
        await asyncio.sleep(self.after)
        return NS(stop_reason=StopReason.MODEL_ERROR)


def test_each_concurrent_run_keeps_only_its_own_failure():
    # The outage is logged while the empty answer's run is still in progress.
    empty = CodeAgent(_Logging(ProviderError(EMPTY_ANSWER.detail), before=0, after=0.02))
    outage = CodeAgent(_Logging(ProviderError("APIConnectionError: reset"), before=0.01, after=0))

    async def both():
        await asyncio.gather(empty.run("q"), outage.run("q"))

    asyncio.run(both())
    assert (empty.failure.kind, outage.failure.kind) == ("empty_answer", "provider")


def test_a_run_that_raises_leaves_no_failure_from_the_one_before():
    class _Raising:
        async def run(self, query):
            raise RuntimeError("kernel died")

    agent = CodeAgent(_Raising())
    agent.failure = EMPTY_ANSWER
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run("q"))
    assert agent.failure is None
