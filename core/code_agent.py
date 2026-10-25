"""The code harness's agent: CaveAgent, reporting why a run failed.

CaveAgent ends a failed run with ``MODEL_ERROR`` and only logs the exception. A
handler on its logger keeps that exception for the run in progress, in a
context variable, so concurrent cases sharing one model each see only their own.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from cave_agent import AgentResponse, StopReason

from core.failure import Failure, classify

if TYPE_CHECKING:
    from cave_agent import CaveAgent

_cave_errors: ContextVar[list[BaseException] | None] = ContextVar("cave_errors", default=None)


class _KeepErrors(logging.Handler):
    """Keeps the exceptions cave-agent logs for the run in progress in this context."""

    def emit(self, record: logging.LogRecord) -> None:
        sink = _cave_errors.get()
        if sink is not None and record.exc_info and record.exc_info[1] is not None:
            sink.append(record.exc_info[1])


logging.getLogger("cave_agent").addHandler(_KeepErrors(level=logging.ERROR))


class CodeAgent:
    """CaveAgent, reporting :attr:`failure` as :class:`~core.tool_agent.ToolAgent` does."""

    def __init__(self, agent: CaveAgent):
        self._agent = agent
        # Why the last run ended with MODEL_ERROR; None when it did not.
        self.failure: Failure | None = None

    @property
    def messages(self) -> list[Any]:
        return self._agent.messages

    @property
    def max_steps(self) -> int:
        return self._agent.max_steps

    @max_steps.setter
    def max_steps(self, steps: int) -> None:
        self._agent.max_steps = steps

    async def run(self, query: str) -> AgentResponse:
        self.failure = None
        errors: list[BaseException] = []
        token = _cave_errors.set(errors)
        try:
            response = await self._agent.run(query)
        finally:
            _cave_errors.reset(token)
        if response.stop_reason is StopReason.MODEL_ERROR:
            self.failure = classify(errors[-1] if errors else None)
        return response
