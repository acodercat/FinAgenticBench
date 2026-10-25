"""An agent that executes Python through one native tool call.

The second harness beside CaveAgent. CaveAgent reads code out of the model's
text (one fenced ```python block per step); this one offers a ``python`` tool
and runs the ``code`` argument of each call. Everything else is held the same,
so a comparison between the two measures the protocol and nothing else:

- the same runtime (``IPyKernelRuntime`` with ``SECURITY_CHECKER``), so the
  same namespace, kernel silence timeout and security rules;
- the same execution deadline, output shaping and feedback texts, imported
  from cave-agent where it defines them;
- the same step accounting (every model call is a step, counted before it is
  sent, so one that fails is counted too) and the same
  :class:`~cave_agent.AgentResponse` / :class:`~cave_agent.StopReason`, so
  ``core.evaluator`` treats both alike.

The loop: send the history, run the step's tool call, answer it, repeat. Like a
CaveAgent step, a step runs one piece of code; the gateway asks for one call,
and any further call an endpoint sends anyway is answered as not run. A
response with no tool call is the final answer, unless the output cap cut it:
then the model is asked to resume, up to cave-agent's recovery limit. A call
the cap cut before its arguments ended runs nothing and counts as a cut too. A
refusal with no other text is the answer, even under a content-filter finish;
text the filter withdrew, or a filtered reply with none, is no answer. A run
that ends ``MODEL_ERROR`` -- a reply with neither text nor a call, a cut past
the last recovery, a filtered reply, a provider failure -- says why in
:attr:`ToolAgent.failure` (:mod:`core.failure`). A runtime failure ends the
run as ``RUNTIME_ERROR``; any other exception is a defect here and propagates.

Known differences from CaveAgent, beyond the protocol. They make results of
the two harnesses not directly comparable, which is why a study holds one:

- the model's own reasoning goes back to it on later steps (DeepSeek in
  thinking mode requires that inside a tool loop; the Responses API keeps
  reasoning as items), where CaveAgent sends text only;
- tool results are the raw execution output, without CaveAgent's
  ``<execution_output>`` wrapper and its per-step reminder;
- no context compaction, and no resumption of a stream that breaks mid-reply:
  that step fails as a provider error and the runner retries the case;
- a stalled stream ends at the client read timeout, not CaveAgent's 120 s
  idle watchdog.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cave_agent import AgentResponse, StopReason
from cave_agent._placeholders import (
    EMPTY_OUTPUT_PLACEHOLDER,
    EXECUTION_TIMEOUT_PROMPT,
    OUTPUT_RECOVERY_PROMPT,
    PERSISTED_OUTPUT_PREFIX,
    RUNTIME_STATE_LOST_NOTICE,
    build_persist_marker,
)
from cave_agent.agent import MAX_OUTPUT_RECOVERIES
from cave_agent.models.base import TokenUsage
from cave_agent.models.errors import ModelError
from cave_agent.prompts import SECURITY_ERROR_PROMPT
from cave_agent.runtime import PreemptibleRuntime, RuntimeExecutionError, RuntimeStateLostError
from cave_agent.security import SecurityError
from cave_agent.utils import sanitize_surrogates

from core.tool_gateway import MALFORMED_CALL, PYTHON_TOOL, Step, ToolCall, ToolGateway
from core.failure import EMPTY_ANSWER, FILTERED, OUTPUT_LIMIT, WITHDRAWN, Failure, classify

logger = logging.getLogger(__name__)

# Replies to a call that was not run. Each says so, so that the model cannot
# read an unexecuted call as a silent success.
UNKNOWN_TOOL = f"There is no tool named {{name}}; the only tool is `{PYTHON_TOOL}`. Nothing ran."
BAD_ARGUMENTS = "The arguments were not a JSON object with a string `code` field. Nothing ran."
CUT_ARGUMENTS = "The call was cut off at the output limit before its arguments ended. Nothing ran."
EXTRA_CALL = "Only one tool call runs per step, and this was not the first. Nothing ran."
UNREADABLE_CALL = (
    f"Your last tool call could not be read, so nothing ran. Call `{PYTHON_TOOL}` again "
    "with a JSON object holding a string `code` field."
)

# CaveAgent appends this to its system instructions; so does this harness.
_TIMEOUT_NOTE = (
    "- Code execution timeout: {timeout} seconds. For network requests and database queries, "
    "always set timeout parameters (e.g. requests.get(url, timeout=10), "
    "pd.read_sql(sql, con, params, timeout=10)) to avoid hanging."
)


@dataclass
class _Run:
    """What one :meth:`ToolAgent.run` has done so far."""

    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    code_snippets: list[str] = field(default_factory=list)
    text: str = ""
    # Text of replies cut at the output cap, which the next reply continues.
    pending: str = ""
    output_recoveries: int = 0

    def stop(self, reason: StopReason) -> AgentResponse:
        return AgentResponse(
            content=self.text,
            stop_reason=reason,
            steps=self.steps,
            elapsed=time.monotonic() - self.started,
            usage=self.usage,
            code_snippets=self.code_snippets,
        )


class ToolAgent:
    """Drop-in for the slice of ``CaveAgent`` that ``core.evaluator`` uses."""

    def __init__(
        self,
        model: ToolGateway,
        *,
        runtime: PreemptibleRuntime,
        system_instructions: str,
        max_steps: int,
        max_exec_output: int = 10000,
        max_exec_timeout: float = 900.0,
        exec_output_preview_chars: int = 2000,
    ):
        self.runtime = runtime
        self.system_instructions = system_instructions
        self.max_steps = max_steps
        self.max_exec_output = max_exec_output
        self.max_exec_timeout = max_exec_timeout
        self.exec_output_preview_chars = exec_output_preview_chars
        # Why the last run ended with MODEL_ERROR; None when it did not.
        self.failure: Failure | None = None
        self._conversation = model.conversation()

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._conversation.messages

    def build_system_prompt(self) -> str:
        return (
            f"{self.system_instructions}\n{_TIMEOUT_NOTE.format(timeout=self.max_exec_timeout)}\n\n"
            f"<variables>\n{self.runtime.describe_variables()}\n</variables>\n"
        )

    async def run(self, prompt: str) -> AgentResponse:
        run = _Run()
        self.failure = None
        # Rebuilt per run: a later turn registers outputs the earlier prompt did not list.
        self._conversation.set_system(self.build_system_prompt())
        self._conversation.add_user(prompt)
        while run.steps < self.max_steps:
            run.steps += 1
            try:
                step = await self._conversation.step()
            except ModelError as error:
                logger.exception("Model error — ending run")
                return self._fail(run, classify(error))
            run.usage = run.usage + step.usage
            run.text = run.pending + step.text
            if step.finish_reason == "content_filter" and not step.refused:
                logger.warning("Provider filtered the response — ending run")
                return self._fail(run, WITHDRAWN if step.text else FILTERED)
            if step.tool_calls:
                run.pending = ""
                try:
                    await self._answer_calls(step, run)
                except RuntimeExecutionError:
                    logger.exception("Runtime error — ending run")
                    return run.stop(StopReason.RUNTIME_ERROR)
                if step.finish_reason == "length" and step.tool_calls[0].code is None:
                    # The cap cut the call before its arguments ended, so nothing
                    # ran: a cut like any other.
                    if not self._spend_recovery(run):
                        return self._fail(run, OUTPUT_LIMIT)
                else:
                    # Only consecutive cuts use up the recoveries.
                    run.output_recoveries = 0
            elif step.finish_reason == MALFORMED_CALL:
                # The provider could not read the call, so there is none to answer:
                # say so, as a call with unreadable arguments is answered.
                run.pending = ""
                self._conversation.add_user(UNREADABLE_CALL)
            elif step.finish_reason == "length":
                # Cut at the output cap with no call: resume, as CaveAgent does.
                if not self._spend_recovery(run):
                    return self._fail(run, OUTPUT_LIMIT)
                run.pending = run.text
                self._conversation.add_user(OUTPUT_RECOVERY_PROMPT)
            elif run.text.strip():
                return run.stop(StopReason.COMPLETED)
            else:
                return self._fail(run, EMPTY_ANSWER)
        return run.stop(StopReason.MAX_STEPS)

    def _fail(self, run: _Run, failure: Failure) -> AgentResponse:
        self.failure = failure
        return run.stop(StopReason.MODEL_ERROR)

    @staticmethod
    def _spend_recovery(run: _Run) -> bool:
        """Count one more consecutive cut; ``False`` once none are left."""
        if run.output_recoveries == MAX_OUTPUT_RECOVERIES:
            logger.warning("Output cap still hit after %d recoveries — ending run",
                           MAX_OUTPUT_RECOVERIES)
            return False
        run.output_recoveries += 1
        return True

    async def _answer_calls(self, step: Step, run: _Run) -> None:
        """Run the step's first call and answer every call it made."""
        first, *extra = step.tool_calls
        cut = step.finish_reason == "length"
        output = await self._answer(first, cut=cut, code_snippets=run.code_snippets)
        self._conversation.add_tool_result(first.id, output)
        for call in extra:
            self._conversation.add_tool_result(call.id, EXTRA_CALL)

    async def _answer(self, call: ToolCall, *, cut: bool, code_snippets: list[str]) -> str:
        if call.name != PYTHON_TOOL:
            return UNKNOWN_TOOL.format(name=call.name)
        if call.code is None:
            return CUT_ARGUMENTS if cut else BAD_ARGUMENTS
        code_snippets.append(call.code)
        return await self._execute(call.code)

    async def _execute(self, code: str) -> str:
        """Run *code* and describe the outcome as CaveAgent does."""
        execution = asyncio.create_task(self.runtime.execute(code))
        try:
            done, _ = await asyncio.wait((execution,), timeout=self.max_exec_timeout)
        except BaseException:
            # The run is being cancelled: stop the execution before leaving.
            execution.cancel()
            with contextlib.suppress(BaseException):
                await execution
            raise
        if not done:
            execution.cancel()
            state_lost = False
            try:
                await execution
            except RuntimeStateLostError:
                state_lost = True
            except asyncio.CancelledError:
                pass
            timeout = EXECUTION_TIMEOUT_PROMPT.format(timeout=self.max_exec_timeout)
            return _with_notice(timeout, state_lost)

        result = execution.result()
        if not result.success and isinstance(result.error, SecurityError):
            return SECURITY_ERROR_PROMPT.format(error=result.error.message)
        if not result.success and not result.stdout and result.error is not None:
            output = f"{type(result.error).__name__}: {result.error}"
        else:
            output = result.stdout or EMPTY_OUTPUT_PLACEHOLDER
        output = sanitize_surrogates(output)
        if len(output) > self.max_exec_output:
            variable = await self.runtime.bind_unique(PERSISTED_OUTPUT_PREFIX, output)
            output = build_persist_marker(output, self.exec_output_preview_chars, variable)
        return _with_notice(output, result.state_lost)


def _with_notice(output: str, state_lost: bool) -> str:
    return f"{output}\n\n{RUNTIME_STATE_LOST_NOTICE}" if state_lost else output
