"""The tool-calling harness runs what the model asks, answers every call, and
reports its stops the way CaveAgent does."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from cave_agent import StopReason, Variable
from cave_agent._placeholders import EXECUTION_TIMEOUT_PROMPT, OUTPUT_RECOVERY_PROMPT
from cave_agent.agent import MAX_OUTPUT_RECOVERIES
from cave_agent.models.base import TokenUsage
from cave_agent.models.errors import PromptTooLongError, ProviderError
from cave_agent.prompts import SECURITY_ERROR_PROMPT
from cave_agent.runtime import IPyKernelRuntime, RuntimeExecutionError
from cave_agent.security import SecurityError

from core.security import SECURITY_CHECKER
from core.tool_agent import (
    BAD_ARGUMENTS,
    CUT_ARGUMENTS,
    EXTRA_CALL,
    UNKNOWN_TOOL,
    UNREADABLE_CALL,
    ToolAgent,
)
from core.tool_gateway import MALFORMED_CALL, Step, ToolCall
from core.failure import EMPTY_ANSWER, OUTPUT_LIMIT, WITHDRAWN


def _call(code, call_id="c1", name="python"):
    """A call running *code*; ``None`` gives arguments that stop short, as a cut call's do."""
    arguments = json.dumps({"code": code}) if code is not None else '{"code": "'
    return ToolCall(call_id, name, arguments)


def _step(text="", calls=(), finish=None):
    finish = finish or ("tool_calls" if calls else "stop")
    return Step(text, tuple(calls), finish, TokenUsage(10, 5, 15))


class _Scripted:
    """A gateway that plays scripted steps and records what the agent sent."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.system = []
        self.users = []
        self.results = []
        self.history = []

    def conversation(self):
        return self

    @property
    def messages(self):
        return self.history

    def set_system(self, prompt):
        self.system.append(prompt)

    def add_user(self, text):
        self.users.append(text)

    def add_tool_result(self, call_id, output):
        self.results.append((call_id, output))

    async def step(self):
        return self.steps.pop(0)


class _Runtime:
    def __init__(self, outputs=None):
        self.ran = []
        self.outputs = dict(outputs or {})
        self.bound = {}

    def describe_variables(self):
        return "- name: answer\n  type: NoneType\n  description: The answer."

    async def execute(self, code):
        self.ran.append(code)
        return NS(success=True, error=None, stdout=self.outputs.get(code, "ok"), state_lost=False)

    async def bind_unique(self, prefix, value, start=1):
        name = f"{prefix}_{len(self.bound) + 1}"
        self.bound[name] = value
        return name


def _agent(steps, runtime=None, max_steps=14):
    gateway = _Scripted(steps)
    agent = ToolAgent(
        gateway, runtime=runtime or _Runtime(), system_instructions="Analyse.", max_steps=max_steps,
    )
    return agent, gateway


def test_runs_each_call_then_returns_the_text_answer():
    agent, gateway = _agent([
        _step(calls=[_call("print(1)")]),
        _step(calls=[_call("answer = 2", "c2")]),
        _step("The answer is 2."),
    ])
    result = asyncio.run(agent.run("question"))
    assert result.stop_reason is StopReason.COMPLETED
    assert result.content == "The answer is 2."
    assert result.steps == 3
    assert result.code_snippets == ["print(1)", "answer = 2"]
    assert result.usage.total_tokens == 45
    assert agent.runtime.ran == ["print(1)", "answer = 2"]
    assert gateway.results == [("c1", "ok"), ("c2", "ok")]
    assert gateway.users == ["question"]
    assert "<variables>" in gateway.system[0] and "answer" in gateway.system[0]


def test_a_step_runs_one_call_and_answers_the_rest_as_not_run():
    """A CaveAgent step runs one code block; a tool step runs one call."""
    agent, gateway = _agent([
        _step(calls=[_call("a = 1", "c1"), _call("b = 2", "c2")]),
        _step("done"),
    ])
    result = asyncio.run(agent.run("q"))
    assert agent.runtime.ran == ["a = 1"]
    assert result.code_snippets == ["a = 1"]
    assert gateway.results == [("c1", "ok"), ("c2", EXTRA_CALL)]


def test_a_call_that_cannot_run_is_answered_without_running():
    agent, gateway = _agent([
        _step(calls=[_call("x", "c1", name="shell")]),
        _step(calls=[_call(None, "c2")]),
        _step(calls=[_call(None, "c3")], finish="length"),
        _step("done"),
    ])
    asyncio.run(agent.run("q"))
    assert agent.runtime.ran == []
    assert gateway.results == [
        ("c1", UNKNOWN_TOOL.format(name="shell")),
        ("c2", BAD_ARGUMENTS),
        ("c3", CUT_ARGUMENTS),
    ]


def _outcome(agent):
    """Run *agent*; return its response and the failure it reports."""
    result = asyncio.run(agent.run("q"))
    return result, agent.failure


def test_an_empty_answer_stops_with_a_model_error_the_failure_explains():
    result, failure = _outcome(_agent([_step("  ")])[0])
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert failure == EMPTY_ANSWER


class _Failing(_Scripted):
    def __init__(self, error):
        super().__init__([])
        self.error = error

    async def step(self):
        raise self.error


def test_a_filtered_step_or_a_provider_failure_is_the_infrastructures_not_the_models():
    filtered = _agent([_step(finish="content_filter")])[0]
    unreachable = ToolAgent(
        _Failing(ProviderError("APIConnectionError: connection reset")),
        runtime=_Runtime(), system_instructions="", max_steps=3,
    )
    for agent, kind in ((filtered, "content_filter"), (unreachable, "provider")):
        result, failure = _outcome(agent)
        assert result.stop_reason is StopReason.MODEL_ERROR
        assert (failure.kind, failure.model_fault) == (kind, False)


def test_a_refusal_is_the_answer_even_under_a_content_filter_finish_as_in_cave_agent():
    refused = Step("I can't help with that.", (), "content_filter", TokenUsage(10, 5, 15),
                   refused=True)
    result, failure = _outcome(_agent([refused])[0])
    assert result.stop_reason is StopReason.COMPLETED and failure is None
    assert result.content == "I can't help with that."
    # Text the filter withdrew is no answer.
    result, failure = _outcome(_agent([_step("partial answer", finish="content_filter")])[0])
    assert result.stop_reason is StopReason.MODEL_ERROR and failure == WITHDRAWN


def test_a_step_that_fails_is_still_counted_as_cave_agent_counts_it():
    agent = ToolAgent(
        _Failing(PromptTooLongError("context_length_exceeded")),
        runtime=_Runtime(), system_instructions="", max_steps=3,
    )
    result, failure = _outcome(agent)
    assert result.steps == 1
    assert (failure.kind, failure.model_fault) == ("prompt_too_long", True)


def test_any_other_failure_is_a_defect_and_propagates():
    agent = ToolAgent(
        _Failing(TypeError("bad parse")), runtime=_Runtime(), system_instructions="", max_steps=3,
    )
    with pytest.raises(TypeError):
        asyncio.run(agent.run("q"))


def test_the_step_budget_ends_the_run_and_can_be_changed_between_runs():
    agent, gateway = _agent([_step(calls=[_call("print(1)")])] * 3, max_steps=2)
    result = asyncio.run(agent.run("q"))
    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 2
    agent.max_steps = 1
    gateway.steps = [_step("final")]
    assert asyncio.run(agent.run("nudge")).steps == 1
    assert len(gateway.system) == 2


def test_oversize_output_is_kept_in_the_runtime_and_previewed():
    runtime = _Runtime({"print(big)": "x" * 12000})
    agent, gateway = _agent([_step(calls=[_call("print(big)")]), _step("done")], runtime=runtime)
    asyncio.run(agent.run("q"))
    (_, output), = gateway.results
    assert runtime.bound == {"_output_1": "x" * 12000}
    assert "_output_1" in output and len(output) < 3000


def test_against_a_real_kernel_the_assigned_output_is_retrievable():
    runtime = IPyKernelRuntime(
        functions=[], variables=[Variable("answer", None, "The answer.")], types=[],
        security_checker=SECURITY_CHECKER,
    )
    agent, gateway = _agent([
        _step(calls=[_call("answer = 6 * 7\nprint(answer)")]),
        _step("42"),
    ], runtime=runtime)

    async def run():
        try:
            result = await agent.run("q")
            return result, await runtime.retrieve("answer")
        finally:
            await runtime.stop()

    result, answer = asyncio.run(run())
    assert result.stop_reason is StopReason.COMPLETED
    assert answer == 42
    assert gateway.results[0][1].strip() == "42"


def test_a_reply_cut_at_the_output_cap_is_resumed_then_ends_at_the_output_limit():
    agent, gateway = _agent([
        _step("The ratio is 1.2 because", finish="length"), _step(" margins rose."),
    ])
    result = asyncio.run(agent.run("q"))
    assert result.stop_reason is StopReason.COMPLETED
    assert result.content == "The ratio is 1.2 because margins rose."
    assert gateway.users == ["q", OUTPUT_RECOVERY_PROMPT]

    cut = [_step("still going", finish="length")] * (MAX_OUTPUT_RECOVERIES + 1)
    agent, gateway = _agent(cut)
    result, failure = _outcome(agent)
    assert result.stop_reason is StopReason.MODEL_ERROR and failure == OUTPUT_LIMIT
    assert gateway.users.count(OUTPUT_RECOVERY_PROMPT) == MAX_OUTPUT_RECOVERIES


class _BrokenKernel(_Runtime):
    async def execute(self, code):
        raise RuntimeExecutionError("kernel would not start")


def test_a_runtime_failure_ends_the_run_as_a_runtime_error():
    agent, _ = _agent([_step(calls=[_call("x = 1")])], runtime=_BrokenKernel())
    assert asyncio.run(agent.run("q")).stop_reason is StopReason.RUNTIME_ERROR


class _SlowKernel(_Runtime):
    async def execute(self, code):
        await asyncio.sleep(10)


def test_an_execution_past_the_deadline_is_stopped_and_reported():
    gateway = _Scripted([_step(calls=[_call("loop()")]), _step("gave up")])
    agent = ToolAgent(gateway, runtime=_SlowKernel(), system_instructions="", max_steps=3,
                      max_exec_timeout=0.05)
    asyncio.run(agent.run("q"))
    assert gateway.results == [("c1", EXECUTION_TIMEOUT_PROMPT.format(timeout=0.05))]


class _GuardedKernel(_Runtime):
    async def execute(self, code):
        return NS(success=False, error=SecurityError("import os is blocked"), stdout=None,
                  state_lost=False)


def test_blocked_code_gets_the_security_text():
    agent, gateway = _agent([_step(calls=[_call("import os")]), _step("ok")],
                            runtime=_GuardedKernel())
    asyncio.run(agent.run("q"))
    assert gateway.results == [("c1", SECURITY_ERROR_PROMPT.format(error="import os is blocked"))]


def test_a_call_cut_before_its_arguments_ended_counts_as_a_cut():
    """Nothing ran, so it must not reset the count: repeated, it ends the run at the output limit."""
    cut_call = _step(calls=[_call(None)], finish="length")
    agent, gateway = _agent([cut_call] * (MAX_OUTPUT_RECOVERIES + 1), max_steps=30)
    result, failure = _outcome(agent)
    assert result.stop_reason is StopReason.MODEL_ERROR and failure == OUTPUT_LIMIT
    assert {output for _, output in gateway.results} == {CUT_ARGUMENTS}
    assert OUTPUT_RECOVERY_PROMPT not in gateway.users

    cut_reply = _step("thinking", finish="length")
    mixed = ([cut_reply, cut_call] * MAX_OUTPUT_RECOVERIES)[:MAX_OUTPUT_RECOVERIES + 1]
    result, failure = _outcome(_agent(mixed, max_steps=30)[0])
    assert result.stop_reason is StopReason.MODEL_ERROR and failure == OUTPUT_LIMIT


def test_only_consecutive_cuts_use_up_the_recoveries():
    """Cuts, a call that runs, then more cuts: the call resets the count."""
    cut = _step("thinking", finish="length")
    cut_call = _step(calls=[_call(None)], finish="length")
    steps = ([cut, cut_call] * MAX_OUTPUT_RECOVERIES)[:MAX_OUTPUT_RECOVERIES]
    steps += [_step(calls=[_call("x = 1")])]
    steps += [cut] * MAX_OUTPUT_RECOVERIES + [_step("done")]
    agent, _ = _agent(steps, max_steps=30)
    result, failure = _outcome(agent)
    assert result.stop_reason is StopReason.COMPLETED and failure is None


def test_an_unreadable_call_is_reported_and_the_run_goes_on():
    agent, gateway = _agent([_step(finish=MALFORMED_CALL), _step("done")])
    result = asyncio.run(agent.run("q"))
    assert result.stop_reason is StopReason.COMPLETED and result.steps == 2
    assert gateway.users == ["q", UNREADABLE_CALL]
