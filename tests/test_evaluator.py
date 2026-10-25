import asyncio
import importlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from cave_agent import StopReason
from cave_agent.models import TokenUsage
from cave_agent.models.errors import ProviderBillingError, ProviderError

import core.evaluator as evaluator_module
from config import BENCHMARKS_JSON
from core.errors import BillingError, InfrastructureError, UnansweredError
from core.evaluator import classify_failure
from core.results import json_default
from core.failure import EMPTY_ANSWER, OUTPUT_LIMIT, REASONS
from core.validation import ValidatorResult


def test_system_instructions_pin_the_runtime_execution_envelope():
    instructions = evaluator_module.SYSTEM_INSTRUCTIONS
    assert "Only a Markdown ```python fence is executable" in instructions
    assert "XML/DSML tool markup" in instructions
    assert "progress updates in Python comments" in instructions
    assert "standalone planning message" in instructions
    # The prompt shows the turn shape, and names the observed failure as the
    # counterexample: a prose-only "I'll start by inspecting" reply that ends
    # the run with nothing computed.
    assert "What a step looks like" in instructions
    assert "This is not a step" in instructions


def test_an_unanswered_reason_names_the_failure_before_missing_variables():
    missing = ValidatorResult(False, "missing", variables_not_set=True)
    for reason in REASONS:
        assert classify_failure(missing, reason) == reason
    assert classify_failure(missing, StopReason.MAX_STEPS.value) == "variables_not_set"


def test_failure_classification_preserves_strict_first_pass_semantics():
    assert classify_failure(
        ValidatorResult(True, "correct"), StopReason.COMPLETED.value
    ) is None
    assert classify_failure(
        ValidatorResult(False, "missing", variables_not_set=True),
        StopReason.COMPLETED.value,
    ) == "variables_not_set"
    assert classify_failure(
        ValidatorResult(False, "wrong"), StopReason.COMPLETED.value
    ) == "wrong_value"
    assert classify_failure(
        ValidatorResult(False, "wrong"), StopReason.MAX_STEPS.value
    ) == "max_steps"


def test_protocol_repair_preserves_first_pass_and_shares_total_budget(monkeypatch):
    seen_budgets = []

    class FakeRuntime:
        def __init__(self, *, functions, variables, types, security_checker, iopub_timeout):
            self.values = {variable.name: variable.value for variable in variables}

        async def retrieve(self, name):
            return self.values[name]

        async def stop(self):
            """The kernel runtime is a child process the evaluator must shut down."""
            self.stopped = True

    class FakeAgent:
        def __init__(self, model, *, runtime, max_steps, **kwargs):
            self.runtime = runtime
            self.max_steps = max_steps
            self.calls = 0
            self.messages = [{"role": "system", "content": "runtime description"}]

        async def run(self, prompt):
            self.calls += 1
            self.messages.append({"role": "user", "content": prompt})
            seen_budgets.append(self.max_steps)
            if self.calls == 1:
                return SimpleNamespace(
                    stop_reason=StopReason.COMPLETED,
                    content="I will inspect the data.",
                    steps=1,
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                    elapsed=1.0,
                    code_snippets=[],
                )
            assert "filing_accession" in prompt
            self.runtime.values["filing_accession"] = "accession"
            self.runtime.values["fiscal_period_end"] = "2025-01-31"
            self.runtime.values["acceptance_datetime"] = "2025-03-14T12:00:00Z"
            return SimpleNamespace(
                stop_reason=StopReason.COMPLETED,
                content="Computed from the runtime.",
                steps=3,
                usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                elapsed=2.0,
                code_snippets=["filing_accession = 'accession'"],
            )

    monkeypatch.setattr(evaluator_module, "IPyKernelRuntime", FakeRuntime)
    monkeypatch.setattr(evaluator_module, "CaveAgent", FakeAgent)
    monkeypatch.setattr(evaluator_module, "runtime_variables", lambda *_: [])
    spec = next(
        spec for spec in evaluator_module.load_specs(BENCHMARKS_JSON)
        if spec["name"] == "walmart_fy2025_revenue"
    )
    # This unit test exercises one protocol-repair cycle, not the case's later
    # state-dependent revenue turn. Keep the fixture aligned when the live case
    # grows additional turns.
    spec = json.loads(json.dumps(spec))
    spec["conversations"][0]["turns"] = spec["conversations"][0]["turns"][:1]
    case_module = importlib.import_module(spec["module"])
    monkeypatch.setattr(case_module, "variables", case_module.variables[:3])
    monkeypatch.setattr(
        case_module,
        "ground_truth",
        lambda: (
            "accession",
            "2025-01-31",
            "2025-03-14T12:00:00Z",
            "2024-02-01",
            680.985,
        ),
    )
    result = asyncio.run(
        evaluator_module.evaluate_case(None, spec, max_protocol_nudges=2, harness="code")
    )
    turn = result.conversations[0].turns[0]
    assert not turn.success
    assert turn.failure_type == "variables_not_set"
    assert turn.protocol_nudges == 1
    assert seen_budgets == [30, 29]
    repair = turn.protocol_repair
    assert repair["success"]
    assert repair["steps"] == 4
    assert repair["total_tokens"] == 37
    assert repair["elapsed"] == 3.0
    assert repair["code_snippets"] == ["filing_accession = 'accession'"]
    assert len(result.conversations[0].messages) == 3
    assert result.conversations[0].messages[-1]["content"].startswith("Your previous response")


def test_executed_code_reveals_which_registered_tables_were_named():
    names = ["facts_df", "filings_df", "companies_df"]
    referenced = evaluator_module.referenced_names(
        ["merged = facts_df.merge(companies_df, on='cik')", "print(merged.head())"],
        names,
    )
    assert referenced == ["companies_df", "facts_df"]


def test_a_table_named_only_in_a_string_or_comment_does_not_count():
    referenced = evaluator_module.referenced_names(
        ["# could have used filings_df\nlabel = 'filings_df'\nx = facts_df.shape"],
        ["facts_df", "filings_df"],
    )
    assert referenced == ["facts_df"]


def test_lazy_handle_calls_name_the_table_they_load():
    """datasets.load("x") is a reference; the same name in any other string is not."""
    referenced = evaluator_module.referenced_names(
        [
            'frame = datasets.load("filings_df")\n'
            'print(datasets.describe("facts_df"))\n'
            "other = 'companies_df'",
        ],
        ["facts_df", "filings_df", "companies_df"],
    )
    assert referenced == ["facts_df", "filings_df"]


def test_system_instructions_follow_the_injection_mode():
    eager = evaluator_module.system_instructions("eager", "code")
    lazy = evaluator_module.system_instructions("lazy", "code")
    assert eager == evaluator_module.SYSTEM_INSTRUCTIONS
    assert "registered there as pandas DataFrames" in eager and "datasets" not in eager
    assert 'datasets.load("table_name")' in lazy and "Only the listed tables can be loaded" in lazy
    with pytest.raises(ValueError):
        evaluator_module.system_instructions("streaming", "code")


def test_code_the_runtime_rejected_still_shows_intent():
    """A syntax error is not evidence the agent ignored the table."""
    referenced = evaluator_module.referenced_names(
        ["totals = filings_df.groupby('form'"], ["facts_df", "filings_df"]
    )
    assert referenced == ["filings_df"]


def test_result_json_default_preserves_numpy_scalar_types():
    assert json_default(np.float64(1.25)) == 1.25
    assert json_default(np.bool_(False)) is False
    # Anything else is kept as text, and named by the detector derived from the
    # same rule so the two cannot disagree about what survived.
    assert json_default(object()).startswith("<object object")


def test_result_json_cites_transcript_without_embedding_messages(tmp_path):
    turn = evaluator_module.TurnResult(
        turn=1, query="q", stores=[], success=True, failure_type=None,
        response="done", outputs={}, validation_message="correct",
        variables_not_set=False, steps=1, prompt_tokens=2, completion_tokens=3,
        total_tokens=5, elapsed=0.1, stop_reason="completed", executed_code=True,
        code_snippets=["x = 1"], protocol_nudges=0, protocol_repair=None,
    )
    result = evaluator_module.CaseResult(
        name="case",
        conversations=[evaluator_module.ConversationResult(
            id="main", turns=[turn],
            transcript="transcripts/model_main.jsonl",
            messages=[{"role": "system", "content": "large prompt"}],
        )],
    )
    path = tmp_path / "model.json"
    evaluator_module.write_result(path, result, {"model_id": "model"})
    saved = json.loads(path.read_text())
    assert saved["conversations"][0]["transcript"] == "transcripts/model_main.jsonl"
    assert "messages" not in saved["conversations"][0]["turns"][0]


def _one_turn_case(monkeypatch):
    """The first turn of walmart_fy2025_revenue, its three outputs, no tables.

    Its ground truth is pinned, so the case's validator reads no data and the
    test runs in a code-only checkout.
    """
    monkeypatch.setattr(evaluator_module, "runtime_variables", lambda *_: [])
    spec = next(
        spec for spec in evaluator_module.load_specs(BENCHMARKS_JSON)
        if spec["name"] == "walmart_fy2025_revenue"
    )
    spec = json.loads(json.dumps(spec))
    spec["conversations"][0]["turns"] = spec["conversations"][0]["turns"][:1]
    case_module = importlib.import_module(spec["module"])
    monkeypatch.setattr(case_module, "variables", case_module.variables[:3])
    monkeypatch.setattr(
        case_module, "ground_truth",
        lambda: ("accession", "2025-01-31", "2025-03-14T12:00:00Z", "2024-02-01", 680.985),
    )
    return spec


def _every_run_fails(monkeypatch, error):
    """Replace the runtime and CaveAgent with ones whose every run ends with a model
    error, having logged *error* as CaveAgent does; return the runtimes made."""
    runtimes = []

    class FakeRuntime:
        def __init__(self, *, functions, variables, types, security_checker, iopub_timeout):
            self.values = {variable.name: variable.value for variable in variables}
            self.iopub_timeout = iopub_timeout
            runtimes.append(self)

        async def retrieve(self, name):
            return self.values[name]

        async def stop(self):
            pass

    class FailingAgent:
        def __init__(self, model, *, runtime, max_steps, **kwargs):
            self.max_steps = max_steps
            self.messages = []

        async def run(self, prompt):
            try:
                raise error
            except type(error):
                logging.getLogger("cave_agent.agent").exception("Model error — ending run")
            return SimpleNamespace(
                stop_reason=StopReason.MODEL_ERROR, content="", steps=1,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=14, total_tokens=24),
                elapsed=1.0, code_snippets=[],
            )

    monkeypatch.setattr(evaluator_module, "IPyKernelRuntime", FakeRuntime)
    monkeypatch.setattr(evaluator_module, "CaveAgent", FailingAgent)
    return runtimes


def test_an_empty_answer_is_retried_rather_than_scored(monkeypatch):
    _every_run_fails(monkeypatch, ProviderError(EMPTY_ANSWER.detail))
    spec = _one_turn_case(monkeypatch)
    with pytest.raises(UnansweredError):
        asyncio.run(evaluator_module.evaluate_case(None, spec, harness="code"))


def test_an_accepted_empty_answer_is_recorded_and_fails(monkeypatch):
    _every_run_fails(monkeypatch, ProviderError(EMPTY_ANSWER.detail))
    spec = _one_turn_case(monkeypatch)
    result = asyncio.run(evaluator_module.evaluate_case(
        None, spec, accept_unanswered=True, harness="code",
    ))
    turn = result.conversations[0].turns[0]
    assert turn.stop_reason == EMPTY_ANSWER.kind
    assert not turn.success
    assert turn.failure_type == EMPTY_ANSWER.kind
    assert turn.response == ""


@pytest.mark.parametrize("error,raised", [
    (ProviderError("APIConnectionError: connection reset"), InfrastructureError),
    (ProviderError("The provider filtered this response and returned no content."),
     InfrastructureError),
    (ProviderBillingError("Your credit balance is too low"), BillingError),
])
def test_a_failure_not_the_models_is_never_scored_and_billing_stops_the_run(
        monkeypatch, error, raised):
    _every_run_fails(monkeypatch, error)
    spec = _one_turn_case(monkeypatch)
    with pytest.raises(raised):
        asyncio.run(evaluator_module.evaluate_case(
            None, spec, accept_unanswered=True, harness="code",
        ))


def test_a_cell_may_compute_silently_as_long_as_it_may_run(monkeypatch):
    """The kernel's default 30 s silence limit would interrupt a quiet cell long
    before the cell limit."""
    runtimes = _every_run_fails(monkeypatch, ProviderError(EMPTY_ANSWER.detail))
    spec = _one_turn_case(monkeypatch)
    asyncio.run(evaluator_module.evaluate_case(
        None, spec, accept_unanswered=True, harness="code",
    ))
    assert runtimes[0].iopub_timeout == evaluator_module.MAX_EXEC_SECONDS


def test_the_tool_harness_runs_the_case_through_tool_calls(monkeypatch):
    """harness="tool" builds a ToolAgent, asks with the tool protocol, and is judged alike."""
    from core.tool_gateway import Step, ToolCall

    class Runtime:
        def __init__(self, *, functions, variables, types, security_checker, iopub_timeout):
            self.values = {variable.name: variable.value for variable in variables}

        def inject_variable(self, variable):
            self.values[variable.name] = variable.value

        def describe_variables(self):
            return "\n".join(self.values)

        async def execute(self, code):
            exec(code, {}, self.values)
            return SimpleNamespace(success=True, error=None, stdout="done", state_lost=False)

        async def retrieve(self, name):
            return self.values[name]

        async def stop(self):
            pass

    code = (
        "filing_accession = 'accession'\n"
        "fiscal_period_end = '2025-01-31'\n"
        "acceptance_datetime = '2025-03-14T12:00:00Z'"
    )
    steps = [
        Step("", (ToolCall("c1", "python", json.dumps({"code": code})),), "tool_calls",
             TokenUsage(10, 5, 15)),
        Step("Computed from the runtime.", (), "stop", TokenUsage(10, 5, 15)),
    ]
    prompts = []

    class Gateway:
        def conversation(self):
            return self

        messages = []

        def set_system(self, prompt):
            prompts.append(prompt)

        def add_user(self, text):
            pass

        def add_tool_result(self, call_id, output):
            pass

        async def step(self):
            return steps.pop(0)

    monkeypatch.setattr(evaluator_module, "IPyKernelRuntime", Runtime)
    spec = _one_turn_case(monkeypatch)
    result = asyncio.run(evaluator_module.evaluate_case(Gateway(), spec, harness="tool"))
    turn = result.conversations[0].turns[0]
    assert turn.success, turn.validation_message
    assert turn.code_snippets == [code]
    assert "`python` tool" in prompts[0] and "```python" not in prompts[0]
    assert "The runtime is persistent." in prompts[0]


def test_an_accepted_output_limit_through_the_tool_harness_is_recorded(monkeypatch):
    """The tool agent's last cut reaches the stored turn as its stop reason."""
    from core.tool_gateway import Step

    class Runtime:
        def __init__(self, *, functions, variables, types, security_checker, iopub_timeout):
            self.values = {variable.name: variable.value for variable in variables}

        def describe_variables(self):
            return "\n".join(self.values)

        async def retrieve(self, name):
            return self.values[name]

        async def stop(self):
            pass

    class Gateway:
        messages = []

        def conversation(self):
            return self

        def set_system(self, prompt):
            pass

        def add_user(self, text):
            pass

        async def step(self):
            return Step("still reasoning", (), "length", TokenUsage(10, 5, 15))

    spec = _one_turn_case(monkeypatch)
    monkeypatch.setattr(evaluator_module, "IPyKernelRuntime", Runtime)
    with pytest.raises(UnansweredError) as raised:
        asyncio.run(evaluator_module.evaluate_case(Gateway(), spec, harness="tool"))
    assert raised.value.reason == OUTPUT_LIMIT.kind
    result = asyncio.run(evaluator_module.evaluate_case(
        Gateway(), spec, harness="tool", accept_unanswered=True,
    ))
    turn = result.conversations[0].turns[0]
    assert turn.stop_reason == OUTPUT_LIMIT.kind and turn.failure_type == OUTPUT_LIMIT.kind
