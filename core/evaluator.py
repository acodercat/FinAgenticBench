from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from cave_agent import CaveAgent, StopReason
from cave_agent.runtime import IPyKernelRuntime

from config import CASE_TAXONOMY_PATH
from core import failure as failures
from core.code_agent import CodeAgent
from core.datastore import HANDLE_NAME, datastore_variable
from core.errors import (
    BenchmarkSpecificationError,
    BillingError,
    InfrastructureError,
    UnansweredError,
)
from core.results import (
    RESULT_SCHEMA,
    atomic_write_json,
    unserializable_outputs,
    utc_now,
)
from core.runtime_catalog import runtime_variables, select_runtime_tables
from core.security import SECURITY_CHECKER
from core.tool_agent import ToolAgent
from core.transcripts import messages_of
from core.types import Conversation, Turn, conversations_of

INJECTION_MODES = ("eager", "lazy")
# "code": CaveAgent reads fenced code from the text. "tool": core.tool_agent
# runs the code argument of a native `python` tool call.
HARNESSES = ("code", "tool")

_DATA_ACCESS = {
    "eager": (
        "The data is registered there as pandas DataFrames that the runtime\n"
        "describes; it is not in this text. Write and execute Python to inspect,\n"
        "filter and calculate from those DataFrames."
    ),
    "lazy": (
        f"The data is reachable through the registered `{HANDLE_NAME}` handle, whose\n"
        f"description lists this task's tables; it is not in this text. Call\n"
        f"`{HANDLE_NAME}.load(name)` to obtain a table as a pandas DataFrame and\n"
        f"`{HANDLE_NAME}.describe(name)` for its columns, units and caveats, then\n"
        "write and execute Python to inspect, filter and calculate from those\n"
        "DataFrames. Only the listed tables can be loaded."
    ),
}

# The first-step example, shown to the model as code. "table" stands for
# whichever registered table the question needs.
_FIRST_STEP_CODE = {
    "eager": (
        "# Orient before filtering: the columns, periods and units in play\n"
        "print(table_df.columns.tolist())\n"
        "print(table_df.head())"
    ),
    "lazy": (
        "# Orient before filtering: the columns, periods and units in play\n"
        f"print({HANDLE_NAME}.describe(\"table_name\"))\n"
        f"table_df = {HANDLE_NAME}.load(\"table_name\")\n"
        "print(table_df.columns.tolist())\n"
        "print(table_df.head())"
    ),
}
_FIRST_STEP_STANDS_FOR = {
    "eager": "table_df stands for whichever registered DataFrame\nthe question needs",
    "lazy": "\"table_name\" stands for whichever listed table\nthe question needs",
}

# How code is executed, per harness. "code" (CaveAgent) is the text every
# stored run before the tool harness was asked under; keep it byte-for-byte.
_EXECUTING = {
    "code": """\
- Each analysis step is exactly one fenced ```python code block with nothing
  after it. The environment runs that block and returns its real output; then
  either issue one further block or, once every required variable is assigned,
  give the final answer.
- Only a Markdown ```python fence is executable. Code inside provider-specific
  tags, special tokens, XML/DSML tool markup or other model-native envelopes is
  treated as prose and will not run. Never merely say that you will inspect the
  data: inspect it in code, and put progress updates in Python comments, never
  in a standalone planning message.""",
    "tool": """\
- Each analysis step is one call of the `python` tool with the code to run. The
  environment runs it and returns its real output; then either call the tool
  again or, once every required variable is assigned, give the final answer as
  plain text without a tool call.
- Only code passed to the `python` tool is executed. Code written in the text of
  a response is treated as prose and will not run. Never merely say that you will
  inspect the data: inspect it with the tool, and put progress updates in Python
  comments, never in a standalone planning message.""",
}


# How state carries between executions, per harness. "code" is again the text
# every stored CaveAgent run was asked under.
_SESSION = {
    "code": """\
- The session is continuous: variables, imports and intermediate results persist
  across blocks and can be referenced directly. Reuse them instead of
  recomputing.
- Keep each block compact. Print only small filtered samples or aggregates,
  never whole tables, and do not plot.""",
    "tool": """\
- The runtime is persistent. Every `python` call runs in the same session, across
  all steps and all questions of this task: variables, imports, loaded tables and
  intermediate results from earlier calls are still there and can be referenced
  directly. Reuse them instead of reloading or recomputing.
- Keep each call compact. Print only small filtered samples or aggregates, never
  whole tables, and do not plot.""",
}

# What a protocol-repair nudge asks for, per harness.
_NUDGE_ACTION = {
    "code": "Output exactly one fenced ```python code block",
    "tool": "Call the `python` tool",
}


def _first_step(injection: str, harness: str) -> str:
    code = _FIRST_STEP_CODE[injection]
    stands_for = _FIRST_STEP_STANDS_FOR[injection]
    if harness == "code":
        return (
            "What a step looks like. A complete first response is only this — one fence,\n"
            f"nothing before or after it ({stands_for}):\n\n```python\n{code}\n```"
        )
    indented = "\n".join("    " + line for line in code.splitlines())
    return (
        "What a step looks like. A complete first response is only one `python` tool\n"
        f"call, with code such as this ({stands_for}):\n\n{indented}"
    )


_SYSTEM_INSTRUCTIONS_TEMPLATE = """
You are a financial data-analysis agent operating in a persistent Jupyter-like
Python runtime. {data_access}

Executing code:
{executing}
- Never predict or invent execution output. Conclude only from results the
  environment actually returned.
{session}
- Use only the data the runtime registers. Do not access files, the network,
  benchmark modules or validator code.

{first_step}

This is not a step. It runs nothing, and because it contains no code the run
ends here with every output variable still unassigned:

I'll start by inspecting the available data to understand the structure.

Analysing:
- Determine the relevant records and their financial meaning from the question
  and the data. Rows with similar labels are not interchangeable: confirm
  period, unit, scope and filing before treating values as comparable.
- Convert units only when the requested output requires it.

Answering:
- Assign every required output variable in the runtime; printing a value does
  not record it.
- The final answer states each result together with the basis it rests on and
  the limits of what it establishes, in plain financial language without
  internal DataFrame or column names. Cover everything the question asks and
  nothing decorative.
"""


def system_instructions(injection: str, harness: str) -> str:
    """The agent's system prompt for one data-injection mode and harness."""
    if injection not in INJECTION_MODES:
        raise ValueError(f"injection must be one of {INJECTION_MODES}, got {injection!r}")
    if harness not in HARNESSES:
        raise ValueError(f"harness must be one of {HARNESSES}, got {harness!r}")
    return _SYSTEM_INSTRUCTIONS_TEMPLATE.format(
        data_access=_DATA_ACCESS[injection],
        executing=_EXECUTING[harness],
        session=_SESSION[harness],
        first_step=_first_step(injection, harness),
    )


SYSTEM_INSTRUCTIONS = system_instructions("eager", "code")


# One code block may run this long before the kernel is interrupted. Model-
# written code is otherwise unbounded: a single case held this process for
# almost four hours. Long enough for an honest bootstrap or tree, short
# enough that one case cannot own the study.
MAX_EXEC_SECONDS = 900.0


@dataclass
class TurnResult:
    """One turn's trajectory and the verdict on the outputs it asked for."""

    turn: int
    query: str
    stores: list[str]
    success: bool
    failure_type: str | None
    response: str
    outputs: dict
    validation_message: str
    variables_not_set: bool
    steps: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed: float
    stop_reason: str
    executed_code: bool
    code_snippets: list[str]
    protocol_nudges: int
    protocol_repair: dict | None
    # Outputs the result file can only store as a repr. Post-hoc verification
    # refuses these instead of scoring the text a lossless run never saw.
    outputs_unserializable: list = field(default_factory=list)
    # Which registered tables the executed code named, and which it never did.
    # Diagnostic only: a turn is not failed for leaving a table untouched.
    tables_referenced: list = field(default_factory=list)
    tables_unreferenced: list = field(default_factory=list)
    # A digest of the storage contract the agent was shown for this turn's
    # outputs. The query guard in post-hoc verification exists because an answer
    # to a different question is not evidence about this validator; the same is
    # true of an answer to a different storage contract, and the Variable
    # description is where this suite puts precision, unit, format and any
    # closed token vocabulary. Empty on runs written before 2026-09-07, which
    # verification then scores as it always did.
    output_contract: str = ""


@dataclass
class ConversationResult:
    id: str
    turns: list[TurnResult]
    transcript: str | None = None
    # Kept in memory until the runner writes the separate JSONL artefact.
    messages: list = field(default_factory=list, repr=False)


@dataclass
class CaseResult:
    name: str
    conversations: list[ConversationResult]

    @property
    def turns(self) -> list[TurnResult]:
        return [turn for item in self.conversations for turn in item.turns]

    def cost(self) -> dict:
        """Summed from the turns, so it can never disagree with them."""
        totals = {
            key: sum(getattr(turn, key) for turn in self.turns)
            for key in ("steps", "prompt_tokens", "completion_tokens",
                        "total_tokens", "elapsed")
        }
        return {"turns": len(self.turns), **totals}


async def _retrieve(runtime, name: str):
    value = runtime.retrieve(name)
    return await value if inspect.isawaitable(value) else value


class _ResolvedTurnRuntime:
    """Expose one awaited output snapshot through the validator interface.

    Cave runtime retrieval is asynchronous, while case validators are deliberately
    synchronous and are replayed later from stored outputs.  The evaluator already
    awaits every requested variable before validation, so validators must read that
    exact snapshot instead of starting a second, un-awaited live-runtime lookup.
    """

    def __init__(self, outputs: dict):
        self._outputs = outputs

    def retrieve(self, name: str):
        return self._outputs[name]


def referenced_names(code_snippets: list[str], names: list[str]) -> list[str]:
    """Which registered runtime names the executed code actually names.

    An agent that answers a two-source question after only ever naming one table
    either found a shortcut the case did not intend or was handed a source it
    does not need, and neither shows up in a pass/fail verdict. This is the
    FinAgenticBench form of a tool-call record: the runtime exposes DataFrames rather
    than callables, so what is worth recovering is which of them the code
    referred to.

    Read from the executed snippets, so it reports a *reference*, not a read: a
    name inside dead code counts, and a value reached some other way does not.
    That is weaker than interception but costs the run nothing and cannot change
    what the agent sees. Diagnostic only; no channel scores on it.
    """
    wanted = set(names)
    seen: set[str] = set()
    for snippet in code_snippets:
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            # The runtime rejected this block, but naming a table is still
            # evidence of intent, so fall back to a word-boundary scan.
            seen.update(
                name for name in wanted
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", snippet)
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in wanted:
                seen.add(node.id)
            elif (table := _handle_argument(node)) in wanted:
                # Under lazy injection a table is named as the string argument
                # of datasets.load(...) or datasets.describe(...); a table name
                # in any other string or in a comment is still not a reference.
                seen.add(table)
    return sorted(seen)


def _handle_argument(node: ast.AST) -> str | None:
    """The table name a `datasets.load("...")`-style call names, if any."""
    if not (isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute)):
        return None
    handle = node.func.value
    if not (isinstance(handle, ast.Name) and handle.id == HANDLE_NAME and node.func.attr in {"load", "describe"}):
        return None
    argument = node.args[0]
    return argument.value if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else None


def _validate(validator, case_name: str, response: str, runtime, turn: Turn):
    """Run a turn's validator, treating its own crash as a case defect.

    A validator that raises is broken benchmark code, not a model failure and
    not an outage: retrying pays for another model call and reproduces it.
    """
    try:
        verdict = validator(response, runtime, turn)
    except Exception as error:
        raise BenchmarkSpecificationError(
            f"{case_name}: validator raised {type(error).__name__}: {error}"
        ) from error
    if not (hasattr(verdict, "success") and hasattr(verdict, "variables_not_set")):
        raise BenchmarkSpecificationError(
            f"{case_name}: validator returned {type(verdict).__name__}, "
            f"not a ValidatorResult"
        )
    return verdict


def classify_failure(verdict, stop_reason: str) -> str | None:
    """Classify a recorded model failure without turning it into a retry.

    ``stop_reason`` is the value stored with the turn: the agent's
    :class:`StopReason`, or the reason an accepted unanswered run ended
    (:data:`core.failure.REASONS`).
    """
    if verdict.success:
        return None
    if stop_reason in failures.REASONS:
        # Why the outputs are missing, which "variables_not_set" would hide.
        return stop_reason
    if verdict.variables_not_set:
        return "variables_not_set"
    if stop_reason != StopReason.COMPLETED.value:
        return stop_reason
    return "wrong_value"


def load_specs(registry_path: Path) -> list[dict]:
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != 2 or not isinstance(registry.get("cases"), dict):
        raise ValueError(f"{registry_path}: expected case registry schema_version 2")
    specs = []
    for case_id, relative in registry["cases"].items():
        path = registry_path.parent / relative
        spec = json.loads(path.read_text())
        if spec.get("name") != case_id:
            raise ValueError(
                f"{path}: name={spec.get('name')!r}, registry case id={case_id!r}"
            )
        task_family = spec.get("task_family")
        if path.parent.name != task_family:
            raise ValueError(
                f"{path}: task_family={task_family!r}, directory={path.parent.name!r}"
            )
        if path.stem != case_id:
            raise ValueError(f"{path}: filename must equal case id {case_id!r}")
        spec["_path"] = str(path)
        specs.append(spec)
    taxonomy = json.loads(CASE_TAXONOMY_PATH.read_text())
    assignments = taxonomy.get("cases", {})
    for spec in specs:
        assignment = assignments.get(spec["name"])
        if not isinstance(assignment, dict):
            raise ValueError(
                f"{CASE_TAXONOMY_PATH}: missing assignment for {spec['name']}"
            )
        domains = assignment.get("financial_domains")
        if not isinstance(domains, list) or not domains:
            raise ValueError(
                f"{CASE_TAXONOMY_PATH}: invalid financial domains for {spec['name']}"
            )
        spec["financial_domains"] = list(domains)
        spec["source_mode"] = (
            "single" if len(set(spec["data_sources"])) == 1 else "multi"
        )
    return specs


def output_contract_fingerprint(module, stores) -> str:
    """Digest the storage contract this turn's outputs declare.

    Post-hoc verification compares this against the case as it stands now.
    Only the outputs the turn registers are digested, in the module's own
    variable order, so editing one turn's precision does not invalidate the
    stored answers of the turns around it.

    The description is the whole contract: this suite puts type, unit,
    precision, date format and any closed token vocabulary there and nowhere
    else, so a change to it changes what a correct answer looks like. Three
    commits in the first week of September 2026 did exactly that -- `25910d9`
    across 42 case modules, `9b41a87` and `1b6b87b` on four more -- and nothing
    could tell afterwards which stored runs had answered which contract.
    """
    wanted = set(stores or ())
    described = [
        f"{variable.name}\0{variable.description or ''}"
        for variable in getattr(module, "variables", [])
        if variable.name in wanted
    ]
    return hashlib.sha256("\0\0".join(described).encode()).hexdigest()[:20]


def _resolve_validator(module, case_name: str, turn: Turn):
    """Find the callable a turn names, refusing to silently auto-pass."""
    if not turn.validator:
        raise BenchmarkSpecificationError(f"{case_name}: turn declares no validator")
    validators = getattr(module, "validators", None)
    if not isinstance(validators, dict) or turn.validator not in validators:
        raise BenchmarkSpecificationError(
            f"{case_name}: module registers no validator named {turn.validator!r}"
        )
    return validators[turn.validator]


async def _evaluate_turn(
    agent,
    runtime,
    module,
    case_name: str,
    turn: Turn,
    index: int,
    stores: list[str],
    table_names: list[str],
    *,
    max_protocol_nudges: int,
    total_step_budget: int,
    accept_unanswered: bool,
    harness: str,
) -> TurnResult:
    """Run one turn and judge only the outputs that turn asked for.

    A run the model ended without an answer raises :class:`UnansweredError`
    for the runner to retry, unless ``accept_unanswered``: then the turn is
    recorded as it ended, with the reason as its stop reason, and judged like
    any other. A run the infrastructure ended raises
    :class:`InfrastructureError`, or :class:`BillingError` for an endpoint out
    of credit.
    """
    validator = _resolve_validator(module, case_name, turn)
    # The validator reads `turn.stores` to know what to check, so it must see
    # the resolved scope: a short-form turn declares none and would otherwise
    # be handed an empty set and pass judgement on nothing.
    turn = replace(turn, stores=list(stores))
    agent.max_steps = total_step_budget
    result = await agent.run(turn.query)
    stop_reason = result.stop_reason.value
    if result.stop_reason is StopReason.MODEL_ERROR:
        failure = agent.failure or failures.classify(None)
        if failure.kind == failures.BILLING.kind:
            raise BillingError(failure.detail)
        if not failure.model_fault:
            raise InfrastructureError(f"{failure.kind}: {failure.detail}")
        if not accept_unanswered:
            raise UnansweredError(failure.kind, failure.detail)
        stop_reason = failure.kind
    elif result.stop_reason is StopReason.RUNTIME_ERROR:
        raise InfrastructureError(f"agent stopped with {stop_reason}")

    async def read() -> dict:
        return {name: await _retrieve(runtime, name) for name in stores}

    values = await read()
    verdict = _validate(
        validator, case_name, result.content, _ResolvedTurnRuntime(values), turn
    )
    usage = result.usage
    protocol_nudges = 0
    protocol_repair = None

    if (
        max_protocol_nudges
        and verdict.variables_not_set
        and result.stop_reason is StopReason.COMPLETED
        and result.steps < total_step_budget
    ):
        totals = {
            "steps": result.steps, "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens, "elapsed": result.elapsed,
        }
        all_code_snippets = list(result.code_snippets)
        attempts = []
        repair_result, repair_values, repair_verdict = result, values, verdict

        while (
            protocol_nudges < max_protocol_nudges
            and repair_verdict.variables_not_set
            and repair_result.stop_reason is StopReason.COMPLETED
            and totals["steps"] < total_step_budget
        ):
            missing = [name for name in stores if repair_values.get(name) is None]
            if not missing:
                break
            protocol_nudges += 1
            remaining_steps = total_step_budget - totals["steps"]
            agent.max_steps = remaining_steps
            nudge = (
                "Your previous response did not contain executable progress and the "
                f"required variables {', '.join(missing)} are still unassigned. "
                f"Execute Python now. {_NUDGE_ACTION[harness]} "
                "and do not explain what you plan to do."
            )
            repair_result = await agent.run(nudge)
            repair_values = await read()
            repair_verdict = _validate(
                validator, case_name, repair_result.content,
                _ResolvedTurnRuntime(repair_values), turn,
            )
            repair_usage = repair_result.usage
            totals["steps"] += repair_result.steps
            totals["prompt_tokens"] += repair_usage.prompt_tokens
            totals["completion_tokens"] += repair_usage.completion_tokens
            totals["total_tokens"] += repair_usage.total_tokens
            totals["elapsed"] += repair_result.elapsed
            all_code_snippets.extend(repair_result.code_snippets)
            attempts.append({
                "nudge_number": protocol_nudges,
                "nudge": nudge,
                "remaining_step_budget_at_start": remaining_steps,
                "response": repair_result.content,
                "outputs": repair_values,
                "validation_message": repair_verdict.message,
                "variables_not_set": repair_verdict.variables_not_set,
                "steps": repair_result.steps,
                "prompt_tokens": repair_usage.prompt_tokens,
                "completion_tokens": repair_usage.completion_tokens,
                "total_tokens": repair_usage.total_tokens,
                "elapsed": repair_result.elapsed,
                "stop_reason": repair_result.stop_reason.value,
                "executed_code": bool(repair_result.code_snippets),
                "code_snippets": repair_result.code_snippets,
            })

        protocol_repair = {
            "attempted": bool(protocol_nudges),
            "protocol_nudges": protocol_nudges,
            "success": bool(repair_verdict.success),
            "failure_type": classify_failure(repair_verdict, repair_result.stop_reason.value),
            "response": repair_result.content,
            "outputs": repair_values,
            "validation_message": repair_verdict.message,
            "variables_not_set": repair_verdict.variables_not_set,
            **totals,
            "stop_reason": repair_result.stop_reason.value,
            "executed_code": bool(all_code_snippets),
            "code_snippets": all_code_snippets,
            "tables_referenced": referenced_names(all_code_snippets, table_names),
            "attempts": attempts,
            "total_step_budget": total_step_budget,
        }

    # Strict first pass only, matching every other headline field; the repair
    # arm records its own reference set inside protocol_repair.
    referenced = referenced_names(result.code_snippets, table_names)
    return TurnResult(
        turn=index + 1, query=turn.query, stores=list(stores),
        success=bool(verdict.success),
        failure_type=classify_failure(verdict, stop_reason),
        response=result.content, outputs=values,
        validation_message=verdict.message,
        variables_not_set=verdict.variables_not_set, steps=result.steps,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens, elapsed=result.elapsed,
        stop_reason=stop_reason,
        executed_code=bool(result.code_snippets), code_snippets=result.code_snippets,
        protocol_nudges=protocol_nudges, protocol_repair=protocol_repair,
        outputs_unserializable=unserializable_outputs(values),
        tables_referenced=referenced,
        tables_unreferenced=[n for n in table_names if n not in set(referenced)],
        output_contract=output_contract_fingerprint(module, stores),
    )


def _turn_scopes(case_name: str, conversation: Conversation, outputs: list) -> list[list[str]]:
    """Which outputs each turn registers, checked before anything runs.

    A turn naming an output the module does not declare, or an output no turn
    ever claims, is a case defect: the second leaves a variable unregistered and
    its validator reading a name that was never bound.
    """
    declared = {variable.name for variable in outputs}
    scoped = any(turn.stores for turn in conversation.turns)
    if not scoped:
        return [sorted(declared) for _ in conversation.turns]
    scopes = [list(turn.stores or []) for turn in conversation.turns]
    unknown = sorted({name for scope in scopes for name in scope} - declared)
    if unknown:
        raise BenchmarkSpecificationError(
            f"{case_name}/{conversation.id}: stores name outputs the module does "
            f"not register: {unknown}"
        )
    uncovered = sorted(declared - {name for scope in scopes for name in scope})
    if uncovered:
        raise BenchmarkSpecificationError(
            f"{case_name}/{conversation.id}: no turn claims {uncovered}; those "
            f"outputs would never be registered and their validator would fail"
        )
    return scopes


async def evaluate_case(
    model,
    spec: dict,
    *,
    max_protocol_nudges: int = 0,
    total_step_budget: int = 30,
    injection: str = "lazy",
    accept_unanswered: bool = False,
    harness: str,
) -> CaseResult:
    """Run one case's conversations and judge every turn.

    ``harness`` picks the agent: ``"code"`` is CaveAgent, which executes the
    fenced code in the model's text, and ``model`` is a cave-agent ``Model``;
    ``"tool"`` is :class:`core.tool_agent.ToolAgent`, which executes native
    ``python`` tool calls, and ``model`` is a :class:`core.tool_gateway.ToolGateway`.

    ``accept_unanswered`` records a run the model ended without an answer as
    the turn's answer instead of raising :class:`UnansweredError`; the runner
    sets it on the last attempt it allows.

    ``injection`` decides how the declared tables reach the agent: ``"eager"``
    registers each as a loaded DataFrame, ``"lazy"`` registers one ``datasets``
    handle that loads a table on request. Both expose exactly the declared
    tables; only memory, prompt size and the trace of what was loaded differ.
    """
    if not 0 <= max_protocol_nudges <= 2:
        raise ValueError("max_protocol_nudges must be between 0 and 2")
    if total_step_budget < 1:
        raise ValueError("total_step_budget must be at least 1")
    if injection not in INJECTION_MODES:
        raise ValueError(f"injection must be one of {INJECTION_MODES}, got {injection!r}")
    if harness not in HARNESSES:
        raise ValueError(f"harness must be one of {HARNESSES}, got {harness!r}")
    try:
        module = importlib.import_module(spec["module"])
    except Exception as error:
        raise BenchmarkSpecificationError(
            f"{spec['name']}: cannot load case module {spec['module']!r}: {error}"
        ) from error

    try:
        tables = select_runtime_tables(spec["data_sources"])
    except ValueError as error:
        raise BenchmarkSpecificationError(f"{spec['name']}: {error}") from error
    table_names = [table.name for table in tables]
    instructions = system_instructions(injection, harness)
    conversations = []
    for conversation in conversations_of(spec):
        # A fresh copy per conversation, so a mutable default cannot carry state
        # from one conversation into the next.
        outputs = deepcopy(module.variables)
        by_name = {variable.name: variable for variable in outputs}
        scopes = _turn_scopes(spec["name"], conversation, outputs)
        registered = set(scopes[0])
        data_variables = (
            runtime_variables(spec["data_sources"]) if injection == "eager"
            else [datastore_variable(tables)]
        )
        # A separate kernel process, not this one: only a preemptible runtime
        # can honour `max_exec_timeout`, and model-written code has no other
        # bound. One case ran a Ho-Lee tree for nearly four hours in-process
        # before it was killed by hand, holding 12 GB and a core the whole
        # time. Isolation also keeps a runaway allocation from taking the
        # study down with it.
        runtime = IPyKernelRuntime(
            functions=[],
            variables=[by_name[name] for name in scopes[0]] + data_variables,
            types=[],
            security_checker=SECURITY_CHECKER,
            # A cell may compute silently as long as it may run; the kernel's
            # default 30 s of silence would interrupt it long before that. The
            # same bound covers reading outputs and injecting variables, so a
            # wedged kernel now takes up to MAX_EXEC_SECONDS to give up on.
            iopub_timeout=MAX_EXEC_SECONDS,
        )
        agent_class = CaveAgent if harness == "code" else ToolAgent
        agent = agent_class(
            model=model,
            runtime=runtime,
            system_instructions=instructions,
            max_steps=total_step_budget,
            max_exec_output=10000,
            max_exec_timeout=MAX_EXEC_SECONDS,
        )
        if harness == "code":
            # CaveAgent only logs why a run failed; this keeps it, as ToolAgent does.
            agent = CodeAgent(agent)
        turns = []
        try:
            for index, (turn, stores) in enumerate(zip(conversation.turns, scopes)):
                # Register this turn's outputs as it begins, so the runtime this
                # turn describes names them and earlier turns' did not.
                for name in stores:
                    if name not in registered:
                        runtime.inject_variable(by_name[name])
                        registered.add(name)
                turns.append(await _evaluate_turn(
                    agent, runtime, module, spec["name"], turn, index, stores,
                    table_names, max_protocol_nudges=max_protocol_nudges,
                    total_step_budget=total_step_budget,
                    accept_unanswered=accept_unanswered,
                    harness=harness,
                ))
        finally:
            # The kernel is a child process and a failed case must not leak one:
            # a long study would accumulate them until the host ran out.
            await runtime.stop()
        conversations.append(ConversationResult(
            id=conversation.id, turns=turns, messages=messages_of(agent) or [],
        ))
    return CaseResult(name=spec["name"], conversations=conversations)


def write_result(
    path: Path,
    result: CaseResult,
    model_config: dict,
    run_metadata: dict | None = None,
) -> None:
    """Persist the trajectory without embedding an authoritative PV verdict."""
    VERDICT_FIELDS = {
        "success", "failure_type", "validation_message", "variables_not_set",
    }

    def without_verdict(record: dict) -> dict:
        clean = {k: v for k, v in record.items() if k not in VERDICT_FIELDS}
        repair = clean.get("protocol_repair")
        if isinstance(repair, dict):
            repair = {k: v for k, v in repair.items() if k not in VERDICT_FIELDS}
            attempts = repair.get("attempts")
            if isinstance(attempts, list):
                repair["attempts"] = [
                    {k: v for k, v in attempt.items() if k not in VERDICT_FIELDS}
                    if isinstance(attempt, dict) else attempt
                    for attempt in attempts
                ]
            clean["protocol_repair"] = repair
        return clean

    payload = {
        "schema": RESULT_SCHEMA,
        "written_at": utc_now(),
        "case": result.name,
        "model": model_config,
        "run": dict(run_metadata or {}),
        # Summed from the turns on every save, so it cannot drift from them.
        "cost": result.cost(),
        "conversations": [
            {
                "id": conversation.id,
                **({"transcript": conversation.transcript}
                   if conversation.transcript else {}),
                "turns": [
                    without_verdict({
                        item.name: getattr(turn, item.name)
                        for item in fields(turn)
                    })
                    for turn in conversation.turns
                ],
            }
            for conversation in result.conversations
        ],
    }
    atomic_write_json(path, payload)
