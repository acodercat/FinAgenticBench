"""The tool-calling harness's connection to the tested model.

:mod:`core.tool_agent` runs an agent that executes Python through one native
tool call, ``python``, instead of fenced code in its text. This module is the
only place it meets a provider: :class:`ToolGateway` opens one
:class:`Conversation` per agent, in the wire protocol ``provider`` names in
``models.toml``.

- ``openai_compatible`` -> :class:`ChatConversation`, Chat Completions with
  ``tools``. An assistant message's ``reasoning_content`` is sent back with it:
  DeepSeek in thinking mode requires that inside a tool loop.
- ``openai_responses`` -> :class:`ResponsesConversation`, the Responses API.
  GPT-6 Luna accepts function tools on Chat Completions only at effort
  ``none``, and its reasoning must be returned, so output items are kept
  verbatim (``store=False``, reasoning encrypted).
- ``google_genai`` -> :class:`GeminiConversation`, the Gemini API through the
  google-genai SDK. Gemini 3 requires each function call's thought signature
  back on the next request; an OpenAI-compatible proxy drops it, so the model's
  own content is kept as the SDK returned it.
- ``anthropic`` -> :class:`AnthropicConversation`, the Messages API through the
  anthropic SDK. Claude's thinking blocks go back with the tool call they led
  to, signed, so the content blocks are kept as returned.

Each OpenAI and Anthropic request asks for at most one tool call
(``parallel_tool_calls=False``, ``disable_parallel_tool_use``): a CaveAgent step
runs one code block, and :class:`~core.tool_agent.ToolAgent` holds a step to one
execution as well, for endpoints -- and the Gemini API, which has no such
switch -- that send more.

Requests stream. A reverse proxy in front of an endpoint ends a request that
sends nothing for too long -- non-streaming steps of DeepSeek V4 Flash on
oneapi got ``504 Gateway Time-out`` from its nginx 4 times in 213 cases -- while
a stream keeps bytes moving as the model writes. The client's timeout
(:meth:`~core.models.ModelConfig.request_timeout`) bounds each phase of a
request, so on a stream the read timeout is the longest silence allowed. The
SDK retries a failure to connect. A provider's failure reaches the agent as
cave-agent's typed ``ModelError`` (``raise_model_error``): a prompt past the
window, an endpoint out of credit, else a ``ProviderError``. Sampling options
come from :class:`~core.models.ModelConfig`, as for the CaveAgent harness. The
system prompt is replaced before each run and followed, on the wire only, by
the date reminder CaveAgent sends.
"""

from __future__ import annotations

import contextlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import anthropic
import httpx
import httpx2
import openai
from cave_agent.compaction import default_token_estimate
from cave_agent.models.base import TokenUsage
from cave_agent.models.errors import PromptTooLongError, ProviderError, raise_model_error
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core.responses_model import response_finish_reason, response_text, response_usage

if TYPE_CHECKING:
    from core.models import ModelConfig

PYTHON_TOOL = "python"
# The finish reason of a response whose tool call could not be read (Gemini reports
# this instead of returning the call): nothing ran, and the model is told so.
MALFORMED_CALL = "malformed_call"
# Failures on the provider's side of a request: the SDK wraps those in opening
# one, and a stream that breaks while it is read raises the transport's own
# (httpx2 under the anthropic SDK).
_PROVIDER_FAILURES = (
    openai.APIError, genai_errors.APIError, anthropic.APIError, httpx.HTTPError, httpx2.HTTPError,
)
_CODE = "code"
_DESCRIPTION = (
    "Run Python in the task's persistent runtime and return what it printed. Every call "
    "runs in the same session: variables, imports and loaded tables from earlier calls "
    "remain available."
)
_PARAMETERS = {
    "type": "object",
    "properties": {_CODE: {"type": "string", "description": "The Python code to execute."}},
    "required": [_CODE],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # As the model wrote them, which need not be JSON: malformed, or cut at the cap.
    # (Gemini returns arguments parsed; :class:`_GeminiReply` serialises them to JSON.)
    arguments: str

    @property
    def code(self) -> str | None:
        """The ``code`` argument, or ``None`` when the arguments hold none."""
        value = (_arguments(self.arguments) or {}).get(_CODE)
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class Step:
    """One model response: its text, the calls it asked for, and why it stopped."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    reasoning: str = ""
    # The text is the model's refusal: the answer when it gave no other, as in
    # CaveAgent, even under a "content_filter" finish.
    refused: bool = False


async def _close(stream: Any) -> None:
    """Release a stream; a failure to close never outranks the reply already read."""
    with contextlib.suppress(*_PROVIDER_FAILURES):
        await stream.close()


def _arguments(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _replayable(raw: str) -> str:
    """Arguments as the history replays them: verbatim when they are a JSON object.

    Anything else -- malformed, or cut at the output cap -- goes back as ``{}``.
    Providers reject a history whose tool-call arguments are not valid JSON
    (DeepSeek: "Assistant tool call function.arguments must be valid JSON"),
    which failed every later request of the case; the agent has already told
    the model that the call was not run.
    """
    return raw if _arguments(raw) is not None else "{}"


def _date_reminder_text() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:00")
    return f"<system-reminder>\nCurrent date and time: {now}\n</system-reminder>"


def _date_reminder() -> dict[str, str]:
    return {"role": "user", "content": _date_reminder_text()}


def _chat_usage(usage: Any) -> TokenUsage:
    """A Chat Completions usage block in cave-agent's terms."""
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return TokenUsage(prompt, completion, getattr(usage, "total_tokens", 0) or prompt + completion)


def _message_text(message: dict[str, Any]) -> str:
    """The text a wire message carries: content, reasoning sent back, a tool result,
    call arguments."""
    parts = [
        message.get("content"), message.get("reasoning_content"),
        message.get("output"), message.get("arguments"),
    ]
    parts += [call["function"]["arguments"] for call in message.get("tool_calls", ())]
    return "".join(part for part in parts if isinstance(part, str))


def _estimated_usage(prompt_texts: list[str], step: Step) -> TokenUsage:
    """Cost a step whose provider reported no usage, with CaveAgent's estimator.

    oneapi drops the usage chunk a stream asks for. CaveAgent then estimates
    with cave-agent's ``default_token_estimate`` -- prompt from the messages
    sent, completion from the answer and its reasoning -- and this harness
    does the same, counting a call's arguments as the model wrote them, cut or
    not. What it sends includes the reasoning it returns to the model, which
    CaveAgent never sends, so on a thinking model its prompt count is higher,
    as its requests are. Each protocol gives the texts of its own wire
    (:meth:`Conversation._wire_texts`).
    """
    prompt = sum(default_token_estimate(text) for text in prompt_texts)
    written = [step.text, step.reasoning, *(call.arguments for call in step.tool_calls)]
    completion = sum(default_token_estimate(text) for text in written)
    return TokenUsage(prompt, completion, prompt + completion)


class Conversation(ABC):
    """One agent's history with the model, in one wire protocol."""

    def __init__(self, client: Any, options: dict[str, Any]):
        self._client = client
        self._options = options
        self._system = ""
        self._history: list[Any] = []

    @classmethod
    @abstractmethod
    def client(cls, cfg: ModelConfig) -> Any:
        """The SDK client this protocol talks through, shared by a study's agents."""

    @classmethod
    @abstractmethod
    def options(cls, cfg: ModelConfig) -> dict[str, Any]:
        """The request options this protocol sends for *cfg*."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self._system}, *self._history]

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

    @abstractmethod
    def add_tool_result(self, call_id: str, output: str) -> None:
        """Answer the call *call_id* with the text *output*."""

    async def step(self) -> Step:
        """Send the history, record the model's response in it, and return the response.

        A failure on the provider's side -- refused, unreachable, timed out --
        raises cave-agent's typed ``ModelError``, so the agent never sees the SDK.
        """
        wire = self._wire()
        try:
            step = await self._request(wire)
        except _PROVIDER_FAILURES as error:
            raise_model_error(error)
        if step.usage.total_tokens == 0:
            step = replace(step, usage=_estimated_usage(self._wire_texts(wire), step))
        return step

    @abstractmethod
    async def _request(self, wire: list[Any]) -> Step:
        """Send *wire* in this protocol, record the response, and return it."""

    def _wire(self) -> list[Any]:
        return [{"role": "system", "content": self._system}, _date_reminder(), *self._history]

    def _wire_texts(self, wire: list[Any]) -> list[str]:
        """The text each wire message carries, for estimating unreported usage."""
        return [_message_text(message) for message in wire]


class _OpenAIConversation(Conversation):
    """A protocol spoken through the OpenAI SDK."""

    @classmethod
    def client(cls, cfg: ModelConfig) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout(),
            max_retries=3,
        )


@dataclass
class _StreamedCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


# The finish reasons a Chat Completions reply ends with on the model's account. Any
# other is the provider's: vLLM reports "abort" for a request the server dropped,
# which must be retried as an outage rather than read as an empty answer.
_CHAT_FINISH = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


class _ChatReply:
    """A streamed Chat Completions reply, put back together.

    Tool calls arrive as fragments keyed by ``index`` (absent on some gateways,
    which send one call; read as 0). ``id`` and ``name`` are assigned rather than
    appended, since some gateways repeat them in every fragment, and
    ``arguments`` are joined in order.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._refusal: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, _StreamedCall] = {}
        self._finish_reason: str | None = None
        self._usage = TokenUsage()

    @property
    def finished(self) -> bool:
        return self._finish_reason is not None

    def add(self, chunk: Any) -> None:
        if getattr(chunk, "usage", None):
            self._usage = _chat_usage(chunk.usage)
        for choice in chunk.choices or ():
            self._finish_reason = choice.finish_reason or self._finish_reason
            delta = choice.delta
            if delta is None:
                # Metadata such as an asynchronous content-filter verdict.
                continue
            self._text.append(delta.content or "")
            self._refusal.append(getattr(delta, "refusal", None) or "")
            self._reasoning.append(
                getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
            )
            for fragment in delta.tool_calls or ():
                call = self._calls.setdefault(fragment.index or 0, _StreamedCall())
                call.id = fragment.id or call.id
                if fragment.function:
                    call.name = fragment.function.name or call.name
                    call.arguments.append(fragment.function.arguments or "")

    def finish(self) -> tuple[dict[str, Any] | None, Step]:
        """The assistant message for the history, if any, and the step for the agent.

        A refusal is the answer when there is no other text, as in CaveAgent. A
        reply with neither text nor a call leaves no message: several providers
        reject empty assistant content on the next request.
        """
        if self._finish_reason is None:
            # A stream that stops without saying why was cut, not finished.
            raise ProviderError("The model stream ended without a terminal finish reason")
        if self._finish_reason not in _CHAT_FINISH:
            raise ProviderError(f"The provider ended the response with {self._finish_reason}")
        written, refusal = "".join(self._text), "".join(self._refusal)
        text = written or refusal
        reasoning = "".join(self._reasoning)
        calls, wire = [], []
        for index, streamed in sorted(self._calls.items()):
            call_id, raw = streamed.id or f"call_{index}", "".join(streamed.arguments)
            call = ToolCall(call_id, streamed.name, raw)
            calls.append(call)
            wire.append({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _replayable(call.arguments)},
            })
        step = Step(text, tuple(calls), self._finish_reason, self._usage, reasoning,
                    refused=bool(refusal) and not written)
        if not text.strip() and not wire:
            return None, step
        assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
        if wire:
            assistant["tool_calls"] = wire
        if reasoning:
            assistant["reasoning_content"] = reasoning
        return assistant, step


class ChatConversation(_OpenAIConversation):
    _TOOL: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {"name": PYTHON_TOOL, "description": _DESCRIPTION, "parameters": _PARAMETERS},
    }

    @classmethod
    def options(cls, cfg: ModelConfig) -> dict[str, Any]:
        return {"model": cfg.api_model, **cfg.chat_options()}

    def add_tool_result(self, call_id: str, output: str) -> None:
        self._history.append({"role": "tool", "tool_call_id": call_id, "content": output})

    async def _request(self, wire: list[dict[str, Any]]) -> Step:
        stream = await self._client.chat.completions.create(
            messages=wire,
            tools=[self._TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=True,
            stream_options={"include_usage": True},
            **self._options,
        )
        reply = _ChatReply()
        try:
            async for chunk in stream:
                reply.add(chunk)
        except _PROVIDER_FAILURES:
            # After the finish reason only the usage chunk can be lost; the
            # answer itself is whole.
            if not reply.finished:
                raise
        finally:
            await _close(stream)
        assistant, step = reply.finish()
        if assistant is not None:
            self._history.append(assistant)
        return step


class ResponsesConversation(_OpenAIConversation):
    _TOOL: ClassVar[dict[str, Any]] = {
        "type": "function",
        "name": PYTHON_TOOL,
        "description": _DESCRIPTION,
        "parameters": _PARAMETERS,
    }
    _FINAL_EVENTS: ClassVar[tuple[str, ...]] = (
        "response.completed", "response.incomplete", "response.failed",
    )

    @classmethod
    def options(cls, cfg: ModelConfig) -> dict[str, Any]:
        return {"model": cfg.api_model, **cfg.responses_options()}

    def add_tool_result(self, call_id: str, output: str) -> None:
        self._history.append({"type": "function_call_output", "call_id": call_id, "output": output})

    async def _request(self, wire: list[dict[str, Any]]) -> Step:
        stream = await self._client.responses.create(
            input=wire,
            tools=[self._TOOL],
            parallel_tool_calls=False,
            store=False,
            include=["reasoning.encrypted_content"],
            stream=True,
            **self._options,
        )
        response = None
        try:
            async for event in stream:
                if event.type in self._FINAL_EVENTS:
                    response = event.response
                    break  # nothing after the final response is needed
                if event.type == "error":
                    raise ProviderError(f"Responses stream error: {event.message}")
        finally:
            await _close(stream)
        if response is None:
            raise ProviderError("Responses stream ended without a final response")
        if response.status == "failed":
            raise ProviderError(f"Responses stream failed: {response.error}")
        return self._record(response)

    def _record(self, response: Any) -> Step:
        """Turn the final response into a step, keeping its items for the next request.

        The items go back verbatim -- reasoning items carry state the next call
        needs -- unless the response holds neither text nor a call. Then it
        leaves nothing, as an empty Chat reply does.
        """
        written, refusal = response_text(response)
        text = written or refusal
        calls = tuple(
            ToolCall(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        )
        if text.strip() or calls:
            self._history.extend(_replayable_item(item) for item in response.output)
        finish = response_finish_reason(response)
        return Step(
            text=text,
            tool_calls=calls,
            finish_reason="tool_calls" if finish == "stop" and calls else finish,
            usage=response_usage(response),
            refused=bool(refusal) and not written,
        )


def _replayable_item(item: Any) -> dict[str, Any]:
    dumped = item.model_dump(exclude_none=True)
    if item.type == "function_call":
        dumped["arguments"] = _replayable(item.arguments)
    return dumped


# How a Gemini finish reason reads to the agent. Unlisted reasons (OTHER,
# LANGUAGE, an unspecified or new one) are the provider's, not the model's.
_GEMINI_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "CONTINUATION": "length",
    **dict.fromkeys(("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"),
                    MALFORMED_CALL),
    **dict.fromkeys(("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                     "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"),
                    "content_filter"),
}


class _GeminiReply:
    """A streamed Gemini response, put back together as the API returns it whole.

    A stream splits the answer into many text parts and can put a thought
    signature on an empty text part at the end. The API rejects a history
    holding an empty text part (400 INVALID_ARGUMENT, signature or not), so
    consecutive text parts of one kind are joined and a signature that arrives
    on empty text moves onto the text it closes. Function-call parts, which
    carry their own signature, are kept as they are. Thought parts are
    reasoning; the rest of the text is the answer.
    """

    def __init__(self) -> None:
        self._parts: list[genai_types.Part] = []
        self._finish_reason: str | None = None
        self._usage = TokenUsage()

    @property
    def finished(self) -> bool:
        return self._finish_reason is not None

    def add(self, chunk: genai_types.GenerateContentResponse) -> None:
        metadata = chunk.usage_metadata
        if metadata and metadata.total_token_count:
            completion = (
                (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0)
            )
            self._usage = TokenUsage(
                metadata.prompt_token_count or 0, completion, metadata.total_token_count,
            )
        feedback = chunk.prompt_feedback
        if feedback and feedback.block_reason:
            # The prompt itself was blocked: no candidate will follow.
            self._finish_reason = "SAFETY"
        for candidate in chunk.candidates or ():
            for part in (candidate.content.parts if candidate.content else None) or ():
                self._take(part)
            if candidate.finish_reason:
                self._finish_reason = candidate.finish_reason.name

    def _take(self, part: genai_types.Part) -> None:
        last = self._parts[-1] if self._parts else None
        if part.function_call:
            self._parts.append(part.model_copy())
        elif last is not None and _continues_text(last, part):
            last.text += part.text or ""
            last.thought_signature = part.thought_signature or last.thought_signature
        elif part.text:
            self._parts.append(part.model_copy())
        elif part.thought_signature and last is not None and not last.thought_signature:
            # A lone signature with no text to join closes the previous part. One
            # that already has its own -- a function call's, the only kind the API
            # checks -- keeps it.
            last.thought_signature = part.thought_signature

    def finish(
        self,
    ) -> tuple[genai_types.Content | None, Step, dict[str, genai_types.FunctionCall]]:
        """The model's content for the history (if any), the step for the agent, and
        each call it made by the id the step gives it."""
        if self._finish_reason is None:
            raise ProviderError("The model stream ended without a terminal finish reason")
        finish = _GEMINI_FINISH.get(self._finish_reason)
        if finish is None:
            raise ProviderError(f"Gemini ended the response with {self._finish_reason}")
        text = "".join(part.text for part in self._parts if part.text and not part.thought)
        reasoning = "".join(part.text for part in self._parts if part.text and part.thought)
        function_calls = [part.function_call for part in self._parts if part.function_call]
        # The id the agent answers by; a call the API sent without one is answered
        # with none, so the response matches it.
        calls = {call.id or f"call_{index}": call for index, call in enumerate(function_calls)}
        tool_calls = tuple(
            ToolCall(call_id, call.name or "", json.dumps(call.args or {}))
            for call_id, call in calls.items()
        )
        if finish == "stop" and tool_calls:
            finish = "tool_calls"
        step = Step(text, tool_calls, finish, self._usage, reasoning)
        if not text.strip() and not tool_calls:
            return None, step, calls
        return genai_types.Content(role="model", parts=self._parts), step, calls


def _continues_text(last: genai_types.Part, part: genai_types.Part) -> bool:
    """Whether *part* belongs to the text part *last*: neither is a call, and both
    are thought or both are answer."""
    return (
        last.function_call is None and last.text is not None
        and bool(part.thought) == bool(last.thought)
    )


class _OwnTimeouts(httpx.AsyncClient):
    """An httpx client whose own timeouts hold for every request.

    google-genai passes its ``HttpOptions.timeout`` with each request, ``None``
    when unset, which httpx reads as no limit at all: the client's timeouts
    would never apply, and a stalled stream would hold its step indefinitely.
    """

    def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
        kwargs.pop("timeout", None)
        return super().build_request(*args, **kwargs)


class GeminiConversation(Conversation):
    _TOOL: ClassVar[genai_types.Tool] = genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name=PYTHON_TOOL, description=_DESCRIPTION, parameters_json_schema=_PARAMETERS,
        ),
    ])

    @classmethod
    def client(cls, cfg: ModelConfig) -> genai.Client:
        # Through httpx rather than the SDK's default aiohttp: aiohttp applies the
        # timeout to the whole request, ending any step that streams for longer
        # than it, where httpx bounds each read, as the OpenAI clients do. The SDK
        # sends HttpOptions.timeout as X-Server-Timeout too, so it is left unset.
        return genai.Client(api_key=cfg.api_key, http_options=genai_types.HttpOptions(
            base_url=cfg.base_url,
            httpx_async_client=_OwnTimeouts(timeout=cfg.request_timeout()),
            # As the OpenAI clients' max_retries=3: a request that fails to open is retried.
            retry_options=genai_types.HttpRetryOptions(attempts=4),
        ))

    @classmethod
    def options(cls, cfg: ModelConfig) -> dict[str, Any]:
        return {"model": cfg.api_model, "config": cfg.gemini_options()}

    def __init__(self, client: genai.Client, options: dict[str, Any]):
        super().__init__(client, options)
        # The last response's calls not yet answered, by the id the agent sees.
        self._unanswered: dict[str, genai_types.FunctionCall] = {}
        # The user turn that holds the answers to those calls, once one is given.
        self._answers: genai_types.Content | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        history = [content.model_dump(mode="json", exclude_none=True) for content in self._history]
        return [{"role": "system", "content": self._system}, *history]

    def add_user(self, text: str) -> None:
        self._answers = None
        self._history.append(genai_types.Content(role="user", parts=[genai_types.Part(text=text)]))

    def add_tool_result(self, call_id: str, output: str) -> None:
        call = self._unanswered.pop(call_id)
        answer = genai_types.Part(function_response=genai_types.FunctionResponse(
            id=call.id, name=call.name, response={"output": output},
        ))
        if self._answers is None:
            # The answers to one response's calls go back together, as Gemini expects.
            self._answers = genai_types.Content(role="user", parts=[])
            self._history.append(self._answers)
        self._answers.parts.append(answer)

    async def _request(self, wire: list[Any]) -> Step:
        config = genai_types.GenerateContentConfig(
            system_instruction=self._system, tools=[self._TOOL], **self._options["config"],
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self._options["model"], contents=wire, config=config,
        )
        reply = _GeminiReply()
        try:
            async with contextlib.aclosing(stream):
                async for chunk in stream:
                    reply.add(chunk)
        except _PROVIDER_FAILURES:
            # After the finish reason only the usage metadata can be lost.
            if not reply.finished:
                raise
        content, step, self._unanswered = reply.finish()
        self._answers = None
        if content is not None:
            self._history.append(content)
        return step

    def _wire(self) -> list[Any]:
        reminder = genai_types.Part(text=_date_reminder_text())
        return [genai_types.Content(role="user", parts=[reminder]), *self._history]

    def _wire_texts(self, wire: list[Any]) -> list[str]:
        texts = [self._system]
        for content in wire:
            for part in content.parts or ():
                texts.append(part.text or "")
                if part.function_call:
                    texts.append(json.dumps(part.function_call.args or {}))
                if part.function_response:
                    texts.append(json.dumps(part.function_response.response or {}))
        return texts


# How a Messages API stop reason reads to the agent. A reply the context window
# cut is a prompt past the window, as other providers report it; other unlisted
# reasons (pause_turn, a new one) are the provider's.
_ANTHROPIC_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class AnthropicConversation(Conversation):
    _TOOL: ClassVar[dict[str, Any]] = {
        "name": PYTHON_TOOL, "description": _DESCRIPTION, "input_schema": _PARAMETERS,
    }

    @classmethod
    def client(cls, cfg: ModelConfig) -> anthropic.AsyncAnthropic:
        # The SDK speaks httpx2, whose Timeout is its own type.
        timeout = httpx2.Timeout(**cfg.request_timeout().as_dict())
        return anthropic.AsyncAnthropic(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=timeout, max_retries=3,
        )

    @classmethod
    def options(cls, cfg: ModelConfig) -> dict[str, Any]:
        return {"model": cfg.api_model, **cfg.anthropic_options()}

    def add_tool_result(self, call_id: str, output: str) -> None:
        result = {"type": "tool_result", "tool_use_id": call_id, "content": output}
        last = self._history[-1] if self._history else None
        if last is not None and last["role"] == "user" and isinstance(last["content"], list):
            # The results for one response's calls go back in one user turn.
            last["content"].append(result)
        else:
            self._history.append({"role": "user", "content": [result]})

    async def _request(self, wire: list[dict[str, Any]]) -> Step:
        # Each call's input as the model wrote it, by block index: the SDK's own
        # ``input`` is parsed leniently and keeps a call cut at the cap.
        written: dict[int, str] = {}
        async with self._client.messages.stream(
            messages=wire,
            system=self._system,
            tools=[self._TOOL],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            **self._options,
        ) as stream:
            try:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                        written[event.index] = written.get(event.index, "") + event.delta.partial_json
            except _PROVIDER_FAILURES:
                # After the stop reason only the closing event can be lost.
                if stream.current_message_snapshot.stop_reason is None:
                    raise
            message = stream.current_message_snapshot
        return self._record(message, written)

    def _record(self, message: anthropic.types.Message, written: dict[int, str]) -> Step:
        """Turn the final message into a step, keeping its blocks for the next request.

        The blocks go back as returned -- a thinking block's signature is
        checked -- except a call's input, which goes back as the model wrote it
        when that is a JSON object and as ``{}`` otherwise, as on the other
        protocols. A message with neither text nor a call leaves nothing.
        """
        if message.stop_reason is None:
            raise ProviderError("The model stream ended without a terminal stop reason")
        finish = _ANTHROPIC_FINISH.get(message.stop_reason)
        if message.stop_reason == "model_context_window_exceeded":
            raise PromptTooLongError("Claude's reply reached the end of its context window")
        if finish is None:
            raise ProviderError(f"Claude ended the response with {message.stop_reason}")
        blocks, calls = [], []
        for index, block in enumerate(message.content):
            dumped = block.model_dump(mode="json", exclude_none=True)
            if block.type == "tool_use":
                raw = written.get(index) or json.dumps(block.input)
                calls.append(ToolCall(block.id, block.name, raw))
                dumped["input"] = _arguments(raw) or {}
            blocks.append(dumped)
        text = "".join(block.text for block in message.content if block.type == "text")
        reasoning = "".join(block.thinking for block in message.content if block.type == "thinking")
        if text.strip() or calls:
            self._history.append({"role": "assistant", "content": blocks})
        usage = message.usage
        prompt = (
            usage.input_tokens + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0)
        )
        return Step(
            text=text,
            tool_calls=tuple(calls),
            finish_reason="tool_calls" if finish == "stop" and calls else finish,
            usage=TokenUsage(prompt, usage.output_tokens, prompt + usage.output_tokens),
            reasoning=reasoning,
        )

    def _wire(self) -> list[Any]:
        """The messages to send, with a prompt-cache breakpoint on the last block.

        A breakpoint caches everything before it -- tools, system prompt,
        history -- so each step reads the previous step's prefix from the cache
        and pays full price only for what is new. The external platform drops
        the request-level ``cache_control``, so the breakpoint is set on the
        block itself, in a copy: history keeps no marks. The system prompt is a
        request parameter of its own, not a message, and the API always
        reports usage, so the estimate's wire texts are never read.
        """
        *earlier, last = [_date_reminder(), *self._history]
        content = last["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        marked = [*blocks[:-1], {**blocks[-1], "cache_control": {"type": "ephemeral"}}]
        return [*earlier, {**last, "content": marked}]


class ToolGateway:
    """One client per study; each agent opens its own :meth:`conversation`."""

    _PROTOCOLS: ClassVar[dict[str, type[Conversation]]] = {
        "openai_compatible": ChatConversation,
        "openai_responses": ResponsesConversation,
        "google_genai": GeminiConversation,
        "anthropic": AnthropicConversation,
    }

    def __init__(self, cfg: ModelConfig):
        self._kind = self._PROTOCOLS[cfg.provider]
        self._options = self._kind.options(cfg)
        self._client = self._kind.client(cfg)

    def conversation(self) -> Conversation:
        return self._kind(self._client, self._options)
