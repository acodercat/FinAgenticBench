"""The tested agent's model over OpenAI's Responses API.

GPT-6 Luna writes each step of an agent turn as its own output message, in a
``commentary`` phase, and only the closing message in the ``final_answer``
phase. Chat Completions returns the final phase alone, so the agent received
"no data-inspection code was executed" in place of the code the model wrote,
or an empty answer. LiteLLM's Responses bridge returns every phase but joins
the messages with nothing between them, which puts the fence that opens the
next message mid-line (``assign variables.```python``), where cave-agent's
parser -- correctly -- does not take it for a fence.

This model reads the Responses API directly and starts each output message on
a new line; nothing else about the text changes. ``models.toml`` selects it
with ``provider = "openai_responses"``, the switch the judge client reads.

The endpoint does not always stream a message: some replies arrive with no
text deltas, the whole text only in the part's closing
``response.output_text.done`` event. Read as deltas alone, such a reply is
empty, and the run would end as the model's empty answer. A part that
streamed nothing is taken from its ``done``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx
import openai
from cave_agent.models.base import Model, ModelResponse, StreamDelta, StreamResponse, TokenUsage
from cave_agent.models.errors import ProviderError

_REASONING_EVENTS = ("response.reasoning_text.delta", "response.reasoning_summary_text.delta")
# Responses ``incomplete_details.reason`` -> the chat finish reason cave-agent reads.
_INCOMPLETE_REASON = {"max_output_tokens": "length", "content_filter": "content_filter"}


def _separator(previous: str) -> str:
    """What goes before the next output message: a newline unless one ended the last."""
    return "" if not previous or previous.endswith("\n") else "\n"


def _join(messages: Iterable[str]) -> str:
    text = ""
    for message in messages:
        text += _separator(text) + message
    return text


def response_text(response: Any) -> tuple[str, str]:
    """A final response's text (its output messages, each on a new line) and its refusal."""
    items = [item for item in response.output if item.type == "message"]
    text = _join(
        "".join(part.text for part in item.content if part.type == "output_text")
        for item in items
    )
    refusal = "".join(
        part.refusal for item in items for part in item.content if part.type == "refusal"
    )
    return text, refusal


def _part(event: Any) -> tuple[int, int]:
    return (getattr(event, "output_index", 0), getattr(event, "content_index", 0))


def response_usage(response: Any) -> TokenUsage:
    """A Responses API response's token usage in cave-agent's terms."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    prompt = usage.input_tokens or 0
    completion = usage.output_tokens or 0
    return TokenUsage(prompt, completion, usage.total_tokens or prompt + completion)


def response_finish_reason(response: Any) -> str:
    """The Chat Completions finish reason equivalent to a Responses API outcome."""
    if response.status != "incomplete":
        return "stop"
    reason = getattr(response.incomplete_details, "reason", None)
    return _INCOMPLETE_REASON.get(reason, "length")


class ResponsesModel(Model):
    """cave-agent ``Model`` backed by ``client.responses``, stateless per call."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str | None,
        timeout: httpx.Timeout,
        options: dict[str, Any],
    ):
        """*options* are as :meth:`core.models.ModelConfig.responses_options` gives them."""
        self.model_id = model_id
        self.max_output_tokens = options["max_output_tokens"]
        self.client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3,
        )
        self._options = dict(options)

    def request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        # ``store=False``: the agent resends the whole history every step, so
        # nothing is kept server-side between calls.
        return {
            "model": self.model_id,
            "input": [{"role": m["role"], "content": m["content"]} for m in messages],
            "store": False,
            **self._options,
        }

    async def _complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        response = await self.client.responses.create(**self.request(messages))
        content, refusal = response_text(response)
        return ModelResponse(
            content=content or refusal,
            usage=response_usage(response),
            finish_reason=response_finish_reason(response),
            refusal=refusal or None,
        )

    def stream(self, messages: list[dict[str, str]]) -> _ResponsesStream:
        return _ResponsesStream(self, messages)

    async def aclose(self) -> None:
        await self.client.close()


class _ResponsesStream(StreamResponse):
    """Responses event stream as cave-agent deltas, one new line per output message."""

    def __init__(self, model: ResponsesModel, messages: list[dict[str, str]]):
        super().__init__()
        self._model = model
        self._messages = messages
        # The last content delta: its final character decides the next separator.
        self._last = ""
        # The (output, content) parts that sent deltas.
        self._streamed: set[tuple[int, int]] = set()

    async def _open_stream(self):
        # Called once per connection attempt; a retry replays from the first event.
        self._last = ""
        self._streamed = set()
        request = self._model.request(self._messages)
        return await self._model.client.responses.create(**request, stream=True)

    def _process_stream_chunk(self, event: Any) -> StreamDelta | None:
        kind = event.type
        if kind == "response.output_item.added" and event.item.type == "message":
            return self._content(_separator(self._last))
        if kind == "response.output_text.delta":
            if event.delta:
                self._streamed.add(_part(event))
            return self._content(event.delta)
        if kind == "response.output_text.done" and _part(event) not in self._streamed:
            return self._content(event.text)
        if kind in _REASONING_EVENTS and event.delta:
            self.thinking += event.delta
            return StreamDelta(thinking=event.delta)
        if kind == "response.refusal.delta":
            self.refusal += event.delta
        elif kind in ("response.completed", "response.incomplete"):
            self.usage = response_usage(event.response)
            self.finish_reason = response_finish_reason(event.response)
        elif kind == "response.failed":
            raise ProviderError(f"Responses stream failed: {event.response.error}")
        elif kind == "error":
            raise ProviderError(f"Responses stream error: {event.message}")
        return None

    def _content(self, text: str) -> StreamDelta | None:
        if not text:
            return None
        self._last = text
        return StreamDelta(content=text)
