"""The tool gateway asks each provider for the python tool, reassembles the
streamed reply, and keeps in history whatever the next request needs."""

import asyncio
import inspect
import json
from types import SimpleNamespace as NS

import anthropic
import httpx
import httpx2
import openai
import pytest
from cave_agent.models.errors import PromptTooLongError, ProviderBillingError, ProviderError
from google.genai import errors as genai_errors
from anthropic import types as anthropic_types
from google.genai import types as genai_types

from core import tool_gateway
from core.models import ModelConfig, load_model
from core.tool_gateway import PYTHON_TOOL, GeminiConversation, Step, ToolCall, ToolGateway


def _config(**overrides):
    values = dict(name="m", api_model="model-id", api_key="k", base_url="http://localhost:1/v1",
                  temperature=0.1, max_tokens=1000)
    values.update(overrides)
    return ModelConfig(**values)


class _Stream:
    """An SDK stream: async iteration over *items*, then *error* if given."""

    def __init__(self, items, error=None):
        self._items = iter(items)
        self._error = error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            if self._error:
                raise self._error from None
            raise StopAsyncIteration from None

    async def close(self):
        self.closed = True


def _gateway(monkeypatch, cfg, reply):
    """A gateway whose client records each request and answers with *reply*:
    a list of stream items, a :class:`_Stream`, or an exception to raise."""
    sent = []

    async def create(**params):
        sent.append(params)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, _Stream) else _Stream(reply)

    def client(**settings):
        return NS(chat=NS(completions=NS(create=create)), responses=NS(create=create))

    monkeypatch.setattr(tool_gateway.openai, "AsyncOpenAI", client)
    return ToolGateway(cfg), sent


def _chunk(content=None, reasoning=None, calls=(), finish=None, usage=None):
    fragments = [
        NS(index=index, id=call_id, function=NS(name=name, arguments=arguments))
        for index, call_id, name, arguments in calls
    ]
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=fragments or None)
    choices = [] if usage else [NS(delta=delta, finish_reason=finish)]
    return NS(choices=choices, usage=usage)


_USAGE = NS(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_chat_request_streams_and_offers_the_tool_with_the_configured_switches(monkeypatch):
    cfg = _config(
        reasoning=True, reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}},
    )
    gateway, sent = _gateway(monkeypatch, cfg, [_chunk("hi", finish="stop")])
    conversation = gateway.conversation()
    conversation.set_system("system prompt")
    conversation.add_user("question")
    asyncio.run(conversation.step())
    request = sent[0]
    assert request["stream"] is True and request["stream_options"] == {"include_usage": True}
    assert request["model"] == "model-id" and request["temperature"] == 0.1
    assert request["max_completion_tokens"] == 1000 and "max_tokens" not in request
    assert request["extra_body"] == {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    assert request["tools"][0]["function"]["name"] == PYTHON_TOOL
    assert request["parallel_tool_calls"] is False
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system", "user", "user"]
    assert "Current date and time" in request["messages"][1]["content"]


def test_a_plain_model_gets_max_tokens(monkeypatch):
    gateway, sent = _gateway(monkeypatch, _config(), [_chunk("hi", finish="stop")])
    asyncio.run(gateway.conversation().step())
    assert sent[0]["max_tokens"] == 1000 and "max_completion_tokens" not in sent[0]


def test_streamed_tool_call_fragments_are_reassembled_and_kept_with_reasoning(monkeypatch):
    stream = _Stream([
        _chunk(reasoning="thi"),
        _chunk(reasoning="nk", calls=[(0, "c1", "python", '{"co')]),
        # Some gateways repeat the id and name in every fragment.
        _chunk(calls=[(0, "c1", "python", 'de": "x = 1"}')]),
        _chunk(finish="tool_calls"),
        _chunk(usage=_USAGE),
    ])
    gateway, _ = _gateway(monkeypatch, _config(), stream)
    conversation = gateway.conversation()
    step = asyncio.run(conversation.step())
    assert [(call.id, call.name, call.code) for call in step.tool_calls] == [
        ("c1", "python", "x = 1"),
    ]
    assert step.finish_reason == "tool_calls" and step.usage.total_tokens == 15
    assert stream.closed
    conversation.add_tool_result("c1", "ok")
    assistant, tool = conversation.messages[-2:]
    assert assistant["reasoning_content"] == "think"
    assert assistant["tool_calls"] == [{
        "id": "c1", "type": "function",
        "function": {"name": "python", "arguments": '{"code": "x = 1"}'},
    }]
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_arguments_that_are_not_json_are_kept_as_a_call_without_code(monkeypatch):
    gateway, _ = _gateway(monkeypatch, _config(), [
        _chunk(calls=[(0, "c1", "python", "{not json")], finish="tool_calls"),
    ])
    (call,) = asyncio.run(gateway.conversation().step()).tool_calls
    assert call.arguments == "{not json" and call.code is None


def _response(items, status="completed", reason=None):
    return NS(output=items, status=status, incomplete_details=NS(reason=reason) if reason else None,
              usage=NS(input_tokens=10, output_tokens=5, total_tokens=15), error=None)


class _Dumpable(NS):
    def model_dump(self, exclude_none=False):
        return dict(vars(self))


def test_responses_request_streams_and_items_round_trip(monkeypatch):
    cfg = _config(
        provider="openai_responses", reasoning=True, reasoning_effort="medium", temperature=1.0,
    )
    items = [
        _Dumpable(type="reasoning", encrypted_content="opaque"),
        _Dumpable(type="message", content=[NS(type="output_text", text="Checking.")]),
        _Dumpable(
            type="function_call", call_id="fc1", name="python", arguments='{"code": "print(1)"}',
        ),
    ]
    events = [
        NS(type="response.output_text.delta", delta="Checking."),
        NS(type="response.completed", response=_response(items)),
    ]
    gateway, sent = _gateway(monkeypatch, cfg, events)
    conversation = gateway.conversation()
    conversation.set_system("system prompt")
    step = asyncio.run(conversation.step())
    request = sent[0]
    assert request["stream"] is True
    assert request["reasoning"] == {"effort": "medium"} and request["max_output_tokens"] == 1000
    assert request["store"] is False and request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"][0]["name"] == PYTHON_TOOL
    assert request["parallel_tool_calls"] is False
    assert step.text == "Checking." and step.tool_calls[0].code == "print(1)"
    assert step.finish_reason == "tool_calls"
    conversation.add_tool_result("fc1", "1")
    history = conversation.messages[1:]
    assert history[0] == {"type": "reasoning", "encrypted_content": "opaque"}
    assert history[-1] == {"type": "function_call_output", "call_id": "fc1", "output": "1"}


def test_responses_cut_at_the_cap_reports_length(monkeypatch):
    cfg = _config(provider="openai_responses")
    final = _response([], status="incomplete", reason="max_output_tokens")
    gateway, _ = _gateway(monkeypatch, cfg, [NS(type="response.incomplete", response=final)])
    assert asyncio.run(gateway.conversation().step()).finish_reason == "length"


def test_a_responses_stream_without_a_final_response_is_a_provider_error(monkeypatch):
    cfg = _config(provider="openai_responses")
    gateway, _ = _gateway(monkeypatch, cfg, [NS(type="response.output_text.delta", delta="x")])
    with pytest.raises(ProviderError):
        asyncio.run(gateway.conversation().step())


def test_a_provider_failure_reaches_the_agent_as_a_provider_error(monkeypatch):
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    failures = [
        openai.APIConnectionError(request=request),
        _Stream([_chunk("partial")], error=httpx.ReadTimeout("silent", request=request)),
    ]
    for failure in failures:
        gateway, _ = _gateway(monkeypatch, _config(), failure)
        with pytest.raises(ProviderError):
            asyncio.run(gateway.conversation().step())


def _status_error(error_class, status, message):
    """An OpenAI SDK error for a request the endpoint answered with *status* and *message*."""
    response = httpx.Response(status, request=httpx.Request("POST", "http://localhost:1/v1"))
    return error_class(message, response=response, body=None)


@pytest.mark.parametrize("provider", ["openai_compatible", "openai_responses"])
def test_a_prompt_past_the_window_or_an_endpoint_out_of_credit_is_typed_as_such(
        monkeypatch, provider):
    failures = [
        (_status_error(openai.BadRequestError, 400,
                       "This model's maximum context length is 128000 tokens"), PromptTooLongError),
        (_status_error(openai.RateLimitError, 429, "You exceeded your current quota"),
         ProviderBillingError),
    ]
    for failure, typed in failures:
        gateway, _ = _gateway(monkeypatch, _config(provider=provider), failure)
        with pytest.raises(typed):
            asyncio.run(gateway.conversation().step())


def test_a_stream_cut_before_its_finish_reason_is_a_provider_error(monkeypatch):
    """A half-written call must not run as if the model had finished it."""
    gateway, _ = _gateway(monkeypatch, _config(), [_chunk(calls=[(0, "c1", "python", '{"co')])])
    with pytest.raises(ProviderError):
        asyncio.run(gateway.conversation().step())


def test_fragments_without_an_index_form_one_call(monkeypatch):
    gateway, _ = _gateway(monkeypatch, _config(), [
        _chunk(calls=[(None, "c1", "python", '{"code": ')]),
        _chunk(calls=[(None, None, None, '"x = 1"}')], finish="tool_calls"),
    ])
    (call,) = asyncio.run(gateway.conversation().step()).tool_calls
    assert (call.id, call.code) == ("c1", "x = 1")


def test_usage_the_provider_withholds_is_estimated_as_cave_agent_does(monkeypatch):
    unreported, _ = _gateway(monkeypatch, _config(), [_chunk("An answer.", finish="stop")])
    conversation = unreported.conversation()
    conversation.set_system("A system prompt of some length.")
    usage = asyncio.run(conversation.step()).usage
    assert usage.prompt_tokens > 0 and usage.completion_tokens > 0

    reported, _ = _gateway(monkeypatch, _config(), [
        _chunk("An answer.", finish="stop"), _chunk(usage=_USAGE),
    ])
    assert asyncio.run(reported.conversation().step()).usage.total_tokens == 15


def test_the_estimate_counts_the_reasoning_sent_back():
    """The history returns the model's reasoning with its message, so the prompt carries it."""
    answer = {"role": "assistant", "content": "An answer."}
    reasoned = {**answer, "reasoning_content": "A line of reasoning. " * 50}
    step = Step("", (), "stop")
    without, with_reasoning = (
        tool_gateway._estimated_usage([tool_gateway._message_text(message)], step).prompt_tokens
        for message in (answer, reasoned)
    )
    assert with_reasoning > without


def test_the_estimate_counts_a_cut_calls_arguments():
    """A call cut at the cap holds no code, but the model wrote every token of it."""
    cut = ToolCall("c1", "python", '{"code": "' + "x = 1\n" * 200)
    assert cut.code is None
    assert tool_gateway._estimated_usage([], Step("", (cut,), "length")).completion_tokens > 0


def test_a_chunk_without_a_delta_is_skipped(monkeypatch):
    metadata = NS(choices=[NS(delta=None, finish_reason=None)], usage=None)
    gateway, _ = _gateway(monkeypatch, _config(), [metadata, _chunk("done", finish="stop")])
    assert asyncio.run(gateway.conversation().step()).text == "done"


def test_a_refusal_is_the_answer_when_there_is_no_other_text(monkeypatch):
    refusal = NS(choices=[NS(delta=NS(content=None, refusal="I can't help with that.",
                                       tool_calls=None), finish_reason="stop")], usage=None)
    gateway, _ = _gateway(monkeypatch, _config(), [refusal])
    step = asyncio.run(gateway.conversation().step())
    assert (step.text, step.refused) == ("I can't help with that.", True)
    gateway, _ = _gateway(monkeypatch, _config(), [_chunk("Partial", finish="content_filter")])
    assert not asyncio.run(gateway.conversation().step()).refused


def test_usage_without_a_total_is_summed_and_a_chunk_without_choices_is_read(monkeypatch):
    usage = NS(prompt_tokens=3000, completion_tokens=2000, total_tokens=None)
    gateway, _ = _gateway(monkeypatch, _config(), [
        _chunk("done", finish="stop"), NS(choices=None, usage=usage),
    ])
    assert asyncio.run(gateway.conversation().step()).usage.total_tokens == 5000


def test_a_break_after_the_finish_reason_keeps_the_reply(monkeypatch):
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    lost = httpx.ReadError("lost", request=request)
    stream = _Stream([_chunk("done", finish="stop")], error=lost)
    gateway, _ = _gateway(monkeypatch, _config(), stream)
    assert asyncio.run(gateway.conversation().step()).text == "done"


def test_history_holds_no_empty_assistant_content(monkeypatch):
    gateway, _ = _gateway(monkeypatch, _config(), _Stream([
        _chunk(calls=[(0, "c1", "python", '{"code": "x = 1"}')], finish="tool_calls"),
    ]))
    conversation = gateway.conversation()
    asyncio.run(conversation.step())
    assert conversation.messages[-1]["content"] is None

    empty, _ = _gateway(monkeypatch, _config(), [_chunk(finish="stop")])
    conversation = empty.conversation()
    asyncio.run(conversation.step())
    assert [m["role"] for m in conversation.messages] == ["system"]


def test_an_empty_responses_answer_leaves_no_items_and_a_refusal_is_read(monkeypatch):
    cfg = _config(provider="openai_responses")
    empty = _response([_Dumpable(type="reasoning", encrypted_content="x"),
                       _Dumpable(type="message", content=[])])
    gateway, _ = _gateway(monkeypatch, cfg, [NS(type="response.completed", response=empty)])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).text == ""
    assert [m["role"] for m in conversation.messages] == ["system"]

    refused = _response([_Dumpable(type="message", content=[NS(type="refusal", refusal="No.")])])
    gateway, _ = _gateway(monkeypatch, cfg, [NS(type="response.completed", response=refused)])
    step = asyncio.run(gateway.conversation().step())
    assert (step.text, step.refused) == ("No.", True)


def test_responses_messages_each_start_on_a_new_line_as_the_code_harness_reads_them(
        monkeypatch):
    """A message that already ended with one gets no blank line after it."""
    cfg = _config(provider="openai_responses")
    messages = [_Dumpable(type="message", content=[NS(type="output_text", text=text)])
                for text in ("Plan.\n", "Checking.", "Done.")]
    gateway, _ = _gateway(monkeypatch, cfg, [
        NS(type="response.completed", response=_response(messages)),
    ])
    assert asyncio.run(gateway.conversation().step()).text == "Plan.\nChecking.\nDone."


def test_whitespace_alone_leaves_no_history(monkeypatch):
    gateway, _ = _gateway(monkeypatch, _config(), [_chunk("   ", finish="stop")])
    conversation = gateway.conversation()
    asyncio.run(conversation.step())
    assert [m["role"] for m in conversation.messages] == ["system"]


def test_events_after_the_final_response_are_not_read(monkeypatch):
    cfg = _config(provider="openai_responses")
    done = _response([_Dumpable(type="message", content=[NS(type="output_text", text="Done.")])])
    gateway, _ = _gateway(monkeypatch, cfg, [
        NS(type="response.completed", response=done), NS(type="error", message="late"),
    ])
    assert asyncio.run(gateway.conversation().step()).text == "Done."


def test_arguments_that_are_not_json_are_replayed_as_an_empty_object(monkeypatch):
    """A provider rejects a history whose call arguments are not JSON."""
    gateway, _ = _gateway(monkeypatch, _config(), [
        _chunk(calls=[(0, "c1", "python", "{not json")], finish="tool_calls"),
    ])
    conversation = gateway.conversation()
    asyncio.run(conversation.step())
    assert conversation.messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"

    cfg = _config(provider="openai_responses")
    item = _Dumpable(type="function_call", call_id="fc1", name="python", arguments='{"code": "x')
    final = NS(type="response.completed", response=_response([item]))
    gateway, _ = _gateway(monkeypatch, cfg, [final])
    conversation = gateway.conversation()
    asyncio.run(conversation.step())
    assert conversation.messages[-1]["arguments"] == "{}"


# --- Gemini API through google-genai --------------------------------------------------------


def _gemini_chunk(*parts, finish=None, usage=None):
    candidate = genai_types.Candidate(
        content=genai_types.Content(role="model", parts=list(parts)) if parts else None,
        finish_reason=finish,
    )
    return genai_types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)


def _gemini_gateway(monkeypatch, replies, **overrides):
    """A Gemini gateway whose client records each request and streams *replies* in turn."""
    sent = []
    replies = list(replies)

    async def generate_content_stream(**params):
        sent.append(params)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        async def stream():
            for chunk in reply:
                yield chunk

        return stream()

    def client(**settings):
        return NS(aio=NS(models=NS(generate_content_stream=generate_content_stream)))

    monkeypatch.setattr(tool_gateway.genai, "Client", client)
    cfg = _config(provider="google_genai", api_model="gemini-3.8-flash", temperature=1.0,
                  reasoning_effort="medium", **overrides)
    return ToolGateway(cfg), sent


def _call_part(code, call_id="fc1", signature=b"signature"):
    call = genai_types.FunctionCall(id=call_id, name="python", args={"code": code})
    return genai_types.Part(function_call=call, thought_signature=signature)


_GEMINI_USAGE = genai_types.GenerateContentResponseUsageMetadata(
    prompt_token_count=10, candidates_token_count=5, thoughts_token_count=20, total_token_count=35,
)


def test_gemini_request_carries_the_tool_thinking_level_and_temperature(monkeypatch):
    gateway, sent = _gemini_gateway(monkeypatch, [[
        _gemini_chunk(genai_types.Part(text="hi"), finish="STOP", usage=_GEMINI_USAGE),
    ]])
    conversation = gateway.conversation()
    conversation.set_system("system prompt")
    conversation.add_user("question")
    step = asyncio.run(conversation.step())
    request = sent[0]
    config = request["config"]
    assert request["model"] == "gemini-3.8-flash"
    assert config.system_instruction == "system prompt"
    assert config.temperature == 1.0 and config.max_output_tokens == 1000
    assert config.thinking_config.thinking_level == "MEDIUM"
    assert config.tools[0].function_declarations[0].name == PYTHON_TOOL
    texts = [content.parts[0].text for content in request["contents"]]
    assert "Current date and time" in texts[0] and texts[1] == "question"
    assert step.text == "hi" and step.finish_reason == "stop"
    assert (step.usage.prompt_tokens, step.usage.completion_tokens) == (10, 25)


def test_gemini_call_goes_back_with_its_thought_signature_and_answer(monkeypatch):
    thought = genai_types.Part(text="Plan the query.", thought=True)
    gateway, sent = _gemini_gateway(monkeypatch, [
        [_gemini_chunk(thought), _gemini_chunk(_call_part("x = 1"), finish="STOP")],
        [_gemini_chunk(genai_types.Part(text="done"), finish="STOP")],
    ])
    conversation = gateway.conversation()
    conversation.add_user("q")
    step = asyncio.run(conversation.step())
    assert [(call.id, call.code) for call in step.tool_calls] == [("fc1", "x = 1")]
    assert step.finish_reason == "tool_calls"
    assert step.reasoning == "Plan the query." and step.text == ""
    conversation.add_tool_result("fc1", "ok")
    asyncio.run(conversation.step())
    model_turn, answer = sent[1]["contents"][-2:]
    assert model_turn.parts[-1].thought_signature == b"signature"
    response = answer.parts[0].function_response
    assert (response.id, response.name, response.response) == ("fc1", "python", {"output": "ok"})
    json.dumps(conversation.messages)


def test_gemini_answers_to_one_responses_calls_go_back_together(monkeypatch):
    unnamed = genai_types.Part(function_call=genai_types.FunctionCall(name="python", args={}))
    gateway, _ = _gemini_gateway(monkeypatch, [
        [_gemini_chunk(_call_part("a = 1"), unnamed, finish="STOP")],
    ])
    conversation = gateway.conversation()
    step = asyncio.run(conversation.step())
    first, second = step.tool_calls
    assert second.id == "call_1" and second.code is None
    conversation.add_tool_result(first.id, "ok")
    conversation.add_tool_result(second.id, "not run")
    answers = conversation._history[-1].parts
    assert [part.function_response.id for part in answers] == ["fc1", None]


def test_gemini_finish_reasons_and_an_empty_reply(monkeypatch):
    gateway, _ = _gemini_gateway(monkeypatch, [
        [_gemini_chunk(genai_types.Part(text="cut"), finish="MAX_TOKENS")],
        [_gemini_chunk(finish="SAFETY")],
        [_gemini_chunk(genai_types.Part(text="  "), finish="STOP")],
        [_gemini_chunk(genai_types.Part(text="no end"))],
    ])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).finish_reason == "length"
    assert asyncio.run(conversation.step()).finish_reason == "content_filter"
    history = len(conversation.messages)
    asyncio.run(conversation.step())
    assert len(conversation.messages) == history
    with pytest.raises(ProviderError):
        asyncio.run(conversation.step())


def test_a_gemini_api_failure_reaches_the_agent_as_a_provider_error(monkeypatch):
    failure = genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
    gateway, _ = _gemini_gateway(monkeypatch, [failure])
    with pytest.raises(ProviderError):
        asyncio.run(gateway.conversation().step())


def test_a_gemini_prompt_past_the_window_is_typed_as_such(monkeypatch):
    failure = genai_errors.ClientError(400, {"error": {
        "message": "The input token count (1200000) exceeds the maximum number of tokens allowed",
    }})
    gateway, _ = _gemini_gateway(monkeypatch, [failure])
    with pytest.raises(PromptTooLongError):
        asyncio.run(gateway.conversation().step())


@pytest.mark.parametrize("provider", ["google_genai", "anthropic"])
def test_only_the_tool_harness_speaks_the_native_apis(tmp_path, provider):
    path = tmp_path / "models.toml"
    path.write_text(f'[g]\napi_model = "native"\napi_key = "k"\nprovider = "{provider}"\n')
    with pytest.raises(ValueError, match="tool harness"):
        load_model(path, "g")


def test_gemini_stream_text_is_joined_and_a_closing_signature_kept_on_it(monkeypatch):
    """The API rejects an empty text part, signed or not, so the signature moves."""
    gateway, sent = _gemini_gateway(monkeypatch, [
        [
            _gemini_chunk(genai_types.Part(text="The answer ")),
            _gemini_chunk(genai_types.Part(text="is 391.")),
            _gemini_chunk(genai_types.Part(text="", thought_signature=b"closing"), finish="STOP"),
        ],
        [_gemini_chunk(genai_types.Part(text="next"), finish="STOP")],
    ])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).text == "The answer is 391."
    conversation.add_user("next question")
    asyncio.run(conversation.step())
    model_turn = sent[1]["contents"][-2]
    assert [(p.text, p.thought_signature) for p in model_turn.parts] == [
        ("The answer is 391.", b"closing"),
    ]


def test_gemini_call_keeps_its_own_signature_and_empty_parts_are_dropped(monkeypatch):
    gateway, _ = _gemini_gateway(monkeypatch, [[
        _gemini_chunk(_call_part("x = 1")),
        _gemini_chunk(genai_types.Part(text=""),
                      genai_types.Part(text="", thought_signature=b"stray"), finish="STOP"),
    ]])
    conversation = gateway.conversation()
    asyncio.run(conversation.step())
    (part,) = conversation._history[-1].parts
    assert part.function_call.id == "fc1" and part.thought_signature == b"signature"


def test_gemini_unreadable_call_and_provider_finish_reasons(monkeypatch):
    gateway, _ = _gemini_gateway(monkeypatch, [
        [_gemini_chunk(finish="MALFORMED_FUNCTION_CALL")],
        [_gemini_chunk(genai_types.Part(text="more"), finish="CONTINUATION")],
        [_gemini_chunk(finish="OTHER")],
        [genai_types.GenerateContentResponse(
            prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
                block_reason="SAFETY",
            ),
        )],
    ])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).finish_reason == tool_gateway.MALFORMED_CALL
    assert asyncio.run(conversation.step()).finish_reason == "length"
    with pytest.raises(ProviderError):
        asyncio.run(conversation.step())
    assert asyncio.run(conversation.step()).finish_reason == "content_filter"


def test_gemini_break_after_the_finish_keeps_the_reply(monkeypatch):
    finished = _gemini_chunk(genai_types.Part(text="done"), finish="STOP")

    class Broken:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield finished
            raise httpx.ReadError("lost")

        async def aclose(self):
            self.closed = True

    broken = Broken()

    async def generate_content_stream(**params):
        return broken

    monkeypatch.setattr(tool_gateway.genai, "Client", lambda **settings: NS(
        aio=NS(models=NS(generate_content_stream=generate_content_stream))))
    gateway = ToolGateway(_config(provider="google_genai", reasoning_effort="medium"))
    assert asyncio.run(gateway.conversation().step()).text == "done"
    assert broken.closed


def test_gemini_client_bounds_each_phase_of_every_request_through_httpx():
    client = GeminiConversation.client(_config(provider="google_genai", timeout=120.0))
    options = client._api_client._http_options
    assert isinstance(options.httpx_async_client, httpx.AsyncClient)
    assert options.timeout is None and options.retry_options.attempts == 4
    # The SDK sends its own unset timeout with each request, which httpx would
    # read as no limit at all.
    request = options.httpx_async_client.build_request("POST", "http://localhost:1", timeout=None)
    assert request.extensions["timeout"] == {
        "connect": 30.0, "read": 120.0, "write": 120.0, "pool": 120.0,
    }


def test_openai_client_connects_within_30_seconds_and_bounds_each_read():
    client = tool_gateway.ChatConversation.client(_config(timeout=120.0))
    assert (client.timeout.connect, client.timeout.read) == (30.0, 120.0)
    assert _config(timeout=10.0).request_timeout().connect == 10.0


class _MessageStream:
    """The anthropic SDK's message stream: *events*, then *error* if given, over a
    final message snapshot."""

    def __init__(self, events, message, error=None):
        self._events, self._error = events, error
        self.current_message_snapshot = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self._events:
            yield event
        if self._error:
            raise self._error


def _anthropic_gateway(monkeypatch, replies, **overrides):
    """An Anthropic gateway whose client records each request and streams *replies*
    (``(events, message)`` or ``(events, message, error)``) in turn."""
    sent = []
    replies = list(replies)

    def stream(**params):
        # Bound against the SDK's own signature, so an argument it lacks fails here.
        inspect.signature(anthropic.resources.messages.AsyncMessages.stream).bind(None, **params)
        sent.append(params)
        return _MessageStream(*replies.pop(0))

    monkeypatch.setattr(tool_gateway.anthropic, "AsyncAnthropic",
                        lambda **settings: NS(messages=NS(stream=stream)))
    cfg = _config(provider="anthropic", api_model="claude-sonnet-5", temperature=1.0, **overrides)
    return ToolGateway(cfg), sent


def _message(*blocks, stop="end_turn"):
    return anthropic_types.Message(
        id="msg", type="message", role="assistant", model="claude-sonnet-5", content=list(blocks),
        stop_reason=stop, stop_sequence=None,
        usage=anthropic_types.Usage(input_tokens=10, output_tokens=7, cache_read_input_tokens=5),
    )


def _thinking():
    return anthropic_types.ThinkingBlock(type="thinking", thinking="Plan.", signature="sig")


def _tool_use(code, call_id="toolu_1"):
    return anthropic_types.ToolUseBlock(type="tool_use", id=call_id, name="python", input={"code": code})


def _written(index, *fragments):
    """The stream events that carry a call's input as the model wrote it."""
    return [NS(type="content_block_delta", index=index,
               delta=NS(type="input_json_delta", partial_json=fragment)) for fragment in fragments]


def test_anthropic_request_carries_the_tool_system_and_switches(monkeypatch):
    gateway, sent = _anthropic_gateway(
        monkeypatch, [([], _message(anthropic_types.TextBlock(type="text", text="hi")))],
        reasoning_effort="high", extra_body={"thinking": {"type": "adaptive"}},
    )
    conversation = gateway.conversation()
    conversation.set_system("system prompt")
    conversation.add_user("question")
    step = asyncio.run(conversation.step())
    request = sent[0]
    assert request["model"] == "claude-sonnet-5" and request["system"] == "system prompt"
    assert request["max_tokens"] == 1000
    assert request["extra_body"] == {
        "thinking": {"type": "adaptive"}, "temperature": 1.0, "output_config": {"effort": "high"},
    }
    assert request["tools"][0]["name"] == PYTHON_TOOL
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    reminder, question = request["messages"]
    assert "Current date and time" in reminder["content"]
    assert question["content"] == [
        {"type": "text", "text": "question", "cache_control": {"type": "ephemeral"}}]
    assert step.text == "hi" and step.finish_reason == "stop"
    assert (step.usage.prompt_tokens, step.usage.completion_tokens) == (15, 7)


def test_anthropic_call_goes_back_with_its_signed_thinking_and_answer(monkeypatch):
    text = anthropic_types.TextBlock(type="text", text="done")
    gateway, sent = _anthropic_gateway(monkeypatch, [
        (_written(1, '{"code": ', '"x = 1"}'), _message(_thinking(), _tool_use("x = 1"), stop="tool_use")),
        ([], _message(text)),
    ])
    conversation = gateway.conversation()
    conversation.add_user("q")
    step = asyncio.run(conversation.step())
    assert [(call.id, call.arguments, call.code) for call in step.tool_calls] == [
        ("toolu_1", '{"code": "x = 1"}', "x = 1")]
    assert step.finish_reason == "tool_calls" and step.reasoning == "Plan."
    conversation.add_tool_result("toolu_1", "ok")
    asyncio.run(conversation.step())
    assistant, answer = sent[1]["messages"][-2:]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "Plan.", "signature": "sig"}
    assert assistant["content"][1]["input"] == {"code": "x = 1"}
    # Only the last block of a request carries the cache breakpoint; history keeps none.
    assert answer == {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "toolu_1", "content": "ok",
        "cache_control": {"type": "ephemeral"}}]}
    assert "cache_control" not in json.dumps(conversation.messages)


def test_anthropic_answers_to_one_responses_calls_go_back_together(monkeypatch):
    gateway, _ = _anthropic_gateway(monkeypatch, [
        ([], _message(_tool_use("a = 1"), _tool_use("b = 2", "toolu_2"), stop="tool_use")),
    ])
    conversation = gateway.conversation()
    conversation.add_user("q")
    first, second = asyncio.run(conversation.step()).tool_calls
    conversation.add_tool_result(first.id, "ok")
    conversation.add_tool_result(second.id, "not run")
    conversation.add_user("next")
    answers, question = conversation.messages[-2:]
    assert [block["tool_use_id"] for block in answers["content"]] == ["toolu_1", "toolu_2"]
    assert question == {"role": "user", "content": "next"}


def test_anthropic_call_cut_at_the_cap_keeps_what_was_written(monkeypatch):
    # The SDK parses a cut input leniently; the call's own text says it was cut.
    gateway, _ = _anthropic_gateway(monkeypatch, [
        (_written(0, '{"code": "x = ', ), _message(_tool_use("x = "), stop="max_tokens")),
    ])
    conversation = gateway.conversation()
    step = asyncio.run(conversation.step())
    (call,) = step.tool_calls
    assert step.finish_reason == "length"
    assert call.arguments == '{"code": "x = ' and call.code is None
    assert conversation.messages[-1]["content"][0]["input"] == {}


def test_anthropic_stop_reasons_and_an_empty_reply(monkeypatch):
    text = anthropic_types.TextBlock(type="text", text="cut")
    gateway, _ = _anthropic_gateway(monkeypatch, [
        ([], _message(text, stop="max_tokens")),
        ([], _message(stop="refusal")),
        ([], _message(anthropic_types.TextBlock(type="text", text="  "))),
        ([], _message(text, stop="pause_turn")),
        ([], _message(text, stop=None)),
        ([], _message(text, stop="model_context_window_exceeded")),
    ])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).finish_reason == "length"
    assert asyncio.run(conversation.step()).finish_reason == "content_filter"
    history = len(conversation.messages)
    asyncio.run(conversation.step())
    assert len(conversation.messages) == history
    for _ in range(2):
        with pytest.raises(ProviderError):
            asyncio.run(conversation.step())
    with pytest.raises(PromptTooLongError):          # the window, as other providers say
        asyncio.run(conversation.step())


def test_anthropic_break_keeps_a_finished_reply_and_fails_an_open_one(monkeypatch):
    done = _message(anthropic_types.TextBlock(type="text", text="done"))
    open_ = _message(anthropic_types.TextBlock(type="text", text="do"), stop=None)
    gateway, _ = _anthropic_gateway(monkeypatch, [
        ([], done, httpx2.ReadError("lost")),
        ([], open_, httpx2.ReadError("lost")),
    ])
    conversation = gateway.conversation()
    assert asyncio.run(conversation.step()).text == "done"
    with pytest.raises(ProviderError, match="ReadError"):
        asyncio.run(conversation.step())


def test_anthropic_client_bounds_each_read_and_retries():
    cfg = _config(provider="anthropic", timeout=120.0)
    client = tool_gateway.AnthropicConversation.client(cfg)
    assert isinstance(client, anthropic.AsyncAnthropic)
    assert isinstance(client.timeout, httpx2.Timeout)
    assert (client.timeout.connect, client.timeout.read) == (30.0, 120.0)
    assert client.max_retries == 3


def test_an_anthropic_endpoint_out_of_credit_is_typed_as_such(monkeypatch):
    request = httpx2.Request("POST", "http://localhost:1/v1/messages")
    credit = anthropic.BadRequestError("Your credit balance is too low",
                                       response=httpx2.Response(400, request=request), body=None)
    gateway, _ = _anthropic_gateway(monkeypatch, [([], _message(stop=None), credit)])
    with pytest.raises(ProviderBillingError):
        asyncio.run(gateway.conversation().step())


def test_a_reply_the_server_aborted_is_a_provider_error(monkeypatch):
    """vLLM ends a request it dropped with "abort"; that is an outage to retry,
    not a reply with no answer."""
    gateway, _ = _gateway(monkeypatch, _config(), [_chunk(reasoning="half a thought", finish="abort")])
    with pytest.raises(ProviderError, match="abort"):
        asyncio.run(gateway.conversation().step())
