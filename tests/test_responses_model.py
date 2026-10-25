"""The Responses-API agent model keeps every output message and its boundaries."""

import asyncio
from types import SimpleNamespace as NS

from cave_agent.parsing.streaming import SegmentType, StreamingTextParser

from core.models import ModelConfig, load_model
from core.responses_model import ResponsesModel

TOML = """
[luna]
api_model = "gpt-6-luna"
api_key = "k"
base_url = "http://localhost:1/v1"
provider = "openai_responses"
max_tokens = 1000
temperature = 0.1
reasoning_effort = "none"
"""


def _message_added():
    return NS(type="response.output_item.added", item=NS(type="message"))


def _text(delta, output=0):
    return NS(type="response.output_text.delta", delta=delta, output_index=output, content_index=0)


def _text_done(text, output=0):
    return NS(type="response.output_text.done", text=text, output_index=output, content_index=0)


def _done(status="completed", reason=None):
    usage = NS(input_tokens=10, output_tokens=5, total_tokens=15)
    details = NS(reason=reason) if reason else None
    response = NS(status=status, incomplete_details=details, usage=usage)
    return NS(type=f"response.{status}", response=response)


class _Events:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


def _model(events):
    cfg = ModelConfig(
        name="luna", api_model="gpt-6-luna", api_key="k", max_tokens=1000,
        temperature=0.1, reasoning_effort="none",
    )
    model = ResponsesModel("gpt-6-luna", "k", "http://localhost:1/v1", cfg.request_timeout(),
                           cfg.responses_options())

    async def create(**params):
        model.last_params = params
        return _Events(events)

    model.client = NS(responses=NS(create=create), close=None)
    return model


def _collect(stream):
    async def run():
        return [d async for d in stream]

    return asyncio.run(run())


def test_each_output_message_starts_on_a_new_line():
    events = [
        _message_added(), _text("Need assign variables."),
        _message_added(), _text("```python\nx = 1\n```"),
        _done(),
    ]
    stream = _model(events).stream([{"role": "user", "content": "q"}])
    text = "".join(d.content or "" for d in _collect(stream))
    assert text == "Need assign variables.\n```python\nx = 1\n```"
    assert stream.finish_reason == "stop"
    assert stream.usage.prompt_tokens == 10 and stream.usage.completion_tokens == 5
    parser = StreamingTextParser()
    segments = parser.process_chunk(text) + parser.flush()
    code = [s.content for s in segments if s.type is SegmentType.CODE]
    assert code == ["x = 1"]


def test_the_bridge_join_hides_the_fence_from_the_parser():
    # What LiteLLM's Responses bridge delivered: the same two messages, joined.
    parser = StreamingTextParser()
    segments = parser.process_chunk("Need assign variables.```python\nx = 1\n```") + parser.flush()
    assert not [s for s in segments if s.type is SegmentType.CODE]


def test_no_newline_is_added_when_a_message_already_ended_one():
    events = [
        _message_added(), _text("```python\nx = 1\n```\n"),
        _message_added(), _text("done"),
        _done(),
    ]
    text = "".join(d.content or "" for d in _collect(_model(events).stream([])))
    assert text == "```python\nx = 1\n```\ndone"


def test_a_message_the_endpoint_did_not_stream_is_taken_from_its_done_event():
    events = [
        _message_added(), _text_done("Plan first.", output=0),
        _message_added(), _text("```python\nx = 1\n```", output=1),
        _text_done("```python\nx = 1\n```", output=1),
        _done(),
    ]
    text = "".join(d.content or "" for d in _collect(_model(events).stream([])))
    # The streamed part is not repeated by its own done event.
    assert text == "Plan first.\n```python\nx = 1\n```"


def test_cap_reached_is_reported_as_length():
    events = [_message_added(), _text("partial"), _done("incomplete", "max_output_tokens")]
    stream = _model(events).stream([])
    _collect(stream)
    assert stream.finish_reason == "length"


def test_request_carries_cap_temperature_and_effort():
    model = _model([_done()])
    _collect(model.stream([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]))
    params = model.last_params
    assert params["max_output_tokens"] == 1000
    assert params["temperature"] == 0.1
    assert params["reasoning"] == {"effort": "none"}
    assert params["input"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert params["stream"] is True


def test_provider_switch_selects_the_responses_model(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    model, config = load_model(path, "luna")
    assert isinstance(model, ResponsesModel)
    assert model.max_output_tokens == config.max_tokens
    assert (model.client.timeout.connect, model.client.timeout.read) == (30.0, config.timeout)


def test_non_streamed_answer_joins_messages_the_same_way():
    def message(text):
        return NS(type="message", content=[NS(type="output_text", text=text)])

    response = NS(
        output=[
            NS(type="reasoning"),
            message("Need assign variables."),
            message("```python\nx = 1\n```"),
        ],
        status="completed", incomplete_details=None,
        usage=NS(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    model = _model([])

    async def create(**params):
        return response

    model.client = NS(responses=NS(create=create))
    answer = asyncio.run(model._complete([{"role": "user", "content": "q"}]))
    assert answer.content == "Need assign variables.\n```python\nx = 1\n```"
    assert answer.finish_reason == "stop" and answer.refusal is None
