import json

from core.transcripts import (
    message_to_dict,
    messages_of,
    read_transcript,
    transcript_path,
    transcript_reference,
    write_transcript,
)


class FakeMessage:
    def __init__(self, content: str, role: str):
        self.content = content
        self.wire_role = role


class SystemMessage(FakeMessage):
    pass


class CodeExecutionMessage(FakeMessage):
    pass


def test_transcript_reference_resolves_from_result_directory(tmp_path):
    result = tmp_path / "case" / "model.json"
    reference = transcript_reference(result, "main")
    assert result.parent / reference == transcript_path(result, "main")
    assert reference == "transcripts/model_main.jsonl"


def test_v3_run_uses_conversation_name_without_model_prefix(tmp_path):
    result = tmp_path / "runs" / "case" / "model" / "run-1" / "run.json"
    assert transcript_reference(result, "main") == "transcripts/main.jsonl"
    assert transcript_path(result, "main") == result.parent / "transcripts/main.jsonl"


def test_transcript_round_trip_preserves_type_content_and_lines(tmp_path):
    path = transcript_path(tmp_path / "case" / "model.json", "main")
    messages = [
        SystemMessage("system\nline two", "system"),
        CodeExecutionMessage("```python\nx = 1\n```", "assistant"),
    ]
    write_transcript(path, messages)
    records = read_transcript(path)
    assert [record["index"] for record in records] == [0, 1]
    assert [record["type"] for record in records] == [
        "SystemMessage", "CodeExecutionMessage",
    ]
    assert [record["content"] for record in records] == [
        message.content for message in messages
    ]
    assert len(path.read_text().splitlines()) == 2
    assert list(path.parent.iterdir()) == [path]


def test_a_writer_never_takes_another_writers_temporary_file(tmp_path):
    path = transcript_path(tmp_path / "case" / "model.json", "main")
    path.parent.mkdir(parents=True)
    # The name every write used, while another writer is still filling it.
    other = path.with_suffix(".jsonl.tmp")
    other.write_text("another writer's transcript\n")
    write_transcript(path, [SystemMessage("system", "system")])
    assert [record["content"] for record in read_transcript(path)] == ["system"]
    assert other.read_text() == "another writer's transcript\n"


def test_mapping_message_is_preserved_whole():
    record = message_to_dict(2, {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1"}],
    })
    assert record["index"] == 2
    assert record["tool_calls"] == [{"id": "call-1"}]


def test_messages_of_copies_history_and_handles_absence():
    message = {"role": "user", "content": "question"}
    agent = type("Agent", (), {"messages": [message]})()
    captured = messages_of(agent)
    agent.messages.append({"role": "assistant", "content": "later"})
    assert captured == [message]
    assert messages_of(object()) is None


def test_jsonl_is_directly_greppable(tmp_path):
    path = tmp_path / "transcript.jsonl"
    write_transcript(path, [{"role": "assistant", "content": "a\nb"}])
    line = path.read_text().strip()
    assert json.loads(line)["content"] == "a\nb"
