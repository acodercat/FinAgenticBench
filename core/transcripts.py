"""Full, greppable conversation transcripts for benchmark runs.

Result JSON records the scored outputs and compact execution summary.  It does
not by itself preserve every prompt, model message, code execution and runtime
result.  Those messages are written here as JSON Lines, one record per message,
and cited by a relative path from the result that owns them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from core.results import atomic_write_text


TRANSCRIPTS_DIR = "transcripts"


def message_to_dict(index: int, message: Any) -> dict[str, Any]:
    """Convert one history item without discarding adapter-specific fields."""
    if isinstance(message, Mapping):
        return {"index": index, **message}

    role = getattr(message, "wire_role", None) or getattr(message, "role", None)
    return {
        "index": index,
        "type": type(message).__name__,
        "role": getattr(role, "value", role),
        "content": getattr(message, "content", ""),
    }


def transcript_reference(result_path: Path, conversation_id: str) -> str:
    """Return the transcript reference stored in a result JSON."""
    if result_path.name == "run.json":
        return f"{TRANSCRIPTS_DIR}/{conversation_id}.jsonl"
    return f"{TRANSCRIPTS_DIR}/{result_path.stem}_{conversation_id}.jsonl"


def transcript_path(result_path: Path, conversation_id: str) -> Path:
    """Resolve a transcript path from its owning result file."""
    return result_path.parent / transcript_reference(result_path, conversation_id)


def write_transcript(path: Path, messages: Iterable[Any]) -> None:
    """Atomically write one JSONL record per conversation message."""
    lines = [
        json.dumps(message_to_dict(index, message), ensure_ascii=False)
        for index, message in enumerate(messages)
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def read_transcript(path: Path) -> list[dict[str, Any]]:
    """Read a transcript back into its ordered records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def messages_of(agent: Any) -> list[Any] | None:
    """Return an agent's linear history, or ``None`` when unavailable."""
    messages = getattr(agent, "messages", None)
    return None if messages is None else list(messages)
