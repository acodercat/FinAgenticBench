"""The case schema: a case is a conversation, a conversation is turns.

A single-turn case is the degenerate form of this shape, not a different one,
so the loader below reads both without a compatibility branch elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """One question, its validator, and the outputs it is allowed to ask for."""

    query: str = ""
    validator: str | None = None
    # Variables THIS turn asks the agent to store. Absent means "all of them",
    # which is right for a single-turn case. Listing them per turn is what stops
    # turn 1 from advertising turn 2's outputs: the runtime registers a turn's
    # variables only when that turn begins, and the agent sees the runtime's
    # live registrations. A static list naming every output would tell the model
    # in turn 1 what turn 2 is going to ask.
    stores: list[str] | None = None
    # Claim Coverage is declared per turn: a claim addressed only in an earlier
    # turn's response is not coverage for this one.
    claims: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            query=data.get("query", ""),
            validator=data.get("validator"),
            stores=data.get("stores"),
            claims=list(data.get("claims", [])),
        )


@dataclass
class Conversation:
    id: str
    turns: list[Turn]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        return cls(
            id=data.get("id", "main"),
            turns=[Turn.from_dict(turn) for turn in data.get("turns", [])],
        )


def conversations_of(spec: dict[str, Any]) -> list[Conversation]:
    """Read a case's conversations, accepting the single-turn short form.

    A case that carries a bare `query` is one conversation of one turn. Keeping
    that expansion here means the evaluator, the scorers and the reports all see
    one shape and never ask which form a case was written in.
    """
    declared = spec.get("conversations")
    if declared:
        return [Conversation.from_dict(item) for item in declared]
    return [Conversation.from_dict({
        "id": "main",
        "turns": [{
            "query": spec.get("query", ""),
            "validator": spec.get("validator"),
            "claims": spec.get("claims", []),
        }],
    })]


def sample_id(case_name: str, conversation_id: str, turn_index: int) -> str:
    """Address one scored turn. `turn_index` is 0-based; files display 1-based."""
    return f"{case_name}/{conversation_id}/{turn_index}"
