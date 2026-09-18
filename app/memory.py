"""In-memory short/long-term memory store, keyed per session."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class SessionMemory:
    session_id: str
    short_term: list[Message] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)  # long-term key facts

    def add(self, role: str, content: str) -> None:
        self.short_term.append(Message(role, content))
        # keep the working window bounded
        if len(self.short_term) > 30:
            self.short_term = self.short_term[-30:]

    def history_for_model(self) -> list[dict]:
        return [
            {
                "role": "model" if m.role == "assistant" else m.role,
                "parts": [{"text": m.content}],
            }
            for m in self.short_term
        ]


class MemoryManager:
    """Holds one SessionMemory per conversation. Swap for Redis/DB later."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id)
        return self._sessions[session_id]


memory_manager = MemoryManager()
