"""robit.agent.conversation — turn-by-turn conversation dataclass.

The :class:`Conversation` is an append-only ledger of canonical messages.
Each mutation returns a fresh instance — never edits in place — so the loop
can hand snapshots to multiple consumers (renderer, audit, session saver)
without defensive copies.

Wave 15.0 reuses :class:`robit.proxy.canonical.Message` and the
``CanonicalContentPart`` union verbatim. Do NOT invent a parallel message
format here — the canonical types are what the proxy pipeline already speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable
from uuid import uuid4

import time

from robit.proxy.canonical import (
    ContentPart as CanonicalContentPart,
    Message as CanonicalMessage,
    TextPart,
    ToolResultPart,
)


def _new_session_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class Conversation:
    """Turn-by-turn agent conversation.

    Append-only. The four ``append_*`` methods each return a NEW instance
    with the appended message; the original is unchanged. This lets callers
    safely capture intermediate states.

    Attributes
    ----------
    system_prompt:
        Default system prompt for the conversation; persisted at session
        start and folded into the canonical ``CanonicalRequest.system`` field
        on every dispatch. ``None`` means no system prompt.
    messages:
        Ordered tuple of canonical messages. First user prompt comes first;
        assistant + tool_result messages interleave after.
    model:
        Current model id (the loop hands this to ``CanonicalRequest.model``).
        Mutable across turns via ``/model <name>``; replaced by the slash
        command via :meth:`with_model`.
    started_ts:
        Unix epoch seconds; set once at construction.
    session_id:
        UUID4 hex; identifies the on-disk JSONL session log.
    """

    system_prompt: str | None
    messages: tuple[CanonicalMessage, ...]
    model: str
    started_ts: float
    session_id: str

    # ----- factory ----------------------------------------------------------

    @staticmethod
    def new(
        *,
        model: str,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> "Conversation":
        """Construct a fresh conversation with no messages yet."""
        return Conversation(
            system_prompt=system_prompt,
            messages=(),
            model=model,
            started_ts=time.time(),
            session_id=session_id or _new_session_id(),
        )

    # ----- mutators (return new instances) ---------------------------------

    def append_user(self, text: str) -> "Conversation":
        msg = CanonicalMessage(role="user", content=(TextPart(text=text),))
        return replace(self, messages=self.messages + (msg,))

    def append_assistant(
        self, content: Iterable[CanonicalContentPart]
    ) -> "Conversation":
        parts = tuple(content)
        msg = CanonicalMessage(role="assistant", content=parts)
        return replace(self, messages=self.messages + (msg,))

    def append_tool_result(
        self, tool_use_id: str, result: str, is_error: bool = False
    ) -> "Conversation":
        """Append a user-role message carrying a single ``tool_result`` part.

        The canonical shape uses the ``"user"`` role for tool results because
        every major provider (Anthropic, OpenAI, Gemini) models tool outputs
        as user-side inputs feeding the next assistant turn.
        """
        part = ToolResultPart(
            tool_use_id=tool_use_id, content=result, is_error=is_error
        )
        msg = CanonicalMessage(role="user", content=(part,))
        return replace(self, messages=self.messages + (msg,))

    def with_model(self, model: str) -> "Conversation":
        """Switch model mid-session. Used by ``/model <name>``."""
        return replace(self, model=model)

    def cleared(self) -> "Conversation":
        """Reset messages but preserve ``session_id`` and ``system_prompt``.

        Used by ``/clear``. The same session log keeps growing — a CLEAR
        sentinel will be emitted by the loop into the JSONL so replay can
        rebuild state.
        """
        return replace(self, messages=())


__all__ = ["Conversation"]
