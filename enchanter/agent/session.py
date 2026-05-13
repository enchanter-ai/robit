"""enchanter.agent.session — JSONL persistence for agent conversations.

Layout::

    $ENCHANTER_HOME/sessions/<session_id>.jsonl    (if env var set)
    %APPDATA%\\enchanter\\sessions\\<session_id>.jsonl   (Windows default)
    ~/.enchanter/sessions/<session_id>.jsonl       (POSIX default)

Format
------

One JSON object per line. The first line is a ``session_start`` header; every
subsequent line is either a ``message`` (one canonical message) or a control
record (``model_changed``, ``clear``, ``usage``).

Replay rebuilds a :class:`Conversation` by streaming the file in order. A
malformed line is logged and skipped — corruption never crashes replay.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from enchanter.proxy.canonical import (
    ContentPart as CanonicalContentPart,
    Message as CanonicalMessage,
    TextPart,
    ToolResultPart,
    ToolUsePart,
)

from .conversation import Conversation


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def session_dir() -> Path:
    """Return the directory holding session JSONL logs.

    Resolution order:

    1. ``$ENCHANTER_HOME/sessions``
    2. Windows: ``%APPDATA%\\enchanter\\sessions``
    3. POSIX:   ``~/.enchanter/sessions``

    The directory is created (parents=True) on first call.
    """
    override = os.environ.get("ENCHANTER_HOME")
    if override:
        root = Path(override)
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        root = Path(appdata) / "enchanter" if appdata else Path.home() / ".enchanter"
    else:
        root = Path.home() / ".enchanter"
    out = root / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    return out


def session_path(session_id: str) -> Path:
    return session_dir() / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# Serialisation helpers (canonical → JSON-safe dict)
# ---------------------------------------------------------------------------


def _part_to_dict(part: CanonicalContentPart) -> dict:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolUsePart):
        return {
            "type": "tool_use",
            "id": part.id,
            "name": part.name,
            "input": part.input,
        }
    if isinstance(part, ToolResultPart):
        return {
            "type": "tool_result",
            "tool_use_id": part.tool_use_id,
            "content": part.content,
            "is_error": part.is_error,
        }
    raise TypeError(f"unsupported content part: {type(part).__name__}")


def _dict_to_part(d: dict) -> CanonicalContentPart:
    t = d.get("type")
    if t == "text":
        return TextPart(text=d.get("text", ""))
    if t == "tool_use":
        return ToolUsePart(
            id=d.get("id", ""),
            name=d.get("name", ""),
            input=d.get("input", {}),
        )
    if t == "tool_result":
        return ToolResultPart(
            tool_use_id=d.get("tool_use_id", ""),
            content=d.get("content", ""),
            is_error=bool(d.get("is_error", False)),
        )
    raise ValueError(f"unknown content part type: {t!r}")


def _message_to_dict(msg: CanonicalMessage) -> dict:
    return {
        "role": msg.role,
        "content": [_part_to_dict(p) for p in msg.content],
    }


def _dict_to_message(d: dict) -> CanonicalMessage:
    parts = tuple(_dict_to_part(p) for p in d.get("content", []))
    return CanonicalMessage(role=d.get("role", "user"), content=parts)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass
class SessionWriter:
    """Append-only JSONL writer for a single conversation session.

    Construct once at session start; call :meth:`write_header` (idempotent
    if the file is empty), then :meth:`write_message` per appended message.
    A single open file handle would simplify things; we open-and-close per
    write so a crashed process leaves a flushed file behind.
    """

    session_id: str
    path: Path

    @staticmethod
    def for_session(session_id: str) -> "SessionWriter":
        return SessionWriter(session_id=session_id, path=session_path(session_id))

    def _append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_header(self, conv: Conversation) -> None:
        """Write a ``session_start`` record. Safe to call once per session."""
        self._append(
            {
                "kind": "session_start",
                "session_id": conv.session_id,
                "model": conv.model,
                "system_prompt": conv.system_prompt,
                "started_ts": conv.started_ts,
            }
        )

    def write_message(self, msg: CanonicalMessage) -> None:
        self._append({"kind": "message", "message": _message_to_dict(msg)})

    def write_model_change(self, model: str) -> None:
        self._append({"kind": "model_changed", "model": model})

    def write_clear(self) -> None:
        self._append({"kind": "clear"})

    def write_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._append(
            {
                "kind": "usage",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )


# ---------------------------------------------------------------------------
# Loader (replay)
# ---------------------------------------------------------------------------


def load_session(session_id: str) -> Conversation:
    """Replay a JSONL session log into a fresh :class:`Conversation`.

    Corrupted lines are logged at WARNING and skipped — the loader never
    crashes on a single bad record. If no ``session_start`` header is
    present the loader returns an empty conversation seeded with
    ``model="unknown"``.
    """
    path = session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"no such session: {session_id} (looked at {path})")

    header: dict | None = None
    messages: list[CanonicalMessage] = []
    model_override: str | None = None
    cleared_count = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                _log.warning(
                    "session %s line %d: skipping corrupted JSONL (%s)",
                    session_id,
                    lineno,
                    exc,
                )
                continue
            kind = rec.get("kind")
            try:
                if kind == "session_start":
                    header = rec
                elif kind == "message":
                    messages.append(_dict_to_message(rec.get("message", {})))
                elif kind == "model_changed":
                    model_override = rec.get("model")
                elif kind == "clear":
                    messages.clear()
                    cleared_count += 1
                elif kind == "usage":
                    pass  # informational only
                else:
                    _log.warning(
                        "session %s line %d: skipping unknown kind %r",
                        session_id,
                        lineno,
                        kind,
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "session %s line %d: skipping malformed record (%s)",
                    session_id,
                    lineno,
                    exc,
                )

    if header is None:
        return Conversation.new(model="unknown", session_id=session_id)

    return Conversation(
        system_prompt=header.get("system_prompt"),
        messages=tuple(messages),
        model=model_override or header.get("model", "unknown"),
        started_ts=float(header.get("started_ts", 0.0)),
        session_id=session_id,
    )


__all__ = [
    "session_dir",
    "session_path",
    "SessionWriter",
    "load_session",
]


# ---------------------------------------------------------------------------
# Conversation.load — convenience proxy on the dataclass.
# ---------------------------------------------------------------------------


def _conv_load(cls, session_id: str) -> Conversation:
    """Implementation of ``Conversation.load(session_id)`` — attached below."""
    return load_session(session_id)


Conversation.load = classmethod(_conv_load)  # type: ignore[attr-defined]


def session_messages(messages: tuple[CanonicalMessage, ...]) -> Iterable[dict]:
    """Render messages as JSON-safe dicts. Public for tests / debugging."""
    return [_message_to_dict(m) for m in messages]
