"""Tests for robit.agent.session — JSONL persistence + corruption tolerance."""

from __future__ import annotations

import json

import pytest

from robit.agent.conversation import Conversation
from robit.agent.session import (
    SessionWriter,
    load_session,
    session_dir,
    session_path,
)
from robit.proxy.canonical import TextPart, ToolUsePart


def test_session_dir_honors_enchanter_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBIT_HOME", str(tmp_path))
    d = session_dir()
    assert str(d).startswith(str(tmp_path))
    assert d.exists()


def test_round_trip_preserves_all_messages():
    c = Conversation.new(model="m", system_prompt="sys")
    c = c.append_user("hello")
    c = c.append_assistant((
        TextPart(text="hi"),
        ToolUsePart(id="tu_1", name="echo", input={"text": "x"}),
    ))
    c = c.append_tool_result("tu_1", "x", is_error=False)

    w = SessionWriter.for_session(c.session_id)
    w.write_header(c)
    for m in c.messages:
        w.write_message(m)

    loaded = Conversation.load(c.session_id)  # type: ignore[attr-defined]
    assert len(loaded.messages) == len(c.messages)
    for orig, restored in zip(c.messages, loaded.messages):
        assert orig.role == restored.role
        assert len(orig.content) == len(restored.content)


def test_corrupted_line_is_skipped_not_crashed(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ROBIT_HOME", str(tmp_path))
    c = Conversation.new(model="m")
    w = SessionWriter.for_session(c.session_id)
    w.write_header(c)
    w.write_message(c.append_user("good").messages[0])

    # Inject a corrupt line directly.
    with w.path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        # And a record with unknown kind.
        f.write(json.dumps({"kind": "mystery"}) + "\n")

    loaded = load_session(c.session_id)
    # The good message survived; bad ones were skipped.
    assert len(loaded.messages) == 1


def test_load_session_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_session("no-such-session-id-xyz")


def test_model_change_record_replays():
    c = Conversation.new(model="m1")
    w = SessionWriter.for_session(c.session_id)
    w.write_header(c)
    w.write_model_change("m2")

    loaded = load_session(c.session_id)
    assert loaded.model == "m2"


def test_clear_record_resets_messages_during_replay():
    c = Conversation.new(model="m")
    w = SessionWriter.for_session(c.session_id)
    w.write_header(c)
    w.write_message(c.append_user("first").messages[0])
    w.write_clear()
    after_clear = c.cleared().append_user("second")
    w.write_message(after_clear.messages[0])

    loaded = load_session(c.session_id)
    # Only the post-clear message survives.
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content[0].text == "second"
