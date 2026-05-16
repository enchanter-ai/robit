"""Tests for robit.agent.conversation — immutability + session_id round-trip."""

from __future__ import annotations

import pytest

from robit.agent.conversation import Conversation
from robit.agent.session import SessionWriter, load_session
from robit.proxy.canonical import TextPart, ToolUsePart


def test_append_user_returns_new_instance():
    c0 = Conversation.new(model="m", system_prompt="sys")
    c1 = c0.append_user("hello")
    assert c0 is not c1
    assert len(c0.messages) == 0
    assert len(c1.messages) == 1
    # original untouched
    assert c0.system_prompt == "sys"


def test_append_assistant_preserves_immutability():
    c0 = Conversation.new(model="m")
    part = TextPart(text="hi")
    c1 = c0.append_assistant((part,))
    assert c0.messages == ()
    assert c1.messages[0].role == "assistant"
    assert c1.messages[0].content == (part,)


def test_append_tool_result_marks_user_role():
    c0 = Conversation.new(model="m")
    c1 = c0.append_tool_result("tu_1", "result text", is_error=False)
    msg = c1.messages[0]
    assert msg.role == "user"
    assert msg.content[0].tool_use_id == "tu_1"
    assert msg.content[0].content == "result text"
    assert msg.content[0].is_error is False


def test_system_prompt_persists_across_appends():
    c = Conversation.new(model="m", system_prompt="be terse")
    c = c.append_user("a")
    c = c.append_assistant((TextPart(text="b"),))
    c = c.append_tool_result("tu_1", "ok")
    assert c.system_prompt == "be terse"


def test_session_id_round_trips_through_jsonl(tmp_path):
    c0 = Conversation.new(model="m", system_prompt="sys")
    c0 = c0.append_user("hello")
    c0 = c0.append_assistant((
        TextPart(text="hi back"),
        ToolUsePart(id="tu_1", name="echo", input={"text": "x"}),
    ))
    c0 = c0.append_tool_result("tu_1", "x")

    w = SessionWriter.for_session(c0.session_id)
    w.write_header(c0)
    for m in c0.messages:
        w.write_message(m)

    loaded = load_session(c0.session_id)
    assert loaded.session_id == c0.session_id
    assert loaded.model == "m"
    assert loaded.system_prompt == "sys"
    assert len(loaded.messages) == 3


def test_cleared_keeps_session_id_and_system_prompt():
    c = Conversation.new(model="m", system_prompt="sys")
    c = c.append_user("hello")
    cleared = c.cleared()
    assert cleared.session_id == c.session_id
    assert cleared.system_prompt == "sys"
    assert cleared.messages == ()


def test_with_model_returns_new_instance():
    c0 = Conversation.new(model="m1")
    c1 = c0.with_model("m2")
    assert c0.model == "m1"
    assert c1.model == "m2"
