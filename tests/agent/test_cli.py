"""Tests for enchanter.agent.cli — version flag + one-shot mode w/ mock LLM."""

from __future__ import annotations

import pytest

from enchanter.agent.cli import main


def test_version_exits_0(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "enchanter" in out
    # Version string from enchanter package.
    from enchanter import __version__
    assert __version__ in out


def test_one_shot_echo_runs_to_completion(capsys):
    rc = main(["echo hello-world"])
    assert rc == 0
    out = capsys.readouterr().out
    # The mock LLM proposes an echo tool_use; the tool returns the text;
    # the loop renders the tool result + a final summary.
    assert "hello-world" in out
    assert "turn done" in out


def test_one_shot_text_only_completes(capsys):
    rc = main(["just say hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "acknowledged" in out  # mock LLM's default response
    assert "turn done" in out
