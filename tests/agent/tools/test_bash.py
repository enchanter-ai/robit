"""Tests for enchanter.agent.tools.bash.BashTool.

These tests exercise the W5 pre-execution veto (the headline feature), the
subprocess plumbing, the timeout + truncation behaviour, the env-allowlist
discipline, and the static contract attributes.

Some tests run a shell command and inspect output. On Windows the underlying
shell is cmd.exe, so we use cmd.exe-syntax (``echo``, ``%VAR%``, ``exit /b``,
``ping -n``) where possible; truly POSIX-specific tests are skipped on Windows.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from enchanter.agent.tools._types import ToolContext
from enchanter.agent.tools.bash import (
    BashTool,
    _ENV_ALLOWLIST,
    _MAX_TIMEOUT_S,
    _clamp_timeout,
)


IS_WINDOWS = sys.platform == "win32"


def _ctx(cwd: Path, *, max_output_bytes: int = 64 * 1024) -> ToolContext:
    return ToolContext(
        cwd=cwd,
        session_id="test-session-bash",
        max_output_bytes=max_output_bytes,
        timeout_s=5.0,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


def test_static_attributes():
    t = BashTool()
    assert t.name == "bash"
    assert t.requires_approval is True
    assert "command" in t.input_schema["properties"]
    assert t.input_schema["required"] == ["command"]
    assert t.input_schema["additionalProperties"] is False
    assert "BLOCKED" in t.description


def test_requires_approval_is_true():
    """The bash tool MUST always require explicit user approval."""
    assert BashTool().requires_approval is True


def test_clamp_timeout_caps_at_max():
    """Even a 999-second request should clamp to the hard ceiling."""
    assert _clamp_timeout(999) == _MAX_TIMEOUT_S
    assert _clamp_timeout(0) == 1.0  # below floor
    assert _clamp_timeout("nope") == 30.0  # default
    assert _clamp_timeout(60) == 60.0
    assert _clamp_timeout(False) == 30.0  # booleans treated as invalid


# ---------------------------------------------------------------------------
# Benign execution
# ---------------------------------------------------------------------------


def test_benign_echo_runs_successfully(tmp_path):
    """echo hello → exit 0, content shows command + payload."""
    tool = BashTool()
    res = _run(tool.execute({"command": "echo hello"}, _ctx(tmp_path)))
    assert res.is_error is False
    assert "$ echo hello" in res.content
    assert "hello" in res.content
    assert "exit_code: 0" in res.content


def test_failing_command_marks_error(tmp_path):
    """A non-zero exit must surface as is_error=True with the right code."""
    tool = BashTool()
    if IS_WINDOWS:
        cmd = "cmd /c exit 7"
    else:
        cmd = "exit 7"
    res = _run(tool.execute({"command": cmd}, _ctx(tmp_path)))
    assert res.is_error is True
    assert "exit_code: 7" in res.content


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_timeout_kills_process(tmp_path):
    """A sleep longer than timeout_s must be killed and reported."""
    tool = BashTool()
    if IS_WINDOWS:
        # ping -n 5 sleeps roughly 4s (it waits between pings).
        cmd = "ping -n 5 127.0.0.1 > NUL"
    else:
        cmd = "sleep 5"
    res = _run(tool.execute({"command": cmd, "timeout_s": 1}, _ctx(tmp_path)))
    assert res.is_error is True
    assert "timed out" in res.content
    # The side-effects should advertise the timeout too.
    assert any("timeout" in s for s in res.side_effects)


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def test_output_exceeding_max_bytes_truncates(tmp_path):
    """Output larger than ctx.max_output_bytes must be truncated with a marker."""
    tool = BashTool()
    # Generate a few KB of output; cap the context to a tiny budget.
    if IS_WINDOWS:
        # `for /L %i in (1,1,200) do @echo XXXXX...` — keep it simple, repeat 200 times.
        cmd = "for /L %i in (1,1,200) do @echo " + ("X" * 80)
    else:
        cmd = "for i in $(seq 1 200); do echo " + ("X" * 80) + "; done"
    res = _run(tool.execute({"command": cmd}, _ctx(tmp_path, max_output_bytes=512)))
    assert "...[truncated]" in res.content
    # Byte budget honoured (allow some slack for the marker itself).
    assert len(res.content.encode("utf-8")) <= 512 + len("\n...[truncated]") + 4
    assert any("truncated" in s for s in res.side_effects)


# ---------------------------------------------------------------------------
# Destructive-op-gate vetoes (THE headline feature)
# ---------------------------------------------------------------------------


def test_rm_rf_is_vetoed_before_execution(tmp_path):
    """The W5 rm-rf pattern must veto BEFORE any subprocess spawns."""
    tool = BashTool()
    # Use a clearly invented path to make the test obvious — but the veto
    # should fire regardless of the target.
    res = _run(
        tool.execute(
            {"command": "rm -rf /nonexistent-test-path-xyz-987"}, _ctx(tmp_path)
        )
    )
    assert res.is_error is True
    assert "veto" in res.content.lower()
    assert "not executed" in res.content
    # The side-effect must name the engine.
    assert any("destructive-op-gate" in s for s in res.side_effects)
    # The pattern id must be the rm-rf one.
    assert any("w5-rm-rf" in s for s in res.side_effects)


def test_sudo_rm_rf_is_vetoed(tmp_path):
    """``sudo rm -rf`` is caught by the rm-rf regex (sudo prefix doesn't help)."""
    tool = BashTool()
    res = _run(
        tool.execute(
            {"command": "sudo rm -rf /var/cache/whatever"}, _ctx(tmp_path)
        )
    )
    assert res.is_error is True
    assert "veto" in res.content.lower()
    assert any("w5-rm-rf" in s for s in res.side_effects)


def test_git_reset_hard_is_vetoed(tmp_path):
    """W5 also vetoes ``git reset --hard``."""
    tool = BashTool()
    res = _run(
        tool.execute({"command": "git reset --hard HEAD~3"}, _ctx(tmp_path))
    )
    assert res.is_error is True
    assert "veto" in res.content.lower()
    assert any("w5-reset-hard" in s for s in res.side_effects)


def test_force_push_is_vetoed(tmp_path):
    """``git push --force`` is the canonical W5 veto target."""
    tool = BashTool()
    res = _run(
        tool.execute({"command": "git push --force origin main"}, _ctx(tmp_path))
    )
    assert res.is_error is True
    assert "veto" in res.content.lower()
    assert any("w5-force-push" in s for s in res.side_effects)


# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------


def test_runs_in_supplied_cwd(tmp_path):
    """A file created by the subprocess lands in ctx.cwd, not wherever pytest runs."""
    tool = BashTool()
    if IS_WINDOWS:
        cmd = "echo landed > marker.txt"
    else:
        cmd = "echo landed > marker.txt"  # POSIX shell accepts this too
    res = _run(tool.execute({"command": cmd}, _ctx(tmp_path)))
    assert res.is_error is False
    assert (tmp_path / "marker.txt").exists()


# ---------------------------------------------------------------------------
# Env hygiene
# ---------------------------------------------------------------------------


def test_arbitrary_env_var_is_stripped(tmp_path, monkeypatch):
    """A non-allow-listed var set in the parent must NOT reach the subprocess."""
    monkeypatch.setenv("ARBITRARY_SECRET_VAR_FOR_TEST", "should-not-leak")
    # Sanity: make sure our allow-list does NOT include this name.
    assert "ARBITRARY_SECRET_VAR_FOR_TEST" not in _ENV_ALLOWLIST

    tool = BashTool()
    if IS_WINDOWS:
        # cmd.exe expands %VAR% to empty string when unset.
        cmd = "echo [%ARBITRARY_SECRET_VAR_FOR_TEST%]"
    else:
        cmd = 'echo "[${ARBITRARY_SECRET_VAR_FOR_TEST}]"'
    res = _run(tool.execute({"command": cmd}, _ctx(tmp_path)))
    assert res.is_error is False
    assert "should-not-leak" not in res.content
    # Either empty brackets [] or the literal %VAR% (Windows when var truly
    # unset prints the literal — but env=dict means it IS unset for the child).
    assert ("[]" in res.content) or ("[%ARBITRARY_SECRET_VAR_FOR_TEST%]" in res.content)


def test_allowlist_membership():
    """Document the exact env vars that propagate. Update this test when changed."""
    for required in ("PATH",):
        assert required in _ENV_ALLOWLIST
    # Sensitive shapes that MUST NOT be on the allow-list.
    forbidden = (
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
    )
    for f in forbidden:
        assert f not in _ENV_ALLOWLIST


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_command_arg_errors(tmp_path):
    tool = BashTool()
    res = _run(tool.execute({}, _ctx(tmp_path)))
    assert res.is_error is True
    assert "required" in res.content


def test_empty_command_arg_errors(tmp_path):
    tool = BashTool()
    res = _run(tool.execute({"command": "   "}, _ctx(tmp_path)))
    assert res.is_error is True
