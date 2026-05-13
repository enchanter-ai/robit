"""enchanter.agent.tools.bash — execute a shell command, gated by W5.

The :class:`BashTool` is the most security-sensitive tool in the wave. Before
spawning any subprocess, the tool builds a synthetic ``mcp.tool.call.requested``
event on the ``trust-gate`` phase and asks the same :mod:`destructive_op_gate`
engine that vetoes prompt-borne destructive commands whether to allow it. If
the engine returns ``status="veto"``, the command is never executed — the
tool returns an error result naming the matched pattern.

Design contracts:

* **Pre-execution veto.** ``rm -rf``, ``git push --force``, ``git reset --hard``
  and the rest of the W5 table get blocked by the same enforcement layer the
  proxy uses on prompt text. This is the "enforcement in tools, not just in
  prompts" payoff.
* **Approval-gated.** ``requires_approval = True`` — every bash call surfaces
  to the user before the loop dispatches the execute coroutine, on top of the
  W5 gate.
* **Environment hygiene.** The subprocess inherits only an explicitly
  allow-listed subset of env vars (PATH, HOME, temp dirs, locale, Windows
  system vars). API keys, SSH credentials, etc. are stripped.
* **Output budget.** Combined stdout+stderr is truncated to
  ``ctx.max_output_bytes`` with a ``...[truncated]`` marker.
* **Timeout cap.** Default 30s; ``args["timeout_s"]`` may extend up to 300s.

Windows limitation: on Windows ``asyncio.create_subprocess_shell`` uses
``cmd.exe``, not bash. POSIX-only commands (``rm``, ``ls`` without aliases,
etc.) require a Unix-style shell environment (WSL, Git Bash on PATH).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from enchanter.core import (
    EnchantedEvent,
    PluginAck,
    create_request_context,
)
from enchanter.core.bus import build_event
from enchanter.engines.destructive_op_gate.adapter import adapter as _dog_adapter

from ._types import ToolContext, ToolResult


# Hard cap on per-invocation timeout. The LLM may not extend beyond this.
_MAX_TIMEOUT_S: float = 300.0
_DEFAULT_TIMEOUT_S: float = 30.0
_MIN_TIMEOUT_S: float = 1.0

# Truncation marker appended when output exceeds ctx.max_output_bytes.
_TRUNCATION_MARKER = "\n...[truncated]"

# Allow-listed env vars the subprocess inherits. Anything not in this set is
# stripped — particularly anything that smells like a credential. Note: PATH
# is required for the shell to find binaries; everything else is best-effort
# convenience.
_ENV_ALLOWLIST: tuple[str, ...] = (
    # Required.
    "PATH",
    # Where the user's home / profile lives (cross-platform).
    "HOME",
    "USERPROFILE",
    # Temp dirs (POSIX + Windows).
    "TMPDIR",
    "TEMP",
    "TMP",
    # Locale.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    # Windows essentials — without SYSTEMROOT/PATHEXT, cmd.exe cannot resolve
    # builtins or find executables; without WINDIR child processes fail to
    # locate kernel32.
    "SYSTEMROOT",
    "WINDIR",
    "PATHEXT",
    "COMSPEC",
    "SYSTEMDRIVE",
)


def _filtered_env() -> dict[str, str]:
    """Build the subprocess env from :data:`_ENV_ALLOWLIST` only."""
    out: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            out[key] = val
    return out


def _clamp_timeout(raw: object) -> float:
    """Clamp ``timeout_s`` into ``[_MIN_TIMEOUT_S, _MAX_TIMEOUT_S]``."""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return _DEFAULT_TIMEOUT_S
    t = float(raw)
    if t != t:  # NaN
        return _DEFAULT_TIMEOUT_S
    if t < _MIN_TIMEOUT_S:
        return _MIN_TIMEOUT_S
    if t > _MAX_TIMEOUT_S:
        return _MAX_TIMEOUT_S
    return t


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_bytes`` UTF-8 bytes; return (text, truncated)."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    cut = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return cut + _TRUNCATION_MARKER, True


def _format_cwd(cwd: Path) -> str:
    """Render ``cwd`` for the side-effect message — basename if possible."""
    try:
        return str(cwd)
    except Exception:
        return repr(cwd)


def _cmd_summary(cmd: str, max_len: int = 60) -> str:
    """First ``max_len`` chars of ``cmd``, with ellipsis if truncated."""
    s = cmd.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


async def _veto_check(cmd: str, session_id: str) -> PluginAck:
    """Ask destructive-op-gate whether ``cmd`` may run.

    Builds a synthetic ``mcp.tool.call.requested`` event at the ``trust-gate``
    phase, mirroring exactly what the proxy emits for prompt-borne tool calls,
    and routes it through the engine's ``on_phase`` coroutine. The engine's
    pattern table is the only authority — this function does no regex of its
    own.
    """
    ctx = create_request_context(
        session_id=session_id,
        budget_tier="HIGH",
        deadline_ms=500,
    )
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source="agent-bash",
        budget_tier=ctx.budget_tier,
        payload={"tool": "bash", "args": [cmd], "source": "agent-bash"},
    )
    return await _dog_adapter.on_phase(event, ctx)


def _pattern_id_from_ack(ack: PluginAck) -> str:
    """Best-effort extraction of the matched W5 ``pattern_id``.

    The adapter stamps the pattern id into the derived event's payload as
    ``pattern_id``; the ack's ``reason`` carries it as ``"<plugin>:<id>"``.
    We prefer the derived-event payload (richer) and fall back to parsing the
    reason string.
    """
    for derived in ack.derived_events or ():
        pid = derived.payload.get("pattern_id") if derived.payload else None
        if isinstance(pid, str) and pid:
            return pid
    reason = ack.reason or ""
    if ":" in reason:
        return reason.split(":", 1)[1]
    return reason or "unknown"


class BashTool:
    """Execute a shell command, gated by the W5 destructive-op engine."""

    name: str = "bash"
    description: str = (
        "Execute a shell command and return its combined stdout/stderr. The command "
        "runs in ctx.cwd. Output is truncated to ctx.max_output_bytes. Commands "
        "matching the destructive-op-gate (rm -rf, sudo, fork bombs, etc.) are "
        "BLOCKED before execution and return an error explaining the veto. On "
        "Windows, runs via cmd.exe — POSIX-only commands (rm, ls, etc.) require "
        "a Unix-style shell environment (WSL, Git Bash on PATH, etc.)."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run.",
            },
            "timeout_s": {
                "type": "number",
                "minimum": 1,
                "maximum": 300,
                "default": 30,
                "description": "Override the default timeout (seconds, capped at 300).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    requires_approval: bool = True  # Always require user approval.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        # 0. Argument validation.
        raw_cmd = args.get("command")
        if not isinstance(raw_cmd, str) or not raw_cmd.strip():
            return ToolResult(
                content="bash: 'command' arg is required and must be a non-empty string",
                is_error=True,
            )
        cmd = raw_cmd
        timeout_s = _clamp_timeout(args.get("timeout_s", _DEFAULT_TIMEOUT_S))

        # 1. PRE-EXECUTION VETO. Ask destructive-op-gate before spawning.
        try:
            ack = await _veto_check(cmd, ctx.session_id)
        except Exception as exc:  # pragma: no cover — engine is fail-closed; surface explicitly.
            return ToolResult(
                content=(
                    f"bash veto check failed: {exc!r} — refusing to execute "
                    f"command without a passing destructive-op-gate ack"
                ),
                is_error=True,
                side_effects=("destructive-op-gate check raised",),
            )

        if ack.status == "veto":
            pattern_id = _pattern_id_from_ack(ack)
            reason = ack.reason or "destructive-op-gate veto"
            return ToolResult(
                content=(
                    f"bash veto: {reason} — command not executed. "
                    f"This command matched the destructive-op-gate pattern "
                    f"{pattern_id!r}. If you genuinely need it, ask the user "
                    f"to run it manually."
                ),
                is_error=True,
                side_effects=(f"destructive-op-gate vetoed: {pattern_id}",),
            )

        # 2. Spawn the subprocess.
        env = _filtered_env()
        cwd = str(ctx.cwd)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
        except (OSError, ValueError) as exc:
            return ToolResult(
                content=f"bash: failed to spawn shell for {_cmd_summary(cmd)!r}: {exc}",
                is_error=True,
            )

        # 3. Wait with timeout. On timeout, kill the process group and report.
        timed_out = False
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Drain whatever the child wrote before we killed it.
            try:
                stdout_bytes, _ = await proc.communicate()
            except Exception:
                stdout_bytes = b""

        rc = proc.returncode if proc.returncode is not None else -1

        # 4. Decode + truncate.
        try:
            output = stdout_bytes.decode("utf-8", errors="replace")
        except Exception:
            output = repr(stdout_bytes)

        if timed_out:
            note = (
                f"\n[bash: command timed out after {timeout_s:g}s — "
                f"process killed]"
            )
            output = output + note

        # Construct the rendered body before truncation so the prelude + body
        # share the byte budget honestly.
        prelude = (
            f"$ {cmd}\n"
            f"exit_code: {rc}\n"
            f"[stdout/stderr below]\n"
        )
        body, truncated = _truncate(prelude + output, ctx.max_output_bytes)

        # 5. Side-effect summary.
        side = f"ran '{_cmd_summary(cmd)}' in {_format_cwd(ctx.cwd)}"
        side_effects: tuple[str, ...]
        if timed_out:
            side_effects = (side, f"timeout after {timeout_s:g}s — process killed")
        elif truncated:
            side_effects = (side, "output truncated")
        else:
            side_effects = (side,)

        is_error = timed_out or (rc != 0)
        return ToolResult(
            content=body,
            is_error=is_error,
            side_effects=side_effects,
        )


__all__ = ["BashTool"]
