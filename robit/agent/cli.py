"""robit.agent.cli — `robit` binary entry point.

Three modes:

* ``robit``                  — Textual REPL (default).
* ``robit --version``        — print version, exit 0.
* ``robit --session <id>``   — resume a saved session in the REPL.
* ``robit "task description"`` — one-shot mode: drive a single turn,
                                     print output, exit. The mock LLM ships
                                     in Wave 15.0 so this works offline.

Mock LLM behaviour
------------------

When the environment variable ``ENCHANTER_AGENT_MOCK`` is set to ``1``
(or no real upstream credentials are configured), the loop's
``dispatch_fn`` is swapped for a deterministic stub: any user text that
starts with the word ``echo`` produces a single ``tool_use`` call to the
``echo`` tool; everything else produces a one-line text response. This
lets Wave 15.0 demonstrate the end-to-end loop without a real provider.

Wave 15.1 will flip the default to the real pipeline once a configurable
upstream is wired in.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Union

import robit
from robit.proxy.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
    ToolUsePart,
)
from robit.proxy.pipeline import PipelineResult, VetoResult

from .conversation import Conversation
from .loop import AgentLoop, AssistantTextDelta, ToolCallExecuted, TurnComplete, VetoFired
from .session import SessionWriter, load_session, session_dir
from .slash import SlashContext, builtin_registry, dispatch_slash
from .tools import EchoTool, ToolRegistry, default_registry


DEFAULT_MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Mock LLM dispatch — Wave 15.0 only.
# ---------------------------------------------------------------------------


async def mock_dispatch(req) -> Union[PipelineResult, VetoResult]:
    """Deterministic stand-in for the proxy pipeline.

    Routing rules:
    * Last user message text starts with ``echo `` (case-insensitive): emit
      one ``tool_use`` part calling ``echo`` with that text.
    * Otherwise: respond with a one-line acknowledgement, no tool use.
    * After a tool_result has come back, respond with a final summary and
      stop (avoid infinite loops).
    """
    # Walk messages backwards to learn the latest situation.
    last_user_text = ""
    saw_tool_result = False
    for msg in reversed(req.messages):
        if msg.role != "user":
            continue
        from robit.proxy.canonical import ToolResultPart
        for part in msg.content:
            if isinstance(part, ToolResultPart):
                saw_tool_result = True
                break
            if isinstance(part, TextPart):
                last_user_text = part.text
        if last_user_text or saw_tool_result:
            break

    if saw_tool_result:
        content = (
            TextPart(text="Done."),
        )
        stop = "end_turn"
    elif last_user_text.lower().startswith("echo"):
        # Strip the leading "echo " when populating the tool args.
        payload = last_user_text[5:] if len(last_user_text) > 5 else last_user_text
        content = (
            TextPart(text="I'll echo that back for you."),
            ToolUsePart(id="mock_tool_1", name="echo", input={"text": payload}),
        )
        stop = "tool_use"
    else:
        content = (TextPart(text=f"(mock) acknowledged: {last_user_text[:80]}"),)
        stop = "end_turn"

    resp = CanonicalResponse(
        model=req.model,
        content=content,
        stop_reason=stop,
        usage=CanonicalUsage(input_tokens=1, output_tokens=1),
    )
    return PipelineResult(response=resp, fired=())


def _should_use_mock() -> bool:
    if os.environ.get("ENCHANTER_AGENT_MOCK") == "1":
        return True
    # No real provider credentials → mock so the CLI is offline-safe.
    return not any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_prompt() -> str | None:
    path = Path(__file__).with_name("prompts") / "system.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _build_loop(
    *, model: str, session_id: str | None, resume: bool
) -> tuple[AgentLoop, SlashContext, SessionWriter]:
    # Wave 15.1: load all production tools (file_read/write/edit, glob, grep,
    # bash, web_fetch). EchoTool stays available for smoke runs.
    registry = default_registry(include_echo=True)

    if resume and session_id is not None:
        conv = load_session(session_id)
        # Resume preserves the previous model unless --model was given here.
        if model != DEFAULT_MODEL:
            conv = conv.with_model(model)
    else:
        conv = Conversation.new(
            model=model,
            system_prompt=_system_prompt(),
            session_id=session_id,
        )

    writer = SessionWriter.for_session(conv.session_id)
    # Header is idempotent-ish: only write if file is empty.
    if writer.path.stat().st_size == 0 if writer.path.exists() else True:
        if not writer.path.exists() or writer.path.stat().st_size == 0:
            writer.write_header(conv)

    dispatch = mock_dispatch if _should_use_mock() else None
    loop = AgentLoop(
        conversation=conv,
        tool_registry=registry,
        session_writer=writer,
    )
    if dispatch is not None:
        loop.dispatch_fn = dispatch  # type: ignore[assignment]

    slash_ctx = SlashContext(
        conversation=conv,
        tool_registry=registry,
        audit_dir=session_dir(),
    )
    return loop, slash_ctx, writer


def _err(msg: str) -> None:
    try:
        sys.stderr.write(msg + "\n")
    except UnicodeEncodeError:
        sys.stderr.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def _write(msg: str) -> None:
    try:
        sys.stdout.write(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(msg.encode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# One-shot driver
# ---------------------------------------------------------------------------


async def _one_shot(loop: AgentLoop, task: str) -> int:
    async for ev in loop.run_turn(task):
        if isinstance(ev, AssistantTextDelta):
            _write(ev.text + "\n")
        elif isinstance(ev, ToolCallExecuted):
            tag = "ERROR" if ev.is_error else "ok"
            _write(f"[tool {ev.tool_name} {tag}] {ev.result}\n")
        elif isinstance(ev, VetoFired):
            _err(f"vetoed by {ev.plugin}: {ev.reason}")
            return 1
        elif isinstance(ev, TurnComplete):
            _write(
                f"\n--- turn done (iters={ev.iterations}, "
                f"tokens={ev.usage.input_tokens}+{ev.usage.output_tokens}) ---\n"
            )
    return 0


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robit",
        description=(
            "Enchanter coding agent — REPL with enchanter enforcement baked in."
        ),
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    p.add_argument(
        "--session",
        metavar="ID",
        help="Resume a saved session by id.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model id for new sessions (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "task",
        nargs="?",
        help="Optional one-shot task. If omitted, drop into the REPL.",
    )
    return p


def _build_login_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robit login",
        description="Authenticate with a subscription provider.",
    )
    p.add_argument(
        "provider",
        nargs="?",
        choices=["chatgpt", "anthropic"],
        help="Provider to authenticate with.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List cached tokens instead of running a flow.",
    )
    return p


def _build_logout_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robit logout",
        description="Delete cached subscription tokens.",
    )
    p.add_argument(
        "provider",
        nargs="?",
        choices=["chatgpt", "anthropic"],
        help="Provider whose token to delete.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Delete every cached token.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env files before anything reads os.environ (e.g. API keys,
    # mock-mode flags). Silent no-op if no .env files exist.
    from robit._env import load_env_files
    load_env_files()

    if argv is None:
        argv = sys.argv[1:]

    # Subcommand dispatch — must run before the positional `task` parser so
    # `robit login` / `robit logout` aren't swallowed as one-shot
    # task strings.
    if argv and argv[0] == "login":
        from .login import run_login
        return run_login(_build_login_parser().parse_args(argv[1:]))
    if argv and argv[0] == "logout":
        from .login import run_logout
        return run_logout(_build_logout_parser().parse_args(argv[1:]))

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        _write(f"robit {robit.__version__}\n")
        return 0

    try:
        loop, slash_ctx, _writer = _build_loop(
            model=args.model,
            session_id=args.session,
            resume=bool(args.session),
        )
    except FileNotFoundError as exc:
        _err(f"error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to initialise agent: {exc}")
        return 2

    if args.task:
        try:
            return asyncio.run(_one_shot(loop, args.task))
        except KeyboardInterrupt:
            return 130

    # REPL mode — import Textual on demand so --version / one-shot don't pay.
    try:
        from .app import launch
    except Exception as exc:  # noqa: BLE001
        _err(f"error: failed to import Textual app: {exc}")
        return 2

    try:
        launch(loop, slash_ctx)
    except KeyboardInterrupt:
        return 0
    return 0


__all__ = ["main", "mock_dispatch", "DEFAULT_MODEL"]
