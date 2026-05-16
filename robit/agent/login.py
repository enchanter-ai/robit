"""robit.agent.login — `robit login` / `logout` command implementations.

Wave 17.1 wires the existing PKCE OAuth helpers (Wave 16.2) into reachable
CLI commands. The PKCE flow itself lives in
``robit.llm._chatgpt_auth.run_pkce_flow`` — this module is the user
interface around it.

Token cache layout (matches ``_chatgpt_auth._default_token_path``):

    $ENCHANTER_HOME/<provider>-token.json     (when ENCHANTER_HOME is set)
    %APPDATA%/enchanter/<provider>-token.json (Windows fallback)
    ~/.enchanter/<provider>-token.json        (POSIX fallback)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from robit.llm._chatgpt_auth import (
    AuthDeniedError,
    AuthError,
    AuthTimeoutError,
    ChatGptToken,
    run_pkce_flow,
    save_token,
)

# Providers that have a token file under the enchanter home dir.
_PROVIDERS: tuple[str, ...] = ("chatgpt", "anthropic")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _enchanter_home() -> Path:
    """Return the directory where ``<provider>-token.json`` files live.

    Mirrors ``_chatgpt_auth._default_token_path``'s rules so the two stay
    in sync without importing a private helper.
    """
    override = os.environ.get("ENCHANTER_HOME")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "enchanter"
    return Path.home() / ".enchanter"


def token_path(provider: str) -> Path:
    """Return the token cache path for ``provider`` (no I/O)."""
    return _enchanter_home() / f"{provider}-token.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write(msg: str) -> None:
    try:
        sys.stdout.write(msg)
        if not msg.endswith("\n"):
            sys.stdout.write("\n")
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def _err(msg: str) -> None:
    try:
        sys.stderr.write(msg + "\n")
    except UnicodeEncodeError:
        sys.stderr.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# login chatgpt
# ---------------------------------------------------------------------------


async def login_chatgpt() -> int:
    """Run the ChatGPT PKCE flow and persist the resulting token.

    Returns:
        0 on success, 1 on user denial, 2 on timeout, 3 on other auth error.
    """
    _write("Opening browser to authorize ChatGPT subscription...")
    _write(
        "If the browser doesn't open, the flow will print the authorize URL"
        " from the PKCE helper."
    )
    try:
        token: ChatGptToken = await run_pkce_flow()
    except AuthTimeoutError as exc:
        _err(f"error: authentication timed out: {exc}")
        _err("retry with: robit login chatgpt")
        return 2
    except AuthDeniedError as exc:
        _err(f"error: authentication denied: {exc}")
        return 1
    except AuthError as exc:
        _err(f"error: authentication failed: {exc}")
        return 3

    save_token(token, path=token_path("chatgpt"))
    acct = token.chatgpt_account_id or "(unknown account)"
    _write(f"Logged in as ChatGPT account {acct}")
    return 0


# ---------------------------------------------------------------------------
# login anthropic — stub for v1
# ---------------------------------------------------------------------------


# TODO Wave 18: implement standalone Anthropic OAuth flow if feasible.
_ANTHROPIC_STUB = """\
Anthropic Pro/Max OAuth currently has no standalone enchanter flow.

To authenticate:
  1. Install Claude Code (https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
  2. Run `claude /login` and complete the browser flow
  3. Copy the token from the Claude Code config to your .env as
     CLAUDE_CODE_OAUTH_TOKEN=<token>

OR use a regular API key:
  Set ANTHROPIC_API_KEY=sk-ant-... in your .env

For ChatGPT subscription: `robit login chatgpt`
"""


def login_anthropic() -> int:
    """Print the v1 instructions for Anthropic auth.

    Returns 0 (informational).
    """
    _write(_ANTHROPIC_STUB)
    return 0


# ---------------------------------------------------------------------------
# login --list
# ---------------------------------------------------------------------------


def _redact(value: str | None) -> str:
    """Return a safe redacted display for a token-ish string."""
    if not value:
        return "(none)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def _summarise_token(path: Path) -> str:
    """Return a one-line summary for a token JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"  (unreadable: {exc})"
    expires_at = data.get("expires_at")
    if isinstance(expires_at, (int, float)):
        delta = expires_at - time.time()
        if delta <= 0:
            expiry = "expired"
        elif delta < 3600:
            expiry = f"expires in {int(delta // 60)}m"
        elif delta < 86400:
            expiry = f"expires in {int(delta // 3600)}h"
        else:
            expiry = f"expires in {int(delta // 86400)}d"
    else:
        expiry = "expiry unknown"
    acct = data.get("chatgpt_account_id")
    access = data.get("access_token")
    bits = [f"access={_redact(access)}", expiry]
    if acct:
        bits.insert(0, f"account={acct}")
    return "  " + ", ".join(bits)


def login_list() -> int:
    """Print which providers have cached tokens and basic metadata."""
    found: list[tuple[str, Path]] = []
    for provider in _PROVIDERS:
        p = token_path(provider)
        if p.exists():
            found.append((provider, p))
    if not found:
        _write("no cached tokens")
        return 0
    _write("cached tokens:")
    for provider, p in found:
        _write(f"- {provider} ({p})")
        _write(_summarise_token(p))
    return 0


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


def _remove_one(provider: str) -> bool:
    """Delete a single provider's token file. Returns True if removed."""
    p = token_path(provider)
    if not p.exists():
        return False
    try:
        p.unlink()
    except OSError as exc:
        _err(f"error: failed to delete {p}: {exc}")
        return False
    return True


def logout(provider: str | None = None, *, all_providers: bool = False) -> int:
    """Delete cached subscription token(s).

    Args:
        provider: a single provider id, or ``None`` when ``all_providers`` is set.
        all_providers: when True, delete every known provider's cache.

    Returns 0 always — missing files are a no-op with a friendly message.
    """
    targets: Iterable[str]
    if all_providers:
        targets = _PROVIDERS
    elif provider is not None:
        targets = (provider,)
    else:
        _err("error: logout requires a provider or --all")
        return 1

    removed_any = False
    for prov in targets:
        if _remove_one(prov):
            _write(f"removed cached token for {prov}")
            removed_any = True
        else:
            _write(f"no token to remove for {prov}")
    # Informational; never an error when --all on an empty cache.
    return 0 if (removed_any or all_providers or provider is not None) else 1


# ---------------------------------------------------------------------------
# argparse dispatch entry points (used by cli.py)
# ---------------------------------------------------------------------------


def run_login(args) -> int:
    """Dispatch the ``robit login`` subcommand."""
    if args.list:
        return login_list()
    if args.provider == "chatgpt":
        try:
            return asyncio.run(login_chatgpt())
        except KeyboardInterrupt:
            return 130
    if args.provider == "anthropic":
        return login_anthropic()
    _err("error: `robit login` requires a provider or --list")
    _err("usage: robit login {chatgpt,anthropic} | robit login --list")
    return 1


def run_logout(args) -> int:
    """Dispatch the ``robit logout`` subcommand."""
    if args.all:
        return logout(all_providers=True)
    if args.provider:
        return logout(args.provider)
    _err("error: `robit logout` requires a provider or --all")
    return 1


__all__ = [
    "login_chatgpt",
    "login_anthropic",
    "login_list",
    "logout",
    "run_login",
    "run_logout",
    "token_path",
]
