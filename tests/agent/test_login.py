"""Tests for robit.agent.login — CLI auth subcommands (Wave 17.1).

The PKCE flow is mocked at the `robit.llm._chatgpt_auth.run_pkce_flow`
seam so no real browser opens. `ROBIT_HOME` is redirected to `tmp_path`
by the autouse `isolated_enchanter_home` fixture in conftest.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robit.agent import login as login_mod
from robit.agent.cli import main as cli_main
from robit.llm._chatgpt_auth import (
    AuthDeniedError,
    AuthTimeoutError,
    ChatGptToken,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_token(account: str = "acct_test_123") -> ChatGptToken:
    return ChatGptToken(
        access_token="access-abcdefgh-1234",
        refresh_token="refresh-zzz",
        id_token="id.tok.en",
        expires_at=time.time() + 3600.0,
        chatgpt_account_id=account,
    )


def _install_pkce(monkeypatch, *, result=None, raises=None) -> None:
    """Patch run_pkce_flow on the login module's import binding."""

    async def fake_flow(**_kwargs):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(login_mod, "run_pkce_flow", fake_flow)


# ---------------------------------------------------------------------------
# login chatgpt — happy path
# ---------------------------------------------------------------------------


def test_login_chatgpt_happy_path(tmp_path, monkeypatch, capsys):
    token = _make_fake_token("acct_happy")
    _install_pkce(monkeypatch, result=token)

    rc = cli_main(["login", "chatgpt"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Opening browser" in out
    assert "acct_happy" in out

    # Token file landed at $ROBIT_HOME/chatgpt-token.json.
    cache = tmp_path / "chatgpt-token.json"
    assert cache.exists()
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["access_token"] == "access-abcdefgh-1234"
    assert data["chatgpt_account_id"] == "acct_happy"


# ---------------------------------------------------------------------------
# login chatgpt — error paths
# ---------------------------------------------------------------------------


def test_login_chatgpt_timeout_returns_2(tmp_path, monkeypatch, capsys):
    _install_pkce(monkeypatch, raises=AuthTimeoutError("no callback in 300s"))

    rc = cli_main(["login", "chatgpt"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "timed out" in err.lower()
    assert "retry" in err.lower()
    # No token persisted on failure.
    assert not (tmp_path / "chatgpt-token.json").exists()


def test_login_chatgpt_denied_returns_1(tmp_path, monkeypatch, capsys):
    _install_pkce(monkeypatch, raises=AuthDeniedError("user denied consent"))

    rc = cli_main(["login", "chatgpt"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "denied" in err.lower()
    assert not (tmp_path / "chatgpt-token.json").exists()


# ---------------------------------------------------------------------------
# login anthropic — v1 stub
# ---------------------------------------------------------------------------


def test_login_anthropic_prints_stub_and_returns_0(capsys):
    rc = cli_main(["login", "anthropic"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Claude Code" in out
    assert "CLAUDE_CODE_OAUTH_TOKEN" in out
    assert "ANTHROPIC_API_KEY" in out


# ---------------------------------------------------------------------------
# login --list
# ---------------------------------------------------------------------------


def test_login_list_empty(tmp_path, capsys):
    rc = cli_main(["login", "--list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no cached tokens" in out


def test_login_list_shows_provider_and_redacted_info(tmp_path, capsys):
    # Seed a token file for chatgpt directly via the public save_token path.
    token = _make_fake_token("acct_list_42")
    login_mod.save_token(token, path=login_mod.token_path("chatgpt"))

    rc = cli_main(["login", "--list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "chatgpt" in out
    assert "acct_list_42" in out
    # Access token is redacted (not the full secret).
    assert "access-abcdefgh-1234" not in out
    # Expiry hint appears (token expires in ~1h).
    assert "expires" in out.lower()


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


def test_logout_chatgpt_deletes_cached_file(tmp_path, capsys):
    token = _make_fake_token()
    login_mod.save_token(token, path=login_mod.token_path("chatgpt"))
    assert (tmp_path / "chatgpt-token.json").exists()

    rc = cli_main(["logout", "chatgpt"])

    assert rc == 0
    assert not (tmp_path / "chatgpt-token.json").exists()
    out = capsys.readouterr().out
    assert "removed" in out.lower()


def test_logout_all_deletes_every_cached_token(tmp_path, capsys):
    login_mod.save_token(_make_fake_token(), path=login_mod.token_path("chatgpt"))
    # Anthropic token cache: shape doesn't have to be valid; logout only unlinks.
    (tmp_path / "anthropic-token.json").write_text("{}", encoding="utf-8")

    rc = cli_main(["logout", "--all"])

    assert rc == 0
    assert not (tmp_path / "chatgpt-token.json").exists()
    assert not (tmp_path / "anthropic-token.json").exists()
    out = capsys.readouterr().out
    assert "chatgpt" in out
    assert "anthropic" in out


def test_logout_chatgpt_missing_file_is_noop(tmp_path, capsys):
    assert not (tmp_path / "chatgpt-token.json").exists()

    rc = cli_main(["logout", "chatgpt"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no token to remove" in out.lower()


# ---------------------------------------------------------------------------
# CLI argparse integration — `robit login --help` shows new subcommands
# ---------------------------------------------------------------------------


def test_login_help_lists_provider_and_list_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["login", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "chatgpt" in out
    assert "anthropic" in out
    assert "--list" in out


def test_logout_help_lists_provider_and_all_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["logout", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "chatgpt" in out
    assert "--all" in out


def test_login_without_provider_or_list_returns_1(capsys):
    """`robit login` with no args is a usage error (exit 1)."""
    rc = cli_main(["login"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "provider" in err.lower() or "list" in err.lower()
