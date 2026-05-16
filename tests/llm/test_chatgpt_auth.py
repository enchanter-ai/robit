"""Tests for robit.llm._chatgpt_auth — mocked; no network, no browser."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from unittest.mock import patch

import pytest

from robit.llm._chatgpt_auth import (
    AuthError,
    ChatGptToken,
    _extract_account_id,
    generate_pkce_pair,
    load_cached_token,
    refresh_if_needed,
    save_token,
)


# ---------------------------------------------------------------------------
# PKCE primitive
# ---------------------------------------------------------------------------


def test_generate_pkce_pair_verifier_and_challenge_are_well_formed() -> None:
    verifier, challenge = generate_pkce_pair()

    # RFC 7636 §4.1: verifier is 43-128 chars from the URL-safe alphabet.
    assert 43 <= len(verifier) <= 128
    assert all(c.isalnum() or c in "-._~" for c in verifier)

    # Challenge must equal base64url(sha256(verifier)) with no padding.
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected


def test_generate_pkce_pair_is_unique_per_call() -> None:
    a, _ = generate_pkce_pair()
    b, _ = generate_pkce_pair()
    assert a != b


# ---------------------------------------------------------------------------
# Token cache I/O
# ---------------------------------------------------------------------------


def test_load_cached_token_returns_none_when_missing(tmp_path) -> None:
    assert load_cached_token(tmp_path / "missing.json") is None


def test_load_cached_token_returns_none_when_malformed(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not-json{", encoding="utf-8")
    assert load_cached_token(p) is None

    # Also malformed: valid JSON but missing required key.
    p.write_text(json.dumps({"refresh_token": "r"}), encoding="utf-8")
    assert load_cached_token(p) is None


def test_save_and_load_token_round_trip(tmp_path) -> None:
    path = tmp_path / "sub" / "chatgpt-token.json"
    expires = time.time() + 3600
    token = ChatGptToken(
        access_token="access-abc",
        refresh_token="refresh-xyz",
        id_token="id-jwt",
        expires_at=expires,
        chatgpt_account_id="acct-123",
    )
    save_token(token, path)

    loaded = load_cached_token(path)
    assert loaded == token


# ---------------------------------------------------------------------------
# JWT claim extraction
# ---------------------------------------------------------------------------


def _make_id_token(payload: dict) -> str:
    """Build a fake JWT (header.payload.signature) — no real signing."""
    def b64(d: dict) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(d).encode("utf-8"))
            .rstrip(b"=")
            .decode("ascii")
        )
    return f"{b64({'alg': 'none'})}.{b64(payload)}.sig"


def test_extract_account_id_reads_namespaced_claim() -> None:
    jwt = _make_id_token(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-xyz"}}
    )
    assert _extract_account_id(jwt) == "acct-xyz"


def test_extract_account_id_returns_none_on_garbage() -> None:
    assert _extract_account_id(None) is None
    assert _extract_account_id("not.a.jwt-payload") is None
    assert _extract_account_id("only-one-segment") is None


# ---------------------------------------------------------------------------
# refresh_if_needed
# ---------------------------------------------------------------------------


def test_refresh_if_needed_returns_unchanged_when_valid() -> None:
    token = ChatGptToken(
        access_token="a",
        refresh_token="r",
        id_token=None,
        expires_at=time.time() + 3600,
        chatgpt_account_id=None,
    )
    out = asyncio.run(refresh_if_needed(token))
    assert out is token


def test_refresh_if_needed_exchanges_refresh_token_near_expiry() -> None:
    token = ChatGptToken(
        access_token="old",
        refresh_token="refresh-zzz",
        id_token=None,
        expires_at=time.time() + 10,  # within 60s skew → refresh
        chatgpt_account_id="acct-old",
    )

    captured: dict = {}

    def fake_post(payload, issuer):
        captured["payload"] = payload
        captured["issuer"] = issuer
        return {
            "access_token": "new-access",
            "refresh_token": None,  # IdP omits → keep old
            "id_token": None,
            "expires_in": 3600,
        }

    with patch("robit.llm._chatgpt_auth._post_token", side_effect=fake_post):
        out = asyncio.run(refresh_if_needed(token))

    assert captured["payload"]["grant_type"] == "refresh_token"
    assert captured["payload"]["refresh_token"] == "refresh-zzz"
    assert captured["payload"]["client_id"]
    assert out.access_token == "new-access"
    # IdP omitted refresh_token → original preserved
    assert out.refresh_token == "refresh-zzz"
    assert out.chatgpt_account_id == "acct-old"
    assert out.expires_at > time.time() + 60


def test_refresh_if_needed_raises_when_no_refresh_token() -> None:
    token = ChatGptToken(
        access_token="a",
        refresh_token=None,
        id_token=None,
        expires_at=time.time() + 5,
        chatgpt_account_id=None,
    )
    with pytest.raises(AuthError):
        asyncio.run(refresh_if_needed(token))
