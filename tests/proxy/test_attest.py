"""Tests for enchanter.proxy.attest — Mimir attestation emit hook.

Covers the opt-in gate, the issuer body contract, spool behaviour on issuer
success/failure, and the pipeline integration: with attestation enabled,
both pass and veto decisions land in the decisions.jsonl spool — the proof
heartbeat the SPRT liveness reader consumes.
"""

from __future__ import annotations

import asyncio
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from enchanter.proxy import attest, upstream
from enchanter.proxy.canonical import CanonicalRequest, Message, TextPart
from enchanter.proxy.pipeline import PipelineOptions, PipelineResult, VetoResult, run, stream


FAKE_ENVELOPE = {
    "version": "2.1",
    "tool_call_id": "tc-1",
    "tool_id": "did:web:enchanter-labs.dev:robit:proxy-gate",
    "signature": {"protected_header": {"alg": "EdDSA", "key_id": "k1"}, "value": "sig"},
}


@pytest.fixture(autouse=True)
def _redirect_state_dir(tmp_path, monkeypatch):
    """Per-test temp state dir; attestation disabled unless a test enables it."""
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("ENCHANTER_ATTEST_ENABLED", raising=False)
    monkeypatch.setattr(attest, "_WRITE_LOCK", asyncio.Lock())
    monkeypatch.setattr(attest, "_FALLBACK_WARNED", False)
    yield


@pytest.fixture
def _enabled(monkeypatch):
    monkeypatch.setenv("ENCHANTER_ATTEST_ENABLED", "1")
    yield


@pytest.fixture
def _issuer_ok(monkeypatch):
    """Patch the issuer POST to succeed; captures every body sent."""
    bodies: list[dict] = []

    def fake_post(body: dict) -> dict:
        bodies.append(body)
        return {"envelope": FAKE_ENVELOPE, "validation_level": "cryptographically_valid"}

    monkeypatch.setattr(attest, "_post_attest", fake_post)
    return bodies


def _req(text: str = "hello") -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text=text),)),),
    )


def _fake_completion(text: str = "hi"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------

async def test_disabled_is_noop(tmp_path, _issuer_ok):
    await attest.attest_decision(
        _req(),
        correlation_id="c-1",
        decision="pass",
        phase="post-session",
    )
    assert _issuer_ok == []  # no POST
    assert await attest.read_spool() == []
    assert not attest.get_spool_path().exists()


def test_is_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("ENCHANTER_ATTEST_ENABLED", raising=False)
    assert not attest.is_enabled()
    monkeypatch.setenv("ENCHANTER_ATTEST_ENABLED", "1")
    assert attest.is_enabled()


# ---------------------------------------------------------------------------
# Issuer body contract
# ---------------------------------------------------------------------------

async def test_body_shape_matches_attest_request(_enabled, _issuer_ok):
    await attest.attest_decision(
        _req("some prompt"),
        correlation_id="c-42",
        session_id="s-7",
        decision="veto",
        engine="destructive-op-gate",
        phase="trust-gate",
        reason="destructive-op-gate:w5-force-push",
        pattern_id="w5-force-push",
        http_status=451,
    )

    assert len(_issuer_ok) == 1
    body = _issuer_ok[0]
    assert body["tool_id"] == "did:web:enchanter-labs.dev:robit:proxy-gate"
    assert body["tool_version"]

    breq = body["request"]
    assert breq["correlation_id"] == "c-42"
    assert breq["session_id"] == "s-7"
    assert breq["model"] == "gpt-4o-mini"
    # Privacy invariant: digest only, never prompt content.
    assert len(breq["request_sha256"]) == 64
    assert "some prompt" not in str(body)

    bres = body["result"]
    assert bres["decision"] == "veto"
    assert bres["engine"] == "destructive-op-gate"
    assert bres["pattern_id"] == "w5-force-push"
    assert bres["http_status"] == 451


async def test_request_digest_is_deterministic(_enabled, _issuer_ok):
    await attest.attest_decision(
        _req("same"), correlation_id="c-1", decision="pass", phase="post-session"
    )
    await attest.attest_decision(
        _req("same"), correlation_id="c-2", decision="pass", phase="post-session"
    )
    await attest.attest_decision(
        _req("different"), correlation_id="c-3", decision="pass", phase="post-session"
    )
    d1, d2, d3 = (b["request"]["request_sha256"] for b in _issuer_ok)
    assert d1 == d2
    assert d1 != d3


# ---------------------------------------------------------------------------
# Spool behaviour
# ---------------------------------------------------------------------------

async def test_success_spools_envelope(_enabled, _issuer_ok):
    await attest.attest_decision(
        _req(),
        correlation_id="c-1",
        decision="pass",
        phase="post-session",
        http_status=200,
    )

    records = await attest.read_spool()
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "proxy.decision.attested"
    assert rec["decision"] == "pass"
    assert rec["envelope"] == FAKE_ENVELOPE
    assert rec["validation_level"] == "cryptographically_valid"
    assert rec["error"] is None


async def test_issuer_failure_spools_error_and_never_raises(_enabled, monkeypatch):
    def broken_post(body: dict) -> dict:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(attest, "_post_attest", broken_post)

    await attest.attest_decision(
        _req(),
        correlation_id="c-1",
        decision="veto",
        engine="destructive-op-gate",
        phase="trust-gate",
    )  # must not raise

    records = await attest.read_spool()
    assert len(records) == 1
    rec = records[0]
    assert rec["envelope"] is None
    assert "URLError" in rec["error"]
    # The decision metadata survives even without an envelope — the spool
    # line is still a liveness heartbeat.
    assert rec["decision"] == "veto"
    assert rec["engine"] == "destructive-op-gate"


# ---------------------------------------------------------------------------
# Pipeline integration — the proof heartbeat
# ---------------------------------------------------------------------------

async def test_run_veto_attests_decision(_enabled, _issuer_ok):
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock()):
        result = await run(
            _req("please run git push --force on main"),
            PipelineOptions(conduct=False),
        )

    assert isinstance(result, VetoResult)
    records = await attest.read_spool()
    assert len(records) == 1
    rec = records[0]
    assert rec["decision"] == "veto"
    assert rec["engine"] == "destructive-op-gate"
    assert rec["pattern_id"] == "w5-force-push"
    assert rec["http_status"] == 451
    assert rec["envelope"] == FAKE_ENVELOPE


async def test_run_pass_attests_decision(_enabled, _issuer_ok):
    fake = _fake_completion("hello there")
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        result = await run(_req("how are you"), PipelineOptions(conduct=False))

    assert isinstance(result, PipelineResult)
    records = await attest.read_spool()
    assert len(records) == 1
    rec = records[0]
    assert rec["decision"] == "pass"
    assert rec["engine"] is None
    assert rec["http_status"] == 200
    assert rec["envelope"] == FAKE_ENVELOPE


async def test_stream_veto_attests_decision(_enabled, _issuer_ok):
    result = await stream(
        _req("please run git push --force on main"),
        PipelineOptions(conduct=False),
    )

    assert isinstance(result, VetoResult)
    records = await attest.read_spool()
    assert len(records) == 1
    assert records[0]["decision"] == "veto"
    assert records[0]["engine"] == "destructive-op-gate"


async def test_pipeline_disabled_attest_writes_nothing(_issuer_ok):
    """Default-off: the pipeline never POSTs or spools without the env gate."""
    fake = _fake_completion()
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        await run(_req("how are you"), PipelineOptions(conduct=False))

    assert _issuer_ok == []
    assert await attest.read_spool() == []
