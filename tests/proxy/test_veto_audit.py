"""Tests for robit.proxy._veto_audit — durable veto audit log.

Covers the writer/reader unit surface and the pipeline integration: a
destructive prompt vetoed at trust-gate must leave one durable JSONL record
answering the delegation-audit section 9 gap (4) question — "why did this
request get a 451, and which pattern fired?"
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from robit.proxy import _veto_audit, upstream
from robit.proxy.canonical import CanonicalRequest, Message, TextPart
from robit.proxy.pipeline import PipelineOptions, VetoResult, run, stream


@pytest.fixture(autouse=True)
def _redirect_state_dir(tmp_path, monkeypatch):
    """Point every test at a per-test temp state dir + reset module state."""
    monkeypatch.setenv("ROBIT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(_veto_audit, "_WRITE_LOCK", asyncio.Lock())
    monkeypatch.setattr(_veto_audit, "_FALLBACK_WARNED", False)
    yield


def _expected_audit_file(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "vetoes.jsonl"


def _req(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text=text),)),),
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_path_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBIT_STATE_DIR", str(tmp_path))
    assert _veto_audit.get_audit_path() == _expected_audit_file(tmp_path)


def test_path_repo_root_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBIT_STATE_DIR", raising=False)
    repo_root = tmp_path / "fakerepo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    sub = repo_root / "deep" / "nested"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    p = _veto_audit.get_audit_path()
    assert p == repo_root / "state" / "audit" / "vetoes.jsonl"


# ---------------------------------------------------------------------------
# Write + read roundtrip
# ---------------------------------------------------------------------------

async def test_record_and_read_roundtrip(tmp_path):
    await _veto_audit.record_veto(
        correlation_id="c-123",
        engine="destructive-op-gate",
        phase="trust-gate",
        reason="destructive-op-gate:w5-force-push",
        pattern_id="w5-force-push",
        http_status=451,
    )

    records = await _veto_audit.read_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "proxy.request.vetoed"
    assert rec["correlation_id"] == "c-123"
    assert rec["engine"] == "destructive-op-gate"
    assert rec["phase"] == "trust-gate"
    assert rec["pattern_id"] == "w5-force-push"
    assert rec["http_status"] == 451
    assert rec["mode"] == "pre-dispatch"
    assert isinstance(rec["ts"], float)


async def test_read_skips_corrupt_lines(tmp_path):
    await _veto_audit.record_veto(
        correlation_id="c-1",
        engine="cve-pattern-gate",
        phase="trust-gate",
        reason="cve-pattern-gate:cve-2024-x",
    )
    path = _expected_audit_file(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    await _veto_audit.record_veto(
        correlation_id="c-2",
        engine="cve-pattern-gate",
        phase="trust-gate",
        reason="cve-pattern-gate:cve-2024-y",
    )

    records = await _veto_audit.read_records()
    assert [r["correlation_id"] for r in records] == ["c-1", "c-2"]


async def test_read_since_filters(tmp_path):
    await _veto_audit.record_veto(
        correlation_id="c-old",
        engine="destructive-op-gate",
        phase="trust-gate",
        reason="r",
    )
    records = await _veto_audit.read_records()
    cutoff = records[0]["ts"] + 0.001

    await _veto_audit.record_veto(
        correlation_id="c-new",
        engine="destructive-op-gate",
        phase="trust-gate",
        reason="r",
    )
    recent = await _veto_audit.read_records(since=cutoff)
    assert [r["correlation_id"] for r in recent] == ["c-new"]


async def test_record_never_raises_on_unwritable_dir(tmp_path, monkeypatch):
    """Audit is best-effort: a broken sink must not abort the veto path."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("ROBIT_STATE_DIR", str(blocker / "sub"))

    await _veto_audit.record_veto(
        correlation_id="c-x",
        engine="destructive-op-gate",
        phase="trust-gate",
        reason="r",
    )  # must not raise; falls back to tempdir


# ---------------------------------------------------------------------------
# Pipeline integration — the actual compliance question
# ---------------------------------------------------------------------------

async def test_run_veto_lands_durable_audit_record(tmp_path):
    """Non-streaming: a destructive prompt yields a VetoResult AND one
    durable vetoes.jsonl line carrying engine + pattern + 451."""
    mock_acomp = AsyncMock()
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        result = await run(
            _req("please run git push --force on main"),
            PipelineOptions(conduct=False),
        )

    assert isinstance(result, VetoResult)
    assert mock_acomp.await_count == 0

    records = await _veto_audit.read_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["engine"] == "destructive-op-gate"
    assert rec["phase"] == "trust-gate"
    assert rec["pattern_id"] == "w5-force-push"
    assert rec["http_status"] == 451
    assert rec["mode"] == "pre-dispatch"
    assert rec["correlation_id"]  # non-empty, joinable to bus events


async def test_stream_veto_lands_durable_audit_record(tmp_path):
    """Streaming: the synchronous trust-gate veto also lands one record."""
    result = await stream(
        _req("please run git push --force on main"),
        PipelineOptions(conduct=False),
    )

    assert isinstance(result, VetoResult)

    records = await _veto_audit.read_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["engine"] == "destructive-op-gate"
    assert rec["http_status"] == 451
    assert rec["mode"] == "pre-dispatch"


async def test_benign_run_writes_no_audit_record(tmp_path):
    """No veto — no record; the sink stays quiet on the happy path."""
    from types import SimpleNamespace

    message = SimpleNamespace(content="hello there", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3)
    fake = SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini")

    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        result = await run(_req("how are you"), PipelineOptions(conduct=False))

    assert not isinstance(result, VetoResult)
    assert await _veto_audit.read_records() == []
    assert not _expected_audit_file(tmp_path).exists()
