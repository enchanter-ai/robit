"""Tests for enchanter.loader.runtimes._audit — sidecar rejection audit log."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from enchanter.loader.runtimes import _audit


@pytest.fixture(autouse=True)
def _redirect_state_dir(tmp_path, monkeypatch):
    """Point every test at a per-test temp state dir + reset module locks/state.

    The autouse env-var redirect protects the production audit file from being
    polluted by tests; the per-test ``asyncio.Lock`` swap keeps tests that
    share an event-loop policy isolated from each other.
    """
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    # Replace the module's lock per test so each test gets a fresh, unlocked
    # lock attached to its own event loop.
    monkeypatch.setattr(_audit, "_WRITE_LOCK", asyncio.Lock())
    monkeypatch.setattr(_audit, "_FALLBACK_WARNED", False)
    yield


def _expected_audit_file(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "sidecar-rejections.jsonl"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_path_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    p = _audit.get_audit_path()
    assert p == tmp_path / "audit" / "sidecar-rejections.jsonl"


def test_path_repo_root_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ENCHANTER_STATE_DIR", raising=False)
    repo_root = tmp_path / "fakerepo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    sub = repo_root / "deep" / "nested"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    p = _audit.get_audit_path()
    assert p == repo_root / "state" / "audit" / "sidecar-rejections.jsonl"


def test_path_platform_default_when_no_env_no_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("ENCHANTER_STATE_DIR", raising=False)
    isolated = tmp_path / "no-repo-here"
    isolated.mkdir()
    monkeypatch.chdir(isolated)

    # Force the resolver to think it's outside any repo by stubbing the walker.
    monkeypatch.setattr(_audit, "_find_repo_root", lambda _start: None)

    p = _audit.get_audit_path()
    if sys.platform.startswith("win"):
        assert "enchanter" in str(p).lower() and "audit" in str(p).lower()
    else:
        assert p.parent.name == "audit"
        assert ".enchanter" in str(p)


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_rejection_writes_one_line(tmp_path):
    await _audit.record_rejection(
        "intent-anchor",
        "source-forgery",
        {"source": "orchestrator", "topic": "evil"},
        expected={"allowed_sources": ["intent-anchor"]},
    )

    f = _expected_audit_file(tmp_path)
    assert f.is_file()
    contents = f.read_text(encoding="utf-8")
    lines = contents.splitlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["kind"] == "sidecar.derived_event.rejected"
    assert rec["adapter_name"] == "intent-anchor"
    assert rec["rejection_reason"] == "source-forgery"
    assert rec["raw_event"] == {"source": "orchestrator", "topic": "evil"}
    assert rec["expected"] == {"allowed_sources": ["intent-anchor"]}
    assert isinstance(rec["ts"], float)
    assert isinstance(rec["ts_iso"], str)


@pytest.mark.asyncio
async def test_two_successive_calls_produce_two_parseable_lines(tmp_path):
    await _audit.record_rejection("a", "source-forgery", {"x": 1})
    await _audit.record_rejection("b", "undeclared-topic", {"y": 2})

    f = _expected_audit_file(tmp_path)
    lines = f.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    r0, r1 = json.loads(lines[0]), json.loads(lines[1])
    assert r0["adapter_name"] == "a"
    assert r1["adapter_name"] == "b"
    assert r0["rejection_reason"] == "source-forgery"
    assert r1["rejection_reason"] == "undeclared-topic"


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_records_round_trip(tmp_path):
    await _audit.record_rejection("alpha", "malformed-event", {"a": 1})
    await _audit.record_rejection("beta", "phase-out-of-scope", {"b": 2})

    records = await _audit.read_records()
    assert len(records) == 2
    assert [r["adapter_name"] for r in records] == ["alpha", "beta"]
    assert [r["rejection_reason"] for r in records] == [
        "malformed-event",
        "phase-out-of-scope",
    ]


@pytest.mark.asyncio
async def test_read_records_since_filters_by_timestamp(tmp_path):
    await _audit.record_rejection("first", "source-forgery", {"n": 1})
    # Capture the cutoff between the two writes.
    all_after_first = await _audit.read_records()
    cutoff = all_after_first[0]["ts"]
    # Small bump so the second record is strictly after the cutoff.
    await asyncio.sleep(0.01)
    await _audit.record_rejection("second", "source-forgery", {"n": 2})

    filtered = await _audit.read_records(since=cutoff + 0.001)
    assert len(filtered) == 1
    assert filtered[0]["adapter_name"] == "second"

    # since=0 -> everything
    everything = await _audit.read_records(since=0)
    assert len(everything) == 2


# ---------------------------------------------------------------------------
# Failure tolerance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disk_full_does_not_raise_and_warns_once(tmp_path, monkeypatch, caplog):
    """Simulate a write failure: caller must not see the exception."""

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    # Patch the sync writer so the asyncio.to_thread call raises.
    monkeypatch.setattr(_audit, "_write_line_sync", _boom)

    with caplog.at_level("WARNING", logger="enchanter.loader.runtimes._audit"):
        # Two calls -- both must complete cleanly.
        await _audit.record_rejection("x", "source-forgery", {"e": 1})
        await _audit.record_rejection("x", "source-forgery", {"e": 2})

    # The write was patched out, so no file ever materialised.
    f = _expected_audit_file(tmp_path)
    assert not f.is_file()
    # At least one WARNING surfaced about the failed write.
    warn_messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("audit write" in m for m in warn_messages)


@pytest.mark.asyncio
async def test_non_serialisable_raw_event_uses_placeholder(tmp_path):
    class WeirdObject:
        def __repr__(self):
            return "<WeirdObject>"

    raw = {"ok": 1, "bad": WeirdObject()}
    await _audit.record_rejection("adapter", "malformed-event", raw)

    f = _expected_audit_file(tmp_path)
    lines = f.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # Top-level raw_event was un-serialisable as a whole, so the placeholder
    # replaces it.
    assert isinstance(rec["raw_event"], dict)
    assert "_encode_error" in rec["raw_event"]


@pytest.mark.asyncio
async def test_complex_number_raw_event_uses_placeholder(tmp_path):
    """A ``complex`` is a stdlib type but still not JSON-serialisable."""
    await _audit.record_rejection("adapter", "malformed-event", {"v": complex(1, 2)})

    f = _expected_audit_file(tmp_path)
    rec = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(rec["raw_event"], dict)
    assert "_encode_error" in rec["raw_event"]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_calls_all_land(tmp_path):
    N = 20

    async def one(i: int) -> None:
        await _audit.record_rejection(
            f"adapter-{i}",
            "source-forgery",
            {"i": i},
        )

    await asyncio.gather(*(one(i) for i in range(N)))

    f = _expected_audit_file(tmp_path)
    raw = f.read_text(encoding="utf-8")
    # File ends in a newline -> splitlines drops the trailing empty.
    lines = raw.splitlines()
    assert len(lines) == N

    parsed = [json.loads(line) for line in lines]
    seen_indices = sorted(r["raw_event"]["i"] for r in parsed)
    assert seen_indices == list(range(N))


# ---------------------------------------------------------------------------
# fsync opt-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fsync_env_var_triggers_fsync(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCHANTER_AUDIT_FSYNC", "1")
    with mock.patch("enchanter.loader.runtimes._audit.os.fsync") as m_fsync:
        await _audit.record_rejection("a", "source-forgery", {"x": 1})
    assert m_fsync.called, "fsync should have been invoked when env var is set"


@pytest.mark.asyncio
async def test_fsync_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv("ENCHANTER_AUDIT_FSYNC", raising=False)
    with mock.patch("enchanter.loader.runtimes._audit.os.fsync") as m_fsync:
        await _audit.record_rejection("a", "source-forgery", {"x": 1})
    assert not m_fsync.called, "fsync must NOT be invoked by default"
