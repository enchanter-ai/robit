"""Tests for PIPELINE-OPS (G7 + G8) — operator dial + durable veto audit.

G7 — operator dial:
  * ``engine_filter`` / ``disabled_engines`` skip an *advisory* engine and
    emit a ``pipeline.engine.skipped`` bus event.
  * disabling a *required* engine raises ``RequiredEngineDisabledError``.

G8 — durable veto audit:
  * a veto appends one well-formed JSON line to ``state/audits/vetoes.jsonl``
    carrying the structured ``pattern_id`` (sourced from the Verdict, not a
    re-parse of the reason string).
  * an audit-write failure is swallowed — the request still produces its
    ``VetoResult``.

The veto-audit tests monkeypatch the inference state-dir env var
(``ROBIT_INFERENCE_STATE``) to a ``tmp_path`` so production state is never
touched.  ``litellm.acompletion`` is mocked throughout so no provider traffic
occurs; the real engine registry is used so destructive-op-gate actually
fires on the crafted prompt.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robit.proxy import audit as audit_mod
from robit.proxy import pipeline as pipeline_mod
from robit.proxy import upstream
from robit.proxy.audit import vetoes_log_path
from robit.proxy.canonical import CanonicalRequest, Message, TextPart
from robit.proxy.pipeline import (
    BusObservation,
    PipelineOptions,
    PipelineResult,
    RequiredEngineDisabledError,
    VetoResult,
    run,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/proxy/test_pipeline.py).
# ---------------------------------------------------------------------------


def _make_completion(text: str | None = "hi", *, model: str = "gpt-4o-mini"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _req(text: str = "hello") -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text=text),)),),
    )


@pytest.fixture
def tmp_audit_dir(tmp_path, monkeypatch):
    """Point the runtime state root at tmp_path so audits land there.

    ``resolve_audits_dir`` hangs ``audits/`` off the parent of the inference
    state dir, which is keyed on ``ROBIT_INFERENCE_STATE``.  Setting it to
    ``<tmp>/inference`` puts the audit log at ``<tmp>/audits/vetoes.jsonl``.
    """
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path / "inference"))
    return tmp_path


# ---------------------------------------------------------------------------
# G7 — operator dial.
# ---------------------------------------------------------------------------


async def test_engine_filter_skips_advisory_engine_and_emits_skipped_event():
    """An engine_filter that omits the advisory ``trust-scorer`` skips it and
    emits a ``pipeline.engine.skipped`` event — while the request still
    completes (the omitted engine is advisory, so no security gate is lost)."""
    # Discover the full set, then build a filter that excludes one advisory
    # engine (trust-scorer) but keeps every required gate so the request runs.
    registry = pipeline_mod.load_engine_registry()
    allow = frozenset(n for n in registry if n != "trust-scorer")
    assert "trust-scorer" in registry  # guard: the engine we drop must exist

    seen_skipped: list[tuple[str, dict]] = []
    original_record = pipeline_mod._BusRecorder.record

    def spy_record(self, event):
        if event.topic == "pipeline.engine.skipped":
            seen_skipped.append((event.source, dict(event.payload)))
        original_record(self, event)

    fake = _make_completion(text="ok")
    with patch.object(pipeline_mod._BusRecorder, "record", spy_record):
        with patch.object(
            upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
        ):
            result = await run(
                _req("benign prompt"),
                PipelineOptions(conduct=False, engine_filter=allow),
            )

    assert isinstance(result, PipelineResult)
    skipped_engines = {payload["engine"] for _src, payload in seen_skipped}
    assert "trust-scorer" in skipped_engines
    # The skip is attributed to the pipeline and labelled correctly.
    for src, payload in seen_skipped:
        if payload["engine"] == "trust-scorer":
            assert src == "proxy-pipeline"
            assert payload["reason"] == "not-in-filter"


async def test_disabled_engines_skips_advisory_and_emits_skipped_event():
    """disabled_engines denylist skips an advisory engine with reason
    ``disabled``."""
    seen_skipped: list[dict] = []
    original_record = pipeline_mod._BusRecorder.record

    def spy_record(self, event):
        if event.topic == "pipeline.engine.skipped":
            seen_skipped.append(dict(event.payload))
        original_record(self, event)

    fake = _make_completion(text="ok")
    with patch.object(pipeline_mod._BusRecorder, "record", spy_record):
        with patch.object(
            upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
        ):
            result = await run(
                _req("benign prompt"),
                PipelineOptions(
                    conduct=False, disabled_engines=frozenset({"trust-scorer"})
                ),
            )

    assert isinstance(result, PipelineResult)
    matched = [p for p in seen_skipped if p["engine"] == "trust-scorer"]
    assert matched and matched[0]["reason"] == "disabled"


async def test_disabling_required_engine_raises():
    """Disabling a REQUIRED engine (destructive-op-gate) raises rather than
    silently dropping a fail-closed security gate."""
    fake = _make_completion(text="ok")
    with patch.object(
        upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
    ):
        with pytest.raises(RequiredEngineDisabledError) as ei:
            await run(
                _req("benign prompt"),
                PipelineOptions(
                    conduct=False,
                    disabled_engines=frozenset({"destructive-op-gate"}),
                ),
            )
    assert ei.value.engine == "destructive-op-gate"


async def test_engine_filter_excluding_required_engine_raises():
    """An engine_filter that omits a REQUIRED engine also raises — the
    allowlist cannot quietly drop a security gate either."""
    fake = _make_completion(text="ok")
    with patch.object(
        upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
    ):
        # Filter that keeps only an advisory engine → all required gates dropped.
        with pytest.raises(RequiredEngineDisabledError):
            await run(
                _req("benign prompt"),
                PipelineOptions(
                    conduct=False, engine_filter=frozenset({"trust-scorer"})
                ),
            )


async def test_no_dial_set_runs_full_registry():
    """Default options (no filter, empty denylist) emit no skip events."""
    seen_skipped: list[dict] = []
    original_record = pipeline_mod._BusRecorder.record

    def spy_record(self, event):
        if event.topic == "pipeline.engine.skipped":
            seen_skipped.append(dict(event.payload))
        original_record(self, event)

    fake = _make_completion(text="ok")
    with patch.object(pipeline_mod._BusRecorder, "record", spy_record):
        with patch.object(
            upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
        ):
            result = await run(_req("benign prompt"), PipelineOptions(conduct=False))

    assert isinstance(result, PipelineResult)
    assert seen_skipped == []


# ---------------------------------------------------------------------------
# G8 — durable veto audit.
# ---------------------------------------------------------------------------


async def test_veto_appends_wellformed_audit_line(tmp_audit_dir):
    """A destructive prompt vetoes and appends one structured JSON line to
    ``state/audits/vetoes.jsonl`` carrying the structured pattern_id."""
    log_path = vetoes_log_path()
    assert not log_path.exists()  # clean tmp dir

    mock_acomp = AsyncMock(return_value=_make_completion())
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        result = await run(
            _req("please run git push --force on main"),
            PipelineOptions(conduct=False),
        )

    assert isinstance(result, VetoResult)
    assert mock_acomp.await_count == 0  # vetoed before upstream

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    # Schema: {ts, correlation_id, engine, pattern_id, phase, payload_summary,
    #          http_status}.
    assert set(entry) == {
        "ts",
        "correlation_id",
        "engine",
        "pattern_id",
        "phase",
        "payload_summary",
        "http_status",
    }
    assert entry["engine"] == "destructive-op-gate"
    assert entry["phase"] == "trust-gate"
    assert entry["http_status"] == 451
    # Structured pattern_id sourced from the Verdict — NOT re-parsed.
    assert entry["pattern_id"] == "w5-force-push"
    assert entry["payload_summary"]["pattern_id"] == "w5-force-push"
    assert isinstance(entry["ts"], int)
    assert entry["correlation_id"]


async def test_multiple_vetoes_append_multiple_lines(tmp_audit_dir):
    """The sink is append-only: two vetoes produce two lines."""
    mock_acomp = AsyncMock(return_value=_make_completion())
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        await run(_req("git push --force origin main"), PipelineOptions(conduct=False))
        await run(_req("please git push --force now"), PipelineOptions(conduct=False))

    lines = vetoes_log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for ln in lines:
        assert json.loads(ln)["pattern_id"] == "w5-force-push"


async def test_audit_write_failure_is_swallowed(tmp_audit_dir):
    """An audit-write failure must never block or crash the request — the
    VetoResult is still returned."""
    # Force the open() inside record_veto to blow up.
    def boom(*_a, **_k):
        raise OSError("disk full (simulated)")

    mock_acomp = AsyncMock(return_value=_make_completion())
    with patch.object(audit_mod.Path, "open", boom):
        with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
            result = await run(
                _req("please run git push --force on main"),
                PipelineOptions(conduct=False),
            )

    # Veto still surfaced even though the audit write failed.
    assert isinstance(result, VetoResult)
    assert result.pattern_id == "w5-force-push"
    assert mock_acomp.await_count == 0
    # Nothing was written (the open failed before any line landed).
    assert not vetoes_log_path().exists()


def test_resolve_audits_dir_follows_state_env(tmp_path, monkeypatch):
    """resolve_audits_dir hangs ``audits/`` off the inference state parent and
    honours the ROBIT_INFERENCE_STATE override at call time."""
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path / "inference"))
    assert audit_mod.resolve_audits_dir() == tmp_path / "audits"
    assert vetoes_log_path() == tmp_path / "audits" / "vetoes.jsonl"
