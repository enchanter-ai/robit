"""Tests for robit.proxy.events.inference_substrate — Wave 13.2 Agent F.

Coverage:

* Discovery: emitter loads via :func:`load_emitters`, alphabetical slot.
* Protocol: name, phases tuple.
* Opt-in gate OFF: both phases are no-ops (no engine calls, no scratch).
* Opt-in gate ON + state initialized:
    - PRE_DISPATCH reads the briefing into ``ctx.scratch``.
    - POST_SESSION (benign) appends a ``proxy-success`` artifact.
    - POST_SESSION (redactions) appends a ``proxy-redaction`` artifact.
    - POST_SESSION (veto in scratch) appends a ``proxy-veto`` artifact.
* State-dir missing: WARNING logged once, no crash.
* Reconcile is NEVER called from the hot loop (regression guard).

All substrate I/O is mocked via :mod:`unittest.mock`.  No real
subprocess calls; no writes to the production state directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from robit.core import InProcessBus
from robit.proxy.canonical import CanonicalRequest, Message, TextPart
from robit.proxy.events import EmitPhase, load_emitters
from robit.proxy.events._types import EmitContext
from robit.proxy.events import inference_substrate as is_module
from robit.proxy.events.inference_substrate import (
    InferenceSubstrateEmitter,
    emitter as is_emitter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(model: str = "claude-3-5-sonnet-20241022") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(
            Message(role="user", content=(TextPart(text="hi"),)),
        ),
        max_tokens=64,
    )


def _ctx(
    *,
    model: str = "claude-3-5-sonnet-20241022",
    redactions: tuple[str, ...] = (),
    scratch: dict | None = None,
) -> EmitContext:
    return EmitContext(
        req=_req(model=model),
        bus=InProcessBus(),
        correlation_id="cid-test",
        session_id="sid-test",
        response=None,
        accumulated_text=None,
        redactions=redactions,
        scratch=dict(scratch or {}),
    )


@pytest.fixture
def gate_on(tmp_path, monkeypatch):
    """Enable the opt-in gate and point state at an isolated tmp dir.

    The state dir is created (empty) so :func:`_state_dir_ok` reports
    True and the emitter proceeds past the safety check.
    """
    state_dir = tmp_path / "inference"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ENCHANTER_INFERENCE_ENABLED", "1")
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(state_dir))
    # Reset the one-shot warning sentinel so each test starts clean.
    monkeypatch.setattr(is_module, "_state_warning_emitted", False, raising=False)
    return state_dir


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.delenv("ENCHANTER_INFERENCE_ENABLED", raising=False)
    monkeypatch.setattr(is_module, "_state_warning_emitted", False, raising=False)


# ---------------------------------------------------------------------------
# 1. Protocol contract + discovery
# ---------------------------------------------------------------------------


def test_emitter_protocol_contract():
    assert is_emitter.name == "inference-substrate"
    assert EmitPhase.PRE_DISPATCH in is_emitter.phases
    assert EmitPhase.POST_SESSION in is_emitter.phases
    assert isinstance(is_emitter, InferenceSubstrateEmitter)


def test_load_emitters_discovers_inference_substrate():
    emitters = load_emitters()
    names = [em.name for em in emitters]
    assert "inference-substrate" in names
    # Discovery is alphabetical by module name.  ``inference_substrate``
    # sorts after ``cost_ledger`` and before ``rate_limiter`` /
    # ``tool_poisoning_scan`` / ``trust_scorer``.
    assert names.index("cost-ledger") < names.index("inference-substrate")
    # The exact sibling ordering depends on which modules exist; assert
    # only the invariants we depend on.
    assert names.index("builtin") < names.index("inference-substrate")


# ---------------------------------------------------------------------------
# 2. Opt-in gate OFF — both phases are no-ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_off_pre_dispatch_is_noop(gate_off):
    ctx = _ctx()
    with patch.object(is_module._inference_engine, "render_briefing") as mock_rb, \
         patch.object(is_module._inference_engine, "emit_unconditional") as mock_emit:
        await is_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)
    mock_rb.assert_not_called()
    mock_emit.assert_not_called()
    assert "inference-substrate" not in ctx.scratch


@pytest.mark.asyncio
async def test_gate_off_post_session_is_noop(gate_off):
    ctx = _ctx()
    with patch.object(is_module._inference_engine, "emit_unconditional") as mock_emit:
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)
    mock_emit.assert_not_called()
    assert "inference-substrate" not in ctx.scratch


# ---------------------------------------------------------------------------
# 3. Opt-in ON — PRE_DISPATCH reads briefing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_dispatch_stashes_briefing_in_scratch(gate_on, tmp_path):
    briefing_path = tmp_path / "briefings" / "agent.md"
    briefing_path.parent.mkdir(parents=True, exist_ok=True)
    briefing_path.write_text("# Agent Briefing\n\nelevated patterns ...\n", encoding="utf-8")

    ctx = _ctx()
    with patch.object(
        is_module._inference_engine, "render_briefing", return_value=briefing_path
    ) as mock_rb:
        await is_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)

    mock_rb.assert_called_once_with("agent")
    assert "inference-substrate" in ctx.scratch
    assert "Agent Briefing" in ctx.scratch["inference-substrate"]["briefing"]


@pytest.mark.asyncio
async def test_pre_dispatch_briefing_read_failure_does_not_crash(gate_on):
    ctx = _ctx()
    with patch.object(
        is_module._inference_engine,
        "render_briefing",
        side_effect=RuntimeError("disk full"),
    ):
        await is_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)

    # Empty briefing still stashed so post-session can reference it.
    assert ctx.scratch["inference-substrate"]["briefing"] == ""


# ---------------------------------------------------------------------------
# 4. Opt-in ON — POST_SESSION emits artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_session_benign_emits_proxy_success(gate_on):
    ctx = _ctx(model="claude-3-5-sonnet-20241022")
    with patch.object(
        is_module._inference_engine, "emit_unconditional"
    ) as mock_emit:
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    mock_emit.assert_called_once()
    artifact = mock_emit.call_args[0][0]
    assert artifact["code"] == "S-PROXY-OK"
    assert artifact["category"] == "proxy-success"
    assert artifact["scope"] == "agent"
    assert "proxy" in artifact["tags"]
    assert "success" in artifact["tags"]
    assert "anthropic" in artifact["tags"]  # wire-format derived from model
    assert artifact["evidence"]["model"] == "claude-3-5-sonnet-20241022"
    # Honest counts — single observation, no pushback.
    assert artifact["evidence"]["iterations"] == 1
    assert artifact["evidence"]["user_rounds_of_pushback"] == 0


@pytest.mark.asyncio
async def test_post_session_with_redactions_emits_proxy_redaction(gate_on):
    ctx = _ctx(
        model="gpt-4o-mini",
        redactions=("aws-secret-key", "github-token"),
    )
    with patch.object(
        is_module._inference_engine, "emit_unconditional"
    ) as mock_emit:
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    mock_emit.assert_called_once()
    artifact = mock_emit.call_args[0][0]
    assert artifact["code"] == "S-MASK-FIRED"
    assert artifact["category"] == "proxy-redaction"
    assert "openai" in artifact["tags"]
    assert artifact["evidence"]["redaction_count"] == 2
    assert artifact["evidence"]["redaction_ids"] == ["aws-secret-key", "github-token"]


@pytest.mark.asyncio
async def test_post_session_with_veto_in_scratch_emits_proxy_veto(gate_on):
    # Duck-type a VetoResult — emitter must not import from pipeline.
    veto = {
        "phase": "pre-dispatch",
        "plugin": "destructive-op-gate",
        "reason": "rm -rf in args",
        "pattern_id": "P-001",
        "pattern_name": "destructive-shell",
    }
    ctx = _ctx(scratch={"veto": veto})

    with patch.object(
        is_module._inference_engine, "emit_unconditional"
    ) as mock_emit:
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    mock_emit.assert_called_once()
    artifact = mock_emit.call_args[0][0]
    assert artifact["code"] == "F-PROXY-VETO"
    assert artifact["category"] == "proxy-veto"
    assert "destructive-op-gate" in artifact["title"]
    assert artifact["evidence"]["veto_phase"] == "pre-dispatch"
    assert artifact["evidence"]["veto_plugin"] == "destructive-op-gate"
    assert artifact["counter"]  # vetos must carry a counter rule


# ---------------------------------------------------------------------------
# 5. State-dir missing → WARNING + no crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_dir_missing_logs_warning_and_no_ops(
    tmp_path, monkeypatch, caplog
):
    # Gate is ON but state dir + its parent don't exist.
    missing = tmp_path / "does" / "not" / "exist" / "inference"
    monkeypatch.setenv("ENCHANTER_INFERENCE_ENABLED", "1")
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(missing))
    monkeypatch.setattr(is_module, "_state_warning_emitted", False, raising=False)

    ctx = _ctx()
    with caplog.at_level(logging.WARNING, logger=is_module._log.name):
        with patch.object(
            is_module._inference_engine, "emit_unconditional"
        ) as mock_emit:
            await is_emitter.emit(EmitPhase.POST_SESSION, ctx)
            # Second call must NOT log again (warn-once).
            await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    mock_emit.assert_not_called()
    state_warnings = [
        r for r in caplog.records if "state dir missing" in r.getMessage()
    ]
    assert len(state_warnings) == 1, (
        f"expected 1 warn-once log; got {len(state_warnings)}: "
        f"{[r.getMessage() for r in state_warnings]}"
    )


# ---------------------------------------------------------------------------
# 6. Reconcile is never called from the hot loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_not_called_from_emitter(gate_on):
    """SPRT needs accumulated observations — per-emit reconcile is noise."""
    ctx = _ctx()
    with patch.object(
        is_module._inference_engine, "reconcile"
    ) as mock_reconcile, patch.object(
        is_module._inference_engine, "render_briefing", return_value=Path("/tmp/missing")
    ), patch.object(
        is_module._inference_engine, "emit_unconditional"
    ):
        await is_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    mock_reconcile.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Emit I/O failure does not crash the proxy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_session_emit_failure_is_swallowed(gate_on, caplog):
    ctx = _ctx()
    with caplog.at_level(logging.WARNING, logger=is_module._log.name):
        with patch.object(
            is_module._inference_engine,
            "emit_unconditional",
            side_effect=OSError("disk full"),
        ):
            # Must NOT raise.
            await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    # No artifact stashed when emit failed.
    sub_scratch = ctx.scratch.get("inference-substrate", {})
    assert "last_artifact" not in sub_scratch
    # The failure surfaces as a WARNING for operators.
    assert any("emit failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 8. Successful emit stashes the artifact in scratch for observability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_session_stashes_last_artifact(gate_on):
    ctx = _ctx()
    with patch.object(is_module._inference_engine, "emit_unconditional"):
        await is_emitter.emit(EmitPhase.POST_SESSION, ctx)

    sub_scratch = ctx.scratch.get("inference-substrate", {})
    assert "last_artifact" in sub_scratch
    assert sub_scratch["last_artifact"]["code"] == "S-PROXY-OK"
