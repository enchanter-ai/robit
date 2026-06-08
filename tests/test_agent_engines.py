"""Tests for the agent-shaped engine contract (audit §8).

Covers the framework + the intent-anchor pilot:

  Manifest:
    * a valid [agent] table parses into an AgentSpec (tier + prompt paths)
    * a malformed [agent] table is rejected with ManifestSchemaError
    * an engine with no [agent] table parses with agent=None (backward-compat)

  PipelineOptions:
    * prompt_overlay defaults to None
    * the overlay is APPENDED after the engine-authored prompt, not replacing it

  intent-anchor pilot (default-OFF):
    * agent flag OFF → deterministic LCS+HMM path runs (existing behaviour)
    * agent flag ON + MOCK llm_call → drift verdict comes from the model output,
      and the operator overlay is applied to the prompt
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    create_request_context,
)
from robit.core.bus import build_event
from robit.core.context import RequestContext
from robit.loader.errors import ManifestSchemaError
from robit.loader.manifest import AgentSpec, EngineManifest, parse_manifest
from robit.proxy.pipeline import PipelineOptions
from robit.engines.intent_anchor import IntentAnchor


# ===========================================================================
# Helpers
# ===========================================================================

def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "engine.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_BASE_TOML = """\
    name = "agenty-engine"
    description = "Engine with an [agent] table for testing."
    version = "1.0.0"
    phases = ["post-session"]
    required = false
    budget_tier = "med-or-higher"
    adapter = "my.module:adapter"

    [topics]
    subscribes = ["user.prompt.submit"]
    emits = ["agenty-engine.drift.detected"]
"""

_AGENT_TABLE = """\

    [agent]
    tier = "executor"

    [agent.prompts]
    post-session = "prompts/drift.md"
"""


async def _fire_phase(
    bus: InProcessBus,
    ctx: RequestContext,
    phase: str,
    prompt: str,
) -> None:
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase=phase,  # type: ignore[arg-type]
        topic="user.prompt.submit",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"user_prompt": prompt},
    )
    await bus.publish(event.topic, event)


# ===========================================================================
# Manifest: [agent] table parsing
# ===========================================================================


class TestAgentManifest:
    def test_valid_agent_table_parses_into_agentspec(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, _BASE_TOML + _AGENT_TABLE)
        m = parse_manifest(p)

        assert isinstance(m, EngineManifest)
        assert isinstance(m.agent, AgentSpec)
        assert m.agent.tier == "executor"
        assert m.agent.prompts == (("post-session", "prompts/drift.md"),)
        assert m.agent.prompt_for("post-session") == "prompts/drift.md"
        assert m.agent.prompt_for("anchor") is None

    def test_no_agent_table_parses_with_agent_none(self, tmp_path: Path) -> None:
        """The common case: no [agent] table → agent=None, fully backward-compat."""
        p = _write_toml(tmp_path, _BASE_TOML)
        m = parse_manifest(p)
        assert m.agent is None

    def test_malformed_agent_bad_tier_rejected(self, tmp_path: Path) -> None:
        bad = _BASE_TOML + textwrap.dedent("""\

            [agent]
            tier = "wizard"

            [agent.prompts]
            post-session = "prompts/drift.md"
        """)
        p = _write_toml(tmp_path, bad)
        with pytest.raises(ManifestSchemaError) as exc:
            parse_manifest(p)
        assert exc.value.field == "agent.tier"

    def test_malformed_agent_missing_prompts_rejected(self, tmp_path: Path) -> None:
        bad = _BASE_TOML + textwrap.dedent("""\

            [agent]
            tier = "executor"
        """)
        p = _write_toml(tmp_path, bad)
        with pytest.raises(ManifestSchemaError) as exc:
            parse_manifest(p)
        assert exc.value.field.startswith("agent")

    def test_malformed_agent_unknown_field_rejected(self, tmp_path: Path) -> None:
        bad = _BASE_TOML + textwrap.dedent("""\

            [agent]
            tier = "executor"
            bogus = "nope"

            [agent.prompts]
            post-session = "prompts/drift.md"
        """)
        p = _write_toml(tmp_path, bad)
        with pytest.raises(ManifestSchemaError) as exc:
            parse_manifest(p)
        assert "bogus" in str(exc.value) or exc.value.field == "agent.bogus"

    def test_malformed_agent_empty_prompts_rejected(self, tmp_path: Path) -> None:
        bad = _BASE_TOML + textwrap.dedent("""\

            [agent]
            tier = "executor"

            [agent.prompts]
        """)
        p = _write_toml(tmp_path, bad)
        with pytest.raises(ManifestSchemaError) as exc:
            parse_manifest(p)
        assert exc.value.field == "agent.prompts"

    def test_real_intent_anchor_manifest_has_agent_table(self) -> None:
        """The pilot's shipped engine.toml must parse with a valid AgentSpec."""
        toml_path = (
            Path(__file__).resolve().parent.parent
            / "robit" / "engines" / "intent_anchor" / "engine.toml"
        )
        m = parse_manifest(toml_path)
        assert m.agent is not None
        assert m.agent.tier == "executor"
        assert m.agent.prompt_for("post-session") == "prompts/drift.md"


# ===========================================================================
# PipelineOptions.prompt_overlay
# ===========================================================================


class TestPromptOverlay:
    def test_prompt_overlay_defaults_none(self) -> None:
        opts = PipelineOptions()
        assert opts.prompt_overlay is None

    def test_prompt_overlay_settable(self) -> None:
        opts = PipelineOptions(prompt_overlay="tenant rule: never drift onto billing")
        assert opts.prompt_overlay == "tenant rule: never drift onto billing"

    def test_overlay_is_appended_not_replacing(self) -> None:
        """The engine-authored prompt body must survive; overlay is appended after."""
        base = IntentAnchor()
        base_system, _ = base.build_drift_prompt("anchor intent", "current prompt")

        overlay = "OPERATOR RULE: treat refactors as on-task."
        engine = IntentAnchor(prompt_overlay=overlay)
        system, user = engine.build_drift_prompt("anchor intent", "current prompt")

        # The full engine-authored body is still present (not replaced).
        assert base_system in system
        # The overlay text is appended after the authored body.
        assert overlay in system
        assert system.index(overlay) > system.index(base_system.strip()[:20])
        # The user message carries both anchor + current prompt.
        assert "anchor intent" in user
        assert "current prompt" in user


# ===========================================================================
# intent-anchor pilot — default-OFF deterministic path
# ===========================================================================


def _set_anchor(engine: IntentAnchor, session_id: str, intent: str) -> None:
    store = engine._get_or_create(session_id)
    store.set_anchor(intent)


class TestPilotDefaultOff:
    async def test_flag_off_runs_deterministic_path(self, monkeypatch) -> None:
        """With the agent flag OFF, a MOCK llm_call must NOT be consulted and
        the deterministic LCS+HMM verdict must be used (existing behaviour)."""
        monkeypatch.delenv("ROBIT_INTENT_ANCHOR_AGENT", raising=False)

        calls: list[tuple[str, str]] = []

        async def mock_llm(system: str, user: str) -> str:
            calls.append((system, user))
            # If the agent path ran it would say "no drift" — opposite of the
            # deterministic verdict below, so we can tell which path fired.
            return '{"drift": false, "confidence": 0.9, "rationale": "n/a"}'

        engine = IntentAnchor(llm_call=mock_llm)
        session_id = "sess-off"
        _set_anchor(engine, session_id, "implement the red-black tree insertion algorithm")

        bus = InProcessBus()
        orch = Orchestrator(OrchestratorConfig(registry={engine.name: engine}, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)
        await _fire_phase(bus, ctx, "post-session", "summarize quarterly finance results")
        await orch.run(ctx, dispatch)

        # Deterministic path: unrelated prompt → drift fired.
        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        assert len(drift_events) == 1
        assert drift_events[0].payload.get("verdict_source") != "agent"
        # The mock was never called — the agent path did not run.
        assert calls == []


# ===========================================================================
# intent-anchor pilot — agent path ON (MOCK llm_call)
# ===========================================================================


class TestPilotAgentOn:
    async def test_flag_on_uses_model_verdict_and_applies_overlay(self, monkeypatch) -> None:
        """With the flag ON + a MOCK llm_call, the drift verdict comes from the
        model output, and the operator overlay is applied to the prompt."""
        monkeypatch.setenv("ROBIT_INTENT_ANCHOR_AGENT", "1")

        seen: dict[str, str] = {}

        async def mock_llm(system: str, user: str) -> str:
            seen["system"] = system
            seen["user"] = user
            # Model says: drift.  (Note the prompts below are SIMILAR, so the
            # deterministic path would NOT fire — proving the verdict is the
            # model's, not the algorithm's.)
            return '```json\n{"drift": true, "confidence": 0.82, "rationale": "topic switched"}\n```'

        overlay = "OPERATOR RULE: billing questions always count as drift."
        engine = IntentAnchor(llm_call=mock_llm, prompt_overlay=overlay)
        session_id = "sess-on"
        _set_anchor(engine, session_id, "fix the authentication token expiry bug")

        bus = InProcessBus()
        orch = Orchestrator(OrchestratorConfig(registry={engine.name: engine}, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)
        # SIMILAR prompt — deterministic LCS would stay above threshold (no drift).
        await _fire_phase(
            bus, ctx, "post-session",
            "fix the authentication token expiry bug in the oauth service",
        )
        await orch.run(ctx, dispatch)

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        # Drift fired because the MODEL said so (the deterministic path would not).
        assert len(drift_events) == 1
        payload = drift_events[0].payload
        assert payload["verdict_source"] == "agent"
        assert payload["agent_confidence"] == pytest.approx(0.82)
        assert payload["agent_rationale"] == "topic switched"

        # Operator overlay was applied (appended) to the system prompt.
        assert overlay in seen["system"]
        # And the engine-authored body is still present (not replaced).
        assert "drift judge" in seen["system"].lower()
        # The user prompt carried both anchor + current prompt.
        assert "fix the authentication token expiry bug" in seen["user"]

    async def test_flag_on_model_says_no_drift_emits_nothing(self, monkeypatch) -> None:
        """Model verdict of no-drift on an UNRELATED prompt suppresses the drift
        event the deterministic path would otherwise emit."""
        monkeypatch.setenv("ROBIT_INTENT_ANCHOR_AGENT", "1")

        async def mock_llm(system: str, user: str) -> str:
            return '{"drift": false, "confidence": 0.7, "rationale": "still on task"}'

        engine = IntentAnchor(llm_call=mock_llm)
        session_id = "sess-on-clean"
        _set_anchor(engine, session_id, "implement the red-black tree insertion algorithm")

        bus = InProcessBus()
        orch = Orchestrator(OrchestratorConfig(registry={engine.name: engine}, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)
        # Unrelated prompt: deterministic path WOULD fire drift; model says no.
        await _fire_phase(bus, ctx, "post-session", "summarize quarterly finance results")
        await orch.run(ctx, dispatch)

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        assert len(drift_events) == 0

    async def test_flag_on_no_seam_uses_real_call_and_fails_open(self, monkeypatch) -> None:
        """Flag ON + NO injected seam → the agent path now builds a REAL
        ``call_upstream``-backed seam (F3(b)).  With no live credentials the
        upstream call raises; the engine is advisory and fails OPEN (clean ack,
        no crash, no drift event).  The network is never actually hit because we
        patch ``call_upstream`` to raise before any provider request."""
        monkeypatch.setenv("ROBIT_INTENT_ANCHOR_AGENT", "1")

        # Stub the real upstream so no network is touched (F3 contract).
        import robit.proxy.upstream as upstream_mod

        async def _boom(*args, **kwargs):
            raise upstream_mod.UpstreamError(
                provider="anthropic", status=401, message="no creds in test"
            )

        monkeypatch.setattr(upstream_mod, "call_upstream", _boom)

        engine = IntentAnchor()  # no llm_call → real seam built, then raises
        session_id = "sess-on-noseam"
        _set_anchor(engine, session_id, "implement the red-black tree insertion algorithm")

        bus = InProcessBus()
        orch = Orchestrator(OrchestratorConfig(registry={engine.name: engine}, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)
        await _fire_phase(bus, ctx, "post-session", "summarize quarterly finance results")
        # Must not raise — advisory fail-open.
        await orch.run(ctx, dispatch)

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        # Fail-open: the agent call errored, so no verdict / no drift event.
        assert len(drift_events) == 0
