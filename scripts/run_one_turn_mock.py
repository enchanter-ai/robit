"""One turn through the enchanter agent using MockLlmClient.

Same wiring as run_one_turn.py but with the LLM replaced by a deterministic
mock that returns a response containing a leaked AWS access key. Exercises the
post-response leg: text comes back from dispatch, post-response event is
published, secret-mask scans and flags it.
"""

from __future__ import annotations

import asyncio
import sys

from robit.composer.conduct import compose_conduct_xml, select_rules
from robit.conduct import load_conduct
from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    SecurityVetoError,
    create_request_context,
)
from robit.core.bus import build_event
from robit.llm.mock_client import MockLlmClient
from robit.llm.types import CompletionRequest, CompletionResponse, Message
from robit.loader import load_engine_registry
from robit.runtime.models_registry import ModelsRegistry
from robit.runtime.tier_router import TierRouter

_SMOKE_RULES: set[str] = {
    "discipline", "verification", "tool-use",
    "refusal-and-recovery", "formatting",
}


def _rule_to_dict(r):
    return {
        "name": r.name, "body": r.body, "enforcement": r.enforcement,
        "package": r.package, "tags": list(r.tags),
    }


async def run_turn(user_prompt: str) -> int:
    # System prompt from a 5-rule conduct subset.
    rules = [_rule_to_dict(r) for r in load_conduct()]
    rules = select_rules(rules, required=_SMOKE_RULES)
    xml = compose_conduct_xml(rules)
    system_prompt = (
        "You are an agent operating under the enchanter conduct framework. "
        "Conduct modules describing your operating rules are provided in the "
        "<conduct> block below. Follow them.\n\n" + xml
    )

    # Engines + orchestrator.
    registry = load_engine_registry()
    bus = InProcessBus()
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    # Tier router for cosmetic logging.
    router = TierRouter(ModelsRegistry.load())
    model_id = router.route("executor")

    # Mock LLM whose response embeds a fake AWS access key to prove the
    # post-response → secret-mask leg.
    leaky_text = (
        "The enchanter agent is an enforcement-first MCP-aware Python runtime "
        "that wraps LLM calls in a 7-phase lifecycle with pluggable security "
        "engines. (For audit trail: AKIAIOSFODNN7EXAMPLE — note this is a known "
        "documentation key.)"
    )
    mock_resp = CompletionResponse(
        text=leaky_text,
        model=model_id,
        stop_reason="end_turn",
        input_tokens=42,
        output_tokens=64,
    )
    client = MockLlmClient(responses=[mock_resp])

    print(f"engines:       {len(registry)} loaded")
    print(f"conduct:       {len(rules)} rules injected ({len(system_prompt)} chars)")
    print(f"model:         {model_id}  (mocked)")
    print(f"prompt:        {user_prompt!r}")
    print("─" * 60)

    ctx = create_request_context()

    # Pre-publish a benign trust-gate event.
    trust_event = build_event(
        correlation_id=ctx.correlation_id, session_id=ctx.session_id,
        phase="trust-gate", topic="llm.completion.requested",
        source="run_one_turn_mock", budget_tier=ctx.budget_tier,
        payload={"prompt": user_prompt, "model": model_id},
    )
    await bus.publish(trust_event.topic, trust_event)

    # Watch the bus for engine activity.
    activity: list = []

    async def trace(ev):
        if any(t in ev.topic for t in ("veto", "mask", "ack", "matched", "applied")):
            activity.append((ev.source, ev.topic, ev.phase))

    bus.subscribe("*", trace)

    async def dispatch(ctx) -> str:
        req = CompletionRequest(
            model=model_id,
            messages=[Message(role="user", content=user_prompt)],
            system=system_prompt,
            max_tokens=1024,
        )
        resp = await client.complete(req)
        post_event = build_event(
            correlation_id=ctx.correlation_id, session_id=ctx.session_id,
            phase="post-response", topic="mcp.tool.result.received",
            source="run_one_turn_mock", budget_tier=ctx.budget_tier,
            payload={"result": resp.text,
                     "input_tokens": resp.input_tokens,
                     "output_tokens": resp.output_tokens,
                     "model": resp.model},
        )
        await bus.publish(post_event.topic, post_event)
        return resp.text

    try:
        text = await orch.run(ctx, dispatch)
    except SecurityVetoError as e:
        print(f"VETOED at {e.phase} by {e.plugin}: {e}")
        return 2

    print(f"\n[mocked response, {mock_resp.input_tokens} in / "
          f"{mock_resp.output_tokens} out]\n")
    print(text)
    print()
    print("─" * 60)
    print("engine bus activity:")
    if not activity:
        print("  (none)")
    for source, topic, phase in activity:
        print(f"  phase={phase:14s} topic={topic:35s} from={source}")
    return 0


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]).strip() or "What is the enchanter agent for?"
    sys.exit(asyncio.run(run_turn(prompt)))
