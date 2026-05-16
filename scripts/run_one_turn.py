"""One real turn through the enchanter agent against the Anthropic API.

Wires conduct injection + lifecycle + engines around a single LlmClient.complete()
call. Picks credentials from the environment (ANTHROPIC_API_KEY for pay-per-token,
CLAUDE_CODE_OAUTH_TOKEN for Pro/Max subscription).

Usage:
    python scripts/run_one_turn.py "your prompt here"
"""

from __future__ import annotations

import asyncio
import sys

from robit.composer.conduct import compose_conduct_xml, select_rules
from robit.conduct import load_conduct

# Minimal conduct subset for smoke runs. The full corpus is ~170KB; for a single
# turn we only need the rules that shape day-to-day agent behavior. Tune freely.
_SMOKE_RULES: set[str] = {
    "discipline",
    "verification",
    "tool-use",
    "refusal-and-recovery",
    "formatting",
}
from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    SecurityVetoError,
    create_request_context,
)
from robit.core.bus import build_event
from robit.llm import AnthropicClient
from robit.llm.types import CompletionRequest, Message
from robit.loader import load_engine_registry
from robit.runtime.models_registry import ModelsRegistry
from robit.runtime.tier_router import TierRouter


def _rule_to_dict(r):
    return {
        "name": r.name,
        "body": r.body,
        "enforcement": r.enforcement,
        "package": r.package,
        "tags": list(r.tags),
    }


def _build_system_prompt(full: bool = False) -> tuple[str, int]:
    rules = load_conduct()
    rule_dicts = [_rule_to_dict(r) for r in rules]
    if not full:
        rule_dicts = select_rules(rule_dicts, required=_SMOKE_RULES)
    xml = compose_conduct_xml(rule_dicts)
    base = (
        "You are an agent operating under the enchanter conduct framework. "
        "Conduct modules describing your operating rules are provided in the "
        "<conduct> block below. Follow them.\n\n"
    )
    return base + xml, len(rule_dicts)


async def run_turn(user_prompt: str, full_conduct: bool = False) -> int:
    # 1) Compose system prompt from conduct.
    system_prompt, n_rules = _build_system_prompt(full=full_conduct)

    # 2) Load engines + orchestrator.
    registry = load_engine_registry()
    bus = InProcessBus()
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    # 3) Pick a model via tier router.
    router = TierRouter(ModelsRegistry.load())
    model_id = router.route("executor")

    # 4) Initialise the LLM client (auto-detects api_key vs oauth from env).
    client = AnthropicClient()

    print(f"engines:       {len(registry)} loaded")
    print(f"conduct:       {n_rules} rules injected ({len(system_prompt)} chars)")
    print(f"model:         {model_id}")
    print(f"auth mode:     {client.auth_mode}")
    print(f"prompt:        {user_prompt!r}")
    print("─" * 60)

    # 5) Build the request context.
    ctx = create_request_context()

    # 6) Pre-publish a benign trust-gate event so trust-gate engines have
    #    something to chew on. (For a chat-only turn there's no tool call,
    #    but the lifecycle still runs every phase; engines just see no work.)
    trust_event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="llm.completion.requested",
        source="run_one_turn",
        budget_tier=ctx.budget_tier,
        payload={"prompt": user_prompt, "model": model_id},
    )
    await bus.publish(trust_event.topic, trust_event)

    # Capture mask/veto events for the post-hoc report.
    fired_events: list = []

    async def trace(ev):
        if any(t in ev.topic for t in ("veto", "mask", "ack")):
            fired_events.append(ev)

    bus.subscribe("*", trace)

    # 7) Dispatch == the actual LLM call.
    completion_result: dict = {}

    async def dispatch(ctx) -> str:
        req = CompletionRequest(
            model=model_id,
            messages=[Message(role="user", content=user_prompt)],
            system=system_prompt,
            max_tokens=1024,
        )
        resp = await client.complete(req)
        completion_result["resp"] = resp

        # Feed the model's text into post-response so secret-mask & friends
        # can scan it.
        post_event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="llm.completion.received",
            source="run_one_turn",
            budget_tier=ctx.budget_tier,
            payload={
                "result": resp.text,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "model": resp.model,
            },
        )
        await bus.publish(post_event.topic, post_event)
        return resp.text

    try:
        text = await orch.run(ctx, dispatch)
    except SecurityVetoError as e:
        print(f"VETOED at {e.phase} by {e.plugin}: {e}")
        return 2

    resp = completion_result["resp"]
    print(f"\n[response — {resp.input_tokens} in / {resp.output_tokens} out, "
          f"stop={resp.stop_reason}]\n")
    print(text)
    print()
    print("─" * 60)

    veto_n = sum(1 for e in fired_events if "veto" in e.topic)
    mask_n = sum(1 for e in fired_events if "mask" in e.topic)
    ack_n = sum(1 for e in fired_events if "ack" in e.topic)
    print(f"engine activity: {veto_n} vetoes, {mask_n} mask events, {ack_n} acks")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full-conduct"]
    full_conduct = "--full-conduct" in sys.argv[1:]
    prompt = " ".join(args).strip() or "Say hi in one short sentence."
    sys.exit(asyncio.run(run_turn(prompt, full_conduct=full_conduct)))
