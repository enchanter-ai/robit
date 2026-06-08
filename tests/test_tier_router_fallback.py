"""tests/test_tier_router_fallback.py — fallback-chain coverage (audit §6 Q5).

Two concerns, both additive:

* :meth:`TierRouter.route_chain` returns an ordered, de-duplicated fallback
  list whose head equals :meth:`TierRouter.route`, and ``route`` itself is
  unchanged for every task class.
* :func:`robit.proxy.upstream.call_upstream` accepts an optional model chain
  and falls through to the next model on a *retryable* upstream error, fails
  fast on a *non-retryable* error, and re-raises once the chain is exhausted.
  A single-model chain (or no chain) behaves exactly like the legacy path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import litellm

from robit.proxy import upstream
from robit.proxy.canonical import CanonicalRequest, Message, TextPart
from robit.proxy.upstream import UpstreamError, call_upstream
from robit.runtime.models_registry import ModelsRegistry
from robit.runtime.tier_router import TierRouter

_TASK_CLASSES = ("orchestrator", "executor", "validator", "image", "embed")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ModelsRegistry:
    return ModelsRegistry.load()


@pytest.fixture(scope="module")
def router(registry: ModelsRegistry) -> TierRouter:
    return TierRouter(registry)


def _make_completion(text: str = "ok", *, model: str = "m"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _req(model: str = "primary-model") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
    )


# ---------------------------------------------------------------------------
# route_chain — ordered, de-duped, head matches route()
# ---------------------------------------------------------------------------


def test_route_chain_head_matches_route(router: TierRouter) -> None:
    for tc in _TASK_CLASSES:
        chain = router.route_chain(tc)
        assert chain[0] == router.route(tc), (
            f"route_chain({tc!r})[0]={chain[0]!r} != route({tc!r})="
            f"{router.route(tc)!r}"
        )


def test_route_chain_is_tuple_of_str(router: TierRouter) -> None:
    for tc in _TASK_CLASSES:
        chain = router.route_chain(tc)
        assert isinstance(chain, tuple)
        assert chain, f"chain for {tc!r} must be non-empty"
        assert all(isinstance(m, str) and m for m in chain)


def test_route_chain_is_deduplicated(router: TierRouter) -> None:
    for tc in _TASK_CLASSES:
        chain = router.route_chain(tc)
        assert len(chain) == len(set(chain)), (
            f"route_chain({tc!r}) has duplicates: {chain}"
        )


def test_route_chain_orchestrator_has_fallback_alternatives(
    router: TierRouter,
) -> None:
    """The bundled registry has >1 Opus model, so orchestrator must offer a
    real fallback tail (not just the primary)."""
    chain = router.route_chain("orchestrator")
    assert chain[0] == "claude-opus-4-7"
    assert len(chain) >= 2, f"expected fallback alternatives, got {chain}"
    # Tail is latest-first within the Opus subfamily.
    assert all("opus" in m.lower() for m in chain)


def test_route_unchanged_for_all_task_classes(router: TierRouter) -> None:
    """route() returns the same primary it always has (regression guard)."""
    assert router.route("orchestrator") == "claude-opus-4-7"
    assert router.route("executor") == "claude-sonnet-4-6"
    assert router.route("validator") == "claude-haiku-4-5"
    # image/embed resolve to a real registry entry (value is data-driven).
    for tc in ("image", "embed"):
        assert isinstance(router.route(tc), str) and router.route(tc)


def test_route_chain_respects_override(registry: ModelsRegistry) -> None:
    """A pinned override yields a single-element chain (no alternatives)."""
    r = TierRouter(registry, overrides={"validator": "claude-haiku-4-5"})
    assert r.route_chain("validator") == ("claude-haiku-4-5",)
    assert r.route("validator") == "claude-haiku-4-5"


def test_route_chain_size_hint_does_not_change_result(router: TierRouter) -> None:
    for tc in _TASK_CLASSES:
        assert router.route_chain(tc) == router.route_chain(tc, size_hint=128_000)


# ---------------------------------------------------------------------------
# call_upstream — fallback iteration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_model_chain_matches_legacy_path() -> None:
    """No chain (or a one-element chain) behaves exactly like before:
    one litellm call, response coerced, model from req.model."""
    fake = _make_completion("hello", model="primary-model")
    with patch.object(
        upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
    ) as mocked:
        resp = await call_upstream(_req("primary-model"))

    assert mocked.await_count == 1
    assert mocked.await_args.kwargs["model"] == "primary-model"
    assert resp.content[0].text == "hello"


@pytest.mark.asyncio
async def test_falls_through_to_second_model_on_retryable_error() -> None:
    """A 529 (overloaded) on the first model falls through to the second,
    which succeeds. Two litellm calls, with the second model on the wire."""
    overloaded = upstream.litellm.RateLimitError(
        message="overloaded", llm_provider="anthropic", model="model-a"
    )
    # RateLimitError carries status_code 429 (retryable) — but force 529 to
    # exercise the Anthropic overloaded path explicitly.
    overloaded.status_code = 529
    success = _make_completion("recovered", model="model-b")

    mocked = AsyncMock(side_effect=[overloaded, success])
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        resp = await call_upstream(
            _req("model-a"), models=["model-a", "model-b"], backoff_s=0
        )

    assert mocked.await_count == 2
    # Second attempt targeted model-b.
    assert mocked.await_args_list[0].kwargs["model"] == "model-a"
    assert mocked.await_args_list[1].kwargs["model"] == "model-b"
    assert resp.content[0].text == "recovered"


@pytest.mark.asyncio
async def test_non_retryable_error_fails_fast_without_trying_next() -> None:
    """A 400 bad request must NOT trigger fallback — only one call, and the
    UpstreamError surfaces immediately."""
    bad = upstream.litellm.BadRequestError(
        message="malformed", llm_provider="anthropic", model="model-a"
    )
    bad.status_code = 400

    mocked = AsyncMock(side_effect=bad)
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        with pytest.raises(UpstreamError) as ei:
            await call_upstream(
                _req("model-a"), models=["model-a", "model-b"], backoff_s=0
            )

    assert mocked.await_count == 1, "fallback must not fire on a 400"
    assert ei.value.status == 400


@pytest.mark.asyncio
async def test_exhausting_chain_raises_upstream_error() -> None:
    """Every model failing with a retryable error exhausts the chain and
    re-raises the last UpstreamError."""
    def _retryable(model: str):
        err = upstream.litellm.ServiceUnavailableError(
            message="503", llm_provider="anthropic", model=model
        )
        err.status_code = 503
        return err

    mocked = AsyncMock(
        side_effect=[_retryable("model-a"), _retryable("model-b")]
    )
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        with pytest.raises(UpstreamError) as ei:
            await call_upstream(
                _req("model-a"), models=["model-a", "model-b"], backoff_s=0
            )

    assert mocked.await_count == 2, "both models in the chain are tried"
    assert ei.value.status == 503


@pytest.mark.asyncio
async def test_no_status_connection_error_is_retryable() -> None:
    """An APIConnectionError with no HTTP status is classified retryable via
    the exception-type path, so fallback fires."""
    conn = litellm.APIConnectionError(
        message="connection reset", llm_provider="anthropic", model="model-a"
    )
    success = _make_completion("recovered", model="model-b")
    mocked = AsyncMock(side_effect=[conn, success])
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        resp = await call_upstream(
            _req("model-a"), models=["model-a", "model-b"], backoff_s=0
        )

    assert mocked.await_count == 2
    assert resp.content[0].text == "recovered"


@pytest.mark.asyncio
async def test_chain_is_deduped_before_dispatch() -> None:
    """Duplicate model ids in the chain collapse so no model is tried twice
    for a single logical slot."""
    overloaded = upstream.litellm.RateLimitError(
        message="overloaded", llm_provider="anthropic", model="model-a"
    )
    overloaded.status_code = 529
    success = _make_completion("recovered", model="model-b")
    mocked = AsyncMock(side_effect=[overloaded, success])
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        resp = await call_upstream(
            _req("model-a"),
            models=["model-a", "model-a", "model-b"],
            backoff_s=0,
        )

    # model-a tried once (dedup), then model-b.
    assert mocked.await_count == 2
    assert resp.content[0].text == "recovered"


@pytest.mark.asyncio
async def test_route_chain_wires_into_call_upstream(
    router: TierRouter,
) -> None:
    """End-to-end: a real route_chain feeds call_upstream and the primary is
    attempted first."""
    chain = router.route_chain("orchestrator")
    fake = _make_completion("ok", model=chain[0])
    mocked = AsyncMock(return_value=fake)
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        await call_upstream(_req(chain[0]), models=list(chain))

    assert mocked.await_args_list[0].kwargs["model"] == chain[0]
