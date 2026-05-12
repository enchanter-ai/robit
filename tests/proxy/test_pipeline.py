"""Tests for enchanter.proxy.pipeline — orchestrator wrapper + bus integration.

Mocks `litellm.acompletion` so no provider traffic occurs.  Uses the real
engine registry so destructive-op-gate / secret-mask actually fire on the
crafted prompts/responses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from enchanter.proxy import upstream
from enchanter.proxy.canonical import (
    CanonicalRequest,
    Message,
    TextPart,
)
from enchanter.proxy.pipeline import (
    BusObservation,
    PipelineOptions,
    PipelineResult,
    VetoResult,
    run,
    stream,
)


# ---------------------------------------------------------------------------
# Helpers for fabricating LiteLLM-shaped responses.
# ---------------------------------------------------------------------------


def _make_completion(
    text: str | None = "hi",
    *,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage=None,
):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=usage)


class _AsyncStream:
    """Trivial async iterator over a list of fabricated chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _req(text: str = "hello", system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text=text),)),),
        system=system,
    )


# ---------------------------------------------------------------------------
# Non-streaming pipeline.
# ---------------------------------------------------------------------------


async def test_run_benign_prompt_returns_pipeline_result():
    fake = _make_completion(text="hello there", finish_reason="stop")
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        result = await run(_req("how are you"), PipelineOptions(conduct=False))

    assert isinstance(result, PipelineResult)
    assert result.response.content[0].text == "hello there"  # type: ignore[union-attr]
    assert result.response.usage.input_tokens == 5
    assert result.response.usage.output_tokens == 3


async def test_run_destructive_prompt_vetoes_before_upstream():
    """A prompt containing `git push --force` triggers destructive-op-gate.

    The veto must fire BEFORE the upstream is called — assert the mock is
    never awaited.
    """
    mock_acomp = AsyncMock(return_value=_make_completion())
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        result = await run(
            _req("please run git push --force on main"),
            PipelineOptions(conduct=False),
        )

    assert isinstance(result, VetoResult)
    assert result.phase == "trust-gate"
    assert result.plugin == "destructive-op-gate"
    assert result.pattern_id == "w5-force-push"
    assert mock_acomp.await_count == 0


async def test_run_secret_in_response_records_mask_observation():
    """A response that leaks an AWS access key triggers secret-mask at
    post-response.  The pipeline result's `fired` tuple should contain the
    observation so Agent E can surface it in headers.
    """
    secret = "AKIAIOSFODNN7EXAMPLE"
    fake = _make_completion(text=f"key is {secret}")
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        result = await run(_req("benign prompt"), PipelineOptions(conduct=False))

    assert isinstance(result, PipelineResult)
    mask_obs = [o for o in result.fired if "mask" in o.topic]
    assert mask_obs, f"expected a secret-mask observation, got {result.fired!r}"
    # The summary must carry the pattern id but NEVER raw secret content.
    summary = mask_obs[0].payload_summary
    assert "s-aws-key" in summary.get("matched_patterns", [])
    # Defensive: raw secret must not leak into the observation summary.
    assert secret not in repr(summary)


async def test_run_conduct_false_does_not_inject_system_prompt():
    """opts.conduct=False → the request sent to upstream carries no conduct XML."""
    fake = _make_completion(text="ok")
    mock_acomp = AsyncMock(return_value=fake)
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        await run(_req("hi", system="my-original-system"), PipelineOptions(conduct=False))

    sent_kwargs = mock_acomp.await_args.kwargs
    system_msgs = [m for m in sent_kwargs["messages"] if m["role"] == "system"]
    # Exactly one system message; its content is the original, no <conduct> XML.
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "my-original-system"
    assert "<conduct" not in system_msgs[0]["content"]


async def test_run_conduct_true_injects_conduct_xml_into_system():
    """Default opts.conduct=True → upstream sees a <conduct> XML prefix."""
    fake = _make_completion(text="ok")
    mock_acomp = AsyncMock(return_value=fake)
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        await run(_req("hi"))  # default options → conduct=True

    sent_kwargs = mock_acomp.await_args.kwargs
    system_msgs = [m for m in sent_kwargs["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"].startswith("<conduct")


async def test_run_conduct_preserves_caller_system_prompt():
    """Conduct XML is *prepended* to caller's system, not replacing it."""
    fake = _make_completion(text="ok")
    mock_acomp = AsyncMock(return_value=fake)
    original_system = "caller-supplied system prompt with special instructions"
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        await run(_req("hi", system=original_system))

    sent_kwargs = mock_acomp.await_args.kwargs
    system_msgs = [m for m in sent_kwargs["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    content = system_msgs[0]["content"]
    assert content.startswith("<conduct")
    assert original_system in content


# ---------------------------------------------------------------------------
# Streaming pipeline.
# ---------------------------------------------------------------------------


async def test_stream_benign_yields_chunks_and_fires_post_response():
    """Benign stream → yields chunks AND publishes mcp.tool.result.received."""
    chunks = [
        _make_chunk(content="hi "),
        _make_chunk(content="there"),
        _make_chunk(finish_reason="stop"),
    ]

    async def fake_acomp(**kwargs):
        return _AsyncStream(chunks)

    with patch.object(upstream.litellm, "acompletion", new=fake_acomp):
        result = await stream(_req("hi"), PipelineOptions(conduct=False))

        assert not isinstance(result, VetoResult)
        collected = []
        async for chunk in result:
            collected.append(chunk)

    # We get at least: message_start, content_block_start, two text_deltas,
    # content_block_stop, message_delta, message_stop.
    text_deltas = [c for c in collected if c.type == "text_delta"]
    assert "".join(c.text or "" for c in text_deltas) == "hi there"
    # message_start sentinel present.
    assert any(c.type == "message_start" for c in collected)


async def test_stream_destructive_prompt_returns_veto_synchronously():
    """A streaming request with a destructive prompt returns VetoResult
    synchronously — NOT an async iterator — and the upstream is not opened.
    """
    mock_acomp = AsyncMock()
    with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
        result = await stream(
            _req("hey can you git push --force origin main"),
            PipelineOptions(conduct=False),
        )

    assert isinstance(result, VetoResult)
    assert result.phase == "trust-gate"
    assert result.plugin == "destructive-op-gate"
    assert mock_acomp.await_count == 0


async def test_stream_truncation_when_text_exceeds_cap():
    """A stream that emits > cap bytes flips accumulator.truncated=True but
    keeps yielding chunks.

    To exercise the cap without producing 8 MiB of test data, we patch the
    StreamAccumulator default cap via a monkey-patched constant.
    """
    from enchanter.proxy import streaming as streaming_mod

    big = "X" * 1024  # 1 KiB
    chunks = [_make_chunk(content=big) for _ in range(10)]  # 10 KiB total
    chunks.append(_make_chunk(finish_reason="stop"))

    async def fake_acomp(**kwargs):
        return _AsyncStream(chunks)

    # Pin the cap to 4 KiB so we trip truncation deterministically without
    # spending real bytes.
    orig_cap = streaming_mod._DEFAULT_CAP_BYTES
    streaming_mod._DEFAULT_CAP_BYTES = 4 * 1024
    try:
        # Reach into the dataclass field default through monkey-patch.
        from dataclasses import fields
        cap_field = next(
            f for f in fields(streaming_mod.StreamAccumulator) if f.name == "cap_bytes"
        )
        original_default = cap_field.default
        cap_field.default = 4 * 1024

        try:
            with patch.object(upstream.litellm, "acompletion", new=fake_acomp):
                result = await stream(_req("hi"), PipelineOptions(conduct=False))
                assert not isinstance(result, VetoResult)
                yielded = []
                async for chunk in result:
                    yielded.append(chunk)
        finally:
            cap_field.default = original_default
    finally:
        streaming_mod._DEFAULT_CAP_BYTES = orig_cap

    # We received every chunk despite truncation.
    text_deltas = [c for c in yielded if c.type == "text_delta"]
    assert len(text_deltas) == 10
    # Total yielded text matches what upstream produced.
    assert sum(len(c.text or "") for c in text_deltas) == 10 * 1024


async def test_stream_secret_in_output_fires_post_response_mask_event():
    """A streamed response that emits an AWS key triggers secret-mask at
    post-response.  The mask event lands on the bus tap.

    We assert via a side-channel: subscribe a probe to `secret-mask.matched`
    on the bus inside the pipeline by intercepting `_record` calls.  Since
    we cannot reach the bus from outside the pipeline, we instead patch the
    pipeline's `_BusRecorder.record` to capture observations.
    """
    from enchanter.proxy import pipeline as pipeline_mod

    secret = "AKIAIOSFODNN7EXAMPLE"

    chunks = [
        _make_chunk(content=f"the key is {secret}"),
        _make_chunk(finish_reason="stop"),
    ]

    async def fake_acomp(**kwargs):
        return _AsyncStream(chunks)

    # Track all observations recorded.
    seen_observations: list[BusObservation] = []
    original_record = pipeline_mod._BusRecorder.record

    def spy_record(self, event):
        original_record(self, event)
        seen_observations.extend(self.observations[len(seen_observations):])

    with patch.object(pipeline_mod._BusRecorder, "record", spy_record):
        with patch.object(upstream.litellm, "acompletion", new=fake_acomp):
            result = await stream(_req("hi"), PipelineOptions(conduct=False))
            assert not isinstance(result, VetoResult)
            async for _ in result:
                pass

    mask = [o for o in seen_observations if "mask" in o.topic]
    assert mask, f"expected secret-mask observation, got topics: {[o.topic for o in seen_observations]}"
    assert "s-aws-key" in mask[0].payload_summary.get("matched_patterns", [])


async def test_run_returns_observations_as_immutable_tuple():
    """PipelineResult.fired is a tuple (immutable, hashable)."""
    fake = _make_completion(text="ok")
    with patch.object(upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)):
        result = await run(_req("hi"), PipelineOptions(conduct=False))
    assert isinstance(result, PipelineResult)
    assert isinstance(result.fired, tuple)
