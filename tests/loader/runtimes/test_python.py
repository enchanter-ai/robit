"""Tests for robit.loader.runtimes.python — Python-runtime adapter resolution."""

from __future__ import annotations

import pytest

from robit.loader.errors import EngineLoadError
from robit.loader.manifest import EngineManifest, EngineTopics
from robit.loader.runtimes import load_runtime
from robit.loader.runtimes.python import load_python_adapter


def _python_manifest(adapter: str, name: str = "test") -> EngineManifest:
    return EngineManifest(
        name=name,
        description="d",
        version="1.0.0",
        phases=("trust-gate",),
        required=False,
        budget_tier="always",
        topics=EngineTopics(subscribes=(), emits=()),
        runtime="python",
        adapter=adapter,
    )


def test_python_runtime_loads_real_engine() -> None:
    """A bundled engine (secret_mask) loads via the registry without error."""
    m = _python_manifest("robit.engines.secret_mask.adapter:adapter")
    adapter = load_runtime(m)
    assert adapter is not None
    assert hasattr(adapter, "on_phase")
    # PluginAdapter contract: name, phases, required, topics, budget_tier present.
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "phases")
    assert hasattr(adapter, "topics")


@pytest.mark.asyncio
async def test_python_runtime_on_phase_works_on_loaded_engine() -> None:
    """on_phase round-trip on a real engine returns a PluginAck."""
    from robit.core import PluginAck, create_request_context
    from robit.core.events import EnchantedEvent

    m = _python_manifest("robit.engines.secret_mask.adapter:adapter")
    adapter = load_runtime(m)

    ctx = create_request_context(session_id="s", budget_tier="HIGH")
    # secret-mask scans payload.result; an event with no secrets ack's clean.
    event = EnchantedEvent(
        id="evt-1",
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="orchestrator",
        budget_tier="HIGH",
        ts=0,
        payload={"result": "nothing sensitive here"},
    )
    ack = await adapter.on_phase(event, ctx)
    assert isinstance(ack, PluginAck)
    assert ack.status == "ack"


def test_python_runtime_missing_module_raises_engine_load_error() -> None:
    m = _python_manifest("definitely.not.a.real.module:adapter")
    with pytest.raises(EngineLoadError) as exc_info:
        load_runtime(m)
    err = exc_info.value
    assert err.engine_name == "test"
    assert "definitely.not.a.real.module" in str(err)


def test_python_runtime_missing_attribute_raises_engine_load_error() -> None:
    # Module exists, attribute does not.
    m = _python_manifest("robit.engines.secret_mask.adapter:not_an_attribute")
    with pytest.raises(EngineLoadError) as exc_info:
        load_runtime(m)
    assert "not_an_attribute" in str(exc_info.value)


def test_python_runtime_invalid_notation_raises_engine_load_error() -> None:
    # No colon — invalid module:attr notation.
    m = _python_manifest("invalid-no-colon")
    with pytest.raises(EngineLoadError):
        load_python_adapter(m)
