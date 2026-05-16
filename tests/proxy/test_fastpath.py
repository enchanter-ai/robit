"""Wave 13.2G v2 — byte pass-through (env-gated + allow-listed)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from robit.proxy import fastpath


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _make_config(
    *,
    enabled: bool = True,
    keys: tuple[str, ...] = ("sk-test-1",),
    models: tuple[str, ...] | None = ("claude-3-5-sonnet-20241022", "gpt-4o-mini", "gemini-1.5-flash"),
    max_body_bytes: int = 1_048_576,
) -> fastpath.FastPathConfig:
    return fastpath.FastPathConfig(
        enabled=enabled,
        allowed_key_hashes=frozenset(_hash(k) for k in keys),
        allowed_models=frozenset(models) if models is not None else None,
        max_body_bytes=max_body_bytes,
    )


def _body(model: str = "claude-3-5-sonnet-20241022", **extra: object) -> bytes:
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "say hi"}],
    }
    payload.update(extra)
    return json.dumps(payload).encode()


# ───────────────────────────────────────────────────────────────────────────
# Eligibility
# ───────────────────────────────────────────────────────────────────────────


async def test_env_gate_off_short_circuits() -> None:
    cfg = _make_config(enabled=False)
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, _body(), cfg)
    assert d.eligible is False
    assert d.reason == "env-gate-off"


async def test_path_not_routed() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate("POST", "/v1/bogus", {"x-api-key": "sk-test-1"}, _body(), cfg)
    assert d.eligible is False
    assert d.reason == "path-not-routed"


async def test_no_auth_header() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate("POST", "/v1/messages", {}, _body(), cfg)
    assert d.eligible is False
    assert d.reason == "no-auth-header"
    assert d.upstream_provider == "anthropic"


async def test_key_not_allowlisted() -> None:
    cfg = _make_config(keys=("sk-other",))
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, _body(), cfg)
    assert d.eligible is False
    assert d.reason == "key-not-allowlisted"


async def test_body_too_large() -> None:
    cfg = _make_config(max_body_bytes=100)
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, b"x" * 200, cfg)
    assert d.eligible is False
    assert d.reason == "body-too-large"


async def test_malformed_body() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, b"not-json", cfg)
    assert d.eligible is False
    assert d.reason == "malformed-body"


async def test_stream_true_rejected() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate(
        "POST", "/v1/messages", {"x-api-key": "sk-test-1"}, _body(stream=True), cfg,
    )
    assert d.eligible is False
    assert d.reason == "stream-true"


async def test_tools_present_rejected() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate(
        "POST", "/v1/messages", {"x-api-key": "sk-test-1"},
        _body(tools=[{"name": "foo"}]), cfg,
    )
    assert d.eligible is False
    assert d.reason == "tools-present"


async def test_model_not_allowlisted() -> None:
    cfg = _make_config(models=("only-this-model",))
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, _body(), cfg)
    assert d.eligible is False
    assert d.reason == "model-not-allowlisted"


async def test_anthropic_eligible() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate("POST", "/v1/messages", {"x-api-key": "sk-test-1"}, _body(), cfg)
    assert d.eligible is True
    assert d.reason == "authorized-bypass"
    assert d.upstream_provider == "anthropic"
    assert d.model == "claude-3-5-sonnet-20241022"


async def test_openai_eligible_with_bearer() -> None:
    cfg = _make_config()
    body = _body(model="gpt-4o-mini")
    d = await fastpath.evaluate(
        "POST", "/v1/chat/completions",
        {"authorization": "Bearer sk-test-1"}, body, cfg,
    )
    assert d.eligible is True
    assert d.upstream_provider == "openai"


async def test_gemini_eligible_with_x_goog() -> None:
    cfg = _make_config()
    d = await fastpath.evaluate(
        "POST", "/v1beta/models/gemini-1.5-flash:generateContent",
        {"x-goog-api-key": "sk-test-1"},
        json.dumps({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).encode(),
        cfg,
    )
    assert d.eligible is True
    assert d.upstream_provider == "gemini"
    assert d.model == "gemini-1.5-flash"


# ───────────────────────────────────────────────────────────────────────────
# Config loading
# ───────────────────────────────────────────────────────────────────────────


async def test_load_config_env_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", raising=False)
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    cfg = fastpath.load_config(force_reload=True)
    assert cfg.enabled is False


async def test_load_config_env_set_but_no_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", "1")
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    caplog.set_level("WARNING", logger="robit.proxy.fastpath")
    cfg = fastpath.load_config(force_reload=True)
    assert cfg.enabled is False
    assert any("does not exist" in rec.message for rec in caplog.records)


async def test_load_config_malformed_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", "1")
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    (tmp_path / "fastpath-allowlist.json").write_text("not-json", encoding="utf-8")
    caplog.set_level("WARNING", logger="robit.proxy.fastpath")
    cfg = fastpath.load_config(force_reload=True)
    assert cfg.enabled is False
    assert any("failed to parse" in rec.message for rec in caplog.records)


async def test_load_config_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", "1")
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    allowlist = {
        "keys": [_hash("sk-prod"), _hash("sk-staging")],
        "models": ["claude-3-5-sonnet-20241022"],
        "max_body_bytes": 524288,
    }
    (tmp_path / "fastpath-allowlist.json").write_text(json.dumps(allowlist), encoding="utf-8")
    cfg = fastpath.load_config(force_reload=True)
    assert cfg.enabled is True
    assert len(cfg.allowed_key_hashes) == 2
    assert cfg.allowed_models == frozenset({"claude-3-5-sonnet-20241022"})
    assert cfg.max_body_bytes == 524288


# ───────────────────────────────────────────────────────────────────────────
# Audit
# ───────────────────────────────────────────────────────────────────────────


async def test_record_bypass_appends_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    await fastpath.record_bypass(
        upstream_provider="anthropic",
        key_hash_short="abc123def456",
        model="claude-3-5-sonnet-20241022",
        body_size=1234,
        upstream_status=200,
    )
    audit_file = tmp_path / "audit" / "fastpath-bypass.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "proxy.fastpath.bypass"
    assert record["upstream_provider"] == "anthropic"
    assert record["model"] == "claude-3-5-sonnet-20241022"
    assert record["key_hash_short"] == "abc123def456"
    assert record["body_size"] == 1234
    assert record["upstream_status"] == 200


# ───────────────────────────────────────────────────────────────────────────
# Server integration: bypass fires end-to-end with mocked upstream
# ───────────────────────────────────────────────────────────────────────────


async def test_integration_bypass_fires_with_x_enchanter_fastpath_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from robit.proxy import ProxyServer

    monkeypatch.setenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", "1")
    monkeypatch.setenv("ENCHANTER_STATE_DIR", str(tmp_path))
    allowlist = {
        "keys": [_hash("sk-trusted")],
        "models": ["claude-3-5-sonnet-20241022"],
    }
    (tmp_path / "fastpath-allowlist.json").write_text(json.dumps(allowlist), encoding="utf-8")
    # Reset cached config so the new env var takes effect
    fastpath.load_config(force_reload=True)

    # Mock passthrough to skip real network
    fake_response = (
        200,
        {"Content-Type": "application/json"},
        b'{"id":"msg_fake","type":"message","content":[{"type":"text","text":"hi"}]}',
    )

    with patch("robit.proxy.fastpath.passthrough", new=AsyncMock(return_value=fake_response)):
        server = ProxyServer(host="127.0.0.1", port=0)
        host, port = await server.start()
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            reader, writer = await asyncio.open_connection(host, port)
            body = _body()
            req = (
                f"POST /v1/messages HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"x-api-key: sk-trusted\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(req)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            await writer.wait_closed()
            assert b"X-Enchanter-FastPath: bypass" in response
            assert b"HTTP/1.1 200" in response
        finally:
            await server.close()
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass


async def test_integration_pipeline_path_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When the env var is unset, the fast path block is skipped entirely;
    the request flows through the normal pipeline. Test by asserting the
    response does NOT carry X-Enchanter-FastPath."""
    from robit.proxy import ProxyServer
    from robit.proxy.canonical import CanonicalResponse, CanonicalUsage, TextPart

    monkeypatch.delenv("ENCHANTER_ALLOW_FASTPATH_BYPASS", raising=False)
    fastpath.load_config(force_reload=True)

    # Mock LiteLLM so we hit the pipeline without network. Patch the
    # binding in pipeline.py (which imports call_upstream by name).
    from robit.proxy import pipeline as pipeline_mod

    async def fake_call(req):  # noqa: ANN001
        return CanonicalResponse(
            model=req.model,
            content=(TextPart(text="hi"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=5, output_tokens=2),
        )

    with patch.object(pipeline_mod, "call_upstream", new=fake_call):
        server = ProxyServer(host="127.0.0.1", port=0)
        host, port = await server.start()
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            reader, writer = await asyncio.open_connection(host, port)
            body = _body()
            req = (
                f"POST /v1/messages HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(req)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            await writer.wait_closed()
            assert b"X-Enchanter-FastPath" not in response
            assert b"HTTP/1.1 200" in response
        finally:
            await server.close()
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
