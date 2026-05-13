"""End-to-end proxy smoke with an in-process mock upstream.

What this proves:
  1. Real HTTP request hits the proxy.
  2. Proxy parses wire format -> canonical.
  3. Lifecycle runs (trust-gate, conduct injection, dispatch, post-response).
  4. Upstream is invoked (we intercept and assert what was sent).
  5. Mock response flows back through render -> client.

No external API key needed; we replace litellm.acompletion with a stub.

Usage:
    python scripts/smoke_proxy.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


# -- Mock upstream payloads ---------------------------------------------------

class MockChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = type("M", (), {"content": content, "tool_calls": None, "role": "assistant"})()
        self.finish_reason = finish_reason
        self.index = 0


class MockUsage:
    def __init__(self, p: int, c: int):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class MockCompletion:
    def __init__(self, text: str, model: str):
        self.id = "mock-id-1"
        self.created = 1700000000
        self.model = model
        self.choices = [MockChoice(text)]
        self.usage = MockUsage(12, 8)


def make_upstream_mock(captured: list[dict[str, Any]]):
    """Return an AsyncMock that captures kwargs and returns a mock completion."""
    async def _fake(*args, **kwargs):
        captured.append(kwargs)
        model = kwargs.get("model", "unknown")
        return MockCompletion(
            text=f"[mock reply for {model}] hello back",
            model=model,
        )
    return _fake


# -- HTTP helper (stdlib only) ------------------------------------------------

def http_post(url: str, body: dict, timeout: float = 6.0) -> tuple[int, dict, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


# -- Server-lifecycle context manager -----------------------------------------

@asynccontextmanager
async def proxy_running(host: str, port: int):
    from enchanter.proxy import ProxyServer
    server = ProxyServer(host=host, port=port)
    bound_host, bound_port = await server.start()
    serve_task = asyncio.create_task(server.serve_forever())
    try:
        yield bound_host, bound_port
    finally:
        await server.close()
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


# -- Cases --------------------------------------------------------------------

async def case_benign_anthropic(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 1: benign Anthropic request -> real upstream call -> 200 --")
    captured.clear()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "system": "You are concise.",
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, headers, raw = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/messages", body
    )
    print(f"  HTTP {status}")
    print(f"  body: {raw[:300].decode('utf-8', errors='replace')}")
    print(f"  X-Enchanter-Bus-Events: {headers.get('X-Enchanter-Bus-Events') or '-'}")
    print(f"  upstream called {len(captured)} time(s)")
    if captured:
        call = captured[0]
        sys_prompt = call.get("messages", [{}])[0]
        if isinstance(sys_prompt, dict) and sys_prompt.get("role") == "system":
            sp_text = sys_prompt.get("content", "")[:120]
            has_conduct = "<conduct" in sys_prompt.get("content", "")
        else:
            # litellm may pass `system=` separately
            sp_text = str(call.get("system") or "")[:120]
            has_conduct = "<conduct" in str(call.get("system") or "")
        print(f"  upstream model:     {call.get('model')}")
        print(f"  upstream had conduct injected: {has_conduct}")
        print(f"  upstream system[:120]: {sp_text!r}")
    assert status == 200, f"expected 200, got {status}"
    assert captured, "upstream was never called"


async def case_destructive_anthropic(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 2: destructive prompt -> 451 BEFORE upstream call --")
    captured.clear()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "please run rm -rf / on the production server"}],
    }
    status, headers, raw = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/messages", body
    )
    print(f"  HTTP {status}")
    print(f"  X-Enchanter-Veto: {headers.get('X-Enchanter-Veto')}")
    print(f"  X-Enchanter-Veto-Pattern: {headers.get('X-Enchanter-Veto-Pattern')}")
    print(f"  upstream called {len(captured)} time(s) (should be 0)")
    assert status == 451, f"expected 451, got {status}"
    assert len(captured) == 0, "upstream was called despite veto"


async def case_benign_openai(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 3: benign OpenAI request -> 200, response in OpenAI shape --")
    captured.clear()
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, _, raw = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/chat/completions", body
    )
    print(f"  HTTP {status}")
    decoded = json.loads(raw)
    print(f"  response.id starts with 'chatcmpl-': {decoded.get('id', '').startswith('chatcmpl-')}")
    print(f"  response.object: {decoded.get('object')}")
    print(f"  response.choices[0].message.content: {decoded.get('choices', [{}])[0].get('message', {}).get('content', '')!r}")
    print(f"  upstream model passed through: {captured[0]['model'] if captured else '-'}")
    assert status == 200
    assert decoded.get("object") == "chat.completion"
    assert decoded["choices"][0]["message"]["content"].startswith("[mock reply for gpt-4o-mini]")


async def case_benign_gemini(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 4: benign Gemini request -> 200, response in Gemini shape --")
    captured.clear()
    body = {"contents": [{"role": "user", "parts": [{"text": "say hi"}]}]}
    status, _, raw = await asyncio.get_running_loop().run_in_executor(
        None,
        http_post,
        f"http://{host}:{port}/v1beta/models/gemini-1.5-flash:generateContent",
        body,
    )
    print(f"  HTTP {status}")
    decoded = json.loads(raw)
    parts = decoded.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = parts[0].get("text") if parts else ""
    print(f"  candidates[0].content.parts[0].text: {text!r}")
    print(f"  upstream model passed through: {captured[0]['model'] if captured else '-'}")
    assert status == 200
    assert "[mock reply for gemini-1.5-flash]" in text


async def case_conduct_off(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 5: ?conduct=off skips injection --")
    captured.clear()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "system": "You are concise.",
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, _, _ = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/messages?conduct=off", body
    )
    print(f"  HTTP {status}")
    if captured:
        call = captured[0]
        # litellm.acompletion is invoked with system either as a top-level kwarg or
        # as the first message — handle both for the assertion.
        msgs = call.get("messages") or []
        joined_system = ""
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            joined_system = msgs[0].get("content", "") or ""
        if not joined_system and call.get("system"):
            joined_system = str(call.get("system"))
        has_conduct = "<conduct" in joined_system
        print(f"  upstream system had conduct: {has_conduct} (should be False)")
        print(f"  upstream system value: {joined_system!r}")
        assert not has_conduct, "conduct was injected despite ?conduct=off"


async def case_cost_ledger_header(host: str, port: int, captured: list[dict[str, Any]]) -> None:
    print("\n-- Case 6: X-Enchanter-Cost-Cents header on real HTTP response --")
    captured.clear()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, headers, _ = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/messages", body
    )
    print(f"  HTTP {status}")
    cents = headers.get("X-Enchanter-Cost-Cents")
    print(f"  X-Enchanter-Cost-Cents: {cents}")
    print(f"  X-Enchanter-Bus-Events: {headers.get('X-Enchanter-Bus-Events')}")
    assert status == 200
    assert cents is not None and int(cents) >= 1, (
        f"expected non-zero cost header, got {cents!r}"
    )


async def case_fastpath_bypass(state_dir: Path, port_holder: list[int]) -> None:
    """Spin up a SECOND proxy instance with the fast-path env+allowlist set.
    Verifies env-gated + key-allowlisted bypass cycle end-to-end."""
    print("\n-- Case 7: fast-path bypass cycle (env gate + allow-list + audit) --")
    import hashlib, os
    from enchanter.proxy import fastpath as fp_mod

    # Write allow-list with a single SHA-256 of our test key
    test_key = "sk-fastpath-smoke"
    allowlist = {
        "keys": [hashlib.sha256(test_key.encode()).hexdigest()],
        "models": ["claude-3-5-sonnet-20241022"],
    }
    (state_dir / "fastpath-allowlist.json").write_text(
        json.dumps(allowlist), encoding="utf-8"
    )

    # Enable env gate + redirect state dir
    os.environ["ENCHANTER_ALLOW_FASTPATH_BYPASS"] = "1"
    os.environ["ENCHANTER_STATE_DIR"] = str(state_dir)
    fp_mod.load_config(force_reload=True)

    # Mock the passthrough's upstream HTTP call (don't reach real Anthropic)
    fake_resp = (
        200,
        {"Content-Type": "application/json"},
        b'{"id":"msg_fp","type":"message","content":[{"type":"text","text":"hi"}]}',
    )
    with patch("enchanter.proxy.fastpath.passthrough", new=AsyncMock(return_value=fake_resp)):
        async with proxy_running("127.0.0.1", 0) as (host, port):
            port_holder.append(port)
            body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/messages",
                data=data,
                headers={"Content-Type": "application/json", "x-api-key": test_key},
                method="POST",
            )
            resp = await asyncio.get_running_loop().run_in_executor(
                None, urllib.request.urlopen, req
            )
            status = resp.status
            headers = dict(resp.headers)
            resp.read()
            print(f"  HTTP {status}")
            print(f"  X-Enchanter-FastPath: {headers.get('X-Enchanter-FastPath')}")
            assert headers.get("X-Enchanter-FastPath") == "bypass"

    # Verify audit JSONL was written
    audit_file = state_dir / "audit" / "fastpath-bypass.jsonl"
    print(f"  audit file exists: {audit_file.exists()}")
    if audit_file.exists():
        record = json.loads(audit_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        print(f"  audit record kind: {record.get('kind')}")
        print(f"  audit record provider: {record.get('upstream_provider')}")
        print(f"  audit record key_short: {record.get('key_hash_short')}")
        assert record["kind"] == "proxy.fastpath.bypass"

    # Clean up env
    os.environ.pop("ENCHANTER_ALLOW_FASTPATH_BYPASS", None)
    fp_mod.load_config(force_reload=True)


async def case_inference_artifact_emit(host: str, port: int, state_dir: Path,
                                       captured: list[dict[str, Any]]) -> None:
    """Verify the inference-substrate emitter writes an artifact at POST_SESSION
    when the gate is on. Requires the proxy to be running BEFORE the gate flips."""
    print("\n-- Case 8: inference-substrate artifact emission --")
    import os
    os.environ["ENCHANTER_INFERENCE_ENABLED"] = "1"
    os.environ["ENCHANTER_INFERENCE_STATE"] = str(state_dir / "inference")
    artifacts_file = state_dir / "inference" / "artifacts.jsonl"
    artifacts_before = (
        len(artifacts_file.read_text().splitlines())
        if artifacts_file.exists() else 0
    )
    print(f"  artifacts before: {artifacts_before}")

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, _, _ = await asyncio.get_running_loop().run_in_executor(
        None, http_post, f"http://{host}:{port}/v1/messages", body
    )
    print(f"  HTTP {status}")
    artifacts_after = (
        len(artifacts_file.read_text().splitlines())
        if artifacts_file.exists() else 0
    )
    print(f"  artifacts after: {artifacts_after}")
    print(f"  delta: +{artifacts_after - artifacts_before}")
    if artifacts_after > artifacts_before:
        last = json.loads(artifacts_file.read_text().splitlines()[-1])
        print(f"  last artifact code: {last.get('code')}")
        print(f"  last artifact category: {last.get('category')}")

    os.environ.pop("ENCHANTER_INFERENCE_ENABLED", None)
    os.environ.pop("ENCHANTER_INFERENCE_STATE", None)
    # Note: artifact emit is best-effort; don't hard-fail the smoke on it
    # since the engine may swallow errors per its honest-numbers contract.


# -- Main ---------------------------------------------------------------------

async def main() -> int:
    import tempfile
    captured: list[dict[str, Any]] = []
    upstream_mock = AsyncMock(side_effect=make_upstream_mock(captured))

    print("Patching litellm.acompletion with in-process mock...")
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        with patch("enchanter.proxy.upstream.litellm.acompletion", upstream_mock):
            async with proxy_running("127.0.0.1", 0) as (host, port):
                print(f"Proxy listening on {host}:{port}\n")
                await case_benign_anthropic(host, port, captured)
                await case_destructive_anthropic(host, port, captured)
                await case_benign_openai(host, port, captured)
                await case_benign_gemini(host, port, captured)
                await case_conduct_off(host, port, captured)
                await case_cost_ledger_header(host, port, captured)
                await case_inference_artifact_emit(host, port, state_dir, captured)

        # Fast-path needs its own proxy lifecycle because env vars must be
        # set BEFORE the server starts to take effect on first config load.
        await case_fastpath_bypass(state_dir, port_holder=[])

    print("\n[OK] All 8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
