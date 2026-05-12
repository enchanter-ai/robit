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


# -- Main ---------------------------------------------------------------------

async def main() -> int:
    captured: list[dict[str, Any]] = []
    upstream_mock = AsyncMock(side_effect=make_upstream_mock(captured))

    print("Patching litellm.acompletion with in-process mock...")
    with patch("enchanter.proxy.upstream.litellm.acompletion", upstream_mock):
        async with proxy_running("127.0.0.1", 0) as (host, port):
            print(f"Proxy listening on {host}:{port}\n")
            await case_benign_anthropic(host, port, captured)
            await case_destructive_anthropic(host, port, captured)
            await case_benign_openai(host, port, captured)
            await case_benign_gemini(host, port, captured)
            await case_conduct_off(host, port, captured)

    print("\n[OK] All 5 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
