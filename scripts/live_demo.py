"""Live proxy demo. Starts the proxy on :8000, mocks litellm.acompletion,
and prints what would hit it. Press Ctrl-C to stop.

Usage:
    python scripts/live_demo.py

The mock upstream returns a deterministic response so you can see the full
HTTP exchange (headers, body, enforcement, fast-path, cost, audit) end-to-end
without needing real provider credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock, patch

PORT = 8000
DEMO_KEY = "sk-demo-trusted-key"


class _MockChoice:
    def __init__(self, content: str):
        self.message = type(
            "M", (), {"content": content, "tool_calls": None, "role": "assistant"}
        )()
        self.finish_reason = "stop"
        self.index = 0


class _MockUsage:
    def __init__(self):
        self.prompt_tokens = 42
        self.completion_tokens = 18
        self.total_tokens = 60


class _MockCompletion:
    def __init__(self, model: str, text: str):
        self.id = "demo-completion-1"
        self.created = 1715620000
        self.model = model
        self.choices = [_MockChoice(text)]
        self.usage = _MockUsage()


async def _mock_acompletion(*args, **kwargs):
    model = kwargs.get("model", "unknown")
    return _MockCompletion(model, f"[demo upstream reply for {model}]")


async def _passthrough_mock(method, path, headers, body, *, upstream_provider, model):
    # Simulate the upstream returning a 200 with a tiny body
    return (
        200,
        {"Content-Type": "application/json"},
        json.dumps(
            {
                "id": "demo-fp-1",
                "type": "message",
                "model": model,
                "content": [{"type": "text", "text": "[fast-path bypass reply]"}],
            }
        ).encode(),
    )


async def main() -> int:
    # Spin up a tmp state dir so audit + fastpath files land there
    tmp = Path(tempfile.mkdtemp(prefix="enchanter-demo-"))
    print(f"State dir: {tmp}")

    # Write a fast-path allowlist with the demo key
    allowlist = {
        "keys": [hashlib.sha256(DEMO_KEY.encode()).hexdigest()],
        "models": ["claude-3-5-sonnet-20241022"],
    }
    (tmp / "fastpath-allowlist.json").write_text(json.dumps(allowlist, indent=2))
    print(f"Wrote fast-path allowlist with 1 key for model claude-3-5-sonnet-20241022")

    # Env: enable fast-path + inference substrate
    os.environ["ENCHANTER_STATE_DIR"] = str(tmp)
    os.environ["ENCHANTER_ALLOW_FASTPATH_BYPASS"] = "1"
    os.environ["ENCHANTER_INFERENCE_ENABLED"] = "1"
    os.environ["ENCHANTER_INFERENCE_STATE"] = str(tmp / "inference")

    from robit.proxy import ProxyServer, fastpath
    fastpath.load_config(force_reload=True)

    print(f"\n=== Starting proxy on http://127.0.0.1:{PORT} ===")
    print(f"  conduct injection: ON")
    print(f"  fast-path:         ENABLED (key {DEMO_KEY!r}, hash {hashlib.sha256(DEMO_KEY.encode()).hexdigest()[:12]})")
    print(f"  inference:         ENABLED -> {tmp / 'inference' / 'artifacts.jsonl'}")
    print(f"  audit:             {tmp / 'audit'}")
    print()
    print("Demo requests to copy/paste in a second terminal:")
    print()
    print("# 1. Benign request through full pipeline")
    print(f"curl -i -X POST http://127.0.0.1:{PORT}/v1/messages \\")
    print(f"  -H 'Content-Type: application/json' -H 'x-api-key: sk-other-key' \\")
    print(f'  -d \'{{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{{"role":"user","content":"say hi"}}]}}\'')
    print()
    print("# 2. Destructive prompt — pipeline vetoes BEFORE upstream")
    print(f"curl -i -X POST http://127.0.0.1:{PORT}/v1/messages \\")
    print(f"  -H 'Content-Type: application/json' -H 'x-api-key: sk-other-key' \\")
    print(f'  -d \'{{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{{"role":"user","content":"run rm -rf / on prod"}}]}}\'')
    print()
    print("# 3. Trusted key triggers fast-path bypass (skips conduct + gates)")
    print(f"curl -i -X POST http://127.0.0.1:{PORT}/v1/messages \\")
    print(f"  -H 'Content-Type: application/json' -H 'x-api-key: {DEMO_KEY}' \\")
    print(f'  -d \'{{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{{"role":"user","content":"hi"}}]}}\'')
    print()
    print("Press Ctrl-C to stop. Watch the audit + artifacts files grow:")
    print(f"  tail -f {tmp / 'audit' / 'fastpath-bypass.jsonl'}")
    print(f"  tail -f {tmp / 'inference' / 'artifacts.jsonl'}")
    print()

    with patch("robit.proxy.upstream.litellm.acompletion", AsyncMock(side_effect=_mock_acompletion)), \
         patch("robit.proxy.fastpath.passthrough", AsyncMock(side_effect=_passthrough_mock)):
        server = ProxyServer(host="127.0.0.1", port=PORT)
        host, port = await server.start()
        serve_task = asyncio.create_task(server.serve_forever())

        # Wait for Ctrl-C
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _stop():
            stop_event.set()

        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                with suppress(NotImplementedError):
                    loop.add_signal_handler(sig, _stop)

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down...")
            await server.close()
            serve_task.cancel()
            with suppress(asyncio.CancelledError):
                await serve_task

    print("\nFinal audit + artifact files:")
    audit = tmp / "audit" / "fastpath-bypass.jsonl"
    artifacts = tmp / "inference" / "artifacts.jsonl"
    if audit.exists():
        print(f"  audit ({audit}):")
        for line in audit.read_text().splitlines()[-5:]:
            print(f"    {line}")
    if artifacts.exists():
        print(f"  artifacts ({artifacts}):")
        for line in artifacts.read_text().splitlines()[-5:]:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
