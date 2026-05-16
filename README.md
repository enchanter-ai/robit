# robit

> **Sister project**: [`enchanter-ai/beholder`](https://github.com/enchanter-ai/beholder) is the TypeScript MCP-client SDK + Rust observability cockpit. Robit is the Python coding agent. Both products enforce the same plugin contracts but at different boundaries — beholder between agent ↔ MCP tool servers, robit on its own LLM calls.

> **Auth setup**: see [`docs/auth.md`](docs/auth.md) for the env-var matrix and `.env` conventions.

## Installation

```bash
pip install robit
```

The `anthropic` SDK is a regular (non-optional) dependency, so `pip install robit` installs it automatically. If you want to use just the mock client in a test environment without network access, the package still imports correctly — `MockLlmClient` has no `anthropic` dep at import time.

Enforcement-first Python coding-agent CLI. Every LLM turn rides a 7-phase lifecycle with conduct injection + engine vetoes + secret-mask. Every tool call (`bash`, `file_write`, `file_edit`, …) runs through the same engines before execution.

- **Conduct injection** — system-prompt XML wrapping of relevant conduct modules per turn
- **Inference substrate** — cross-session accumulation via `inference-engine.py` (Wald SPRT, Beta-Binomial)

## Three-layer architecture

```
Conduct injection (NEW)         per-rule enforcement: tag (code|prompt|hybrid)
Enforcement runtime (PORTED)    7-phase lifecycle + plugin protocol + trust-pin + transports
Inference substrate (WIRE-IN)   inference-engine.py + catalog.json + briefings
```

## Status

| Phase | Component | Status |
|---|---|---|
| 0 | Lifecycle + bus + plugin protocol + context | ✅ |
| 1 | 14 engines ported (destructive-op-gate, secret-mask, cve-pattern-gate, trust-scorer, intent-anchor, token-runway, structural-fingerprint, cost-ledger, rate-limiter, import-graph-pagerank, tool-poisoning-scan, boundary-segmenter, inference-substrate, deep-research) | ✅ |
| 2 | Conduct injection layer | ✅ |
| 3 | Inference substrate wire-in | ✅ |
| 4 | First engine: `deep-research` (6-phase pipeline) | ✅ |
| 5 | Packaging + CLI inspection | moved to beholder (0.8.0) |
| 5 | MCP server mode | moved to beholder (0.8.0) |
| 6 | LLM proxy HTTP server (Anthropic + OpenAI + Gemini + Codex wire formats) | moved to beholder (0.8.0) |
| 7 | Streaming secret-mask + event-emitter scaffold + 4 engine wire-ins (rate-limiter, cost-ledger, trust-scorer, tool-poisoning) | ✅ |
| 7 | Polyglot runtime (Python + sidecar via JSON-RPC stdio) | ✅ |
| 7 | Sidecar trust hardening + JSONL audit log (source-allowlist, topic-allowlist, forgery detection) | ✅ |
| 7 | **Rust** Aho-Corasick sidecar engine (proof-of-concept polyglot engine) | ✅ |
| 7 | Inference substrate live wire-in | ✅ |
| 7 | Byte pass-through fast path | moved to beholder (0.8.0) |
| 7 | Opt-in parallel plugin dispatch (`concurrent_safe = true` in `engine.toml`) | ✅ |
| 8 | Pass-through auth on enforced proxy path | moved to beholder (0.8.0) |
| 8 | `ChatGptClient` for ChatGPT subscription (Plus/Team/Enterprise) auth | ✅ |
| 8 | Codex CLI adapter | moved to beholder (0.8.0) |
| 9 | `.env` auto-loading (cwd + `~/.robit/.env`; legacy `~/.enchanter/.env` honored); shell env wins | ✅ |
| 9 | `robit login chatgpt` / `robit logout` / `robit login --list` | ✅ |
| 9 | ChatGPT-login through proxy (`--passthrough-auth` handles ChatGPT JWTs end-to-end via direct stdlib HTTP) | ✅ |
| 9 | Authentication docs at [`docs/auth.md`](docs/auth.md) | ✅ |

**0.8.0** — One binary, focused: `robit` is the interactive coding-agent CLI. REPL + 7 built-in tools (`file_read`, `file_write`, `file_edit`, `glob`, `grep`, `bash`, `web_fetch`), MCP client for external tool servers, plan mode, subagent dispatch, live cost ticker, enforcement chips. Supports Anthropic Pro/Max OAuth, OpenAI API key, and ChatGPT subscription. Run `robit login chatgpt` to authenticate via your ChatGPT Plus/Team subscription.

`robit` auto-loads `.env` from cwd and `~/.robit/.env` at startup (legacy `~/.enchanter/.env` still honored with a one-shot deprecation notice). Shell env wins over `.env`. See [`docs/auth.md`](docs/auth.md) for the full env-var matrix.

898 tests passing across the coding-agent + engines + conduct + lifecycle + inference + auth + tools surfaces.

## Quickstart

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...    # or CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat...

# One-shot
robit "scan auth.py for hardcoded credentials and propose fixes"

# Interactive REPL (Textual UI with live cost ticker, enforcement chips, diff approvals)
robit

# ChatGPT subscription (one-time)
robit login chatgpt
```

## Inspection + proxy (separate product: beholder)

Wave 20 (0.8.0) removed the proxy HTTP server, MCP server, and inspector CLI from robit. The inspector/observability story lives in [`enchanter-ai/beholder`](https://github.com/enchanter-ai/beholder) — a TypeScript MCP-client SDK + Rust cockpit. To inspect what robit (or any MCP-speaking process) is doing, install beholder.

## License

Apache-2.0
