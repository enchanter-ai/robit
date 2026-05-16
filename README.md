# enchanter-agent

> **Auth setup**: see [`docs/auth.md`](docs/auth.md) for the full env-var matrix, `.env` conventions, and how to point claude-code / codex / cursor at the proxy.

## Installation

```bash
pip install enchanter-agent
```

The `anthropic` SDK is a regular (non-optional) dependency, so `pip install enchanter-agent` installs it automatically. If you want to use just the mock client in a test environment without network access, the package still imports correctly — `MockLlmClient` has no `anthropic` dep at import time.

Enforcement-first MCP-aware agent runtime. Python port of the TypeScript MCP client at `client/enchanter/`, with two layers added:

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
| 5 | Packaging + CLI inspection (`insighter version|status|engines|conduct|inference|tier|serve`) | ✅ |
| 5 | MCP server mode (stdio + Streamable-HTTP, engines as MCP tools) | ✅ |
| 6 | LLM proxy mode (Anthropic + OpenAI + Gemini wire formats, streaming, LiteLLM upstream) | ✅ |
| 7 | Streaming secret-mask + event-emitter scaffold + 4 engine wire-ins (rate-limiter, cost-ledger, trust-scorer, tool-poisoning) | ✅ |
| 7 | Polyglot runtime (Python + sidecar via JSON-RPC stdio) | ✅ |
| 7 | Sidecar trust hardening + JSONL audit log (source-allowlist, topic-allowlist, forgery detection) | ✅ |
| 7 | **Rust** Aho-Corasick sidecar engine (proof-of-concept polyglot engine) | ✅ |
| 7 | Inference substrate live wire-in (proxy emits cross-session artifacts) | ✅ |
| 7 | Byte pass-through fast path (env-gated + key allow-listed + audit JSONL) | ✅ |
| 7 | Opt-in parallel plugin dispatch (`concurrent_safe = true` in `engine.toml`) | ✅ |
| 8 | Pass-through auth on enforced proxy path (`--passthrough-auth`) | ✅ |
| 8 | `ChatGptClient` for ChatGPT subscription (Plus/Team/Enterprise) auth | ✅ |
| 8 | Codex CLI adapter — `/v1/responses` (Responses API, not chat completions) | ✅ |
| 9 | `.env` auto-loading (cwd + `~/.enchanter/.env`); shell env wins | ✅ |
| 9 | `enchanter login chatgpt` / `enchanter logout` / `enchanter login --list` | ✅ |
| 9 | ChatGPT-login through proxy (`--passthrough-auth` handles ChatGPT JWTs end-to-end via direct stdlib HTTP) | ✅ |
| 9 | Authentication docs at [`docs/auth.md`](docs/auth.md) | ✅ |

**0.7.0** — Two binaries:
- `enchanter` — interactive coding-agent CLI (REPL, 7 built-in tools, MCP client, plan mode, subagent dispatch, live cost ticker, enforcement chips). Supports Anthropic Pro/Max OAuth, OpenAI API key, and ChatGPT subscription. Run `enchanter login chatgpt` to authenticate via your ChatGPT Plus/Team subscription.
- `insighter` — runtime inspector + proxy. The `--passthrough-auth` flag forwards host agents' subscription tokens (Anthropic OAuth, OpenAI Bearer, Gemini API key, ChatGPT JWTs) to upstream so claude-code / codex can use their own billing while still getting enforcement.

Both binaries auto-load `.env` from cwd and `~/.enchanter/.env` at startup. Shell env wins over `.env`. See [`docs/auth.md`](docs/auth.md) for the full env-var matrix.

1119 tests passing across all suites.

## LLM proxy quickstart

> **CLI binaries (0.5.0):** `enchanter` is the coding-agent CLI; `insighter` is the runtime inspector (engines, conduct, inference, proxy server).

`insighter serve --proxy 127.0.0.1:8000` runs a wire-format proxy that accepts requests on three endpoints and routes upstream via LiteLLM:

- `POST /v1/messages` — Anthropic Messages API shape
- `POST /v1/chat/completions` — OpenAI Chat Completions shape (also covers OpenAI-compatible providers: Groq, Together, Mistral, Ollama, vLLM, OpenRouter)
- `POST /v1beta/models/{model}:generateContent` and `:streamGenerateContent` — Gemini shape

Every request runs through the 7-phase lifecycle: `destructive-op-gate` + `cve-pattern-gate` veto destructive prompts (HTTP 451), conduct rules are injected into the system prompt, `secret-mask` scans the response. Streaming SSE is supported on all three endpoints.

Point a host agent's base URL at the proxy and it gets enforcement transparently:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 cursor  # or any OpenAI-compatible host
```

Per-request override: append `?conduct=off` to skip conduct injection. Disable a wire format globally: `--accept openai,gemini`. Upstream auth: LiteLLM reads `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` from the environment.

## License

Apache-2.0
