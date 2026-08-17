# enchanter-agent

## Installation

> **Not published to PyPI.** `enchanter-agent` is unclaimed on the public
> PyPI registry (404) — do **not** run `pip install enchanter-agent`, it
> will not resolve to this project. Build from source in this repo
> instead:

```bash
git clone https://github.com/enchanter-ai/robit.git
cd robit
pip install -e .
```

The `anthropic` SDK is a regular (non-optional) dependency, so `pip install -e .` installs it automatically. If you want to use just the mock client in a test environment without network access, the package still imports correctly — `MockLlmClient` has no `anthropic` dep at import time.

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

**0.5.0** — Two binaries:
- `enchanter` — interactive coding-agent CLI (REPL, 7 built-in tools, MCP client, plan mode, subagent dispatch, live cost ticker, enforcement chips)
- `insighter` — runtime inspector (engines, conduct, inference, tier, audit, serve)

1029 tests passing across engines, conduct, lifecycle, inference, integration, MCP server, proxy, fastpath, events, runtimes, audit, agent core, tools, widgets, MCP client, plan mode, and subagents suites.

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
