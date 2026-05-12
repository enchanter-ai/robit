# enchanter-agent

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
| 5 | Packaging + CLI inspection (`enchanter version|status|engines|conduct|inference|tier|serve`) | ✅ |
| 5 | MCP server mode (stdio + Streamable-HTTP, engines as MCP tools) | ✅ |
| 6 | LLM proxy mode (Anthropic + OpenAI + Gemini wire formats, streaming, LiteLLM upstream) | ✅ |

606 tests passing across engines, conduct, lifecycle, inference, integration, MCP server, and proxy suites.

## LLM proxy quickstart

`enchanter serve --proxy 127.0.0.1:8000` runs a wire-format proxy that accepts requests on three endpoints and routes upstream via LiteLLM:

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
