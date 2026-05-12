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

445 tests passing across engines, conduct, lifecycle, inference, integration, and MCP server suites.

## License

Apache-2.0
