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

## Phase 0 — done when

A no-op plugin registered against all 7 phases, a request dispatched through the lifecycle, every phase emits the right event, every phase collects ACKs correctly, and the test goes green.

## Status

| Phase | Component | Status |
|---|---|---|
| 0 | Lifecycle + bus + plugin protocol + context | in progress |
| 0 | Validation test | in progress |
| 1 | Port 9 plugin business logic (crow, djinn, emu, gorgon, hydra, lich, naga, pech, sylph) | not started |
| 2 | Conduct injection layer | not started |
| 3 | Inference substrate wire-in | not started |
| 4 | First engine: `deep-research` | not started |
| 5 | Packaging + CLI | not started |

## License

Apache-2.0
