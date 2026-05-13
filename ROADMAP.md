# Enchanter Agent — Build Roadmap

Managed-split execution: each row is a self-contained subagent dispatch.
Waves run in parallel within a wave; serial across waves so contract issues
surfaced in wave N can be fixed before wave N+1 fires.

## Status legend

- ✅ ported, tests green
- 🟡 in flight
- ⬜ not started
- ❌ failed (needs retry)

## Wave 0 — Spine ✅

| Component | Source | Target | Status |
|---|---|---|---|
| Lifecycle (7 phases) | `client/enchanter/src/orchestration/lifecycle.ts` | `enchanter/core/lifecycle.py` | ✅ |
| In-process bus + ACK tracker | `client/enchanter/src/bus/pubsub.ts` | `enchanter/core/bus.py` | ✅ |
| PluginAdapter Protocol | `client/enchanter/src/plugins/plugin-contract.ts` | `enchanter/core/plugin.py` | ✅ |
| RequestContext | `client/enchanter/src/orchestration/request-context.ts` | `enchanter/core/context.py` | ✅ |
| EnchantedEvent / PluginAck | `client/enchanter/src/bus/event-types.ts` | `enchanter/core/events.py` | ✅ |
| destructive-op-gate (sylph W5) | `client/enchanter/src/plugins/sylph.adapter.ts` | `enchanter/engines/destructive_op_gate/` | ✅ |

## Wave 1 — Stateless security engines ✅

Pure regex / payload scan. No external state. Lowest risk for validating the engine contract at scale.

| Engine | Source (TS) | Target | Complexity |
|---|---|---|---|
| secret-mask | `plugins/hydra.adapter.ts` + `plugins/hydra/cve-patterns.ts` SECRET_PATTERNS_V0_1 | `enchanter/engines/secret_mask/` | small (regex + replace) |
| cve-pattern-gate | `plugins/hydra.adapter.ts` + `plugins/hydra/cve-patterns.ts` CVE_PATTERNS_V0_1 | `enchanter/engines/cve_pattern_gate/` | medium (severity-tiered) |

## Wave 2 — Algorithmic engines ✅

Stateful algorithms. Each engine carries its own state machine.

| Engine | Source | Target | Complexity |
|---|---|---|---|
| trust-scorer | `plugins/crow.adapter.ts` + crow's Beta-Bernoulli math | `enchanter/engines/trust_scorer/` | medium |
| intent-anchor | `plugins/djinn.adapter.ts` + `plugins/djinn/hmm.ts` + `djinn/shared/scripts/engines/c1_lcs.py`, `c2_hmm.py`, `c5_gauss.py` | `enchanter/engines/intent_anchor/` | medium-high |
| token-runway | `plugins/emu.adapter.ts` (Markov drift + runway forecast) | `enchanter/engines/token_runway/` | medium |
| structural-fingerprint | `plugins/naga.adapter.ts` + naga's TF-IDF/Levenshtein | `enchanter/engines/structural_fingerprint/` | medium |

## Wave 3 — Cost + dependency engines ✅

| Engine | Source | Target | Complexity |
|---|---|---|---|
| cost-ledger | `plugins/pech.adapter.ts` + `plugins/pech/ledger-store.ts` | `enchanter/engines/cost_ledger/` | medium |
| rate-limiter | pech's rate-shield logic | `enchanter/engines/rate_limiter/` | small |
| import-graph-pagerank | `plugins/gorgon.adapter.ts` + `plugins/gorgon/tarjan.ts` + python-extractor | `enchanter/engines/import_graph_pagerank/` | high (Tarjan SCC + pagerank) |

## Wave 4 — Lich (code-review engines) ✅

| Engine | Source | Target | Complexity |
|---|---|---|---|
| tool-poisoning-scan | `plugins/lich.adapter.ts` + `plugins/lich/sandbox.ts` | `enchanter/engines/tool_poisoning_scan/` | medium |
| boundary-segmenter | `plugins/sylph.adapter.ts` W2 (Jaccard sliding window) | `enchanter/engines/boundary_segmenter/` | medium |

## Wave 5 — Infrastructure (port from TS) ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| JSON-RPC protocol | `src/protocol/jsonrpc.ts` | `enchanter/protocol/jsonrpc.py` | small |
| stdio transport + 8MB cap | `src/transport/stdio.ts` | `enchanter/transport/stdio.py` | medium |
| streamable-http transport + SSE | `src/transport/streamable-http.ts` + `tls-pin.ts` | `enchanter/transport/http.py` | high |

## Wave 6 — Registry + trust-pin ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| Namespace registry + collision guard + schema-digest pinning | `src/registry/namespace.ts` | `enchanter/registry/namespace.py` | medium |
| Trust-pin (TOFU + mismatch veto + JSONL store) | `src/registry/trust-pin.ts` | `enchanter/registry/trust_pin.py` | medium |

## Wave 7 — Engine manifest loader ✅

| Component | Target | Complexity |
|---|---|---|
| `engine.toml` parser + discovery glob + registry builder | `enchanter/loader/manifest.py` | small |
| Migrate all ported engines to manifest-based registration | (touches every engine dir) | refactor |

## Wave 8 — Conduct injection layer (NEW, not from TS) ✅

| Component | Target | Complexity |
|---|---|---|
| Conduct loader from enchanter-foundations | `enchanter/conduct/loader.py` | small |
| Per-rule `enforcement:` tag parser | `enchanter/conduct/tags.py` | small |
| System-prompt XML composer (prompt-injection layer) | `enchanter/conduct/composer.py` | medium |
| Code-enforced rules registry → middleware plugins | `enchanter/conduct/middleware.py` | medium |

## Wave 9 — Inference substrate wire-in ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| inference-engine port (already Python) | `wixie/shared/scripts/inference-engine.py` | `enchanter/inference/engine.py` | small (mostly copy) |
| Catalog/briefings access | `wixie/plugins/inference-engine/state/*` | `enchanter/inference/state.py` | small |
| Wire as `post-session` + `cross-session` plugin | new | `enchanter/engines/inference_substrate/` | medium |

## Wave 10 — Deep-research engine (multi-phase, the first real "engine") ✅

The deep-research pipeline is the most contract-heavy engine: 6 phases, 3 tier dispatches, structured artifacts. Port last because it exercises everything above.

| Component | Source | Target |
|---|---|---|
| Decompose (Opus, inline) | `wixie/plugins/deep-research/skills/deep-research/SKILL.md` | `enchanter/engines/deep_research/phases/decompose.py` |
| Cast (parallel Haiku fetchers) | fetcher agent prompt | `phases/cast.py` |
| Triangulate (Sonnet) | triangulator agent | `phases/triangulate.py` |
| Gap-fill | orchestrator decision | `phases/gap_fill.py` |
| Synthesize | Opus inline | `phases/synthesize.py` |
| Verify | Haiku verifier | `phases/verify.py` |

## Wave 11 — Packaging + CLI + MCP server mode ✅

| Component | Target | Status |
|---|---|---|
| CLI entry point (`enchanter version|status|engines|conduct|inference|tier|serve`) | `enchanter/cli/__init__.py` | ✅ |
| pip-installable package metadata | `pyproject.toml` | ✅ |
| MCP server mode — stdio + Streamable-HTTP transports, tools/list + tools/call, JSON-RPC error mapping, 8 MiB body cap | `enchanter/mcp_server/` | ✅ |

Default tool set wraps `secret-mask` and `destructive-op-gate` adapters as `enchanter.scan_secrets` and `enchanter.check_destructive_op`. Additional engine wrappers are registered ad-hoc by passing `Tool` instances to `ToolRegistry.register()`. `deep_research` requires an injected LLM client + tier router and is left to the operator to register.

## Concurrency policy

- Max 5 parallel subagents per wave (learned the hard way from the format tournament — 13+ stalls).
- Each subagent runs its OWN pytest invocation to validate its engine. Parent does not re-run.
- If a wave produces a contract-shape revelation, pause; fix the contract; then resume.
- File-write isolation: each engine writes only to `enchanter/engines/<its_name>/` + `tests/engines/test_<its_name>.py`. No shared file edits.

## Verification gate per wave

A wave is "done" when:
1. All engines in the wave register cleanly with the orchestrator.
2. Engine-specific pytest passes inside each subagent.
3. Parent runs the FULL suite (`pytest tests/`) end-to-end and it stays green.
4. No regressions on prior waves.

## Wave 12 — LLM proxy mode (Anthropic + OpenAI + Gemini wire formats) ✅

Managed-split execution across 5 subagents in 3 waves.

| Wave | Agents | Files | Status |
|---|---|---|---|
| 12.0 — Foundation | 1 | `proxy/canonical.py`, `proxy/upstream.py` (LiteLLM bridge), `proxy/conduct.py` | ✅ |
| 12.1 — Adapters (× 3 parallel) | 3 | `proxy/adapters/anthropic.py`, `proxy/adapters/openai.py`, `proxy/adapters/gemini.py` | ✅ |
| 12.1.5 — Contract fix | 1 | `CanonicalChunk` gains `block_kind`/`tool_id`/`tool_name` for streaming tool fidelity | ✅ |
| 12.2 — Integration (× 2 parallel) | 2 | `proxy/pipeline.py` + `proxy/streaming.py` · `proxy/server.py` + CLI `--proxy` flag | ✅ |
| 12.2.5 — Hoist `AdapterParseError` | 1 | Shared `proxy/adapters/errors.py` — server now correctly returns 400 on malformed JSON from every wire format | ✅ |

Test count: 576 → 606 (+30 proxy-specific). Total: 606 passing. Live HTTP smoke verified: 451 veto path (cve-pattern-gate on `rm -rf /`), 404 unknown path, 400 malformed JSON with per-family error envelopes, conduct injection toggleable.

New dependency: `litellm>=1.40` for upstream provider routing.


## Wave 13 + 14.1 — Enforcement hardening + polyglot runtime ✅

Managed-split execution across 11 subagents in 7 waves. Net delta:
616 → 751 tests (+135), version 0.3.0 → 0.4.0.

| Wave | Agents | Files | Status |
|---|---|---|---|
| 13.0 — Streaming secret-leak fix + event-emitter scaffold | 1 | `proxy/streaming.py`, `proxy/pipeline.py`, `proxy/events/` (new package) | ✅ |
| 13.1 — Engine wire-ins (× 4 parallel) | 4 | `proxy/events/{rate_limiter,cost_ledger,trust_scorer,tool_poisoning_scan}.py` + `server.py` cost-cents header | ✅ |
| 13.1.5 — Polyglot runtime contract | 1 | `loader/manifest.py` (`runtime` field), `loader/runtimes/` (new package: python + sidecar JSON-RPC) | ✅ |
| 14.1 — Sidecar trust-boundary hardening (× 2 parallel) | 2 | `loader/runtimes/sidecar.py` (source/topic validation), `loader/runtimes/_audit.py` (JSONL log) | ✅ |
| 13.2E — Rust Aho-Corasick sidecar engine | 1 | `engines/pattern_scanner_rust/` (Cargo crate, 6 patterns ported, 633 KiB binary), `tests/integration/test_rust_sidecar.py` | ✅ |
| 13.2F — Inference substrate live wire-in | 1 | `proxy/events/inference_substrate.py` (PRE_DISPATCH briefing read + POST_SESSION artifact emit) | ✅ |
| 13.2G — Byte pass-through fast path (v2) | me (after agent refusal of v1) | `proxy/fastpath.py` (env gate + per-key SHA-256 allow-list + body sniff + JSONL audit), `server.py` hook | ✅ |
| 13.3 — Opt-in parallel plugin dispatch | 1 | `core/lifecycle.py` (two-bucket dispatch), `core/plugin.py` (Protocol field), manifests | ✅ |

## Delegation-of-authority audit ✅

Sequential two-agent audit produced `docs/architecture/delegation-of-authority.md` and the canonical brief at `docs/architecture/audits/delegation-prompt.md`. The audit's Wave 14 plan (5 entries + deferred bucket) is the source of truth for the next sprint cycle. Wave 14.1 from that plan landed this cycle; 14.0, 14.2, 14.3, 14.4 are deferred.

