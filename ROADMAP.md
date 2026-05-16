# Robit — Build Roadmap

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
| Lifecycle (7 phases) | `client/enchanter/src/orchestration/lifecycle.ts` | `robit/core/lifecycle.py` | ✅ |
| In-process bus + ACK tracker | `client/enchanter/src/bus/pubsub.ts` | `robit/core/bus.py` | ✅ |
| PluginAdapter Protocol | `client/enchanter/src/plugins/plugin-contract.ts` | `robit/core/plugin.py` | ✅ |
| RequestContext | `client/enchanter/src/orchestration/request-context.ts` | `robit/core/context.py` | ✅ |
| EnchantedEvent / PluginAck | `client/enchanter/src/bus/event-types.ts` | `robit/core/events.py` | ✅ |
| destructive-op-gate (sylph W5) | `client/enchanter/src/plugins/sylph.adapter.ts` | `robit/engines/destructive_op_gate/` | ✅ |

## Wave 1 — Stateless security engines ✅

Pure regex / payload scan. No external state. Lowest risk for validating the engine contract at scale.

| Engine | Source (TS) | Target | Complexity |
|---|---|---|---|
| secret-mask | `plugins/hydra.adapter.ts` + `plugins/hydra/cve-patterns.ts` SECRET_PATTERNS_V0_1 | `robit/engines/secret_mask/` | small (regex + replace) |
| cve-pattern-gate | `plugins/hydra.adapter.ts` + `plugins/hydra/cve-patterns.ts` CVE_PATTERNS_V0_1 | `robit/engines/cve_pattern_gate/` | medium (severity-tiered) |

## Wave 2 — Algorithmic engines ✅

Stateful algorithms. Each engine carries its own state machine.

| Engine | Source | Target | Complexity |
|---|---|---|---|
| trust-scorer | `plugins/crow.adapter.ts` + crow's Beta-Bernoulli math | `robit/engines/trust_scorer/` | medium |
| intent-anchor | `plugins/djinn.adapter.ts` + `plugins/djinn/hmm.ts` + `djinn/shared/scripts/engines/c1_lcs.py`, `c2_hmm.py`, `c5_gauss.py` | `robit/engines/intent_anchor/` | medium-high |
| token-runway | `plugins/emu.adapter.ts` (Markov drift + runway forecast) | `robit/engines/token_runway/` | medium |
| structural-fingerprint | `plugins/naga.adapter.ts` + naga's TF-IDF/Levenshtein | `robit/engines/structural_fingerprint/` | medium |

## Wave 3 — Cost + dependency engines ✅

| Engine | Source | Target | Complexity |
|---|---|---|---|
| cost-ledger | `plugins/pech.adapter.ts` + `plugins/pech/ledger-store.ts` | `robit/engines/cost_ledger/` | medium |
| rate-limiter | pech's rate-shield logic | `robit/engines/rate_limiter/` | small |
| import-graph-pagerank | `plugins/gorgon.adapter.ts` + `plugins/gorgon/tarjan.ts` + python-extractor | `robit/engines/import_graph_pagerank/` | high (Tarjan SCC + pagerank) |

## Wave 4 — Lich (code-review engines) ✅

| Engine | Source | Target | Complexity |
|---|---|---|---|
| tool-poisoning-scan | `plugins/lich.adapter.ts` + `plugins/lich/sandbox.ts` | `robit/engines/tool_poisoning_scan/` | medium |
| boundary-segmenter | `plugins/sylph.adapter.ts` W2 (Jaccard sliding window) | `robit/engines/boundary_segmenter/` | medium |

## Wave 5 — Infrastructure (port from TS) ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| JSON-RPC protocol | `src/protocol/jsonrpc.ts` | `robit/protocol/jsonrpc.py` | small |
| stdio transport + 8MB cap | `src/transport/stdio.ts` | `robit/transport/stdio.py` | medium |
| streamable-http transport + SSE | `src/transport/streamable-http.ts` + `tls-pin.ts` | `robit/transport/http.py` | high |

## Wave 6 — Registry + trust-pin ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| Namespace registry + collision guard + schema-digest pinning | `src/registry/namespace.ts` | `robit/registry/namespace.py` | medium |
| Trust-pin (TOFU + mismatch veto + JSONL store) | `src/registry/trust-pin.ts` | `robit/registry/trust_pin.py` | medium |

## Wave 7 — Engine manifest loader ✅

| Component | Target | Complexity |
|---|---|---|
| `engine.toml` parser + discovery glob + registry builder | `robit/loader/manifest.py` | small |
| Migrate all ported engines to manifest-based registration | (touches every engine dir) | refactor |

## Wave 8 — Conduct injection layer (NEW, not from TS) ✅

| Component | Target | Complexity |
|---|---|---|
| Conduct loader from vis | `robit/conduct/loader.py` | small |
| Per-rule `enforcement:` tag parser | `robit/conduct/tags.py` | small |
| System-prompt XML composer (prompt-injection layer) | `robit/conduct/composer.py` | medium |
| Code-enforced rules registry → middleware plugins | `robit/conduct/middleware.py` | medium |

## Wave 9 — Inference substrate wire-in ✅

| Component | Source | Target | Complexity |
|---|---|---|---|
| inference-engine port (already Python) | `wixie/shared/scripts/inference-engine.py` | `robit/inference/engine.py` | small (mostly copy) |
| Catalog/briefings access | `wixie/plugins/inference-engine/state/*` | `robit/inference/state.py` | small |
| Wire as `post-session` + `cross-session` plugin | new | `robit/engines/inference_substrate/` | medium |

## Wave 10 — Deep-research engine (multi-phase, the first real "engine") ✅

The deep-research pipeline is the most contract-heavy engine: 6 phases, 3 tier dispatches, structured artifacts. Port last because it exercises everything above.

| Component | Source | Target |
|---|---|---|
| Decompose (Opus, inline) | `wixie/plugins/deep-research/skills/deep-research/SKILL.md` | `robit/engines/deep_research/phases/decompose.py` |
| Cast (parallel Haiku fetchers) | fetcher agent prompt | `phases/cast.py` |
| Triangulate (Sonnet) | triangulator agent | `phases/triangulate.py` |
| Gap-fill | orchestrator decision | `phases/gap_fill.py` |
| Synthesize | Opus inline | `phases/synthesize.py` |
| Verify | Haiku verifier | `phases/verify.py` |

## Wave 11 — Packaging + CLI + MCP server mode ✅

| Component | Target | Status |
|---|---|---|
| CLI entry point (`insighter version|status|engines|conduct|inference|tier|serve`, renamed from `robit` in Wave 15.rename) | `robit/insighter/__init__.py` | ✅ |
| pip-installable package metadata | `pyproject.toml` | ✅ |
| MCP server mode — stdio + Streamable-HTTP transports, tools/list + tools/call, JSON-RPC error mapping, 8 MiB body cap | `robit/mcp_server/` | ✅ |

Default tool set wraps `secret-mask` and `destructive-op-gate` adapters as `robit.scan_secrets` and `robit.check_destructive_op`. Additional engine wrappers are registered ad-hoc by passing `Tool` instances to `ToolRegistry.register()`. `deep_research` requires an injected LLM client + tier router and is left to the operator to register.

## Concurrency policy

- Max 5 parallel subagents per wave (learned the hard way from the format tournament — 13+ stalls).
- Each subagent runs its OWN pytest invocation to validate its engine. Parent does not re-run.
- If a wave produces a contract-shape revelation, pause; fix the contract; then resume.
- File-write isolation: each engine writes only to `robit/engines/<its_name>/` + `tests/engines/test_<its_name>.py`. No shared file edits.

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

## Wave 15.rename — Inspector CLI rename ✅

`robit` → `insighter`. The `robit` name is reserved for
the 0.5.0 coding-agent CLI. All inspection commands move verbatim:

  insighter version
  insighter status
  insighter engines list
  ... etc

Mechanical scope only: `robit/cli/` → `robit/insighter/`,
`[project.scripts]` entry renamed, internal imports + docstrings updated,
test imports updated. No behavior change. Version stays at 0.4.0.


## Wave 15 — Coding-agent CLI + inspector rename (0.5.0) ✅

Managed-split execution across 14 subagents in 6 waves. Net delta: 751 → 1029
tests (+278), version 0.4.0 → 0.5.0.

| Wave | Agents | Scope | Tests |
|---|---|---|---|
| 15.rename | 1 | Renamed `robit` → `insighter`. `robit` reserved for coding-agent. | 0 |
| 15.0 — Agent foundation | 1 | `robit/agent/`: Conversation, Tool Protocol, ToolRegistry, SlashCommand Protocol, AgentLoop with event stream, session persistence, bare Textual app skeleton | +36 |
| 15.1 — Core tools (× 5 parallel) | 5 | file_read · file_write+file_edit (atomic, unified-diff output) · glob+grep · bash (destructive-op-gate vetoes BEFORE execution) · web_fetch (HTTPS-only, SSRF-guarded) | +113 |
| 15.2 — UX layer (× 4 parallel) | 4 | F: REPL polish (RichLog + Input + history) · G: DiffView + ApprovalPrompt · H: EnforcementChip (5 kinds: veto/redaction/conduct/audit/cost) · I: live CostTicker | +49 |
| 15.3 — Advanced (× 3 parallel) | 3 | J: MCP client (`robit/agent/mcp/`, stdio JSON-RPC, allow-listed servers) · K: Plan mode (`/plan`, `/edit`, `/cancel`, `/execute`) · L: Subagent dispatch (`subagent` tool with 3 built-in roles: deep-research, find-references, review-diff; depth cap = 2) | +81 |
| 15.4 — Ship 0.5.0 | me | Bump version, smoke, commit, push | n/a |

New dep: `textual>=0.50`. Two binary entry points: `robit` (coding-agent
REPL) + `insighter` (inspector). All LLM calls in the agent go
through `robit.proxy.pipeline.run` so conduct injection + trust-gate +
post-response secret-mask + audit JSONL apply to every turn automatically.


## Wave 16 — Subscription auth + Codex compat (0.6.0) ✅

Managed-split across 4 waves + 1 research-only. 1023 → 1073 tests (+50).

| Wave | Agents | Scope | Tests |
|---|---|---|---|
| 16.0 — Research | 1 (parallel-safe) | Inventory Codex CLI wire protocol → `docs/architecture/audits/codex-protocol.md`. Found: Codex uses Responses API (`/v1/responses`), not chat completions; API-key mode hits `api.openai.com`, ChatGPT-login mode hits `chatgpt.com/backend-api/codex/responses` with PKCE-flow JWTs. | 0 |
| 16.1 v2 — Pass-through auth | 1 (parallel-safe) | Proxy normal enforced path now forwards inbound auth header verbatim to upstream via LiteLLM `extra_headers` + `api_key`. New `--passthrough-auth` flag on `insighter serve --proxy`. Supports anthropic-api-key, anthropic-oauth, openai-bearer, gemini-api-key kinds. | +8 |
| 16.2 v2 — ChatGptClient | 1 (parallel-safe) | New `robit/llm/chatgpt_client.py` + PKCE OAuth helpers at `robit/llm/_chatgpt_auth.py`. Token cached at `~/.enchanter/chatgpt-token.json`. Real endpoints from 16.0: `https://auth.openai.com`, client_id `app_EMoamEEZ73f0CkXaXp7hrann`, redirect `http://localhost:1455/auth/callback`. | +18 |
| 16.3 — Codex adapter + ChatGptClient.complete() | 1 sequential (after 16.0) | New `CodexAdapter` at `/v1/responses` with Responses-API ↔ canonical translation. New `_codex_responses.py` shared helpers. ChatGptClient.complete() now talks to `chatgpt.com/backend-api/codex/responses` directly via stdlib HTTP. | +24 |
| 16.4 — Ship | me | Bump version, smoke, commit, push | n/a |

Honest v1 limitations documented in code:
- No WebSocket transport (Codex prefers WS; we ship HTTP-SSE fallback only)
- No `x-oai-attestation` header generation
- Developer-role collapse (Codex's `"developer"` → canonical `"system"`)
- No tool-call streaming in CodexAdapter (text-only deltas)
- ChatGPT-login mode through the proxy path requires upstream URL override (LiteLLM doesn't override base URL per-request) — direct `ChatGptClient.complete()` works; proxy passthrough for ChatGPT JWTs deferred
- ChatGptClient streaming deferred to Wave 17+


## Wave 17 — Auth UX (0.7.0) ✅

Managed-split across 4 parallel agents. 1073 → 1119 tests (+46).

| Wave | Agents | Scope | Tests |
|---|---|---|---|
| 17.0 — `.env` auto-loading | 1 (parallel-safe) | `robit/_env.py` loader; cwd + user-home precedence; shell env wins. Stdlib-only (no python-dotenv). Wired into both `robit` and `insighter` `main()` before any env reads. | +19 |
| 17.1 — `robit login` CLI | 1 (parallel-safe) | `robit login chatgpt` runs PKCE flow, saves token. `robit login --list` shows cached tokens. `robit logout chatgpt` / `--all` removes them. `login anthropic` is an informational stub (no standalone Claude.ai OAuth flow exists). | +12 |
| 17.2 — ChatGPT-login through proxy | 1 (parallel-safe) | Inbound `_extract_inbound_auth` shape-matches JWTs → kind `"chatgpt-jwt"` with `ChatGPT-Account-ID`. Outbound `_call_chatgpt_internal` posts directly to `chatgpt.com/backend-api/codex/responses` via stdlib `urllib`, bypassing LiteLLM. Streaming on this path deferred to Wave 18+. | +15 |
| 17.3 — Auth docs | 1 (parallel-safe, docs only) | `docs/auth.md` (1929 words, 53 source citations). Env-var matrix, `.env` conventions, three pass-through patterns for the proxy, host-agent base-URL examples, honest limitations. | 0 |
| 17.4 — Ship | me | Bump 0.6.0 → 0.7.0, smoke, commit, push | n/a |


## Wave 20 — Strip robit down to the coding agent (0.8.0) ✅

`enchanter-ai/beholder` is the canonical observability product (TypeScript MCP-client SDK + Rust cockpit, already live). Robit is just the Python coding agent. Removed everything inspection/proxy/MCP-server-shaped.

| Removed | What it did |
|---|---|
| `robit/insighter/` (entire package) | The status / engines list / conduct list / inference / tier / serve CLI |
| `robit/mcp_server/` (entire package) | MCP server mode (engines exposed as MCP tools over stdio/HTTP) |
| `robit/proxy/server.py` | HTTP proxy server (Anthropic/OpenAI/Gemini/Codex endpoint listener) |
| `robit/proxy/adapters/` | Wire-format adapters (anthropic, openai, gemini, codex) for the proxy's inbound side |
| `robit/proxy/fastpath.py` | Env-gated key-allowlisted byte pass-through |
| `robit/registry/` | Namespace registry + trust-pin (only used by mcp_server) |
| `scripts/smoke_proxy.py`, `scripts/live_demo.py` | Proxy-only smoke/demo |
| `tests/cli/`, `tests/mcp_server/`, `tests/registry/`, proxy adapter+server+fastpath tests | Tests for the removed surfaces |
| `insighter` binary entry point in `pyproject.toml` | — |

What stayed (substrate the coding agent uses internally):
- `robit/agent/` — REPL, tools, sessions, login, slash commands
- `robit/proxy/{canonical,pipeline,upstream,conduct,events,streaming}.py` — pipeline.run wraps every LLM turn with conduct injection + lifecycle + post-response gates
- `robit/engines/` (14 engines), `robit/conduct/`, `robit/core/`, `robit/composer/`, `robit/loader/`, `robit/runtime/`, `robit/inference/`, `robit/llm/`, `robit/protocol/`, `robit/transport/`, `robit/_env.py`, `robit/_compat.py`

Tests: 1130 → 898 (–232 from the deletion of mcp_server / proxy-server / adapters / fastpath / registry / insighter test suites). The 898 cover the coding agent's full active surface.

