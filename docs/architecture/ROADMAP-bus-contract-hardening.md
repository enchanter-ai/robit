# Roadmap — Bus & Contract Hardening

> **Author:** Enchanter Labs
> **Branch:** `roadmap/bus-contract-hardening`
> **Source:** Gaps surfaced while auditing how robit's engines communicate, cross-referenced
> against [`delegation-of-authority.md`](delegation-of-authority.md).
> **Goal:** Close every identified gap in the plugin-communication substrate without
> regressing the existing test suite, landing the work in dependency-ordered waves.

## Why this exists

Robit's engines are decoupled through an in-process pub/sub bus driven by a 7-phase
lifecycle orchestrator. The substrate works, but it carries contract debt that the
project's own audit catalogues: five ways to say "veto", no central topic registry, an
unversioned event contract, a sidecar trust boundary that trusts self-reported topics, a
recursive re-publish path with no cycle guard, silently-swallowed subscriber failures, no
operator dial, and no durable veto audit. Separately, the shared wire contracts are
hand-ported between the Python (`robit`) and TypeScript (`beholder`) runtimes, guaranteeing
drift.

This roadmap closes all of it.

## Gap inventory

| # | Gap | Primary files | Audit ref | Severity |
|---|-----|---------------|-----------|----------|
| G1 | Five representations of "veto"; `pattern_id` recovered by string-slicing `reason` | `core/verdict.py` (new), `core/lifecycle.py`, `proxy/pipeline.py`, `proxy/server.py`, `mcp_server/dispatcher.py` | §5 Q3 | **High (correctness)** |
| G2 | No central topic registry; emit/subscribe mismatches silently never match | `core/topics.py` (new), `loader/discovery.py`, all `engine.toml` | §5 Q5 | High |
| G3 | `EnchantedEvent` / `PluginAck` carry no `schema_version` | `core/events.py` | §3 #1 | Medium |
| G4 | Sidecar self-reports topics + can forge `source`/`derived_events` | `loader/runtimes/sidecar.py`, `loader/manifest.py` | §7b | **High (trust)** |
| G5 | Recursive `derived_events` re-publish has no hop/cycle guard | `core/bus.py`, `core/events.py` | new | Medium (DoS) |
| G6 | Subscriber exceptions swallowed with bare `except: pass` | `core/bus.py`, `core/context.py` | §7a / new | Medium (observability) |
| G7 | No operator dial: cannot filter/disable engines at runtime | `core/context.py`, `proxy/pipeline.py` | §6 Q4 | Medium |
| G8 | Veto decisions are header-only / in-memory; not durable | `proxy/pipeline.py`, `proxy/server.py`, `state/audits/` (new) | §9 #4 | Medium |
| G9 | Shared wire contracts hand-ported across Python/TS → drift | `schema/` (new), `tools/codegen/` (new) | language analysis | Medium (maintenance) |

## Dependency graph

```
                 ┌─────────────────────────────────────┐
   WAVE 1        │ G1 Verdict · G3 schema_version       │   (single agent, core/ only)
   FOUNDATION    │ G5 cycle guard · G6 subscriber log   │
                 └───────────────┬─────────────────────┘
                                 │ merged + reviewed
          ┌──────────────────────┼──────────────────────┐
   WAVE 2 │ G2+G4 LOADER-TRUST    │ G7+G8 PIPELINE-OPS    │ G9 CODEGEN
          │ loader/, sidecar.py   │ proxy/, state/audits  │ schema/, tools/
          └───────────────────────┴──────────────────────┘
                       (3 agents, disjoint directories)
```

**Why this ordering:** G1's `Verdict` type and G3's `schema_version` are consumed by the
sidecar coercion (G4), the durable veto log (G8), and the codegen schema (G9). They must
land and be reviewed first. Wave 2's three packages touch disjoint top-level directories
(`loader/` vs `proxy/` vs `schema/`+`tools/`), so they parallelize without collision; each
only *reads* the Wave-1 core.

## Waves

### Wave 1 — Foundation (`core/`)  ·  one agent

**G1 — Unified `Verdict` type.** New `robit/core/verdict.py` with frozen
`Verdict(plugin, phase, reason, pattern_id, pattern_name, severity)`. `PluginAck(status="veto")`
carries an optional `verdict`. `lifecycle.py` raises `SecurityVetoError` wrapping a `Verdict`
(no more string-slicing in `_veto_from_error`). `proxy/server.py` renders it to HTTP 451;
`mcp_server/dispatcher.py` renders it to JSON-RPC `-32099`. Keep `VetoResult` as a thin
deprecated alias for one release (dual-run).

**G3 — Contract versioning.** Add `schema_version: int = 1` (ClassVar or field) to
`EnchantedEvent` and `PluginAck`. Decoders tolerate a missing field (treat as 1).

**G5 — Cycle/depth guard.** Add `hop_count: int = 0` to `EnchantedEvent`. In
`InProcessBus.publish`, derived events inherit `hop_count + 1`; drop (and record) any event
exceeding `MAX_DERIVED_HOPS` (default 8).

**G6 — Subscriber-failure surfacing.** Replace the bare `except Exception: pass` in
`InProcessBus.publish` with a callback that appends a `DegradedFinding` (or emits a
`bus.subscriber.failed` event) so dropped handlers are observable, while still isolating the
bus from crashing.

**Exit criteria:** full suite green; new unit tests for Verdict rendering (451 + -32099),
hop-cap drop, subscriber-failure surfacing; `VetoResult` alias still imports.

### Wave 2 — Parallel packages  ·  three agents

**Package LOADER-TRUST (G2 + G4)** — `loader/`, `sidecar.py`, `manifest.py`
- G2: `robit/core/topics.py` registry (canonical topic → {owner, payload-schema-ref,
  expected phase}). `loader/discovery.load_engine_registry` cross-checks every engine's
  declared `emits`/`subscribes` against the registry at boot; unknown or unsubscribed topics
  raise a `TopicRegistryError`. Retire the `mcp.tool.call.requested` / `llm.proxy.request`
  synonym (pick one canonical name).
- G4: in `SidecarAdapter`, cross-check the `initialize`-reply topics against the parsed
  manifest's topics (mismatch → init failure). Source/topic-allowlist `derived_events`
  before re-publish; reject any with a reserved `source` (`orchestrator`) or an undeclared
  topic. Make the timeout/crash → veto coercion **conditional on `required`** (advisory
  sidecars fail open with `status="error", degraded=True`).

**Package PIPELINE-OPS (G7 + G8)** — `proxy/pipeline.py`, `proxy/server.py`, `state/audits/`
- G7: `PipelineOptions.engine_filter: frozenset[str] | None` (allowlist) and
  `disabled_engines: frozenset[str]`. The orchestrator skips non-matching engines and emits
  `pipeline.engine.skipped`. A `required` engine that would be disabled raises rather than
  silently dropping a security gate.
- G8: durable JSONL sink `state/audits/vetoes.jsonl` writing
  `{ts, correlation_id, engine, pattern_id, phase, payload_summary, http_status}` on every
  veto, consuming the Wave-1 `Verdict`.

**Package CODEGEN (G9)** — `schema/`, `tools/codegen/`
- Single source of truth: `schema/contracts.json` (JSON Schema) defining `EnchantedEvent`,
  `PluginAck`, `Verdict`, the JSON-RPC error codes, and the `initialize`/`on_phase` sidecar
  shapes — matching the Wave-1 final shapes exactly.
- `tools/codegen/generate.py` emits Python dataclasses and TS types; a check mode
  (`--check`) fails CI if `core/events.py` / `core/verdict.py` drift from the schema. Wire a
  test that runs `--check`.

**Exit criteria per package:** full suite green; package-specific new tests; no edits outside
the package's directories except read-only imports of Wave-1 core.

## Verification protocol

- Baseline test count captured before Wave 1 (see `state/baseline-tests.txt`).
- Every agent runs `python -m pytest -q` and reports pass/fail counts **honestly** — a
  refactor that drops tests is HOLD, not done.
- Additive-where-possible: new fields default-valued, deprecated types kept as aliases for
  one release (no-regression contract).
- Human review + merge between Wave 1 and Wave 2.

## Out of scope (logged, not closed here)

- Collapsing the three per-request scratch buckets (audit §5 Q4) — needs its own design.
- Pricing-table consolidation into `models-registry.json` (audit §5 Q2) — orthogonal.
- Agent-shaped engines / `[agent]` manifest table (audit §8) — separate initiative.
- Tier-router fallback chains (audit §6 Q5) — separate initiative.
</content>
</invoke>
