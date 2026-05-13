# Delegation-of-Authority Audit — enchanter-agent

> Source brief: [delegation-prompt.md](audits/delegation-prompt.md).
> This report was produced by a two-agent sequential split. Agent X
> wrote sections 2–7; Agent Y wrote sections 1, 8, 9, 10.

### 1. Executive summary

The highest-leverage honey spot is
[`EnchantedEvent`+`PluginAck`](../../enchanter/core/events.py) — every
engine and emitter rides on these two frozen dataclasses, so adding a
`schema_version` literal there delivers more durability per change unit
than any other contract harden. The biggest consolidation opportunity
is the five-way fragmentation of "veto" semantics
(`SecurityVetoError`, `PluginAck(status="veto")`, `VetoResult`, HTTP
451, JSON-RPC `-32099`), which should collapse into a single
`enchanter/core/verdict.py` typed value carrying `pattern_id`/`severity`
so the proxy and MCP renderers stop best-effort-parsing `reason`
strings. The biggest delegation opportunity is the absence of any
operator dial reaching the running pipeline: sidecar
[`MAX_RESTARTS=3`/`DEFAULT_TIMEOUT_S=5`](../../enchanter/loader/runtimes/sidecar.py)
and engine `required=True` cannot be reshaped without a code edit, so
hoisting both into `EngineManifest` + adding `PipelineOptions
.engine_filter` recovers operator authority that the precedence pyramid
in section 4 today simply omits. The most under-protected authority is
[`SidecarAdapter._parse_ack`](../../enchanter/loader/runtimes/sidecar.py)
admitting `PluginAck.derived_events` from a subprocess with no
source-allowlist and no topic cross-check against the parsed manifest —
a `trust-escalation` failure mode that has neither a durable audit log
nor an override (section 7b confirms a sidecar can forge
`source="orchestrator"` or emit on undeclared topics today). Ship the
one agent-shaped delegation first as a paired wave: hoist
`timeout_s`/`max_restarts` into `EngineManifest` and pilot a
**Sonnet-tier `intent-anchor`** sidecar — the engine is already
advisory, already per-session, and is the lowest-risk surface for
proving the agent-engine contract before letting Opus loose on
post-session reasoning.

#### Notes on landmarks

Every landmark path in the brief resolved on the current tree, with
two shape notes carried forward into the report. The brief enumerates
`enchanter/engines/*/adapter.py` as "14 engines"; the engines tree at
[enchanter/engines/](../../enchanter/engines) actually contains 15
directories today (`boundary_segmenter`, `cost_ledger`,
`cve_pattern_gate`, `deep_research`, `destructive_op_gate`,
`import_graph_pagerank`, `inference_substrate`, `intent_anchor`,
`rate_limiter`, `secret_mask`, `structural_fingerprint`,
`token_runway`, `tool_poisoning_scan`, `trust_scorer`, plus the
`__init__`). `deep_research` declares `phases = []` so it's an
on-demand engine, not a phase-wired one — relevant for section 8.
Wave 12.5's `AdapterParseError` consolidation **is** real:
[errors.py](../../enchanter/proxy/adapters/errors.py) defines the
single class and the three adapters import from it. No drift to
report.

---

### 2. Authority map

Ordering rule from the brief: failure-mode severity (data-leak >
trust-escalation > silent-corruption > DoS > cost-runaway >
compliance-gap > operational-opacity), then leverage within bucket.
"Surface" cells link directly to the defining file and symbol.

| Surface | Authority | Originator | Decider | Executor | Auditor | Override | Trust boundary | Precedence | Observability | Failure mode |
|---|---|---|---|---|---|---|---|---|---|---|
| [secret_mask/adapter.py](../../enchanter/engines/secret_mask/adapter.py) | Redact secrets in upstream response text before client sees them | engine author | `secret-mask` (advisory) | `SecretSanitizingStream` mid-stream + post-response scan | `X-Enchanter-Mask-Matched` header + bus observation | `?conduct=off` does **not** disable; only an env/manifest edit removes it | in-process / network egress | beats nothing — last gate; loses to streaming chunks already shipped (pipeline limitation #1 in [pipeline.py](../../enchanter/proxy/pipeline.py#L51)) | header only, no durable log of which pattern fired | data-leak |
| [streaming.py:SecretSanitizingStream](../../enchanter/proxy/streaming.py) | Decide which mid-stream tokens to flush vs. hold | framework default | rolling-window regex | `SecretSanitizingStream.wrap` | `sanitizer.redactions` tuple, surfaced in `BusObservation` | none — implicit always-on for `stream()` | in-process | beats post-response engine for partial-stream output | in-memory (per-request `redactions` tuple) | data-leak |
| [proxy/adapters/](../../enchanter/proxy/adapters)/{anthropic,openai,gemini}.py | Parse inbound wire format → `CanonicalRequest`; render `CanonicalResponse` → wire | framework | adapter module | `parse_request` / `render_response` | server-layer 400 envelope on `AdapterParseError` | none for parse; format is path-routed by host header | network (untrusted bytes in) | wire format is final at the boundary | exception → HTTP 400; no structured log of malformed inputs | data-leak (echo of attacker-crafted prompt) |
| [destructive_op_gate/adapter.py](../../enchanter/engines/destructive_op_gate/adapter.py) | Veto requests containing W5 destructive shell patterns | engine author (`required=True`) | regex scan over `mcp.tool.call.requested` payload | orchestrator `SecurityVetoError` raise | HTTP 451 + `X-Enchanter-Veto` (see [proxy/server.py#L511](../../enchanter/proxy/server.py)) | none — `required=True` and no operator dial | in-process | beats operator if engine declares `required=True` (orchestrator has no operator override path) | response header + recorder observation; no durable per-request audit log | trust-escalation |
| [cve_pattern_gate](../../enchanter/engines/cve_pattern_gate)/adapter.py | Veto on CVE-pattern matches | engine author (`required=True`) | regex scan | orchestrator veto path | 451 + bus observation | none | in-process | tied with `destructive-op-gate`; first to ack wins | header only | trust-escalation |
| [sidecar.py:SidecarAdapter](../../enchanter/loader/runtimes/sidecar.py) | Admit a subprocess as a `PluginAdapter` and coerce its `PluginAck` into the in-process trust pool | engine manifest (`runtime="sidecar"`) | `SidecarAdapter._parse_ack` | subprocess via JSON-RPC over stdio | stderr ring (50 lines), `_restart_count`, `_failed` flag | manifest edit only | subprocess | manifest declares `required` → subprocess can veto in-process; auto-restart cap is the only operator escape | in-memory `_stderr_ring`; veto reasons land in `PluginAck.reason` | trust-escalation |
| [conduct.py:apply_conduct_to_request](../../enchanter/proxy/conduct.py) | Inject conduct XML before client's system prompt | framework default `DEFAULT_PROXY_RULES` | `PipelineOptions.conduct` boolean + `conduct_rules` set | `_prepend_conduct` | none — silent rewrite of `req.system` | `PipelineOptions(conduct=False)` or empty `frozenset()` | in-process | operator (caller of `run`) beats engine-author defaults | none — no record that conduct was injected for a given correlation_id | silent-corruption (system prompt mutated invisibly) |
| [pipeline.py:run](../../enchanter/proxy/pipeline.py#L469) | Drive the 7-phase lifecycle around every upstream call | framework | `Orchestrator.run` | per-request fresh `InProcessBus` + `Orchestrator` | `_BusRecorder.observations` (in-memory) | per-call `PipelineOptions` only | in-process | wraps every other authority on this list | recorder is per-request only — discarded after response | silent-corruption (slip in lifecycle → engine never fires) |
| [orchestrator.py:Orchestrator](../../enchanter/core/lifecycle.py) | Decide phase ordering, ack-collection, veto translation | framework constants `LIFECYCLE_PHASES` + `DEFAULT_PHASE_TIMEOUTS_MS` | engine `required` flag + ack `status` | event-loop scheduling | raises `SecurityVetoError` / `PhaseTimeoutError` | `OrchestratorConfig.timeouts` override | in-process | final for lifecycle structure — engines cannot reorder | exceptions only; phase events go to bus but aren't persisted | silent-corruption |
| [conduct/loader.py:load_conduct](../../enchanter/conduct/loader.py) | Decide which conduct rules exist at all | `vis` markdown files on disk | filesystem glob | `_load_file` | exception on bad frontmatter | `root` kwarg | filesystem (in-process trust) | vis files beat everything; no operator override | none — rule set is silently re-derived per request | silent-corruption (drift between checked-in conduct and runtime) |
| [tier_router.py:TierRouter](../../enchanter/runtime/tier_router.py) | Map semantic task class → concrete `model_id` | `models-registry.json` + `_PREFERRED_MODEL` defaults | `TierRouter._resolve_default` at construction | `route()` returns string | `UnknownTaskClassError`/`MissingDefaultFamilyError` | `overrides={...}` ctor kwarg | in-process | operator overrides beat defaults; no fallback chain | logger.debug only for `size_hint` | silent-corruption (retired model still routed) |
| [loader/manifest.py:parse_manifest](../../enchanter/loader/manifest.py) | Decide whether an engine is loadable, which runtime owns it, what `env_allowlist` it gets | engine author (TOML on disk) | strict schema validator | `EngineManifest` returned to discovery | `ManifestSchemaError` on any drift | edit the TOML | filesystem | manifest is final; loader cannot reject a valid manifest | exception messages only | trust-escalation (sidecar manifest = subprocess admission) |
| [loader/discovery.py:load_engine_registry](../../enchanter/loader/discovery.py) | Decide which engines are wired into the registry | filesystem glob + manifest | `_topological_sort` over `depends_on` | imports `module:attr` or returns `SidecarAdapter` | exceptions | env-var-free; manifest presence is the dial | filesystem | beats per-request opts | none — registry is constructed silently per pipeline call (in `_build_orchestrator`) | DoS (a slow `import` blocks every request) |
| [sidecar.py timeout + restart budget](../../enchanter/loader/runtimes/sidecar.py) | Decide when a sidecar is "failed" forever | framework constants `DEFAULT_TIMEOUT_S=5`, `MAX_RESTARTS=3` | `_handle_crash` | flips `self._failed=True` | logger error on final kill | manifest cannot raise the cap | subprocess | beats engine author — no way to extend budget | `_stderr_ring` + restart counter; in-memory only | DoS / cost-runaway (no budget on retries) |
| [upstream.py:call_upstream](../../enchanter/proxy/upstream.py) | Make the only egress call to a provider via LiteLLM | framework (`litellm.drop_params=True`) | `litellm.acompletion` | LiteLLM SDK | `UpstreamError` | env vars (`ANTHROPIC_API_KEY` etc.) — but no per-request override | network | provider answer is final | LiteLLM internal logs only | cost-runaway |
| [proxy/events/cost_ledger.py](../../enchanter/proxy/events/cost_ledger.py) | Compute per-request cents from a hardcoded price table | framework `_PRICE_CENTS_PER_M_TOKENS` (module constant) | longest-prefix match on model name | publish `cost.ledger.recorded` | `X-Enchanter-Cost-Cents` header | none — table edit only | in-process | overrides nothing; operator can't tune pricing | per-request observation; no durable cumulative ledger here | cost-runaway |
| [cost_ledger/store.py](../../enchanter/engines/cost_ledger/store.py) | Per-vendor cumulative budget tracking + tier transitions | engine ctor (`set_budget`) | `check_threshold_crossed` | derived `cost-ledger.threshold.crossed` events | optional JSONL `ledger_path` | construct with custom thresholds | in-process | beats nothing — emits only, doesn't gate | durable iff `ledger_path` set; **default in-memory** | cost-runaway |
| [rate_limiter](../../enchanter/engines/rate_limiter)/adapter.py | Decide whether to admit a `pre-dispatch` request given budget | engine config | budget check | publishes `rate.limit.*` events | bus observation | manifest only | in-process | wins on `required=True` if so configured | in-memory | cost-runaway / DoS |
| [proxy/server.py:_is_veto+_send_veto](../../enchanter/proxy/server.py#L511) | Translate `VetoResult` → HTTP 451 with sanitized header content | framework | `VetoResult` shape | `aiohttp` response builder | response itself is the audit | none | network | translates orchestrator decisions to wire | HTTP response only — not persisted server-side | compliance-gap |
| [mcp_server/dispatcher.py:Dispatcher](../../enchanter/mcp_server/dispatcher.py) | Map MCP JSON-RPC methods → handlers; translate exceptions → error codes | framework constants (`SERVER_INFO`, `PROTOCOL_VERSION`) | per-method dispatch | tool registry handler call | `ErrorObject` returned in JSON-RPC envelope | none for method names; tool registry is the dial | network / human-input | tool author beats dispatcher (handler return is trusted) | `logger.exception` on unhandled errors | compliance-gap |
| [proxy/events/_types.py:EmitContext.scratch](../../enchanter/proxy/events/_types.py#L96) | Per-request shared dict between emitters | framework (initialized to `{}` per request) | emitters by namespaced key | dict mutation | none | none | in-process | nothing — convention-only namespace separation | none — dict is GC'd at end of request | silent-corruption (key collision between emitters) |
| [inference/engine.py](../../enchanter/inference/engine.py) | Append/reconcile/render-briefing the cross-session learning substrate | substrate-author docs + `ENCHANTER_INFERENCE_ENABLED` gate | `inference-engine.py emit/reconcile` | atomic catalog write + briefing render | `state/artifacts.jsonl` append log | env var `ENCHANTER_INFERENCE_ENABLED=0` makes `emit` a no-op | filesystem / cross-session | beats nothing live; advisory at session start only | durable JSONL + JSON catalog | operational-opacity |
| [proxy/conduct.py:DEFAULT_PROXY_RULES](../../enchanter/proxy/conduct.py#L34) | Decide which conduct rules are "the default" | framework constant (frozenset of 5 names) | constant | propagates into every request | none | `conduct_rules` ctor kwarg per request | in-process | operator beats default | none — silent set difference | operational-opacity |

22 authority surfaces. The map deliberately mixes "decision authorities"
(the brief's category) with the "type definers" that gate them, because
the right way to read this audit is one row at a time: each row is a
place where the system commits to behaviour without further appeal.

---

### 3. Honey spots (preserve as stable contracts)

Six surfaces with many consumers, few definers, and large blast radius
on change. Listed in leverage order.

1. **`EnchantedEvent` + `PluginAck`** —
   [core/events.py](../../enchanter/core/events.py). The lingua franca
   of the in-process bus. 15 engines + every emitter read or build
   these. Already frozen dataclasses with `Mapping[str, object]` payload.
   Posture is strong: contract is typed and immutable. *Hardening
   recommendation:* document `payload` keys per topic in a single
   side-file (engine.toml schemas don't enforce payload shape) and add a
   `schema_version` literal field for breaking-change detection.
   `[cost: 2 files, additive, parallel-safe, one-shot]`

2. **`PluginAdapter` Protocol** —
   [core/plugin.py](../../enchanter/core/plugin.py). Every engine
   (Python or sidecar) is duck-typed to this. The `runtime_checkable`
   Protocol is the only place that says "an engine has `name`, `phases`,
   `required`, `topics`, `budget_tier`, `on_phase`". Sidecar mirrors
   the same five attrs after `initialize`. *Hardening:* add an
   `@final` shim that takes an adapter and a `RequestContext` and
   returns the validated `PluginAck` — defends against engines that
   silently drop fields (e.g., the `derived_events` list reverted to
   `None`). `[cost: 2 files, additive, parallel-safe, one-shot]`

3. **`CanonicalRequest`/`CanonicalResponse`/`CanonicalChunk`** —
   [proxy/canonical.py](../../enchanter/proxy/canonical.py). All three
   wire adapters parse into these; pipeline + emitters consume them;
   LiteLLM bridge renders out. Frozen, tuple-backed, comprehensive
   docstring naming Anthropic's streaming model as the design reference.
   *Hardening:* add an explicit dataclass-level version sentinel
   (`schema_version: ClassVar[int] = 1`) and a tuple of allowed
   `stop_reason` values exposed as a public constant so adapters all
   read the same set. `[cost: 1 file, additive, parallel-safe, one-shot]`

4. **`EmitContext` + `EmitPhase` + `EventEmitter`** —
   [proxy/events/_types.py](../../enchanter/proxy/events/_types.py).
   Wave 13.0's emitter chain rides on these. Builtin + 4 wired emitters
   already exist; future operator plugins (rate-limiter scaling,
   tenant-cost surfacing) will land on the same Protocol. Strong
   posture: `EmitContext` is frozen, `scratch` is the only mutable
   path. *Hardening:* freeze `scratch` to a `dict[str, dict]` of
   per-emitter sub-dicts created at context construction so the
   namespace-by-convention rule is enforced by structure, not docstring.
   `[cost: 1 file, breaking (signature of `scratch`), parallel-safe,
   incremental]`

5. **`EngineManifest`** —
   [loader/manifest.py](../../enchanter/loader/manifest.py). The single
   schema-pinned, strict-validated definition of an engine — including
   the new `runtime` discriminator. This is the *only* contract that
   crosses the trust boundary at load time. Strong posture: strict
   `_ALL_KNOWN_FIELDS`, runtime-branch validation, kebab-case enforced
   implicitly. *Hardening:* add a manifest signature field
   (`manifest_sha`) that the loader verifies against a registry
   `trust_pin.py` allowlist before admitting a sidecar runtime — the
   `enchanter/registry/trust_pin.py` module already exists. `[cost: 3
   files, additive (new field, default-empty), parallel-safe, dual-run]`

6. **JSON-RPC `Request`/`Response`/`ErrorCode`** —
   [protocol/jsonrpc.py](../../enchanter/protocol/jsonrpc.py). Shared
   wire types between MCP server, MCP client, and the sidecar runtime —
   three independent consumers, one defining module. Custom error code
   range `-32099..-32000` is reserved and the IntEnum is the only
   listing. *Hardening:* add a "reverse" table (code → human label) and
   export both, so the MCP server and the sidecar can render the same
   label and the dispatcher's catch-all doesn't have to know `int(...)`.
   `[cost: 1 file, additive, parallel-safe, one-shot]`

Beyond the top six, `RequestContext`, `LifecyclePhase` literal, and
the `models-registry.json` schema all qualify as honey spots but
deliver smaller leverage gains per change unit; flag-only mention.

---

### 4. Authority hierarchy & precedence

The five precedence layers in the brief, ordered as they actually
resolve today (highest = wins in conflict):

1. **Framework defaults** —
   [`LIFECYCLE_PHASES`](../../enchanter/core/context.py),
   `DEFAULT_PHASE_TIMEOUTS_MS`, `MAX_RESTARTS=3`,
   `DEFAULT_TIMEOUT_S=5.0`, `PER_MESSAGE_BODY_MAX_BYTES=8 MiB`,
   `_PRICE_CENTS_PER_M_TOKENS`. These are the immovable floor — there
   is no operator dial for any of them today. Effectively highest
   precedence in places where the brief expected operator authority to
   win.
2. **Engine author** — manifest declares `required` (fail-closed vs.
   fail-open), `phases`, `budget_tier`, runtime, `env_allowlist`.
   `required=True` cannot be overridden by the operator at request
   time; the orchestrator [veto-checks required acks in lifecycle.py
   :100–105](../../enchanter/core/lifecycle.py) with no inspect-pre-
   raise hook.
3. **Operator** — exists only as construction-time arguments:
   `OrchestratorConfig.timeouts`, `TierRouter(overrides=...)`,
   `CostLedger(thresholds=..., ledger_path=...)`. There is no
   `settings.json` / env-driven operator dial reaching the proxy
   pipeline today.
4. **End-user / host agent** — only one operator-visible dial:
   `PipelineOptions(conduct=False)` and `conduct_rules`. Inbound
   requests cannot disable specific engines, cannot change phase
   timeouts, cannot override the trust-gate.
5. **Cross-session learning** — the inference substrate
   ([inference/engine.py](../../enchanter/inference/engine.py)) is
   currently **advisory at session start**, not a live authority. With
   `ENCHANTER_INFERENCE_ENABLED=0` it is silently inert. Highest
   priority *in theory*, lowest in practice.

**Conflict pair resolutions in the code today:**

- **Operator wants to disable `secret-mask` ↔ engine declares
  `required=True`.** Today: engine wins; there is no proxy-time
  override. Should change: introduce a `disabled_engines`
  `PipelineOptions` field, plus a write-once audit event when used,
  so operator authority becomes legible at the runtime layer.
- **Engine declares `runtime="sidecar"` ↔ framework's
  `MAX_RESTARTS=3`.** Today: framework wins; restart cap is hardcoded.
  Should change: hoist `MAX_RESTARTS` + `DEFAULT_TIMEOUT_S` into the
  manifest with a clamped range, so a CVE-pattern sidecar can declare
  "I need 30s and 5 retries because I'm doing network IO" without
  forking the framework.
- **`PipelineOptions(conduct=False)` ↔ engine subscribes to
  conduct-derived events.** Today: conduct injection is silently
  skipped, no event indicates so. Should change: publish a
  `pipeline.conduct.bypassed` bus event when conduct=False so
  downstream auditors can correlate later anomalies to the bypass.
- **Inference substrate briefing says "rate-limiter must veto on
  pattern X" ↔ rate-limiter engine is `required=False`.** Today:
  substrate has no live channel to flip an engine to `required=True`.
  Should remain: substrate stays advisory, but the rate-limiter could
  read briefings at construction time and adjust thresholds.
- **`TierRouter.overrides` pins a model that is later retired in
  `models-registry.json` ↔ default routing.** Today: override wins
  blindly (constructor validates only at construction). Should
  change: revalidate on `route()` call, or publish a
  `model.deprecated.routed` event when a retired model is asked for.

The precedence rule is **implicit everywhere**: not a single module
declares "framework > engine-author > operator > end-user". The brief
identified this as a finding, and we agree — the precedence pyramid
should be encoded in a single module (`enchanter/runtime/precedence.py`)
and consulted by every override site.

---

### 5. Consolidation opportunities (verify; don't assume)

Each investigation question from the brief, verified against current
state.

**Q1: Is `AdapterParseError` defined in exactly one place?**
**Verified, single source.**
[proxy/adapters/errors.py](../../enchanter/proxy/adapters/errors.py)
is the sole definition; every other site (`anthropic.py`, `openai.py`,
`gemini.py`, `__init__.py`, `proxy/server.py`, plus three test
modules) imports it. Wave 12.5 closed this correctly. *No action.*

**Q2: Is the cost-pricing table defined in one place?**
**Refuted — pricing is duplicated and disjoint.**
[proxy/events/cost_ledger.py#L96](../../enchanter/proxy/events/cost_ledger.py)
holds `_PRICE_CENTS_PER_M_TOKENS` (15 model prefixes). The
`engines/cost_ledger/` engine
([store.py](../../enchanter/engines/cost_ledger/store.py),
[adapter.py](../../enchanter/engines/cost_ledger/adapter.py)) tracks
*token totals* but never prices them; it consumes `tool_call_cost` from
the inbound payload instead. Meanwhile
[runtime/data/models-registry.json](../../enchanter/runtime/data/models-registry.json)
also carries pricing fields. *Three separate sources of price truth.*
**Consolidated owner: `models-registry.json`** — it already has the
schema, the engine and emitter should look up by `model_id`.
`[cost: 3 files, breaking (emitter signature unchanged but data source
shift), parallel-safe (engine + emitter can be done in parallel),
incremental]`

**Q3: How many distinct mechanisms express "veto this request"?**
**Five identifiable mechanisms.** All semantically the same intent
("don't proceed"), but with different observability and different
audit shapes:

1. `SecurityVetoError` — raised by the orchestrator in
   [lifecycle.py#L31](../../enchanter/core/lifecycle.py).
2. `PluginAck(status="veto", reason=...)` — the engine-side primitive
   in [events.py#L31](../../enchanter/core/events.py).
3. `VetoResult` dataclass — pipeline-layer translation in
   [pipeline.py#L145](../../enchanter/proxy/pipeline.py).
4. HTTP 451 + `X-Enchanter-Veto` header — wire shape in
   [proxy/server.py#L511](../../enchanter/proxy/server.py).
5. JSON-RPC `ErrorCode.SECURITY_VETO=-32099` — MCP-side shape in
   [protocol/jsonrpc.py#L32](../../enchanter/protocol/jsonrpc.py).

They **should** converge into a single `Verdict` type that the
orchestrator emits, the proxy server renders to 451, and the MCP
dispatcher renders to JSON-RPC `-32099`. Today the `pattern_id`
parsing in `_veto_from_error`
([pipeline.py#L426](../../enchanter/proxy/pipeline.py)) is best-effort
string slicing of `reason` — a real fault line.
**Consolidated owner: a new `enchanter/core/verdict.py` module** with
typed `Verdict(plugin, phase, reason, pattern_id, pattern_name,
severity)`. `[cost: 6-10 files (rename + import migration), breaking,
parallel-safe-no (one agent owns the rename), dual-run with deprecation
window for `VetoResult`]`

**Q4: How many per-request scratch buckets exist?**
**Three.**

- `EmitContext.scratch: dict[str, Any]` — Wave 13.0 emitter chain,
  [proxy/events/_types.py#L96](../../enchanter/proxy/events/_types.py).
  Convention: namespace by emitter `name`.
- `RequestContext.degraded_findings: list[DegradedFinding]` —
  [core/context.py#L46](../../enchanter/core/context.py). Owned by the
  orchestrator; one append per advisory plugin failure.
- `BusObservation.payload_summary: dict` —
  [pipeline.py#L130](../../enchanter/proxy/pipeline.py). Whitelisted
  scalars only; downstream renders these as response headers.

These have **distinct semantics today** (scratch=cross-emitter
exchange; degraded_findings=advisory-failure log; payload_summary
=safe-to-echo summary) but they conflate two concerns the bus already
expresses ("did this engine see anything") and one that probably
belongs on a context object ("scratch shared between emitters"). The
honest answer: scratch and payload_summary are *accidental* parallel
paths because `cost-ledger.py` had to abuse the `score` key
([cost_ledger.py#L72](../../enchanter/proxy/events/cost_ledger.py))
to surface `cents` through the whitelist. **Consolidated owner: a
`RequestScratchpad` dataclass on `RequestContext`** with a typed
`per_emitter: dict[str, dict]` and a typed `findings: list[Finding]`;
deprecate `EmitContext.scratch` in favor of a thin view. `[cost: 4
files, breaking, parallel-safe-no, dual-run]`

**Q5: Is there a single registry of valid bus topics?**
**Refuted — no central topic registry.** Each engine declares its
`topics.subscribes` / `topics.emits` in its `engine.toml`; the loader
validates the manifest shape but never cross-checks that emitted topics
are subscribed somewhere or that subscribed topics are emitted somewhere.
Topic naming is unenforced kebab-with-dots
(`mcp.tool.call.requested`, `cost-ledger.appended`,
`llm.proxy.accumulator-truncated`). The pipeline picks both
`mcp.tool.call.requested` *and* `llm.proxy.request` as parallel
synonyms (pipeline.py docstring at "Topic choices" explicitly admits
this) — exactly the inconsistency a registry would prevent.
**Consolidated owner: a `topics.toml` (or `enchanter/core/topics.py`)
manifest mirroring engine.toml's strictness** — declares the canonical
topic name, payload schema reference, the phase it's expected on, and
which engines (by name) own emit-vs-subscribe rights. The loader can
then cross-check at boot. `[cost: 6-10 files (one new + every engine
.toml gains a topic-id reference), additive, parallel-safe (one agent
per engine pair), incremental]`

**Q6: Is the 21-code failure-mode taxonomy enforced anywhere in the
agent runtime?**
**Refuted — the taxonomy exists only in wixie's conduct modules and is
referenced by name in `deep_research/phases/verify.py` (an `F02` mention
in a comment) and inside `inference_substrate/adapter.py`.** It is
*not* enforced as a `code` field on any agent-runtime artifact, error
type, `PluginAck.reason`, or audit log. The wixie inference substrate
expects a `code` field
(see [wixie/shared/conduct/inference-substrate.md](../../../wixie/shared/conduct/inference-substrate.md))
but the agent emits artifacts only via a separate enchanter inference
engine that doesn't validate the code namespace. **Consolidated
owner: a `enchanter/core/failure_codes.py` Enum** (re-export of the
21 codes), referenced by `SecurityVetoError`, `PluginAck.reason`
prefix convention, and the inference engine `emit` validator. `[cost:
6-10 files, additive (new enum + opt-in reason prefix), parallel-safe,
incremental]`

**Additional finding (not seeded):** **Trust-gate event topic
schizophrenia.**
[pipeline.py#L355](../../enchanter/proxy/pipeline.py) publishes
*both* `mcp.tool.call.requested` and `llm.proxy.request` at the same
phase carrying near-identical payloads — explicitly because the wave-2
brief asked for one name and the security engines key off the other.
The duplicate publish is a smell, not a feature. **Recommendation:
deprecate one synonym.** Pick `mcp.tool.call.requested` (engines
already key off it), retire `llm.proxy.request`, give Agent E a
3-release migration window to read `mcp.*` topics for header
synthesis. `[cost: 2 files, breaking, parallel-safe, dual-run]`

---

### 6. Delegation opportunities (investigate; refute when applicable)

**Q1: Is phase progression in the orchestrator truly fixed?**
**Verified fixed.** `Orchestrator.run`
([lifecycle.py#L75](../../enchanter/core/lifecycle.py)) iterates
`LIFECYCLE_PHASES` linearly. There is no skip-phase, no conditional
phase entry, no engine-requested branching. The streaming code path
(`_stream_body` in
[pipeline.py#L668](../../enchanter/proxy/pipeline.py)) even
*reimplements* the loop inline because it needs to yield chunks
mid-dispatch. **Delegation primitive:** add a `Phase.gate(ctx)
-> bool` indirection so the orchestrator can ask "is this phase needed
for this request?" before publishing. Use cases: tool-result-only
payloads can skip `anchor`; pure-image-gen requests can skip
`post-response` secret-mask. Keeps the seven phases as the *canonical*
ordering but lets specific engines opt out cheaply. `[cost: 2 files,
additive (new method on the orchestrator), parallel-safe, one-shot]`

**Q2: Does the conduct injector treat every request the same?**
**Verified — `apply_conduct_to_request` is request-shape-blind.**
[proxy/conduct.py#L45](../../enchanter/proxy/conduct.py) takes only
`req` and `rules`; it does not look at message roles, tool definitions,
or whether the request is an image-gen prompt. A
`tools=[{name: "image_gen"}]` request gets the same 5-rule conduct
header as a coding request. **Delegation primitive:** introduce a
`ConductSelector` Protocol with a `select(req) -> frozenset[str]`
method; ship a default `BlanketSelector` (today's behaviour) and an
opt-in `ShapeAwareSelector` that strips `verification` for pure-image
prompts and `tool-use` for tool-result-only payloads. `[cost: 2
files, additive, parallel-safe, one-shot]`

**Q3: Are the DEPLOY-bar criteria hardcoded numerics?**
**Confirmed hardcoded in wixie, not in agent runtime.** The
DEPLOY-bar (σ < 0.45, overall ≥ 9.0, all axes ≥ 7.0, 8/8 SAT) is
declared in [wixie/CLAUDE.md](../../../wixie/CLAUDE.md) as a global
constant. There is no per-target-model dial (a Haiku-tier prompt
faces the same bar as an Opus-tier one) and no per-domain dial.
**Should it change?** Yes — image-gen and adversarial-robustness
prompts already exercise the bar with weaker statistical power
(fewer test cases). **Delegation primitive:** lift to a
`deploy-bar.toml` per target/family, default to today's values.
`[cost: 2 files in wixie (out of agent scope but called out),
additive, parallel-safe, one-shot]`

**Q4: Which engines fire on which phases — is there an operator
policy surface?**
**Verified — wiring is fully code/manifest-driven.** The phases on
each engine are declared in `engine.toml` and parsed verbatim into
`EngineManifest.phases`. An operator cannot say "for tenant X,
disable `import-graph-pagerank` during cross-session". `deep_research`
declares `phases = []` and is invoked manually — proof the model
*can* express on-demand engines, but there is no operator API.
**Delegation primitive:** a `PipelineOptions.engine_filter: frozenset[str]`
allowlist, plus a `pipeline.engine.skipped` bus event when an engine
would have fired but didn't. `[cost: 2 files, additive, parallel-safe,
one-shot]`

**Q5: Does the tier router map task class to a single model, or
support fallback chains?**
**Verified — single model, no fallback chain.**
[tier_router.py#L139](../../enchanter/runtime/tier_router.py) returns
exactly one `model_id` per `route()` call. `_DEFAULT_FAMILY_MAP`
contains a *family* list but only for *bootstrap* preference; once
resolved at constructor time, the chain is collapsed to one entry in
`self._defaults`. There is no runtime fallback: if Anthropic returns
529 (overloaded), `LiteLLM` raises, `UpstreamError` bubbles, and the
request returns HTTP 502. **Delegation primitive:** `route()` returns
a `tuple[str, ...]` (ordered fallback list); `call_upstream`
iterates with backoff. The model registry already carries the family
data; the change is in tier_router's resolution shape. `[cost: 4
files (tier_router + upstream + 2 callers), breaking, parallel-safe-no
(one agent owns the upstream rework), incremental]`

**Additional finding:** **Sidecar restart/timeout knobs are not
delegable.** `MAX_RESTARTS=3` and `DEFAULT_TIMEOUT_S=5.0` are module
constants in [sidecar.py#L73](../../enchanter/loader/runtimes/sidecar.py).
A network-IO-heavy sidecar (e.g., a future CVE database lookup)
cannot extend the timeout via manifest. **Primitive:** hoist
`timeout_s` and `max_restarts` into `EngineManifest` with clamp ranges
(`1 ≤ timeout_s ≤ 60`, `0 ≤ max_restarts ≤ 10`). `[cost: 3 files,
additive (default to current values), parallel-safe, one-shot]`

---

### 7. Trust boundary posture

**a. In-process Python authorities** — fully trusted, no validation
at the boundary:

- `Orchestrator` ← `PluginAdapter.on_phase` returns `PluginAck`. Today
  the orchestrator trusts the entire return shape verbatim except for
  `status ∈ {"ack", "veto", "error"}`. A buggy engine returning
  `derived_events=[<malformed-event>]` would be republished verbatim.
- `_BusRecorder.record` ← every bus event. Source-allowlisted, but
  payload is trusted to be small/scalar after `_summarise_payload`.
- `TierRouter` ← `ModelsRegistry` entries. Trusted to be well-typed
  (validated at registry construction).
- `apply_conduct_to_request` ← `load_conduct()` result. Trusts
  markdown frontmatter on disk to be well-formed (would raise
  `ConductFrontmatterError` otherwise).
- `Pipeline` emitter chain ← every emitter module under
  `proxy/events/`. **Discovery is by glob + `module.emitter` attribute
  inspection**
  ([proxy/events/__init__.py](../../enchanter/proxy/events/__init__.py)).
  An attacker who can drop a `.py` file into the package directory can
  inject an emitter that fires on every request. Should NOT be
  trusted blindly if external plugin loading is ever added.
- The cost-ledger emitter's per-request scratch dict mutation is
  unguarded — see section 5 Q4.

**Flag for review:** every engine module load
([loader/discovery.py:_import_adapter](../../enchanter/loader/discovery.py))
calls `importlib.import_module(module_path)` on a string from
`engine.toml`. If a manifest with `adapter = "evil.module:adapter"`
were ever admitted, the import runs arbitrary Python at registry-
build time. The current root inference (`_default_root` in
discovery.py) constrains manifests to the canonical engines dir, but
no signature check exists. Sidecar's manifest path is *strict-
validated* but Python's `adapter` field is not bounded to a known
module prefix.

**b. Subprocess / sidecar authorities** —
[`SidecarAdapter`](../../enchanter/loader/runtimes/sidecar.py).

What the framework already validates:

- 8 MiB per-message body cap (incoming and outgoing).
- No embedded newlines on outgoing JSON-RPC.
- `initialize` handshake schema (5 required keys, type-checked).
- `on_phase` result must be a dict with `status ∈ {ack, veto, error}`.
- env_allowlist filters parent env down to allowlisted keys plus
  Windows-required minimums.

What the framework does **NOT** validate:

- Sidecar `topics.emits` and `topics.subscribes` are trusted verbatim
  from the `initialize` reply — a malicious sidecar can claim to emit
  to *any* topic, including topics another required engine subscribes
  to, and bypass the engine.toml manifest's declared topic set.
  *This is a bypass of the manifest contract.* The handshake should
  cross-check the runtime-reported topics against the parsed
  manifest's topics.
- `derived_events` returned by `on_phase` are deserialised via
  `_dict_to_event` with no source/topic allowlisting — a sidecar can
  forge events with `source="orchestrator"` or topics it didn't
  declare.
- The current default of **coercing any malformed/timeout/crash
  response to a veto** is a sensible *for-required-plugins* default
  ("if I can't talk to my security gate, fail closed"). But for
  *advisory* sidecars it's wrong — an advisory plugin should fail
  open with `status="error"`, not veto the whole request. Today
  [sidecar.py#L168–192](../../enchanter/loader/runtimes/sidecar.py)
  vetoes regardless of `required`. **Fix:** make the coercion
  conditional on `self.required`; for advisory adapters, return
  `PluginAck(status="error", degraded=True)` instead of a veto.
  `[cost: 1 file, additive, parallel-safe, one-shot]`

**c. Network / human-input authorities** —

- `proxy/server.py` HTTP — Anthropic / OpenAI / Gemini wire formats.
  Each format's `parse_request` ingests JSON, validates structurally,
  raises `AdapterParseError` for malformed input → HTTP 400. The
  validation **is** sufficient for shape; it is **not** sufficient for
  semantics (a request with `system` set to 1 MiB of attack text
  passes parse). Recommend a configurable byte-cap at the server
  layer.
- `mcp_server` over Streamable-HTTP + stdio — JSON-RPC envelope
  validation lives in `protocol/jsonrpc.py:decode`. `Dispatcher`
  short-circuits on `JsonRpcParseError` and on `Notification` (no
  reply expected). What's missing: an upper bound on `params` size
  before `_dispatch` reaches the tool handler. A malicious client can
  POST a 50 MB tool-call payload and force the dispatcher to allocate.
- CLI args — `enchanter/cli/` is tiny (`format.py` only). Today no
  destructive CLI flags reach engine state.

**Closing — substrate trust model in 3-4 sentences:**
*The substrate's implicit trust model is: code on disk and manifests
in the engines tree are fully trusted; everything off-process is
sanitised at the JSON-RPC boundary; everything across the network is
sanitised at the wire-adapter boundary. The orchestrator trusts every
`PluginAck` field that survives those gates, including
`derived_events`, which is the strongest implicit assumption in the
system.* The trust model is **not documented anywhere** as a single
statement — it is reconstructed by walking the code and inferring
from defaults — which is itself a finding.

---

### 8. Agent-shaped delegations

The sidecar runtime ([sidecar.py](../../enchanter/loader/runtimes/sidecar.py))
admits any subprocess that speaks JSON-RPC; an "agent-engine" is just
a sidecar whose body is a model call. The question is which of the 15
shipping engines should remain regex/algorithm-shaped and which would
benefit from an Opus/Sonnet/Haiku call sitting between `on_phase` and
the ack.

**Engines that should stay deterministic** (regex / cryptographic /
rule-based; adversarial manipulability or latency disqualifies an
agent):

- [`secret_mask`](../../enchanter/engines/secret_mask/adapter.py) —
  regex redaction, runs *mid-stream* (per section 2's
  `SecretSanitizingStream` row). A model call on every chunk is a
  latency and cost catastrophe and the regex is provably exhaustive
  on declared pattern set.
- [`destructive_op_gate`](../../enchanter/engines/destructive_op_gate/adapter.py)
  — `required=True`, fail-closed shell-pattern scan. The whole point
  is determinism: an agent might be talked out of vetoing `rm -rf /`.
- [`cve_pattern_gate`](../../enchanter/engines/cve_pattern_gate/adapter.py)
  — same reasoning as destructive-op-gate; CVE matching must be
  literal and prompt-injection-resistant.
- [`structural_fingerprint`](../../enchanter/engines/structural_fingerprint/adapter.py)
  — N1 shape hash + N3 naming-convention drift; pure cryptographic /
  string operations. An agent would underperform `hashlib`.
- [`trust_scorer`](../../enchanter/engines/trust_scorer/adapter.py)
  — Beta-Bernoulli posterior update; honest-numbers math the agent
  cannot improve.
- [`rate_limiter`](../../enchanter/engines/rate_limiter/adapter.py) —
  budget arithmetic, fires on `pre-dispatch`. Agent latency would
  defeat the purpose.

**Engines that would benefit from agent reasoning** (judgment calls,
context-sensitive classification, multi-step inference). Tier
selected against the brief's cost-vs-quality framing and wixie's
tier-sizing module: Haiku for high-frequency narrow classification,
Sonnet for per-session judgment, Opus only for cross-session
synthesis:

- [`intent_anchor`](../../enchanter/engines/intent_anchor/adapter.py)
  — currently LCS + HMM forward labelling. "Did the user's intent
  drift?" is exactly the judgment a small model handles better than
  Jaccard. **Tier: Sonnet.** Fires at `post-session` (low frequency,
  one call per session) so token cost is bounded and quality of the
  drift verdict directly affects the inference substrate's posterior.
- [`tool_poisoning_scan`](../../enchanter/engines/tool_poisoning_scan/adapter.py)
  — M1 today is 5 regex `SUSPICION_PATTERNS`; the actual attacks in
  the wild use natural-language disguises that regex can't see. **Tier:
  Haiku.** Fires at `post-response` per tool registration; per-tool
  cost is one short prompt, and the engine is `required` so the
  honest-numbers downside of an agent miss is offset by the
  deterministic M1 still running in parallel as a safety net.
- [`boundary_segmenter`](../../enchanter/engines/boundary_segmenter/adapter.py)
  — Jaccard sliding window is a proxy for "is this a coherent work
  session?" — a question Sonnet answers more cheaply with semantics.
  **Tier: Sonnet.** Post-session frequency, advisory.
- [`inference_substrate`](../../enchanter/engines/inference_substrate/adapter.py)
  — at `cross-session` it currently runs SPRT reconcile + briefing
  render; the *briefing prose* itself is the obvious agent surface
  (Opus tier, once per cross-session tick). Today the briefing is
  templated; an Opus pass over the elevated catalog would produce
  briefings that future sessions actually want to read. **Tier: Opus.**

**Reference architecture —
[`deep_research`](../../enchanter/engines/deep_research/pipeline.py).**
What's right about its shape and reusable: (i) multi-phase pipeline
with named phases (decompose/cast/triangulate/gap_fill/synthesize/verify)
so each tier-shift is explicit and a single phase can be retried; (ii)
tier resolution via
[`tier_router.route("orchestrator"|"executor"|"validator")`](../../enchanter/runtime/tier_router.py)
keeps the model IDs out of the engine and into the registry — exactly
the right place for the precedence pyramid; (iii) prompts live as
sibling `.md` files
([prompts/](../../enchanter/engines/deep_research/prompts)) for
fetcher/triangulator/verifier *and* as inline `_SYSTEM`/`_USER_TMPL`
constants in [phases/decompose.py](../../enchanter/engines/deep_research/phases/decompose.py)
and [phases/cast.py](../../enchanter/engines/deep_research/phases/cast.py)
— the inconsistency is itself a finding. What's specific to
`deep_research` and not reusable: it declares `phases=()` and lives
entirely off the `research.requested` event, which works only because
research is a long-running on-demand task; an agent-shaped
`intent_anchor` still fires on `post-session` and needs the standard
phase-wired adapter shape.

**Cross-cutting answers** (concrete picks, not options):

- **Where does an agent-engine's prompt live?** External `.md` files
  inside `engine_dir/prompts/`, referenced by name in `engine.toml`
  via a new `[agent]` table (`prompts = { decompose = "prompts/decompose.md" }`).
  Inline string constants drift between phase files; the conduct
  package is the wrong home because conduct rules are *cross-engine*
  invariants, while an engine prompt is the engine's authored work.
  The manifest field is the right honey-spot because section 3 #5
  already names `EngineManifest` as the contract that crosses the
  trust boundary.
- **Who owns the prompt's authority?** Engine author owns the prompt
  body. The operator owns a `prompt_overlay` field in
  `PipelineOptions` (additive, not replacement — the overlay is
  appended after the engine's authored block) so a tenant can inject
  domain rules without forking the engine. Precedence rule, mirroring
  section 4: **framework < engine-author < operator overlay**;
  end-user inbound requests cannot edit prompts. The conflict pair
  (operator overlay vs. engine `required=True`) resolves the same way
  as section 4's secret-mask pair: log a `pipeline.agent.overlay.applied`
  bus event and proceed.
- **How is agent-engine cost accounted?** Route every agent-engine
  upstream call through the same `call_upstream` LiteLLM path that
  the proxy uses, so the existing
  [`cost_ledger.py`](../../enchanter/proxy/events/cost_ledger.py)
  emitter sees the model+tokens and emits `cost.ledger.recorded` with
  the correlation_id of the *triggering* request. This forces the
  consolidation called out in section 5 Q2 (`_PRICE_CENTS_PER_M_TOKENS`
  must move to `models-registry.json`) — agent-engines make the
  divergence intolerable.
- **How are agent-engine vetoes audited differently?** A
  deterministic engine's veto carries a `pattern_id` that compresses
  to a fixed string; an agent veto's reason is open-text and can be
  wrong in ways regex can't. Tie this to section 9's gap: every agent
  veto must additionally persist the prompt+response pair to a
  durable `state/agent-verdicts.jsonl` keyed on correlation_id, and
  publish a `pipeline.agent.verdict.recorded` event so the bus
  recorder sees it. A regex veto needs only the pattern_id; an agent
  veto needs the full trace, because that's the only way a later
  reviewer can answer "did this Sonnet hallucinate the violation?".

**Recommendation — ship Sonnet-tier `intent_anchor` first.** Pair it
with section 6's "Sidecar restart/timeout knobs are not delegable"
finding: ship the manifest hoist in the same wave so the agent-sidecar
can declare `timeout_s = 20` for its model call. Rationale: `intent_anchor`
is `required=False` (fail-open if the agent fails or times out — no
gate breakage), fires once per session (cost is one Sonnet call per
session, not per request — bounded by section 5 Q2's not-yet-fixed
pricing table), and the wins are visible (the inference substrate's
posterior quality improves immediately). It's also the lowest-risk
proof of the prompt-in-manifest contract: a single `[agent]` table on
one engine's `engine.toml` validates the approach before
`tool_poisoning_scan` (the more interesting Haiku-tier case) inherits
it. `[cost: 4 files, additive, parallel-safe-no (one agent owns the
end-to-end pilot), incremental]`

### 9. Observability & audit gaps

Walking section 2's table, the following rows have `Observability`
cells of `in-memory` or `none`. Each is a question that cannot be
answered tomorrow morning from a durable log on the box.

1. **[secret_mask](../../enchanter/engines/secret_mask/adapter.py)
   — header-only, no log of which pattern matched.** Audit question:
   *"Yesterday at 14:32 a tenant's response shipped with three secret
   redactions — which patterns fired, and on what model output?"*
   Today the only trace is the `X-Enchanter-Mask-Matched` header on a
   response the client already discarded. Minimum durable shape: a
   JSONL sink at `state/audits/secret-mask.jsonl` writing
   `{ts, correlation_id, pattern_id, byte_offset_range}` per
   redaction. Row count expectation: O(redactions per request) — low.
   `[cost: 1 file, additive, parallel-safe, one-shot]`

2. **[SecretSanitizingStream](../../enchanter/proxy/streaming.py)
   — `redactions` tuple is per-request and GC'd.** Audit question:
   *"Did the mid-stream redaction window hold-and-flush hide a
   partial leak before the post-response engine caught up?"* Today
   `BusObservation` carries the tuple but the recorder itself is
   in-memory per request. Same JSONL sink as (1), keyed by `mid_stream=True`.
   `[cost: shares the (1) sink, additive, parallel-safe, one-shot]`

3. **[wire adapters](../../enchanter/proxy/adapters) malformed-input
   path — exception only, no structured log.** Audit question: *"How
   many `AdapterParseError` 400s did the Anthropic adapter return
   this week, and were they shaped like an attacker probing the
   parser?"* Today the only signal is an `aiohttp` access log line.
   Minimum shape: `state/audits/wire-parse-errors.jsonl` with
   `{ts, adapter, err_class, byte_len, sha256_of_body}` (hash, not
   body — body could be PII). Row count: low-to-moderate. `[cost: 4
   files (3 adapters + server), additive, parallel-safe, one-shot]`

4. **[destructive_op_gate](../../enchanter/engines/destructive_op_gate/adapter.py)
   & [cve_pattern_gate](../../enchanter/engines/cve_pattern_gate/adapter.py)
   vetoes — header + bus observation but no per-request audit log.**
   Audit question: *"Why did this request get a 451 yesterday at
   14:32? Which pattern? What was the redacted payload context?"*
   The bus recorder is in-memory; the response is the audit. Minimum
   shape: `state/audits/vetoes.jsonl` with `{ts, correlation_id,
   engine, pattern_id, phase, payload_summary, http_status}`.
   This is the heart of the report's most under-protected authority.
   `[cost: 2 files, additive, parallel-safe, one-shot]`

5. **[SidecarAdapter](../../enchanter/loader/runtimes/sidecar.py)
   stderr ring + restart counter — in-memory only.** Audit question:
   *"Why did the sidecar for `cve_pattern_gate` enter the `_failed`
   state at 14:32, and what was the last 50 lines of stderr?"* Today
   the ring evicts on process death. Minimum shape:
   `state/audits/sidecar-crashes.jsonl` flushed on every restart and
   on `_failed=True` transition; `{ts, plugin, restart_count,
   final_failure, stderr_tail}`. `[cost: 1 file, additive,
   parallel-safe, one-shot]`

6. **[conduct.py:apply_conduct_to_request](../../enchanter/proxy/conduct.py)
   — no record that conduct was injected.** Audit question: *"Did
   request X get the `verification` rule injected? Was conduct
   bypassed via `PipelineOptions(conduct=False)`?"* Today: nothing.
   Minimum shape: publish `pipeline.conduct.applied` and
   `pipeline.conduct.bypassed` bus events (already proposed in
   section 4); land in the same JSONL sink the bus recorder feeds.
   `[cost: 1 file, additive, parallel-safe, one-shot]`

7. **[pipeline.py:run](../../enchanter/proxy/pipeline.py#L469)
   `_BusRecorder` — per-request, discarded.** Audit question: *"Show
   me the full bus trace for correlation_id `c-123`."* Today: only
   the response headers survive. Minimum shape: a bus-recorder sink
   (`state/audits/bus.jsonl`) writing observations on response
   finalisation, gated by a `PipelineOptions.persist_bus=True` dial
   to keep the cost-runaway risk bounded. Row count: high; rotation
   required. `[cost: 2 files, additive, parallel-safe, dual-run
   (gated on the option)]`

8. **[orchestrator.py:Orchestrator](../../enchanter/core/lifecycle.py)
   phase events — exceptions only, no persisted phase log.** Audit
   question: *"Which phase did the orchestrator skip on request X
   because of a `PhaseTimeoutError`?"* Same bus-sink solution as (7);
   no separate cost. Covered by (7).

9. **[conduct/loader.py:load_conduct](../../enchanter/conduct/loader.py)
   — rule set silently re-derived per request.** Audit question:
   *"Did the conduct rule set change between request A at 14:30 and
   request B at 14:32?"* Today: nothing. Minimum shape: a
   construction-time hash (`sha256` over the loaded rule names +
   bodies) cached on the loader and emitted as
   `conduct.loaded` once per orchestrator construction; persist that
   single event to a small `state/audits/conduct-loads.jsonl`. Row
   count: very low (one per orchestrator boot). `[cost: 1 file,
   additive, parallel-safe, one-shot]`

10. **[tier_router.py:TierRouter](../../enchanter/runtime/tier_router.py)
    — logger.debug only.** Audit question: *"Why did task class
    `executor` route to model `claude-3-5-sonnet-20241022` on request
    X instead of the registry default?"* Today: only the `route()`
    return value, lost after response. Minimum shape:
    `state/audits/routing.jsonl` `{ts, correlation_id, task_class,
    chosen_model, size_hint, override_in_force}` — row count high
    (one per request), so should share the bus-recorder sink from
    (7). `[cost: shares (7), additive, parallel-safe, one-shot]`

11. **[discovery.py:load_engine_registry](../../enchanter/loader/discovery.py)
    — no record of which engines were wired into a given orchestrator.**
    Audit question: *"Did this orchestrator instance load the
    `tool_poisoning_scan` engine, or was it filtered out by
    manifest validation?"* Today: only debug logs. Minimum shape:
    one-time `engine.registry.loaded` bus event carrying the engine
    names, manifests-hash, and any rejection reasons — persist
    alongside the conduct hash in (9). `[cost: shares (9), additive,
    parallel-safe, one-shot]`

12. **[sidecar.py restart budget decisions](../../enchanter/loader/runtimes/sidecar.py)
    — restart counter in-memory.** Audit question: *"How many
    restarts did sidecar X exhaust today, and at what rate?"*
    Covered by (5) when crash events include the final restart count.

13. **[upstream.py:call_upstream](../../enchanter/proxy/upstream.py)
    — LiteLLM internal logs only.** Audit question: *"Did the
    Anthropic 529 from 14:32 retry once, twice, or fail outright?"*
    Today: depends on LiteLLM's log config. Minimum shape: wrap the
    `acompletion` call in a `pipeline.upstream.attempt` /
    `pipeline.upstream.response` event pair; persist via bus-sink (7).
    `[cost: 1 file, additive, parallel-safe, one-shot]`

14. **[cost_ledger emitter](../../enchanter/proxy/events/cost_ledger.py)
    — per-request observation only.** Audit question: *"What did
    tenant T spend this week?"* Today: only individual
    `cost.ledger.recorded` events on the in-memory bus. Minimum
    shape: persist via bus-sink (7); the cumulative aggregation can
    stay in
    [engines/cost_ledger/store.py](../../enchanter/engines/cost_ledger/store.py)
    once that engine's `ledger_path` is wired by default rather than
    optionally. `[cost: 2 files, additive, parallel-safe, one-shot]`

15. **[cost_ledger/store.py](../../enchanter/engines/cost_ledger/store.py)
    — durable only if `ledger_path` is set.** Audit question: *"What
    cost-tier transition events fired this month?"* The fix is the
    default-on flip of `ledger_path`. Covered by (14).

16. **[rate_limiter](../../enchanter/engines/rate_limiter/adapter.py)
    — bus observation, in-memory.** Audit question: *"Did the rate
    limiter throttle tenant T at 14:32?"* Covered by bus-sink (7).

17. **[proxy/server.py veto rendering](../../enchanter/proxy/server.py)
    — HTTP response is the audit.** Audit question: *"Show me every
    451 returned by the proxy this week with the originating
    correlation_id."* Today: web-server access log only. Covered by
    veto-sink (4); no extra cost.

18. **[mcp_server/dispatcher.py](../../enchanter/mcp_server/dispatcher.py)
    — `logger.exception` only.** Audit question: *"Which MCP method
    on which tool raised an unhandled error yesterday?"* Minimum
    shape: a small `state/audits/mcp-errors.jsonl` writing
    `{ts, method, tool_name, err_class, jsonrpc_code}`. `[cost: 1
    file, additive, parallel-safe, one-shot]`

19. **[EmitContext.scratch](../../enchanter/proxy/events/_types.py#L96)
    — dict GC'd at end of request.** Audit question: *"Did emitter
    A overwrite emitter B's namespace?"* Already addressed in section
    5 Q4's `RequestScratchpad` recommendation; no separate
    observability work needed.

20. **[DEFAULT_PROXY_RULES](../../enchanter/proxy/conduct.py#L34)
    — silent set difference when `conduct_rules` is overridden.**
    Covered by (6).

**Closing — closest to a `compliance-gap` failure mode.** Three rows
are candidates: the veto-rendering surface (17, classified
`compliance-gap` in section 2), the MCP dispatcher (18, same), and
the secret-mask audit (1, classified `data-leak` but the missing log
is the *compliance* dimension of the same surface). The strongest
compliance-gap is **(4) destructive-op-gate + cve-pattern-gate veto
logging**: these are the only `required=True` engines that gate
production traffic at HTTP 451, and the only durable trace of "we
refused this request at this time for this reason" today is the
response that the rejected client already discarded. A regulator —
or a customer asking why their request was blocked — cannot get an
answer. Land `state/audits/vetoes.jsonl` first. (17 and 18 fall out
for free once that sink format exists.) The other gaps are
`operational-opacity`, not compliance — important but not the same
priority.

### 10. Recommendations as a wave plan

Sorted by failure-mode severity first (data-leak > trust-escalation >
silent-corruption > DoS > cost-runaway > compliance-gap >
operational-opacity), then by lowest cost within bucket. Five waves,
matching `ROADMAP.md`'s "managed-split execution" pattern: each row is
a self-contained subagent dispatch, max 5 parallel agents per wave.

#### Wave 14.0 — Data-leak audit & wire-input hardening

- **Agents:** 3 parallel.
- **Scope:**
  - Agent A: `state/audits/secret-mask.jsonl` JSONL sink, wired from
    [secret_mask/adapter.py](../../enchanter/engines/secret_mask/adapter.py)
    and [streaming.py:SecretSanitizingStream](../../enchanter/proxy/streaming.py).
    Schema `{ts, correlation_id, pattern_id, byte_offset_range,
    mid_stream}`. Closes section 9 gaps (1) and (2).
  - Agent B: `state/audits/wire-parse-errors.jsonl` from the three
    wire adapters
    ([anthropic.py](../../enchanter/proxy/adapters/anthropic.py),
    [openai.py](../../enchanter/proxy/adapters/openai.py),
    [gemini.py](../../enchanter/proxy/adapters/gemini.py)) +
    [server.py](../../enchanter/proxy/server.py) on the
    `AdapterParseError` path. Hash bodies, do not store them. Closes
    section 9 gap (3).
  - Agent C: configurable byte-cap at the server layer for inbound
    request bodies and for `_dispatch` MCP `params` size (section 7c
    finding).
- **Files touched:** `enchanter/engines/secret_mask/adapter.py`,
  `enchanter/proxy/streaming.py`, three `proxy/adapters/*.py`,
  `proxy/server.py`, `mcp_server/dispatcher.py`, plus four new audit-
  sink modules.
- **Parallel-safe:** yes — each agent owns disjoint files.
- **Dependencies:** none.
- **Cost suffix:** `[cost: 8 files, additive, parallel-safe, one-shot]`.

#### Wave 14.1 — Sidecar trust-boundary hardening

- **Agents:** 2 parallel.
- **Scope:**
  - Agent A: in
    [sidecar.py:_parse_ack](../../enchanter/loader/runtimes/sidecar.py),
    enforce that returned `derived_events` carry an allowlisted
    `source` (the engine's own name only) and that the event `topic`
    is in the manifest's declared `topics.emits`. Reject mismatches
    with `degraded=True`. Make the malformed-response → veto coercion
    conditional on `self.required` (section 7b fix). Add
    `state/audits/sidecar-crashes.jsonl` for stderr + restart-counter
    durability (section 9 gap 5).
  - Agent B: hoist `timeout_s` and `max_restarts` into
    [`EngineManifest`](../../enchanter/loader/manifest.py) with
    clamps `1 ≤ timeout_s ≤ 60` and `0 ≤ max_restarts ≤ 10`; default
    to today's values (no breakage). Section 4 and section 6
    additional finding.
- **Files touched:**
  `enchanter/loader/runtimes/sidecar.py`,
  `enchanter/loader/manifest.py`, every sidecar engine.toml that
  wants the new field, plus the new audit sink.
- **Parallel-safe:** yes — Agent A owns runtime, Agent B owns
  manifest schema; they share only the `SidecarAdapter` ctor which
  Agent B extends additively.
- **Dependencies:** none.
- **Cost suffix:** `[cost: 4 files, additive, parallel-safe,
  one-shot]`.

#### Wave 14.2 — Verdict consolidation + veto audit log

- **Agents:** 1 (rename + import migration must be serially owned per
  the codebase's wave-12 lesson).
- **Scope:**
  - Introduce `enchanter/core/verdict.py` with typed `Verdict(plugin,
    phase, reason, pattern_id, pattern_name, severity)`. Migrate
    [`SecurityVetoError`](../../enchanter/core/lifecycle.py#L31),
    [`PluginAck(status="veto")`](../../enchanter/core/events.py#L31),
    [`VetoResult`](../../enchanter/proxy/pipeline.py#L145), HTTP 451
    rendering in
    [`proxy/server.py`](../../enchanter/proxy/server.py#L511), and
    JSON-RPC `ErrorCode.SECURITY_VETO`
    ([protocol/jsonrpc.py#L32](../../enchanter/protocol/jsonrpc.py))
    to consume the typed value. `_veto_from_error` string-slicing
    deletes.
  - Add `state/audits/vetoes.jsonl` (the section 9 closing-paragraph
    compliance-gap priority): `{ts, correlation_id, engine,
    pattern_id, phase, payload_summary, http_status}`. Wired from
    the orchestrator veto path so destructive-op-gate and
    cve-pattern-gate are both covered. Closes section 9 gap (4) and
    (17) and (18).
- **Files touched:**
  `enchanter/core/verdict.py` (new),
  `enchanter/core/lifecycle.py`,
  `enchanter/core/events.py`,
  `enchanter/proxy/pipeline.py`,
  `enchanter/proxy/server.py`,
  `enchanter/protocol/jsonrpc.py`,
  `enchanter/mcp_server/dispatcher.py`, plus tests across the rename.
- **Parallel-safe:** no — single agent owns the rename end-to-end.
- **Dependencies:** Wave 14.0 (the JSONL sink convention must exist).
- **Cost suffix:** `[cost: 6-10 files, breaking, parallel-safe-no,
  dual-run]`.

#### Wave 14.3 — Precedence pyramid + operator dials

- **Agents:** 3 parallel.
- **Scope:**
  - Agent A: new `enchanter/runtime/precedence.py` declaring the
    hierarchy (framework < engine-author < operator < end-user < —
    inference advisory only). Every override site
    ([conduct.py](../../enchanter/proxy/conduct.py),
    [tier_router.py](../../enchanter/runtime/tier_router.py),
    [pipeline.py](../../enchanter/proxy/pipeline.py)) imports the
    typed resolution helpers. Add the
    `pipeline.conduct.bypassed`/`pipeline.conduct.applied` events
    from section 4 (closes section 9 gap 6).
  - Agent B: add `PipelineOptions.engine_filter: frozenset[str]`
    allowlist + `pipeline.engine.skipped` event (section 6 Q4
    primitive). Add `PipelineOptions.disabled_engines` honouring
    `required=False`; refuse to disable `required=True` engines and
    publish `pipeline.disable.refused` instead.
  - Agent C: tier router fallback chain — `route()` returns
    `tuple[str, ...]`;
    [upstream.py](../../enchanter/proxy/upstream.py) iterates with
    backoff (section 6 Q5 primitive). Adds
    `state/audits/routing.jsonl` covering section 9 gap (10) and
    `pipeline.upstream.attempt` events covering gap (13).
- **Files touched:**
  `enchanter/runtime/precedence.py` (new),
  `enchanter/proxy/conduct.py`,
  `enchanter/proxy/pipeline.py`,
  `enchanter/runtime/tier_router.py`,
  `enchanter/proxy/upstream.py`.
- **Parallel-safe:** yes — each agent owns disjoint surfaces;
  precedence module is consumed read-only by B and C.
- **Dependencies:** Wave 14.2 (Verdict must exist so a refused
  disable can render properly).
- **Cost suffix:** `[cost: 5 files, additive, parallel-safe,
  incremental]`.

#### Wave 14.4 — First agent-shaped engine (Sonnet `intent_anchor`)

- **Agents:** 1 (end-to-end pilot; pairs with Wave 14.1's manifest
  hoist).
- **Scope:**
  - Extend [`EngineManifest`](../../enchanter/loader/manifest.py)
    with an `[agent]` table: `tier`, `prompts.<phase> = "prompts/<file>.md"`,
    `prompt_overlay_allowed` flag.
  - Add `PipelineOptions.prompt_overlay: Mapping[str, str]` keyed by
    engine name; operator-authored, appended after the engine's
    authored prompt (per section 8's precedence rule).
  - Convert
    [`intent_anchor/adapter.py`](../../enchanter/engines/intent_anchor/adapter.py)
    to dispatch a Sonnet call at `post-session` for the drift verdict,
    keeping LCS+HMM as a parallel deterministic check; veto only if
    both signal drift (defence in depth). Prompt lives in
    `enchanter/engines/intent_anchor/prompts/drift.md`.
  - Route upstream cost through the existing `cost_ledger` path; emit
    `pipeline.agent.verdict.recorded` with the prompt/response pair
    persisted to `state/audits/agent-verdicts.jsonl`
    (section 8's agent-veto-audit answer; also covers section 9
    needs).
- **Files touched:**
  `enchanter/loader/manifest.py`,
  `enchanter/proxy/pipeline.py`,
  `enchanter/engines/intent_anchor/adapter.py`,
  `enchanter/engines/intent_anchor/engine.toml`,
  `enchanter/engines/intent_anchor/prompts/drift.md` (new).
- **Parallel-safe:** no — one agent owns the contract debut.
- **Dependencies:** Wave 14.1 (manifest field shape), Wave 14.3
  (operator dial precedence rule).
- **Cost suffix:** `[cost: 4 files, additive, parallel-safe-no,
  incremental]`.

#### Deferred — re-evaluate next quarter

- **Cost-pricing consolidation onto `models-registry.json`**
  (section 5 Q2). Important for cost-runaway accounting, but the
  emitter works today and the consolidation forces a dual-run window
  — sequence after the agent-engine pilot makes the divergence
  intolerable. `[cost: 3 files, breaking, parallel-safe, incremental]`
- **Topic registry (`topics.toml` or `enchanter/core/topics.py`)**
  (section 5 Q5). High leverage but every engine `.toml` gains a
  reference; defer until the agent-engine pattern has shaken out the
  topic-naming conventions. `[cost: 6-10 files, additive,
  parallel-safe, incremental]`
- **21-code failure-codes Enum** (section 5 Q6 + sections 3/4
  honest-numbers tie-in). The wixie substrate already validates the
  codes; the agent runtime adoption is a follow-on once Verdict (Wave
  14.2) provides the home. `[cost: 6-10 files, additive,
  parallel-safe, incremental]`
- **`RequestScratchpad` consolidation** (section 5 Q4). Deferred —
  the parallel-paths smell is real but the namespaces are
  convention-isolated today; cost is breaking and dual-run, and
  nothing in waves 14.0–14.4 unblocks because of it. `[cost: 4
  files, breaking, parallel-safe-no, dual-run]`
- **Trust-gate topic-synonym deprecation
  (`llm.proxy.request` → `mcp.tool.call.requested`)** (section 5
  additional finding). Mechanical migration; defer until Wave 14.3's
  operator-dial events stabilise topic-naming taste.
  `[cost: 2 files, breaking, parallel-safe, dual-run]`
- **Bus-recorder durable sink behind `PipelineOptions.persist_bus`
  flag** (section 9 gap 7). Worth doing but rotation/retention
  policy needs a real conversation about cost first; the
  per-surface sinks in Waves 14.0–14.3 cover the highest-priority
  observability shortfalls without that conversation.
  `[cost: 2 files, additive, parallel-safe, dual-run]`
- **Honey-spot hardenings from section 3 #1, #4, #5, #6** (schema_version
  on `EnchantedEvent`, frozen `scratch` shape, manifest signature
  field, JSON-RPC reverse-code table). Each is small and additive but
  belongs in a single "contract-hardening" wave after the higher-
  severity work lands. `[cost: 5 files cumulative, additive,
  parallel-safe, one-shot]`
