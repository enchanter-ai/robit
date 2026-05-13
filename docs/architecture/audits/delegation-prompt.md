# Delegation-of-Authority Audit — Source Prompt

This is the canonical brief that produced
`docs/architecture/delegation-of-authority.md`. Saved here so the
audit is reproducible: re-run this prompt against a future version
of the codebase and the report regenerates with comparable shape.

The audit was executed as a two-agent sequential split:

- **Agent X** produced sections 2–7 (authority map, honey spots,
  hierarchy & precedence, consolidation, delegation, trust posture).
- **Agent Y** produced sections 1, 8, 9, 10 (executive summary,
  agent-shaped delegations, observability gaps, wave plan) by
  reading Agent X's output as input context.

Both agents shared this brief verbatim.

---

You are auditing the codebase at `c:/git/enchanter-ai/agent/` (and its
sibling `c:/git/enchanter-ai/wixie/` for context on conduct modules and
the inference substrate) through the lens of **delegation of authority**.

Your job is NOT to write code. Your job is to produce a single markdown
report at:

  `c:/git/enchanter-ai/agent/docs/architecture/delegation-of-authority.md`

The report identifies where authority lives, where it leaks, where it's
over-concentrated, what should be preserved as a stable contract, and
what cost it takes to fix each finding.

## Rubric

For every authority surface you map, apply these 8 dimensions:

1. **Single source of truth.** Does exactly one place own this decision?
2. **Stable contract surface.** Many consumers, few definers? (honey spot)
3. **Explicit delegation chain.** Originator → Decider → Executor → Auditor
   — are all four legible?
4. **Override mechanism.** Is there a logged, explicit way to bypass?
5. **Failure escalation.** When the delegate fails, who picks up?
6. **Trust posture.** Does this surface cross a trust boundary
   (in-process Python / subprocess / network / human input)? What
   posture does the receiver assume (trust / verify / sanitize / reject)?
7. **Authority precedence.** When this authority conflicts with another
   (engine-author vs operator vs end-user vs framework defaults), who wins?
   Is the precedence rule documented or implicit?
8. **Observability.** Where is the exercise of this authority logged
   such that a later audit can reconstruct what happened? Is the log
   durable across sessions, or in-memory only?

## Failure-mode taxonomy

For every authority surface, classify the worst-case failure mode if
that authority is exercised wrongly. Use exactly one of these labels
(invent a new one only if none fit):

- `silent-corruption` — data drift, posterior poisoning, stale state
- `DoS` — resource exhaustion (memory, CPU, file handles, API quota)
- `data-leak` — secrets or PII leave the wrong door
- `trust-escalation` — untrusted code gains in-process privilege
- `compliance-gap` — audit trail missing where regulators would require one
- `cost-runaway` — unbounded spend (no rate limit, no cap)
- `operational-opacity` — decision made, no one can reconstruct why

The classification drives prioritization in the report's
recommendations section.

## Cost-of-change rubric

For every recommendation, attach a cost weight covering four axes:

- **File count**: 1 / 2-5 / 6-10 / >10
- **Breakage**: additive (new fields/methods only) / breaking (rename, signature change, removal)
- **Parallel-safe**: yes (multiple agents can split this) / no (one agent must own it end-to-end)
- **Migration shape**: one-shot / incremental / dual-run with deprecation window

Render this as a compact suffix on each recommendation:
`[cost: 3 files, additive, parallel-safe, one-shot]`.

## Codebase landmarks — verify each exists before relying on it

If any path or symbol below is wrong or has moved, STOP and report it
back in the executive summary — do not hand-wave around it.

### Protocol / contract types (honey-spot candidates)

- `enchanter/core/plugin.py` — `PluginAdapter` Protocol
- `enchanter/core/events.py` — `EnchantedEvent`, `PluginAck`
- `enchanter/core/lifecycle.py` — the 7 lifecycle phases
- `enchanter/proxy/canonical.py` — `CanonicalRequest/Response/Chunk`
- `enchanter/proxy/events/_types.py` — `EmitContext`, `EmitPhase`,
  `EventEmitter` (added in Wave 13.0)
- `enchanter/loader/manifest.py` — `EngineManifest` (extended in Wave
  13.1.5 with the `runtime` field)
- `enchanter/protocol/jsonrpc.py` — JSON-RPC types shared by MCP server
  and sidecar runtime

### Decision authorities

- `enchanter/runtime/tier_router.py`
- `enchanter/proxy/conduct.py`
- `enchanter/proxy/pipeline.py` — `run`, `stream`, `VetoResult`,
  `BusObservation`
- `enchanter/loader/runtimes/sidecar.py` — subprocess restart budget,
  timeout policy
- `enchanter/engines/*/adapter.py` (14 engines)
- `enchanter/mcp_server/dispatcher.py`

### Failure-mode and audit authorities

- The 21-code failure taxonomy (referenced in agent CLAUDE.md and
  wixie's `foundations/packages/core/conduct/failure-modes.md`)
- `enchanter/inference/engine.py` and the wixie inference substrate
- The DEPLOY bar criteria (declared in wixie CLAUDE.md)

### External authority surfaces (crossing trust boundaries)

- `enchanter/proxy/upstream.py` — LiteLLM
- `enchanter/proxy/adapters/{anthropic,openai,gemini}.py` — three wire
  formats; each owns its format
- `enchanter/loader/runtimes/sidecar.py` — subprocess gateway

## Report structure

The report must have exactly these sections in this order. Section
headings (## Authority map, etc.) must match.

### 1. Executive summary (top of file, 5 sentences max)

Cover: (a) highest-leverage honey spot, (b) biggest consolidation
opportunity, (c) biggest delegation opportunity, (d) most
under-protected authority (no audit + worst failure mode), (e) one
agent-shaped delegation to ship first.

### 2. Authority map

A single table with this header, one row per authority surface:

| Surface | Authority | Originator | Decider | Executor | Auditor | Override | Trust boundary | Precedence | Observability | Failure mode |
|---|---|---|---|---|---|---|---|---|---|---|

- Surface: `file_path:symbol` markdown link
- Authority: 1-line description of the decision this surface owns
- Override: how to bypass (or "none — implicit always-on")
- Trust boundary: `in-process` / `subprocess` / `network` / `human-input`
- Precedence: who beats this surface when authorities conflict (or
  "none — final")
- Observability: where the decision is logged + durability
  (`durable` / `in-memory` / `none`)
- Failure mode: one label from the taxonomy

Cover at minimum every landmark listed above. Order by failure-mode
severity (data-leak > trust-escalation > silent-corruption > DoS >
cost-runaway > compliance-gap > operational-opacity), then by leverage.

### 3. Honey spots (preserve as stable contracts)

For each surface that qualifies (many consumers, few definers, large
blast radius on change), write 3-5 lines: why it's a honey spot, its
current versioning posture (typed? frozen dataclass? schema-pinned?
documented?), one recommendation to harden the contract.

Order by leverage. Limit to the top 6 — if you find more, mention but
don't detail.

### 4. Authority hierarchy & precedence

A meta-section. Articulate the precedence order between:

- **Framework defaults** (what the code does with no config)
- **Engine author** (what `engine.toml` and `adapter.py` declare)
- **Operator** (what enchanter CLI flags / settings.json / env
  vars set)
- **End user / host agent** (what the inbound request asserts —
  `?conduct=off`, `system` prompt content, etc.)
- **Cross-session learning** (inference substrate briefings)

When two of these conflict, who wins? Is the precedence rule
documented or implicit? Where is the conflict observable?

For each conflict pair you identify (e.g., operator says "disable
secret-mask" vs engine author says "required=True"), state the
current resolution and whether it should change.

### 5. Consolidation opportunities (investigate; don't assume)

Investigate the codebase for places where the same authority is
exercised by multiple sites, the same type is defined more than once,
or the same concept is named inconsistently. **Do not assume the
examples below exist — verify each one and report what you find,
including refutations.**

Investigation questions:

- Is `AdapterParseError` defined in exactly one place, or scattered
  across adapters? (Wave 12.5 was supposed to fix this — confirm or
  refute.)
- Is the cost-pricing table defined in one place, or duplicated
  between an engine and an emitter?
- How many distinct mechanisms express "veto this request"
  (`SecurityVetoError`, `PluginAck(status="veto")`, HTTP 451,
  `VetoResult`, `X-Enchanter-Veto` headers)? Are they semantically
  identical? Should they converge?
- How many per-request scratch buckets exist (`EmitContext.scratch`,
  `RequestContext.degraded_findings`, `BusObservation.payload_summary`,
  others)? Do they have distinct semantics or are they accidental
  parallel paths?
- Is there a single registry of valid bus topics, or does each engine
  invent its own naming convention?
- Is the 21-code failure-mode taxonomy enforced anywhere in the agent
  runtime, or is it only documented in wixie's conduct modules?

For each finding: where the scatter is, why it matters, who the
consolidated owner should be, cost suffix.

### 6. Delegation opportunities (investigate; don't assume)

Investigate places where authority is centralized but consumers would
benefit from being able to extend or override it. **Verify each
question; refute when applicable.**

Investigation questions:

- Is phase progression in the orchestrator truly fixed, or could an
  engine request to skip a phase under specific conditions?
- Does the conduct injector treat every request the same, or can a
  request shape (image-gen prompts, tool-result-only payloads, etc.)
  opt out of irrelevant rules?
- Are the DEPLOY-bar criteria hardcoded numerics, or per-domain /
  per-target-model? Should they be?
- Which engines fire on which phases — is this wired in code, or is
  there a policy surface where an operator could disable a noisy
  engine for a specific traffic class?
- Does the tier router map task class to a single model, or does it
  support fallback chains for outage handling?

For each finding: where authority is concentrated, what flexibility
is lost, what the right delegation primitive is, cost suffix.

### 7. Trust boundary posture

Three lists:

a. **In-process Python authorities** — surfaces where the framework
   trusts the caller fully (engine returns `PluginAck`, orchestrator
   trusts the verdict). List them; flag any that should NOT be
   trusted (e.g., engines loaded from outside the canonical engines
   directory).

b. **Subprocess / sidecar authorities** — surfaces that cross a
   process boundary (sidecar engines, MCP servers, external LiteLLM
   processes). For each, state what the framework should validate,
   sanitize, or reject before admitting the response. Today's
   `SidecarAdapter` coerces malformed responses to vetoes —
   is that the right default for every receiver?

c. **Network / human-input authorities** — surfaces that admit
   untrusted input (proxy HTTP requests, MCP HTTP transport, CLI
   args). For each, state the validation posture and whether it's
   sufficient.

End with one paragraph: what the substrate's overall trust model is,
in 3-4 sentences. If the trust model is implicit (no documentation
states it), say so — that's a finding.

### 8. Agent-shaped delegations

The sidecar runtime (Wave 13.1.5) admits arbitrary subprocesses.
A subprocess can be a deterministic Rust binary — or an AI agent
(model call inside a JSON-RPC loop).

Investigate which existing engines would benefit from being
agent-shaped vs deterministic, with a brief rationale per category:

- Engines that should stay deterministic (regex / cryptographic /
  rule-based). Why agents would make these worse (adversarial
  manipulability, latency, non-determinism).
- Engines that would benefit from agent reasoning (judgment calls,
  context-sensitive classification, multi-step inference). What
  tier (Opus / Sonnet / Haiku) fits each.
- Engines that ARE already agent-shaped today (look at
  `deep-research` — multi-phase, multi-tier). Use it as the
  reference architecture.

Identify cross-cutting questions the substrate should answer before
shipping agent-engines at scale:

- Where does an agent-engine's prompt live (manifest field, external
  file, conduct package)?
- Who owns the prompt's authority (engine author? operator? both
  with merge rules)?
- How does an agent-engine's cost get accounted to the right
  tenant/request/user?
- How are agent-engine vetoes audited differently from
  deterministic-engine vetoes (since agents can be wrong in ways
  regex can't)?

Recommend ONE agent-shaped delegation to ship first, with rationale
and cost suffix.

### 9. Observability & audit gaps

From the authority map in section 2, list every surface whose
`Observability` cell is `in-memory` or `none`. For each, write:

- What audit question can't be answered today (e.g., "why did this
  request get a 451 yesterday at 14:32?")
- The minimum durable log shape needed
- Cost suffix

End with one paragraph: which observability gap is closest to a
compliance-gap failure mode, and what should be done first.

### 10. Recommendations as a wave plan

Synthesize sections 3-9 into an actionable wave plan in the codebase's
own pattern (see `ROADMAP.md` for examples). Sort by:

1. Highest-failure-mode severity first
2. Then by lowest cost (additive + parallel-safe + one-shot first)

Each wave entry should specify: name, agents (count + scope), files
touched (specific paths), parallel-safety, dependencies on prior
waves. This section is the report's payload — it converts findings
into a queue the project can dispatch.

Limit to 5 waves. If more than 5 worth-doing items emerge, list the
overflow under "Deferred — re-evaluate next quarter."

## Constraints

- Research only. Do NOT edit any production code. The only file you
  create is the markdown report at the path above.
- Read the codebase liberally. Don't trust this brief's descriptions
  of files — verify them. If a landmark is wrong, report it and
  continue.
- Cite specific files and lines. `[pipeline.py:run](enchanter/proxy/pipeline.py#L42)`
  is the convention.
- Keep the report opinionated. A list of findings without
  recommendations is wasted work. Each recommendation carries a cost
  suffix.
- Length budget: 4000-6000 words including tables. Long enough to be
  useful, short enough to read in one sitting.
- The executive summary must fit in 5 sentences. If you can't, the
  audit isn't done yet.

## Return

Return only:

1. The path to the created file.
2. The 5-sentence executive summary verbatim.
3. The top-3 wave-plan entries from section 10 (name, agent count,
   cost suffix only — not the full description).

Nothing else.
