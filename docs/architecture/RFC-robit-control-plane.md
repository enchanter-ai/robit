# RFC: Robit as Ecosystem Control Plane

- **Status:** Draft (proposal) — 2026-06-17
- **Author:** Enchanter Labs
- **Supersedes:** the name "Robit" as used for the executor coding-agent (positioning lock 2026-06-08). The independent assurance *verifier* remains a separate product with its own coinage; only the name "Robit" is reassigned.
- **Evidence base:** deep-research 2026-06-17 (24 primary sources, 22/25 claims confirmed; NIST SP 800-162, XACML, OpenID AuthZEN, Istio ambient, SLSA/in-toto VSA) + prior golden-product finding 2026-06-09.

## 1. Problem

Robit's original charter — *enforce the enchanter-ai governance ecosystem deterministically* — snowballed into a full homegrown coding-agent CLI (REPL/TUI, 7 tools, subagents, plan mode, multi-provider auth) plus a spun-off observability product (beholder). The charter was right; the implementation grew without impact. A CLI is not the determinism mechanism — it is invoked on demand and enforces nothing on its own. We do not need another coding agent; we need the layer that makes the governance everyone already has actually fire across the CLIs people actually use (Claude Code, Codex, opencode).

## 2. Decision

**Robit becomes the ecosystem control plane.** It owns zero enforcement *primitives* and zero new agent surface. It turns on the existing governance bundle (vis hooks + engines + mimir + inference-substrate) across heterogeneous CLIs it does not own, and attests they are actually working. The coding-agent surface is **frozen** (§7).

## 3. Architecture: PEP/PDP split

This is the decades-stable Policy-Enforcement-Point / Policy-Decision-Point pattern. Both halves already exist in `agent/robit/`; we are naming and repositioning them, not building them.

| Role | Component | Existing module |
|---|---|---|
| **PEP** (enforce) | Wire-format proxy — a reverse-proxy "authorization firewall" gating cc/codex/oc LLM traffic with zero per-client change | `robit/proxy/` (`pipeline.py`, `canonical.py`, `upstream.py`, `streaming.py`) |
| **PDP** (decide) | 14 deterministic engines — centralized decision, distributed enforcement | `robit/engines/*` |
| Policy injection | conduct → system-prompt XML | `robit/conduct/`, `robit/composer/` |
| Attestation | mimir VSA-style signed proofs (separate repo) | `mimir/` |
| Liveness | SPRT/Beta-Binomial cross-session accumulation | `robit/inference/`, `robit/engines/inference_substrate/` |

**Why the proxy boundary wins over in-process hooks (decisive, 3-0):** out-of-process enforcement *survives a compromised client* — a compromised workload cannot disable enforcement on its own traffic; in-process hooks can be switched off by the thing they govern (Istio ambient argument).

### 3.1 Control-plane / data-plane split with beholder (design hypothesis — needs confirmation)

Istio's model maps cleanly onto the existing product division: **beholder = data plane** (the proxy/inspector that sits in traffic; it already absorbed the HTTP proxy server + MCP server in robit 0.8.0), **Robit = control plane** (decides policy, projects per-CLI adapters, attests liveness). This avoids re-absorbing the HTTP serving into Robit. **OPEN:** confirm against the `enchanter-ai/beholder` repo before committing — beholder is not vendored locally.

## 4. Three enforcement tiers

Two CLI-agnostic boundaries + one accelerator. Hooks **cannot** be the foundation (Codex has no hook system; Gemini's differs) — confirmed twice independently.

1. **Proxy (PEP)** — gates generation *in-flight* (runtime, pre-execution): secret-mask, destructive-op, CVE, trust before it happens.
2. **Git hooks (pre-commit/pre-push) + CI/PR required status check** — gates the *produced change* (post-generation). The truly worker-independent boundary.
3. **cc-native hooks** — in-process sensor/accelerator **only**, for intra-CLI context the proxy is structurally blind to. "Every product's hooks" = aggregate them as *signals into one PDP*, not Robit owning every hook.

## 5. Attestation (mimir) and the core hard problem

mimir maps onto SLSA/in-toto's three layers (Envelope/Statement/Predicate) and the **Verification Summary Attestation** pattern: Robit is the trusted verifier emitting cacheable, delegable signed proofs that a gate ran; consumers trust Robit's signer identity without re-running. (Layering is portable; the wire format is **not** drop-in — SLSA wants DSSE, mimir uses RFC-8785/JCS. Decide before claiming compatibility.)

**The wall — unsolved by any researched pattern:** you cannot cryptographically prove a gate *actually fired* on a CLI you do not own. in-toto/SLSA attestations only prove what the gate **self-reports**; a disabled or spoofed gate emitting a well-formed signed envelope defeats it. SPIFFE/SPIRE infra-rooted attestation breaks without host control (Robit owns no cc/codex/oc hosts). **Practical ceiling = self-reported portable proofs, with the inference-substrate's SPRT detecting a gate that *stopped emitting proofs* (the dead-policy / absence-of-proof case) statistically — not a per-call guarantee.**

## 6. Activation model: manifest + per-CLI adapters

One policy **manifest** (the PDP's source of truth) projected onto each CLI by **thin adapters** (cc hook adapter, codex/oc proxy config). Not a one-shot install (fails "follow if they're working") and not a resident control-loop daemon (that is how this becomes Istio). Liveness is a **light** layer: SPRT-on-attestation, not a daemon.

## 7. Freeze list

The coding-agent surface is frozen — no new features, name surrendered. **Freeze** `agent/robit/agent/`:
`app.py`, `loop.py`, `conversation.py`, `repl`/`footer`/`cost`/`diff`/`enforcement` widgets, `tools/*` (file_read/write/edit/glob/grep/bash/web_fetch), `subagents/*`, `slash_commands/*`, `plan.py`, `session.py`, `login.py`, `mcp/client.py`.

**Keep (the control-plane kernel):** `robit/proxy/`, `robit/engines/*`, `robit/conduct/`, `robit/composer/`, `robit/inference/`, `robit/core/`, `robit/loader/`, `robit/transport/`, `robit/runtime/`, `robit/protocol/`. **`robit/llm/`:** keep upstream clients the proxy needs; the ChatGPT-subscription executor auth path is frozen with the agent.

## 8. Snowball guards (normative — pin these)

1. Do **not** treat a mimir envelope as proof a gate *ran* when it only proves what the gate *reported*.
2. Do **not** ask the proxy to enforce policy needing CLI-internal state it cannot see (coarse-grained ceiling: it sees only method/body/token).
3. **Single L7 owner per CLI** — never run Robit's L7 enforcement concurrent with another L7 layer on the same traffic (split-brain).
4. **Re-attach originating-CLI identity at the proxy** — an interposing proxy masks source identity, breaking per-CLI policy and attribution.
5. **No resident control-loop daemon.** If Robit grows a persistent process beyond decide-policy + project-adapters + read-substrate-liveness, it is becoming a mesh.

## 9. Open questions

- Confirm the beholder = data-plane / Robit = control-plane split against the repo (§3.1).
- Unspoofable originating-CLI identification at the wire boundary when the proxy terminates the connection.
- DSSE vs RFC-8785 envelope decision for mimir.
- Is a minimal per-client hook adapter justified as a lower-trust complementary signal for CLI-internal policy, reconciled with the single-L7-owner constraint?
