# Robit → Mimir Attestation Integration

Status: v1, 2026-07-05. Companion to
[RFC-robit-control-plane.md](../../../agent/docs/architecture/RFC-robit-control-plane.md) § 5
(attestation) and the delegation-of-authority audit § 9 gap (4).
Implementation: [`enchanter/proxy/attest.py`](../../enchanter/proxy/attest.py),
wired in [`enchanter/proxy/pipeline.py`](../../enchanter/proxy/pipeline.py).

## What this closes

The RFC names the control plane's practical ceiling: **self-reported
portable proofs + SPRT detecting absence-of-proof**. Before this
integration neither half existed — robit's engines decided, but no signed
proof stream left the process, so the inference substrate's SPRT liveness
detector had nothing to watch. This document specifies the first half: every
proxy enforcement decision (pass or veto) is attested via a Mimir issuer
(`POST /v1/attest`) and spooled locally.

## Decision model

One attestation per pipeline decision, at four emit sites:

| Site | Decision | Notes |
|---|---|---|
| `run()` lifecycle veto | `veto` | after the durable `vetoes.jsonl` record; `http_status: 451` |
| `run()` completion | `pass` | `http_status: 200` |
| `stream()` synchronous trust-gate veto | `veto` | `http_status: 451` |
| `_stream_body()` mid-stream veto / phase-timeout / completion | `veto` / `pass` | mid-stream vetoes carry no HTTP status (stream already open) |

**Pass decisions attest too.** This is load-bearing: the SPRT liveness
signal is *absence of proof*. If envelopes fired only on vetoes, an empty
stream would be indistinguishable from "no attacks today". Attesting every
decision makes the stream a heartbeat; a gate that goes dead produces a
statistically detectable gap.

## Wire contract

`POST {MIMIR_ISSUER_URL}/v1/attest` with the issuer's `AttestRequest` shape
(mimir `issuer/types/types.go`):

```json
{
  "tool_id": "did:web:enchanter-labs.dev:robit:proxy-gate",
  "tool_version": "<enchanter.__version__>",
  "request": {
    "kind": "enchanter.proxy.decision.request",
    "correlation_id": "c-...",
    "session_id": "s-...",
    "request_sha256": "<sha256 of the canonical request>",
    "model": "gpt-4o-mini"
  },
  "result": {
    "kind": "enchanter.proxy.decision",
    "decision": "pass | veto",
    "engine": "destructive-op-gate | null",
    "phase": "trust-gate | post-session | ...",
    "reason": "<plugin>:<pattern_id> | null",
    "pattern_id": "w5-force-push | null",
    "http_status": 451,
    "decided_at": "<RFC 3339>"
  }
}
```

The issuer canonicalizes (RFC 8785), digests, and signs Ed25519; the
returned envelope binds `(request, result)` under one signature.

**Privacy invariant:** the issuer never receives prompt or response
content — only the SHA-256 of the canonical request plus decision
metadata (pattern identifiers, phases, counts). Same content-free rule as
`BusObservation` and `vetoes.jsonl`.

## Local spool

Every attestation attempt — success or failure — appends one line to
`<state_dir>/attest/decisions.jsonl` (same path precedence as the audit
sinks): `{ts, kind: "proxy.decision.attested", correlation_id, decision,
engine, phase, pattern_id, http_status, issuer_url, envelope,
validation_level, error}`.

- On issuer success the line carries the signed envelope.
- On issuer failure the line carries `envelope: null` + the error — the
  decision metadata still lands, so the heartbeat survives issuer outages.
- This spool is the stream the SPRT liveness reader (next wave) consumes.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `ENCHANTER_ATTEST_ENABLED` | unset (off) | opt-in gate; when unset, `attest_decision` is a no-op |
| `MIMIR_ISSUER_URL` | `http://localhost:8080` | issuer base URL |
| `ENCHANTER_ATTEST_TOOL_ID` | `did:web:enchanter-labs.dev:robit:proxy-gate` | envelope `tool_id` |
| `ENCHANTER_ATTEST_TIMEOUT_S` | `3` | issuer POST timeout |

Failure posture: best-effort end to end. Issuer down, spool unwritable,
encode error — all swallowed (logged at WARNING); the request path is
never aborted or blocked beyond the POST timeout, and the timeout only
applies when the operator opted in.

## Snowball guard #1, restated

An envelope proves what this process **reported**, not that the gate
**ran**. A disabled or spoofed proxy emits nothing — which is exactly what
the SPRT absence detector is for — but a *compromised* proxy could emit
well-formed envelopes for decisions it never enforced. This integration
does not claim otherwise; it implements the RFC's stated ceiling, not a
per-call cryptographic guarantee.

## Deliberate non-goals (this wave)

- **No per-engine attestation.** One envelope per pipeline decision, not
  per engine ack — row volume and issuer RPS stay bounded. Per-engine
  detail lives in the bus observations and `vetoes.jsonl`.
- **No DPoP / ClientIdentityProof.** The issuer accepts it optionally
  (spec § 6.11); wiring robit's proxy identity into `invoked_by` waits for
  the originating-CLI identity work (RFC § 9 open question, snowball
  guard #4).
- **No on-chain anchoring from robit.** `MimirValidationRegistry.register`
  is an issuer/operator concern, not the proxy's.
- **No SPRT reader.** Next wave: `enchanter/inference` gains a
  `liveness` consumer over `decisions.jsonl` that flags a gate whose
  heartbeat stopped.
