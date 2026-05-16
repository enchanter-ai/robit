# Authentication — `robit` and `insighter`

> One page covering: where each auth mode applies, how tokens resolve,
> where they cache, what's logged. Accurate as of 0.7.0; if behavior
> drifts, file an issue.

## TL;DR

Two binaries (`robit` coding agent + `insighter` inspector/proxy)
support three first-class auth modes: Anthropic API key, Claude.ai
OAuth (Pro/Max subscription), and ChatGPT subscription (Plus/Team/
Enterprise). The proxy adds three honest pass-through patterns —
operator-pays (default), host-agent-pays (`--passthrough-auth`), and
fast-path bypass (env-gated, key-allowlisted, audit-logged). Both
binaries auto-load `.env` from cwd and the user config dir at startup.
Tokens cache under `~/.enchanter/` (POSIX) or `%APPDATA%\enchanter\`
(Windows); none of them leave your machine or the upstream provider.

## Quick start

Three example flows, copy-pasteable.

### "I have an Anthropic API key"

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

```bash
robit "refactor auth.py"
```

### "I want to use my Claude.ai Pro/Max subscription"

```bash
# .env  (get CLAUDE_CODE_OAUTH_TOKEN from Claude Code's config)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat...
```

```bash
robit "refactor auth.py"
```

### "I want to use my ChatGPT subscription"

```bash
robit login chatgpt    # opens browser, saves token to ~/.enchanter/chatgpt-token.json
robit "refactor auth.py"
```

`robit login --list` shows cached tokens; `robit logout chatgpt`
(or `--all`) clears them.

## Environment variables — the full matrix

| Env var | Used by | Auth mode | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | robit, insighter | API key | Highest precedence in `AnthropicClient` resolution (`robit/llm/anthropic_client.py:53`) |
| `CLAUDE_CODE_OAUTH_TOKEN` | robit, insighter | Claude.ai OAuth | Sent as `Authorization: Bearer …` plus `anthropic-beta: oauth-2025-04-20` (`robit/llm/anthropic_client.py:56`, `:72`) |
| `ANTHROPIC_AUTH_TOKEN` | robit, insighter | Claude.ai OAuth | Alt env name; same code path (`robit/llm/anthropic_client.py:57`) |
| `OPENAI_API_KEY` | robit (when model is `gpt-*` / `o*`), insighter proxy upstream | API key | LiteLLM consumes (`robit/proxy/upstream.py:15`) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | robit (when model is `gemini-*`), insighter proxy upstream | API key | LiteLLM consumes (`robit/proxy/upstream.py:17`) |
| `CHATGPT_SESSION_TOKEN` | robit | ChatGPT subscription | JSON blob or bare access_token. Rarely set directly — prefer `robit login chatgpt` (`robit/llm/chatgpt_client.py:83`) |
| `ENCHANTER_HOME` | robit, insighter | (config dir override) | Default `~/.enchanter` POSIX, `%APPDATA%\enchanter` Windows (`robit/llm/_chatgpt_auth.py:41`, `robit/_env.py:36`) |
| `ENCHANTER_ALLOW_FASTPATH_BYPASS` | insighter proxy | (operator gate) | Must be exactly `1` to enable fast path (`robit/proxy/fastpath.py:72`, `:151`) |
| `ENCHANTER_STATE_DIR` | insighter proxy | (state path override) | Overrides where audit JSONLs and allowlist live (`robit/proxy/fastpath.py:73`, `:109`) |
| `ENCHANTER_AGENT_MOCK` | robit | (test) | Use deterministic mock LLM; no real network call (`robit/agent/cli.py:15`) |

## `.env` loading

Wave 17.0 added stdlib `.env` auto-loading. Both `robit` and
`insighter` call `robit._env.load_env_files()` at the top of
`main()` (`robit/agent/cli.py:284`, `robit/insighter/__init__.py:630`).

**Lookup precedence** (highest wins; `robit/_env.py:202`):

1. `<cwd>/.env`
2. `<user_dir>/.env` — `ENCHANTER_HOME/.env` if set, else `%APPDATA%\enchanter\.env`
   (Windows) or `~/.enchanter/.env` (POSIX) (`robit/_env.py:36`).

The shell still wins by default — `os.environ` values already present
are not overwritten (`robit/_env.py:218`). Inside one file, the
last definition of a key wins.

**Syntax supported** (`robit/_env.py:97`):
- `KEY=value`, `# comments`, blank lines.
- Double-quoted strings: `\n \t \\ \"` escapes.
- Single-quoted strings: literal.
- `export KEY=value` prefix tolerated.
- **No interpolation:** `$VAR` in a value is literally `$VAR`.

Invalid lines are logged at WARNING and skipped — parsing continues.

## Token cache files

| File | Created by | Lifetime |
|---|---|---|
| `~/.enchanter/chatgpt-token.json` | `robit login chatgpt` | Refreshed automatically until refresh fails (`robit/llm/_chatgpt_auth.py:196`) |
| `~/.enchanter/anthropic-token.json` | (none — placeholder; see "Honest limitations") | n/a |
| `<state_dir>/fastpath-allowlist.json` | operator (manual) | Persistent (`robit/proxy/fastpath.py:125`) |
| `<state_dir>/audit/fastpath-bypass.jsonl` | proxy fast-path | Persistent, append-only (`robit/proxy/fastpath.py:129`) |

Default `<state_dir>` is `<repo>/state` when a `pyproject.toml` is
detected nearby, else `~/.enchanter` (POSIX) or `%APPDATA%\enchanter\`
(Windows) — see `robit/proxy/fastpath.py:109`. Override with
`ENCHANTER_STATE_DIR`.

## `robit` coding agent — auth resolution

### Anthropic (API key + Claude.ai OAuth)

`AnthropicClient.__init__` (`robit/llm/anthropic_client.py:35`)
resolves credentials in this order when neither constructor arg is
given:

1. `ANTHROPIC_API_KEY` → x-api-key mode (`auth_mode = "api_key"`).
2. `CLAUDE_CODE_OAUTH_TOKEN` → OAuth bearer mode.
3. `ANTHROPIC_AUTH_TOKEN` → OAuth bearer mode.

API-key and OAuth are mutually exclusive (`ValueError` if both passed
explicitly, `:50`). OAuth mode pins the
`anthropic-beta: oauth-2025-04-20` header — required by the upstream
to accept OAuth-issued tokens (`:72`).

### ChatGPT subscription

`ChatGptClient.__init__` (`robit/llm/chatgpt_client.py:69`) walks
this resolution chain:

1. Explicit `token=` argument.
2. `CHATGPT_SESSION_TOKEN` env var — JSON blob (matching the cache
   shape) or bare `access_token` (treated as 1-hour expiry, no refresh).
3. Cache file at `~/.enchanter/chatgpt-token.json`
   (`robit/llm/_chatgpt_auth.py:41`).
4. Otherwise raises `ConfigurationError`.

Tokens are refreshed automatically when within 60 s of expiry
(`robit/llm/_chatgpt_auth.py:196`). On a 401, the client attempts
one refresh + retry; a second 401 raises with a
"re-run `codex login`" message (`robit/llm/chatgpt_client.py:174`).

The upstream endpoint is hardcoded:
`https://chatgpt.com/backend-api/codex/responses`
(`robit/llm/chatgpt_client.py:43`). Headers added: `Authorization:
Bearer <jwt>`, `ChatGPT-Account-ID: <acct>` (when present in the JWT
claim — `robit/llm/_chatgpt_auth.py:112`). This shape is mirrored
from Codex CLI; see `docs/architecture/audits/codex-protocol.md` for
the audit.

### `robit login` / `logout` (Wave 17.1)

`robit.agent.login` (`robit/agent/login.py`) provides:

- `robit login chatgpt` — runs the PKCE flow
  (`robit/llm/_chatgpt_auth.py:280`), saves the token to
  `~/.enchanter/chatgpt-token.json`.
- `robit login anthropic` — prints a stub explaining there's no
  standalone OAuth flow today; use Claude Code's `/login` and export
  `CLAUDE_CODE_OAUTH_TOKEN` (`robit/agent/login.py:126`).
- `robit login --list` — summarises cached tokens, redacts secret
  prefixes, prints expiry (`robit/agent/login.py:192`).
- `robit logout <provider>` or `--all` — deletes token files
  (`robit/agent/login.py:227`).

Exit codes: 0 success; 1 user denied / generic; 2 timeout; 3 other
auth error; 130 Ctrl-C.

## `insighter serve --proxy` — the three pass-through patterns

The proxy is started with `insighter serve --proxy HOST:PORT`. It
accepts requests in four wire formats (Anthropic Messages, OpenAI Chat
Completions, Gemini Generate, Codex Responses) and forwards them
upstream. Auth is decided per `ProxyServer` flag (see
`robit/proxy/server.py:135`).

### Pattern A: operator pays (default)

The operator sets provider keys in the proxy process's environment.
LiteLLM picks them up internally (`robit/proxy/upstream.py:15-18`),
and any auth header the host agent sent is **ignored** for upstream
routing.

Use when: you're paying for shared inference for a team, or running a
lab proxy that enforces conduct on third-party traffic.

### Pattern B: host agent pays (`--passthrough-auth`)

When the server is started with `passthrough_auth=True`
(`robit/proxy/server.py:169`), the inbound auth header is
extracted by `_extract_inbound_auth`
(`robit/proxy/server.py:640`):

| Family | Header read | Resulting kind |
|---|---|---|
| Anthropic | `x-api-key` | `anthropic-api-key` |
| Anthropic | `Authorization: Bearer …` | `anthropic-oauth` |
| OpenAI | `Authorization: Bearer …` | `openai-bearer` |
| Codex | `Authorization: Bearer eyJ…` (JWT shape) | `chatgpt-jwt` |
| Codex | `Authorization: Bearer sk-…` | `openai-bearer` |
| Gemini | `x-goog-api-key` | `gemini-api-key` |

The credential is stashed on `canonical_req.metadata`
(`_enchanter_passthrough_auth`) and consumed by `upstream.py`'s
`_passthrough_auth_kwargs` (`robit/proxy/upstream.py:144`):

- `anthropic-api-key`, `openai-bearer`, `gemini-api-key` → `api_key`
  kwarg on LiteLLM.
- `anthropic-oauth` → `extra_headers={"Authorization": "Bearer …"}`
  plus a placeholder `api_key` (LiteLLM ≤ 1.50.x requires `api_key` to
  be non-empty even when `extra_headers` carries the real auth).
- `chatgpt-jwt` → recognised at the inbound layer (Wave 17.2 in
  flight); the dedicated non-LiteLLM upstream path
  (`_call_chatgpt_internal`) is not yet wired in `upstream.py` —
  requests with a JWT auth kind fall through to LiteLLM, which has no
  provider for `chatgpt.com/backend-api/codex/responses` and will
  error. See "Honest limitations".

Use when: each host agent has its own subscription / billing, but you
still want enforcement (conduct injection, secret mask, cost ledger,
veto-able pattern gates) on every request.

### Pattern C: fast-path bypass (skip enforcement)

`robit/proxy/fastpath.py` implements a byte-pass-through that
**skips conduct injection and the lifecycle trust-gate entirely**. It
fires only when **all of**:

1. `ENCHANTER_ALLOW_FASTPATH_BYPASS=1` at process start
   (`robit/proxy/fastpath.py:72`, `:151`).
2. SHA-256 of the caller's auth header value is listed in
   `<state_dir>/fastpath-allowlist.json` (`:253`).
3. Method is `POST`, path is one of `/v1/messages`,
   `/v1/chat/completions`, or `/v1beta/models/<model>:(generate|stream
   Generate)Content` (`:195`).
4. Body parses as a JSON object, no `tools` field, `stream: true`
   absent, model in `allowed_models` if specified (`:262`).
5. Body ≤ `max_body_bytes` (default 1 MiB; `:258`).

Auth header is forwarded **verbatim** to the upstream
(`:307`): `x-api-key` for Anthropic, `Authorization: Bearer …` for
OpenAI, `x-goog-api-key` for Gemini. Every bypass appends a record to
`<state_dir>/audit/fastpath-bypass.jsonl` and the response carries
`X-Enchanter-FastPath: bypass` (`robit/proxy/server.py:298`).

**This bypasses pattern vetos, secret mask, cost ledger.** Treat
`fastpath-allowlist.json` as a sensitive file (use 0600 on POSIX).

### When to use which

| Need | Pattern |
|---|---|
| Single operator key, enforcement everywhere | A (default) |
| Multi-tenant proxy, per-user billing, enforcement everywhere | B (`--passthrough-auth`) |
| Trusted internal caller, hot path, willing to log + accept zero enforcement | C (fast path) |

## How to point host agents at enchanter

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude
```

Sends OAuth bearer (if Claude Code is logged in) or `x-api-key`. The
proxy reads `/v1/messages`.

### Codex CLI

```bash
codex --config openai_base_url=http://127.0.0.1:8000/v1
# or in ~/.codex/config.toml:
#   openai_base_url = "http://127.0.0.1:8000/v1"
```

Codex CLI speaks the Responses API only (`POST /responses`, not
`/v1/chat/completions`) — see
`docs/architecture/audits/codex-protocol.md` for the audit. The
`CodexAdapter` matches that path.

### Cursor / Cline / any OpenAI-compatible host

Set the host's base URL config to `http://127.0.0.1:8000`. The proxy
will see `/v1/chat/completions` requests.

## What's logged

- **Fast-path bypass audit**
  (`<state_dir>/audit/fastpath-bypass.jsonl`,
  `robit/proxy/fastpath.py:356`): timestamp (epoch + ISO),
  upstream provider, **short** key hash (first 12 chars of SHA-256;
  not the credential), model, body size, upstream status. **No prompt
  text. No response text. No full credential.**
- **Bus headers on non-streaming responses** (`robit/proxy/server.py:702`):
  `X-Enchanter-Bus-Events`, `X-Enchanter-Mask-Matched` (when secret
  mask fires), `X-Enchanter-Cost-Cents`.
- **Veto** (`:577`): when a pattern gate vetoes, the 451 response body
  includes `phase`, `plugin`, `reason`, `pattern_id`, `pattern_name` —
  no inbound credential is included.
- **`_passthrough_auth_kwargs` honesty note** (`robit/proxy/upstream.py:154`):
  the credential is held only on `req.metadata["_enchanter_passthrough_auth"]`,
  stripped before being forwarded to LiteLLM's `metadata` bag (`:208`),
  and never logged.

Streaming responses do **not** carry bus headers, because bus
observations fire after the iterator is exhausted and the headers are
already on the wire (`robit/proxy/server.py:33`). Documented
intentionally; do not expect parity.

## Honest limitations (read this)

- **Anthropic OAuth via proxy passthrough** carries a TODO marker on
  LiteLLM's `extra_headers` acceptance
  (`robit/proxy/upstream.py:160`). Works in tests; not verified
  against a real Anthropic LiteLLM round-trip across multiple LiteLLM
  versions.
- **ChatGPT pass-through ships in Wave 17.2.** Inbound side
  shape-matches JWTs and returns `kind=chatgpt-jwt` with the
  `ChatGPT-Account-ID` header captured. Outbound side branches in
  `call_upstream` and `stream_upstream` to `_call_chatgpt_internal`,
  which posts directly to `https://chatgpt.com/backend-api/codex/responses`
  via stdlib `urllib` — bypassing LiteLLM entirely. Non-streaming works
  end-to-end; **streaming over this path is deferred** —
  `_stream_chatgpt_internal` raises `NotImplementedError` with a clear
  Wave 18 marker (stdlib `urllib` has no async chunked-read; needs a
  worker-thread + `asyncio.Queue` pump).
- **Direct `ChatGptClient` works today** (Wave 16.3): it bypasses
  LiteLLM entirely and uses stdlib `urllib`
  (`robit/llm/chatgpt_client.py:154`). Streaming over the
  ChatGPT-internal endpoint is deferred to a later wave —
  `req.stream = True` raises `NotImplementedError`
  (`robit/llm/chatgpt_client.py:136`).
- **`robit login anthropic` is a stub.** There is no standalone
  PKCE flow for Claude.ai today; the command prints instructions to
  use Claude Code's `/login` and export `CLAUDE_CODE_OAUTH_TOKEN`
  manually (`robit/agent/login.py:142`).
- **Fast path skips enforcement.** Pattern vetos, secret mask, cost
  ledger do **not** run on bypassed requests. The env gate + per-key
  allowlist bound _who_ can bypass; they do not change _what_ is
  bypassed for them (`robit/proxy/fastpath.py:25`).
- **`.env` shell precedence.** `load_env_files()` does not override
  shell-set variables by default (`robit/_env.py:218`). If you
  change a `.env` value but the same name is exported in your shell,
  the shell wins. Either `unset` the shell var or call
  `load_env_files(override=True)` (no CLI flag for this in 0.7.0).
- **Tokens never phone home.** Enchanter writes tokens only to the
  configured cache dir and sends them only to the upstream provider's
  API (api.anthropic.com, api.openai.com, generativelanguage.googleapis.com,
  chatgpt.com). There is no telemetry endpoint.

## See also

- `docs/architecture/delegation-of-authority.md` — broader
  architectural audit including the proxy / sidecar trust boundary.
- `docs/architecture/audits/codex-protocol.md` — Codex CLI / ChatGPT-
  internal endpoint research (Wave 16.0).
- `ROADMAP.md` — wave history (Wave 17 series: `.env` autoload, login
  commands, ChatGPT proxy passthrough).
