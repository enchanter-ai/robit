# Codex CLI Wire Protocol — Inventory for Enchanter Interop

> Research date: 2026-05-15
> Researcher: Wave 16.0
> Codex CLI version surveyed: 0.130.0 (latest stable, 2026-05-08); spot-checked against pre-release 0.131.0-alpha.19 (2026-05-15)
> Repo: https://github.com/openai/codex (Rust crate `codex-rs`, Apache-2.0)

## TL;DR

Codex CLI does **not** speak `/v1/chat/completions`. It speaks the **OpenAI Responses API** (`POST /responses`) only — `wire_api = "responses"` is the sole accepted value in 0.130, and `wire_api = "chat"` was hard-removed (sources: `codex-rs/model-provider-info/src/lib.rs`, error message references `discussions/7782`). Two auth modes exist:

1. **API key:** `Authorization: Bearer <sk-...>` against `https://api.openai.com/v1/responses`.
2. **ChatGPT login (Plus/Pro/Business/Edu/Enterprise):** an OAuth-2.0 + PKCE flow against `https://auth.openai.com`, producing a JWT access token, sent as `Authorization: Bearer <access_token>` plus `ChatGPT-Account-ID: <workspace_id>` (and optionally `X-OpenAI-Fedramp: true`) against `https://chatgpt.com/backend-api/codex/responses` — a non-public ChatGPT-internal endpoint.

The base URL is overridable only via the config.toml field `openai_base_url` (built-in `openai` provider) or per-provider `[model_providers.<id>] base_url = "..."`. There is **no `OPENAI_BASE_URL` environment variable**. Streaming is standard SSE (`text/event-stream`) carrying Responses-API events (not chat-completions deltas). Tool calls follow the Responses API tool/`function_call` shape, not the chat-completions `tool_calls` array.

## Auth modes

### API key

- **Default endpoint:** `https://api.openai.com/v1` + `/responses` → `https://api.openai.com/v1/responses`.
  - Source: `codex-rs/model-provider-info/src/lib.rs` `to_api_provider` — `"https://api.openai.com/v1"` is the default when auth_mode is not Chatgpt; `RESPONSES_ENDPOINT = "/responses"` in `codex-rs/core/src/client.rs`.
- **Auth header:** `Authorization: Bearer <api_key>` exactly.
  - Source: `codex-rs/model-provider/src/bearer_auth_provider.rs` `BearerAuthProvider::add_auth_headers` inserts `format!("Bearer {token}")` into `AUTHORIZATION`.
- **Key sources (env vars):**
  - `OPENAI_API_KEY` (default) — `pub const OPENAI_API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";` in `codex-rs/login/src/auth/manager.rs`.
  - `CODEX_API_KEY` is also read when `enable_codex_api_key_env` is true (used by `codex login --with-api-key` plumbing and the SDK).
  - The key can also be supplied via `codex login --with-api-key` reading from stdin (per `developers.openai.com/codex/auth`).
- **Cache location:** `~/.codex/auth.json`, field `"OPENAI_API_KEY"` (struct: `AuthDotJson { openai_api_key: Option<String>, tokens: Option<TokenData> }` — `codex-rs/login/src/auth/storage.rs`). `CODEX_HOME` overrides `~/.codex`. With `cli_auth_credentials_store = "keyring"` (or `"auto"` on supported OS), credentials live in the OS keyring instead.

### ChatGPT login (OAuth + PKCE)

- **OAuth flow** (`codex-rs/login/src/server.rs`):
  - Issuer: `https://auth.openai.com` (`DEFAULT_ISSUER`).
  - Client id: `app_EMoamEEZ73f0CkXaXp7hrann` (`CLIENT_ID` constant in `codex-rs/login/src/auth/manager.rs`).
  - Redirect URI: `http://localhost:1455/auth/callback` (fallback port 1457). Codex CLI starts a local HTTP server, then opens the browser.
  - Authorize URL pattern (built by `build_authorize_url`):
    ```
    https://auth.openai.com/oauth/authorize
      ?response_type=code
      &client_id=app_EMoamEEZ73f0CkXaXp7hrann
      &redirect_uri=http://localhost:1455/auth/callback
      &scope=openid%20profile%20email%20offline_access%20api.connectors.read%20api.connectors.invoke
      &code_challenge=<S256>
      &code_challenge_method=S256
      &id_token_add_organizations=true
      &codex_cli_simplified_flow=true
      &state=<random>
      &originator=<codex_cli|codex_cli_rs>
    ```
  - Token exchange: `POST https://auth.openai.com/oauth/token` with `grant_type=authorization_code&code=...&redirect_uri=...&client_id=...&code_verifier=...`.
  - Refresh: `POST https://auth.openai.com/oauth/token` (URL overridable via `CODEX_REFRESH_TOKEN_URL_OVERRIDE`); refresh interval is 8 hours (`TOKEN_REFRESH_INTERVAL = 8`).
  - Revoke: `https://auth.openai.com/oauth/revoke` (overridable via `CODEX_REVOKE_TOKEN_URL_OVERRIDE`).
  - Headless device-code flow also exists (`codex-rs/login/src/device_code_auth.rs`) — same client_id, no localhost callback.

- **Token data** (`codex-rs/login/src/token_data.rs`):
  ```rust
  pub struct TokenData {
      pub id_token: IdTokenInfo,   // JWT, parsed for chatgpt_account_id, plan_type, user_id, is_fedramp
      pub access_token: String,    // JWT, used as the request Bearer
      pub refresh_token: String,
      pub account_id: Option<String>,
  }
  ```
  Claim namespace: `https://api.openai.com/auth.chatgpt_account_id`, `chatgpt_user_id`, `chatgpt_plan_type`, `chatgpt_account_is_fedramp`.

- **Token cache:** `~/.codex/auth.json` (or OS keyring). Plaintext when `cli_auth_credentials_store = "file"`. Docs explicitly warn to treat the file as a password.

- **Endpoint used with the ChatGPT token:** `https://chatgpt.com/backend-api/codex` + `/responses` → `https://chatgpt.com/backend-api/codex/responses` (constant `CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"` in `codex-rs/model-provider-info/src/lib.rs`; selected when `AuthMode::Chatgpt | ChatgptAuthTokens | AgentIdentity`). Other paths on the same base used by Codex: `/responses/compact`, `/memories/trace_summarize`.

- **Auth headers** when in ChatGPT mode (`codex-rs/model-provider/src/bearer_auth_provider.rs`):
  - `Authorization: Bearer <access_token>` (JWT from auth.json).
  - `ChatGPT-Account-ID: <chatgpt_account_id>` (from JWT claim or refresh response).
  - `X-OpenAI-Fedramp: true` when the workspace is FedRAMP.

## Request format

Body is a Responses-API JSON payload, **not** chat-completions. Captured/reverse-engineered representative body (Simon Willison's Nov-2025 trace; matches `ResponsesApiRequest` in `codex-rs/codex-api`):

```json
{
  "model": "gpt-5-codex-mini",
  "instructions": "<system instructions>",
  "input": [
    { "type": "message", "role": "developer",
      "content": [{ "type": "input_text", "text": "..." }] },
    { "type": "message", "role": "user",
      "content": [{ "type": "input_text", "text": "..." }] }
  ],
  "tools": [],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "reasoning": { "summary": "auto" },
  "store": false,
  "stream": true,
  "include": ["reasoning.encrypted_content"],
  "prompt_cache_key": "<uuid>"
}
```

Key deviations from `/v1/chat/completions`:
- `messages: [...]` becomes `input: [{type:"message", role, content:[{type:"input_text", text}]}]`.
- System prompt is a top-level string field `instructions` (omitting it returns HTTP 400).
- A third role `developer` is supported alongside `user`/`assistant`.
- `store: false` is set by default (no server-side history); reasoning state is carried back via `include: ["reasoning.encrypted_content"]` and `previous_response_id`.
- Additional Codex-only headers on every request: `OpenAI-Beta: responses_websockets=2026-02-06` (when WebSocket transport is enabled), `x-codex-installation-id`, `x-codex-turn-state` (sticky routing within a turn), `x-codex-turn-metadata`, `x-codex-parent-thread-id`, `x-codex-window-id`, `x-openai-subagent` (sub-agent calls), `x-responsesapi-include-timing-metrics`, optionally `x-oai-attestation`.

## Streaming

- `Accept: text/event-stream`; response is standard SSE (`data: <json>\n\n`).
  - Source: `codex-rs/codex-api/src/endpoint/responses.rs` (`text/event-stream`), `codex-rs/codex-api/src/sse/responses.rs` (uses `eventsource_stream::Eventsource`).
- Events are **Responses-API event types**, not chat-completion deltas: `response.created`, `response.output_item.added`, `response.output_text.delta`, `response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`, `response.completed`, `response.failed`, etc.
- The 0.130+ Codex CLI also supports a **WebSocket transport** for Responses (`OpenAI-Beta: responses_websockets=2026-02-06`, opcode-based prewarm with `generate=false`, sticky routing via `x-codex-turn-state`). HTTP-SSE is the fallback and the canonical wire format any compliant server must support.

## Tool calls

- Follows the Responses API tool shape: tools are declared in a top-level `tools: [...]` array (function tools have `{type:"function", name, description, parameters}` — note: flat fields, **not** wrapped in a `function:{...}` sub-object as chat-completions does).
- Tool invocations stream as `response.output_item.added` of `type:"function_call"` with `name`, `call_id`, and `arguments` (the arguments string is appended via `response.function_call_arguments.delta`).
- Tool outputs are returned in the next request's `input` array as `{type:"function_call_output", call_id, output}` items.
- Codex also uses Responses-API native tools (`web_search`, custom MCP function tools surfaced through Codex's tool registry — see `codex-rs/tools/src/...` and `create_tools_json_for_responses_api`).

## Base URL override

- **Config-only**, no env var.
- `~/.codex/config.toml`:
  ```toml
  # Override the built-in openai provider's base URL (replaces api.openai.com/v1)
  openai_base_url = "https://enchanter.local/v1"

  # Or define a fully custom provider
  [model_providers.enchanter]
  name = "Enchanter Proxy"
  base_url = "https://enchanter.local/v1"
  env_key = "ENCHANTER_API_KEY"          # provider reads bearer token from this env var
  wire_api = "responses"                  # only legal value
  # optional:
  http_headers = { "X-Foo" = "bar" }
  env_http_headers = { "X-Auth" = "ENCHANTER_AUTH" }
  requires_openai_auth = false            # set true to reuse Codex's ChatGPT/API-key auth

  model_provider = "enchanter"
  ```
- CLI override: `codex --config openai_base_url="https://enchanter.local/v1" ...` (this is also how the official TypeScript SDK injects `baseUrl` — `sdk/typescript/src/exec.ts`).
- Custom CA bundle: `CODEX_CA_CERTIFICATE` env var.
- Token-endpoint overrides (relevant only if proxying OAuth too): `CODEX_REFRESH_TOKEN_URL_OVERRIDE`, `CODEX_REVOKE_TOKEN_URL_OVERRIDE`.

## Enchanter interop verdict

1. **Can the existing OpenAI adapter (`/v1/chat/completions`) handle Codex's API-key-mode requests as-is?**
   **No.** Codex sends `POST /v1/responses` with a Responses-API body (`instructions` + `input[]` + reasoning/include), not chat-completions. The adapter needs a `/v1/responses` route that accepts the Responses request schema and emits Responses SSE events. Re-mapping Responses↔chat-completions on the fly is possible but loses fidelity around reasoning summaries, `developer` role, `previous_response_id`, and encrypted reasoning content.

2. **Can pass-through auth (Wave 16.1) forward Codex's ChatGPT-login token through to the actual ChatGPT-internal endpoint?**
   **Yes, mechanically — but be aware of two non-OpenAI-standard headers.** Forward `Authorization: Bearer <jwt>` plus `ChatGPT-Account-ID: <workspace>` (and `X-OpenAI-Fedramp: true` when present) to upstream URL `https://chatgpt.com/backend-api/codex/responses`. The `Authorization` JWT is opaque to enchanter (signed by `auth.openai.com`), so the proxy cannot mint or rotate it — it must accept Codex's token verbatim and let the user re-`codex login` when refresh fails. Caveat: this endpoint is undocumented/private and OpenAI's ToS for ChatGPT subscriptions may forbid third-party routing.

3. **Does Codex need a dedicated adapter or can it ride on the OpenAI adapter?**
   **Dedicated adapter (or a Responses-API extension to the OpenAI adapter).** Reasoning: the wire format is structurally different (Responses, not chat-completions), `wire_api = "chat"` was explicitly removed in Codex, and the request includes Codex-specific headers (`x-codex-turn-state`, `x-codex-installation-id`, optional WebSocket upgrade with `OpenAI-Beta: responses_websockets=...`) that benefit from first-class handling. Minimum viable adapter is HTTP-only Responses SSE; WebSocket prewarm can be punted to v2 since Codex falls back to HTTP transparently.

## Sources

- https://github.com/openai/codex — repo root, Apache-2.0, Rust crate `codex-rs`, latest stable `rust-v0.130.0` (2026-05-08).
- https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs — `RESPONSES_ENDPOINT = "/responses"`, Codex-specific request headers, WebSocket-v2 beta header.
- https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs — `CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"`, default `https://api.openai.com/v1`, `WireApi::Responses` (sole accepted value), `ModelProviderInfo` config fields (`base_url`, `env_key`, `http_headers`, `env_http_headers`, `requires_openai_auth`, etc.), `to_api_provider` base-URL selection by auth_mode.
- https://github.com/openai/codex/blob/main/codex-rs/model-provider/src/bearer_auth_provider.rs — `Authorization: Bearer <token>` + `ChatGPT-Account-ID` + `X-OpenAI-Fedramp` header construction.
- https://github.com/openai/codex/blob/main/codex-rs/model-provider/src/auth.rs — wires `CodexAuth::{ApiKey, Chatgpt, ChatgptAuthTokens, AgentIdentity}` to `BearerAuthProvider`/`AgentIdentityAuthProvider`.
- https://github.com/openai/codex/blob/main/codex-rs/login/src/server.rs — OAuth authorize-URL builder, scopes, PKCE, redirect URI (`http://localhost:1455/auth/callback`), token exchange against `<issuer>/oauth/token`.
- https://github.com/openai/codex/blob/main/codex-rs/login/src/auth/manager.rs — `CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"`, `DEFAULT_CHATGPT_BACKEND_BASE_URL = "https://chatgpt.com/backend-api"`, `REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"`, `REVOKE_TOKEN_URL`, env var `OPENAI_API_KEY`, refresh-URL override env vars.
- https://github.com/openai/codex/blob/main/codex-rs/login/src/token_data.rs — `TokenData` shape (`id_token`, `access_token`, `refresh_token`, `account_id`); JWT claim parsing for `chatgpt_account_id`, `chatgpt_plan_type`, `chatgpt_user_id`, `chatgpt_account_is_fedramp`.
- https://github.com/openai/codex/blob/main/codex-rs/login/src/auth/storage.rs — `AuthDotJson { OPENAI_API_KEY, tokens }` schema for `~/.codex/auth.json`; keyring backend.
- https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/endpoint/responses.rs — `ResponsesClient::stream_request` posts to path `responses` with `Content-Type` JSON, `Accept: text/event-stream`.
- https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/sse/responses.rs — SSE parser using `eventsource_stream::Eventsource`.
- https://developers.openai.com/codex/auth — official authentication overview (ChatGPT sign-in + API key, `~/.codex/auth.json` cache, `cli_auth_credentials_store` strategies).
- https://developers.openai.com/codex/config-reference — config fields: `openai_base_url`, `model_provider`, `model_providers.<id>` (base_url, env_key, env_http_headers, http_headers, auth, wire_api).
- https://developers.openai.com/codex/cli — built-in providers (`openai`, `ollama`, `lmstudio`).
- https://simonwillison.net/2025/Nov/9/gpt-5-codex-mini/ — reverse-engineering of the `chatgpt.com/backend-api/codex/responses` endpoint and the literal request body (used above as the representative payload).
- https://github.com/openai/codex/discussions/7782 — `wire_api = "chat"` removal announcement; Responses API is the only supported wire format going forward.

## Open questions

1. **WebSocket transport surface.** Codex 0.130 prefers `OpenAI-Beta: responses_websockets=2026-02-06` over HTTP-SSE for ChatGPT-auth sessions. The exact upgrade handshake (subprotocol name, message framing for `response.create`/incremental requests, `previous_response_id` propagation) needs a deeper read of `codex-rs/codex-api/src/...` (`ResponsesWebsocketClient`, `ResponsesWsRequest`) before enchanter decides whether to implement WS or rely on Codex's HTTP fallback path. Material for Wave 16.3 if it wants parity with first-party transport.
2. **`x-codex-turn-state` server contract.** The header is described as opaque sticky-routing state set by the server on first response and replayed on subsequent in-turn requests. An enchanter proxy that load-balances across upstreams must either (a) terminate Responses turns locally and mint its own token, or (b) pin a turn to one upstream. Unclear from docs which is acceptable without breaking Codex.
3. **Attestation header (`x-oai-attestation`).** `AttestationProvider` and `X_OAI_ATTESTATION_HEADER` are referenced from `client.rs`, gated by `include_attestation` config. Source for the attestation payload (TPM? Apple DCAppAttest? a JWT?) is in `codex-rs/core/src/attestation.rs` (not read this pass). If the ChatGPT backend ever begins to require attestation, pass-through proxying will break — needs explicit verification before Wave 16.3 ships.
