"""enchanter.llm.chatgpt_client — ChatGptClient (ChatGPT subscription auth).

Wave 16.3 wires the Responses-API upstream call to
``https://chatgpt.com/backend-api/codex/responses`` using the cached
subscription JWT (and ``ChatGPT-Account-ID`` header). Token refresh, body
construction, and response parsing are shared with the proxy CodexAdapter via
:mod:`enchanter.llm._codex_responses`.

v1 limitations
--------------
* **Non-streaming only.** ``CompletionRequest`` has no ``stream`` flag today;
  if a caller ever sets ``getattr(req, "stream", False)`` to ``True``, this
  client raises :class:`NotImplementedError`. SSE streaming over the
  ChatGPT-internal endpoint lands in Wave 17+.
* **Stdlib HTTP path.** We use ``urllib.request`` (via
  ``asyncio.to_thread``), matching :class:`AnthropicClient`'s discipline. No
  LiteLLM — the ChatGPT-internal endpoint is non-standard and LiteLLM has no
  provider for it.
* **One refresh retry on 401.** If the upstream returns 401 we attempt a
  single refresh-token round-trip and retry; a second 401 raises with a
  clear "please re-run codex login" message.
* **No attestation header** (``x-oai-attestation``) — Wave 16.0 flagged.
* **No ``x-codex-turn-state``** — we don't propagate Codex's per-turn sticky
  routing state because we don't have a turn manager here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ._chatgpt_auth import ChatGptToken, load_cached_token, refresh_if_needed
from ._codex_responses import build_responses_request, parse_responses_completion
from .types import CompletionRequest, CompletionResponse

# Upstream endpoint per Wave 16.0 codex-protocol audit. Hardcoded; the
# ChatGPT-internal base URL has no public override.
CHATGPT_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
USER_AGENT = "enchanter-agent/0.6 (codex-responses)"


class ConfigurationError(ValueError):
    """Raised when no ChatGPT subscription credentials can be resolved."""


class ChatGptClient:
    """LlmClient backed by a ChatGPT Plus/Team/Enterprise subscription token.

    Resolution order for the token:

    1. Explicit ``token=`` argument.
    2. ``CHATGPT_SESSION_TOKEN`` env var — JSON blob matching the cache shape,
       or a bare access_token string (treated as expiring in 1 hour with no
       refresh capability).
    3. Cache file at ``~/.enchanter/chatgpt-token.json`` (or APPDATA / ``ENCHANTER_HOME``).
    4. Otherwise ``ConfigurationError``.

    ``complete()`` posts to ``chatgpt.com/backend-api/codex/responses``. See
    the module docstring for v1 limitations.
    """

    auth_mode: str = "chatgpt-subscription"

    def __init__(self, token: ChatGptToken | None = None) -> None:
        resolved = token or self._from_env() or load_cached_token()
        if resolved is None:
            raise ConfigurationError(
                "No ChatGPT subscription credentials found. Provide token=, set "
                "CHATGPT_SESSION_TOKEN, or run the PKCE flow to populate the "
                "cache at ~/.enchanter/chatgpt-token.json."
            )
        self._token: ChatGptToken = resolved

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_env() -> ChatGptToken | None:
        """Parse ``CHATGPT_SESSION_TOKEN`` (JSON blob or bare access_token)."""
        raw = os.environ.get("CHATGPT_SESSION_TOKEN")
        if not raw:
            return None
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
                return ChatGptToken(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    id_token=data.get("id_token"),
                    expires_at=float(data.get("expires_at", time.time() + 3600)),
                    chatgpt_account_id=data.get("chatgpt_account_id"),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None
        # Bare access_token — short-lived assumption, no refresh available.
        return ChatGptToken(
            access_token=raw,
            refresh_token=None,
            id_token=None,
            expires_at=time.time() + 3600,
            chatgpt_account_id=None,
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def has_valid_token(self) -> bool:
        """True iff a token is loaded and not yet expired."""
        return self._token is not None and self._token.expires_at > time.time()

    @property
    def token(self) -> ChatGptToken:
        """Return the loaded token (for diagnostics / Wave 16.3 consumers)."""
        return self._token

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Post the request to the ChatGPT-internal Responses endpoint.

        Refreshes the token if it's near expiry, builds the Responses-API
        body via :mod:`enchanter.llm._codex_responses`, POSTs to
        ``chatgpt.com/backend-api/codex/responses``, and parses the
        non-streaming JSON reply.

        On HTTP 401: attempts one refresh + retry. If the second attempt
        also returns 401, raises :class:`urllib.error.HTTPError` with a
        clear "re-run codex login" message.
        """
        if getattr(req, "stream", False):
            raise NotImplementedError(
                "ChatGptClient streaming pending Wave 17+ "
                "(chatgpt.com/backend-api/codex/responses SSE)."
            )

        # 1. Refresh the token if it's within the skew window. Refresh is a
        #    no-op when the cached token has plenty of headroom.
        try:
            self._token = await refresh_if_needed(self._token)
        except Exception:  # noqa: BLE001 — refresh failures should not mask
            # the actual upstream attempt; we'll surface 401 naturally below.
            pass

        body_dict = build_responses_request(req, stream=False)
        body_bytes = json.dumps(body_dict).encode("utf-8")

        try:
            response_json = await asyncio.to_thread(
                _post_responses, self._token, body_bytes
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
            # One refresh + retry on 401.
            try:
                self._token = await refresh_if_needed(self._token, skew_s=10_000_000)
            except Exception as refresh_exc:  # noqa: BLE001
                raise urllib.error.HTTPError(
                    exc.url, 401,
                    "ChatGPT token rejected and refresh failed — "
                    "re-run `codex login`.",
                    exc.headers, exc.fp,
                ) from refresh_exc
            try:
                response_json = await asyncio.to_thread(
                    _post_responses, self._token, body_bytes
                )
            except urllib.error.HTTPError as retry_exc:
                if retry_exc.code == 401:
                    raise urllib.error.HTTPError(
                        retry_exc.url, 401,
                        "ChatGPT token rejected after refresh — "
                        "re-run `codex login`.",
                        retry_exc.headers, retry_exc.fp,
                    ) from retry_exc
                raise

        return parse_responses_completion(response_json, requested_model=req.model)


def _post_responses(token: ChatGptToken, body: bytes) -> dict[str, Any]:
    """POST the Responses-API body using the supplied token; return parsed JSON.

    Sync helper called via :func:`asyncio.to_thread`. Adds the two non-
    OpenAI-standard headers (``ChatGPT-Account-ID``, ``X-OpenAI-Fedramp``
    when applicable) per Wave 16.0's audit.
    """
    headers = {
        "Authorization": f"Bearer {token.access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if token.chatgpt_account_id:
        headers["ChatGPT-Account-ID"] = token.chatgpt_account_id
    req = urllib.request.Request(  # noqa: S310 — fixed https URL
        CHATGPT_RESPONSES_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))
