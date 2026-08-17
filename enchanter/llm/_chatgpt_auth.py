"""enchanter.llm._chatgpt_auth — OAuth device-code helper for ChatGPT subscriptions.

This module provides the scaffolding for authenticating against a ChatGPT
Plus / Team / Enterprise subscription using the OAuth device-code flow. The
end-to-end flow against the real OpenAI servers is **unverified in this
wave** (Wave 16.2 of 0.6.0). The endpoint constants and default client id
are placeholders pending Wave 16.0's research. The HTTP call sites are
isolated so tests can mock them.

Public surface:

* ``ChatGptToken``        — frozen dataclass for an issued OAuth token.
* ``DeviceCodeFlow``      — frozen dataclass for the device-code init result.
* ``DEFAULT_TOKEN_PATH``  — per-OS cache location.
* ``start_device_code_flow``  — initiate flow with OpenAI.
* ``poll_for_token``      — block until user authorizes or expiry.
* ``load_cached_token`` / ``save_token`` — disk persistence.
* ``refresh_if_needed``   — proactively exchange refresh_token.
* ``AuthError`` and subclasses ``AuthTimeoutError``, ``AuthDeniedError``.

All HTTP traffic uses stdlib ``urllib.request`` wrapped in
``asyncio.to_thread`` for the async surface — consistent with the rest of
the codebase, no new dependency surface.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Endpoint constants — UNVERIFIED placeholders
# ---------------------------------------------------------------------------

# TODO: verify against real OpenAI OAuth docs (Wave 16.0 research)
_DEVICE_CODE_URL = "https://auth.openai.com/oauth/device/code"
# TODO: verify against real OpenAI OAuth docs (Wave 16.0 research)
_TOKEN_URL = "https://auth.openai.com/oauth/token"
# TODO: verify against real OpenAI OAuth docs (Wave 16.0 research)
_DEFAULT_CLIENT_ID = "app_chatgpt_subscription_placeholder"

# Refresh proactively when the token has fewer than this many seconds left.
_REFRESH_LEAD_S: float = 60.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Base exception for ChatGPT OAuth failures."""


class AuthTimeoutError(AuthError):
    """User did not authorize within the device-code flow's expiry window."""


class AuthDeniedError(AuthError):
    """User explicitly denied the device-code authorization request."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatGptToken:
    """An issued ChatGPT subscription access token."""

    access_token: str
    refresh_token: str | None
    expires_at: float
    token_type: str = "Bearer"

    def is_expired(self, now: float | None = None, lead_s: float = _REFRESH_LEAD_S) -> bool:
        """True if the token expires within ``lead_s`` seconds."""
        return (now if now is not None else time.time()) + lead_s >= self.expires_at


@dataclass(frozen=True)
class DeviceCodeFlow:
    """Result of initiating an OAuth device-code flow."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


# ---------------------------------------------------------------------------
# Token cache location (per-OS)
# ---------------------------------------------------------------------------

def _default_token_path() -> Path:
    """Resolve the platform-appropriate token cache path.

    * POSIX: ``~/.enchanter/chatgpt-token.json``
    * Windows: ``%APPDATA%/enchanter/chatgpt-token.json``
      (falls back to ``~/.enchanter/chatgpt-token.json`` if APPDATA unset)
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "enchanter" / "chatgpt-token.json"
    return Path.home() / ".enchanter" / "chatgpt-token.json"


DEFAULT_TOKEN_PATH: Path = _default_token_path()


# ---------------------------------------------------------------------------
# Stdlib HTTP helper (isolated so tests can patch urllib.request.urlopen)
# ---------------------------------------------------------------------------

def _http_post_form(url: str, fields: dict[str, str], timeout: float = 30.0) -> dict[str, Any]:
    """POST ``fields`` as ``application/x-www-form-urlencoded`` and return JSON.

    Raises ``AuthError`` on transport failures or non-JSON bodies.
    HTTP errors (4xx/5xx) are NOT raised here — the JSON body (which OAuth
    servers use to communicate flow-control errors like ``authorization_pending``)
    is returned to the caller. Truly malformed responses raise.
    """
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
    except urllib.error.HTTPError as e:
        # OAuth servers return structured JSON even on 4xx (e.g., authorization_pending).
        # Read the error body and pass it back to the caller for flow-control parsing.
        try:
            body = e.read()
        except Exception as inner:
            raise AuthError(f"HTTP {e.code} from {url} with unreadable body") from inner
    except urllib.error.URLError as e:
        raise AuthError(f"Network error contacting {url}: {e}") from e

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AuthError(f"Malformed JSON response from {url}") from e


# ---------------------------------------------------------------------------
# Device-code flow
# ---------------------------------------------------------------------------

async def start_device_code_flow(client_id: str = _DEFAULT_CLIENT_ID) -> DeviceCodeFlow:
    """Initiate the OAuth device-code flow.

    Calls the OpenAI device-code endpoint with ``client_id`` and returns the
    flow parameters the user needs to complete authorization in a browser.

    Raises ``AuthError`` on transport / protocol failure.
    """
    def _call() -> dict[str, Any]:
        return _http_post_form(
            _DEVICE_CODE_URL,
            {"client_id": client_id, "scope": "openai.api"},
        )

    payload = await asyncio.to_thread(_call)

    if "error" in payload:
        raise AuthError(f"device-code init failed: {payload.get('error')!r}")

    try:
        return DeviceCodeFlow(
            device_code=payload["device_code"],
            user_code=payload["user_code"],
            verification_uri=payload.get("verification_uri") or payload["verification_url"],
            expires_in=int(payload["expires_in"]),
            interval=int(payload.get("interval", 5)),
        )
    except KeyError as e:
        raise AuthError(f"device-code response missing field {e!s}") from e


async def poll_for_token(
    flow: DeviceCodeFlow,
    client_id: str = _DEFAULT_CLIENT_ID,
    *,
    max_wait_s: float = 300.0,
) -> ChatGptToken:
    """Poll OpenAI's token endpoint until the user authorizes or it expires.

    Honors the server's ``interval`` between polls. Recognized OAuth flow
    errors (``authorization_pending``, ``slow_down``) are handled transparently.
    ``access_denied`` raises ``AuthDeniedError``; expiry raises
    ``AuthTimeoutError``.

    The ``max_wait_s`` cap protects against runaway poll loops even if the
    server-reported ``expires_in`` is unreasonably large.
    """
    deadline = time.time() + min(float(flow.expires_in), float(max_wait_s))
    interval = max(1, int(flow.interval))

    while True:
        if time.time() >= deadline:
            raise AuthTimeoutError("Device-code authorization expired before user completed it.")

        def _call() -> dict[str, Any]:
            return _http_post_form(
                _TOKEN_URL,
                {
                    "client_id": client_id,
                    "device_code": flow.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )

        payload = await asyncio.to_thread(_call)

        err = payload.get("error")
        if err is None:
            # Success — token issued.
            try:
                expires_in = float(payload.get("expires_in", 3600))
                return ChatGptToken(
                    access_token=payload["access_token"],
                    refresh_token=payload.get("refresh_token"),
                    expires_at=time.time() + expires_in,
                    token_type=payload.get("token_type", "Bearer"),
                )
            except KeyError as e:
                raise AuthError(f"token response missing field {e!s}") from e

        if err == "authorization_pending":
            await asyncio.sleep(interval)
            continue
        if err == "slow_down":
            interval += 5
            await asyncio.sleep(interval)
            continue
        if err == "access_denied":
            raise AuthDeniedError("User denied the device-code authorization.")
        if err == "expired_token":
            raise AuthTimeoutError("Device-code expired before user completed authorization.")

        # Unknown error — bubble up with context.
        raise AuthError(f"OAuth token poll failed: {err!r}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_cached_token(path: Path = DEFAULT_TOKEN_PATH) -> ChatGptToken | None:
    """Load a cached token from disk.

    Returns ``None`` if the file is missing OR malformed (e.g., JSON parse
    error, missing required fields). Never raises for an unreadable cache —
    the caller is expected to fall through to fresh auth.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    try:
        return ChatGptToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=float(data["expires_at"]),
            token_type=data.get("token_type", "Bearer"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_token(token: ChatGptToken, path: Path = DEFAULT_TOKEN_PATH) -> None:
    """Persist a token to disk with 0600 permissions on POSIX.

    Creates parent directories as needed. On Windows the chmod is a no-op
    (Windows ACLs are not addressed in this wave).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(asdict(token), sort_keys=True)
    path.write_text(serialized, encoding="utf-8")
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Best-effort — if we can't chmod, the file is still written.
            pass


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

async def refresh_if_needed(
    token: ChatGptToken,
    client_id: str = _DEFAULT_CLIENT_ID,
) -> ChatGptToken:
    """Exchange ``token.refresh_token`` for a new access token if near expiry.

    Returns the original ``token`` unchanged when:
    * the token is not yet within the refresh lead window, OR
    * there is no ``refresh_token`` to exchange.

    Raises ``AuthError`` if the refresh exchange itself fails.
    """
    if not token.is_expired():
        return token
    if not token.refresh_token:
        return token

    def _call() -> dict[str, Any]:
        return _http_post_form(
            _TOKEN_URL,
            {
                "client_id": client_id,
                "refresh_token": token.refresh_token or "",
                "grant_type": "refresh_token",
            },
        )

    payload = await asyncio.to_thread(_call)

    if "error" in payload:
        raise AuthError(f"refresh_token exchange failed: {payload.get('error')!r}")

    try:
        expires_in = float(payload.get("expires_in", 3600))
        return ChatGptToken(
            access_token=payload["access_token"],
            # Some providers rotate refresh tokens; others keep the same one.
            refresh_token=payload.get("refresh_token") or token.refresh_token,
            expires_at=time.time() + expires_in,
            token_type=payload.get("token_type", token.token_type),
        )
    except KeyError as e:
        raise AuthError(f"refresh response missing field {e!s}") from e


__all__ = [
    "AuthDeniedError",
    "AuthError",
    "AuthTimeoutError",
    "ChatGptToken",
    "DEFAULT_TOKEN_PATH",
    "DeviceCodeFlow",
    "load_cached_token",
    "poll_for_token",
    "refresh_if_needed",
    "save_token",
    "start_device_code_flow",
]
