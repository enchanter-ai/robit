"""robit.llm._chatgpt_auth — PKCE OAuth helpers for ChatGPT subscription auth.

Wave 16.2 v2 — auth + token storage layer. The `complete()` upstream call
is deferred to Wave 16.3 (Responses API adapter for chatgpt.com/backend-api/codex/responses).

Constants verified against Wave 16.0's codex-protocol audit:
* AUTH_ISSUER = https://auth.openai.com
* CLIENT_ID   = app_EMoamEEZ73f0CkXaXp7hrann
* REDIRECT_URI = http://localhost:1455/auth/callback (loopback only)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants (verified against Codex CLI source via Wave 16.0 audit)
# ---------------------------------------------------------------------------

AUTH_ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid email profile offline_access"
LOOPBACK_PORT = 1455


def _default_token_path() -> Path:
    """Return the default token cache path, honoring ENCHANTER_HOME + APPDATA."""
    override = os.environ.get("ENCHANTER_HOME")
    if override:
        return Path(override) / "chatgpt-token.json"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "enchanter" / "chatgpt-token.json"
    return Path.home() / ".enchanter" / "chatgpt-token.json"


TOKEN_PATH = _default_token_path()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Base class for OAuth errors."""


class AuthDeniedError(AuthError):
    """The user denied consent, or the IdP returned an error."""


class AuthTimeoutError(AuthError):
    """The PKCE redirect did not arrive within max_wait_s."""


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatGptToken:
    """JWT triplet from ChatGPT OAuth, matching Codex CLI's auth.json shape."""

    access_token: str
    refresh_token: str | None
    id_token: str | None
    expires_at: float
    chatgpt_account_id: str | None = None


# ---------------------------------------------------------------------------
# PKCE primitives (RFC 7636)
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` per RFC 7636 (SHA-256, base64url).

    Verifier: 43-128 chars URL-safe (we use ~86 chars via token_urlsafe(64)).
    Challenge: base64url(sha256(verifier)) with padding stripped.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _b64url_decode_segment(seg: str) -> bytes:
    """Decode a base64url JWT segment, restoring missing padding."""
    pad = (-len(seg)) % 4
    return base64.urlsafe_b64decode(seg + ("=" * pad))


def _extract_account_id(id_token: str | None) -> str | None:
    """Pull ``chatgpt_account_id`` from JWT id_token claims (no signature check)."""
    if not id_token:
        return None
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        claims = json.loads(_b64url_decode_segment(parts[1]))
    except (ValueError, json.JSONDecodeError):
        return None
    # Codex CLI reads chatgpt_account_id from the namespaced claim bag.
    auth_claims = claims.get("https://api.openai.com/auth", {}) or {}
    acct = auth_claims.get("chatgpt_account_id") or claims.get("chatgpt_account_id")
    return acct if isinstance(acct, str) else None


# ---------------------------------------------------------------------------
# Token cache (JSON file)
# ---------------------------------------------------------------------------


def load_cached_token(path: Path | None = None) -> ChatGptToken | None:
    """Read the token cache. Return ``None`` if missing or malformed."""
    p = path or _default_token_path()
    try:
        raw = p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        data = json.loads(raw)
        return ChatGptToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=data.get("id_token"),
            expires_at=float(data["expires_at"]),
            chatgpt_account_id=data.get("chatgpt_account_id"),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_token(token: ChatGptToken, path: Path | None = None) -> None:
    """Atomically persist the token cache. Creates parent dirs as needed."""
    p = path or _default_token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(token), indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Token endpoint exchange
# ---------------------------------------------------------------------------


def _post_token(payload: dict[str, str], issuer: str = AUTH_ISSUER) -> dict[str, Any]:
    """POST x-www-form-urlencoded to ``{issuer}/oauth/token`` and parse JSON."""
    body = urllib.parse.urlencode(payload).encode("ascii")
    req = urllib.request.Request(  # noqa: S310 — fixed https issuer
        f"{issuer}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _token_from_response(data: dict[str, Any]) -> ChatGptToken:
    """Build a ChatGptToken from a `/oauth/token` response body."""
    access = data["access_token"]
    refresh = data.get("refresh_token")
    id_token = data.get("id_token")
    expires_in = float(data.get("expires_in", 3600))
    return ChatGptToken(
        access_token=access,
        refresh_token=refresh,
        id_token=id_token,
        expires_at=time.time() + expires_in,
        chatgpt_account_id=_extract_account_id(id_token),
    )


async def refresh_if_needed(
    token: ChatGptToken,
    *,
    client_id: str = CLIENT_ID,
    issuer: str = AUTH_ISSUER,
    skew_s: float = 60.0,
) -> ChatGptToken:
    """Return a fresh token if ``expires_at`` is within ``skew_s`` of now."""
    if token.expires_at > time.time() + skew_s:
        return token
    if not token.refresh_token:
        raise AuthError("Token expired and no refresh_token available.")

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": token.refresh_token,
        "client_id": client_id,
        "scope": SCOPE,
    }
    data = await asyncio.to_thread(_post_token, payload, issuer)
    refreshed = _token_from_response(data)
    # Some IdPs omit refresh_token on refresh — keep the old one.
    if refreshed.refresh_token is None and token.refresh_token:
        refreshed = ChatGptToken(
            access_token=refreshed.access_token,
            refresh_token=token.refresh_token,
            id_token=refreshed.id_token or token.id_token,
            expires_at=refreshed.expires_at,
            chatgpt_account_id=refreshed.chatgpt_account_id or token.chatgpt_account_id,
        )
    return refreshed


# ---------------------------------------------------------------------------
# Localhost PKCE callback receiver
# ---------------------------------------------------------------------------


class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot handler that captures ``?code=`` / ``?error=`` on the redirect."""

    result: dict[str, str] = {}  # populated by subclass per-server instance

    def do_GET(self) -> None:  # noqa: N802 — stdlib API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.__class__.result.update(params)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>enchanter: ChatGPT auth complete</h2>"
            b"<p>You can close this tab.</p></body></html>"
        )

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ARG002
        return  # Silence default stderr logging.


def _build_authorize_url(
    code_challenge: str,
    state: str,
    *,
    client_id: str,
    redirect_uri: str,
    issuer: str,
) -> str:
    """Construct the /oauth/authorize URL with PKCE params."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{issuer}/oauth/authorize?" + urllib.parse.urlencode(params)


async def run_pkce_flow(  # pragma: no cover — live flow; covered by unit-mocked helpers
    *,
    client_id: str = CLIENT_ID,
    redirect_uri: str = REDIRECT_URI,
    issuer: str = AUTH_ISSUER,
    port: int = LOOPBACK_PORT,
    max_wait_s: float = 300.0,
) -> ChatGptToken:
    """Run the full PKCE redirect flow and return a fresh ChatGptToken.

    Spawns a loopback HTTP server on ``port`` (127.0.0.1 only), opens the
    browser, waits for the ``?code=`` callback, exchanges for tokens.
    """
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    # Per-flow handler subclass — isolates `result` between concurrent flows.
    class _Handler(_CallbackHandler):
        result: dict[str, str] = {}

    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        webbrowser.open(
            _build_authorize_url(
                challenge,
                state,
                client_id=client_id,
                redirect_uri=redirect_uri,
                issuer=issuer,
            )
        )
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            if _Handler.result:
                break
            await asyncio.sleep(0.1)
        else:
            raise AuthTimeoutError(
                f"No PKCE callback received within {max_wait_s:.0f}s."
            )
    finally:
        server.shutdown()
        server.server_close()

    result = _Handler.result
    if "error" in result:
        raise AuthDeniedError(result.get("error_description") or result["error"])
    if result.get("state") != state:
        raise AuthError("PKCE state mismatch — possible CSRF.")
    code = result.get("code")
    if not code:
        raise AuthError("No authorization code in callback.")

    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    data = await asyncio.to_thread(_post_token, payload, issuer)
    return _token_from_response(data)
