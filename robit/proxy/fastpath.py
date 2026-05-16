"""robit.proxy.fastpath — byte pass-through for trusted callers.

Skips the parse-translate-render cycle (conduct injection + lifecycle
trust-gate + secret-mask post-response) when ALL of the following hold:

  1. Server-side env var ENCHANTER_ALLOW_FASTPATH_BYPASS=1 is set at
     startup. Without it, the entire fast-path code is unreachable.
  2. The caller's API key (SHA-256 of the credential header value) is
     in <state_dir>/fastpath-allowlist.json under key "keys".
  3. The request's wire format maps to a known upstream provider.
  4. The body is a well-formed JSON object, no "tools" field, no
     "stream": true, model in allowed_models (if the allow-list
     specifies a model list).
  5. Body size <= max_body_bytes (default 1 MiB).
  6. Caller has a recognizable auth header (we forward it verbatim).

Auth header mapping (forwarded unchanged to the upstream provider):

| Provider  | Header                          |
|-----------|---------------------------------|
| Anthropic | x-api-key: <key>                |
| OpenAI    | Authorization: Bearer <key>     |
| Gemini    | x-goog-api-key: <key>           |

KNOWN LIMITATIONS — read before enabling.

The conduct injection and trust-gate engines do NOT run on a bypassed
request. That is the point of the fast path. But it means:

  - Pattern vetos (rm -rf, prompt injection) the engines would have
    caught are NOT caught.
  - Secret masking in the response does NOT run; secrets stream through.
  - Cost-ledger emission does NOT happen; X-Enchanter-Cost-Cents header
    is absent on bypassed responses.
  - Body validation is shallow: rejecting `tools` and `stream:true`
    and capping size catches obvious abuse but does NOT inspect prompt
    content for injection patterns, embedded secrets, or model spoofing
    within the model allow-list.

These gaps are real. The env gate + per-key allow-list bound WHO can
request the bypass; they do not change what enforcement is skipped FOR
those who get it. Treat fastpath-allowlist.json as a sensitive file
(0600 perms, not world-readable on shared hosts).

Honest audit: every bypass appends one JSONL line to
<state_dir>/audit/fastpath-bypass.jsonl with timestamp, short key
hash (first 12 chars), upstream provider, model, body size, upstream
status. The bypass response also carries X-Enchanter-FastPath: bypass.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────

_ENV_GATE: str = "ENCHANTER_ALLOW_FASTPATH_BYPASS"
_ENV_STATE_DIR: str = "ENCHANTER_STATE_DIR"
_DEFAULT_MAX_BODY_BYTES: int = 1 * 1024 * 1024  # 1 MiB

_UPSTREAM_URL: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "gemini": "https://generativelanguage.googleapis.com",
}


# ───────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FastPathConfig:
    enabled: bool
    allowed_key_hashes: frozenset[str] = frozenset()
    allowed_models: frozenset[str] | None = None
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES


@dataclass(frozen=True)
class FastPathDecision:
    eligible: bool
    reason: str
    upstream_provider: str | None = None
    model: str | None = None


# ───────────────────────────────────────────────────────────────────────────
# State dir + audit-path resolution (mirrors loader/runtimes/_audit.py)
# ───────────────────────────────────────────────────────────────────────────


def _resolve_state_dir() -> Path:
    env = os.environ.get(_ENV_STATE_DIR)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "state"
    # Platform default
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "enchanter"
    return Path.home() / ".enchanter"


def _allowlist_path() -> Path:
    return _resolve_state_dir() / "fastpath-allowlist.json"


def _audit_path() -> Path:
    return _resolve_state_dir() / "audit" / "fastpath-bypass.jsonl"


# ───────────────────────────────────────────────────────────────────────────
# Config loader
# ───────────────────────────────────────────────────────────────────────────


_CACHED_CONFIG: FastPathConfig | None = None
_CACHED_CONFIG_KEY: tuple[str, str] | None = None


def load_config(*, force_reload: bool = False) -> FastPathConfig:
    """Read env var + allow-list file. Cached on (env_value, allowlist_path)."""
    global _CACHED_CONFIG, _CACHED_CONFIG_KEY
    env_value = os.environ.get(_ENV_GATE, "")
    allow_path = _allowlist_path()
    cache_key = (env_value, str(allow_path))
    if not force_reload and _CACHED_CONFIG is not None and _CACHED_CONFIG_KEY == cache_key:
        return _CACHED_CONFIG

    if env_value != "1":
        cfg = FastPathConfig(enabled=False)
        _CACHED_CONFIG, _CACHED_CONFIG_KEY = cfg, cache_key
        return cfg

    if not allow_path.exists():
        logger.warning(
            "fastpath: ENCHANTER_ALLOW_FASTPATH_BYPASS=1 but %s does not exist; "
            "fast path disabled.",
            allow_path,
        )
        cfg = FastPathConfig(enabled=False)
        _CACHED_CONFIG, _CACHED_CONFIG_KEY = cfg, cache_key
        return cfg

    try:
        raw = json.loads(allow_path.read_text(encoding="utf-8"))
        keys = frozenset(str(k) for k in raw.get("keys", []))
        models_raw = raw.get("models")
        models: frozenset[str] | None = (
            frozenset(str(m) for m in models_raw) if models_raw else None
        )
        max_body = int(raw.get("max_body_bytes", _DEFAULT_MAX_BODY_BYTES))
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("fastpath: failed to parse %s (%s); fast path disabled.", allow_path, exc)
        cfg = FastPathConfig(enabled=False)
        _CACHED_CONFIG, _CACHED_CONFIG_KEY = cfg, cache_key
        return cfg

    cfg = FastPathConfig(
        enabled=True,
        allowed_key_hashes=keys,
        allowed_models=models,
        max_body_bytes=max_body,
    )
    _CACHED_CONFIG, _CACHED_CONFIG_KEY = cfg, cache_key
    return cfg


# ───────────────────────────────────────────────────────────────────────────
# Path → provider + auth extraction
# ───────────────────────────────────────────────────────────────────────────


def _classify_path(method: str, path: str) -> str | None:
    """Return the upstream provider for a path, or None if not routed."""
    if method != "POST":
        return None
    pure = path.split("?", 1)[0]
    if pure == "/v1/messages":
        return "anthropic"
    if pure == "/v1/chat/completions":
        return "openai"
    if pure.startswith("/v1beta/models/") and (":generateContent" in pure or ":streamGenerateContent" in pure):
        return "gemini"
    return None


def _extract_auth_credential(headers: dict[str, str], provider: str) -> str | None:
    """Return the raw credential string for hashing. None if absent."""
    # headers keys are lowercased by the HTTP parser
    if provider == "anthropic":
        return headers.get("x-api-key")
    if provider == "openai":
        bearer = headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            return bearer[7:].strip()
        return None
    if provider == "gemini":
        return headers.get("x-goog-api-key")
    return None


def _hash_key(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


# ───────────────────────────────────────────────────────────────────────────
# Eligibility
# ───────────────────────────────────────────────────────────────────────────


async def evaluate(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    config: FastPathConfig,
) -> FastPathDecision:
    # Step 1: env gate
    if not config.enabled:
        return FastPathDecision(eligible=False, reason="env-gate-off")

    # Step 2: path routing
    provider = _classify_path(method, path)
    if provider is None:
        return FastPathDecision(eligible=False, reason="path-not-routed")

    # Step 3: auth + allow-list
    credential = _extract_auth_credential(headers, provider)
    if not credential:
        return FastPathDecision(eligible=False, reason="no-auth-header", upstream_provider=provider)
    key_hash = _hash_key(credential)
    if key_hash not in config.allowed_key_hashes:
        return FastPathDecision(eligible=False, reason="key-not-allowlisted", upstream_provider=provider)

    # Step 4: body size cap
    if len(body) > config.max_body_bytes:
        return FastPathDecision(eligible=False, reason="body-too-large", upstream_provider=provider)

    # Step 5: cheap JSON sniff
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return FastPathDecision(eligible=False, reason="malformed-body", upstream_provider=provider)
    if not isinstance(parsed, dict):
        return FastPathDecision(eligible=False, reason="body-not-object", upstream_provider=provider)
    if parsed.get("stream") is True:
        return FastPathDecision(eligible=False, reason="stream-true", upstream_provider=provider)
    if "tools" in parsed:
        return FastPathDecision(eligible=False, reason="tools-present", upstream_provider=provider)

    # Step 5b: model
    model = parsed.get("model") if provider != "gemini" else _extract_gemini_model(path)
    if model is None:
        return FastPathDecision(eligible=False, reason="no-model", upstream_provider=provider)
    if not isinstance(model, str):
        return FastPathDecision(eligible=False, reason="invalid-model", upstream_provider=provider)
    if config.allowed_models is not None and model not in config.allowed_models:
        return FastPathDecision(eligible=False, reason="model-not-allowlisted", upstream_provider=provider, model=model)

    # Step 6: eligible
    return FastPathDecision(
        eligible=True,
        reason="authorized-bypass",
        upstream_provider=provider,
        model=model,
    )


def _extract_gemini_model(path: str) -> str | None:
    """Pull model name out of /v1beta/models/{model}:generateContent."""
    pure = path.split("?", 1)[0]
    if not pure.startswith("/v1beta/models/"):
        return None
    rest = pure[len("/v1beta/models/"):]
    if ":" not in rest:
        return None
    return rest.split(":", 1)[0]


# ───────────────────────────────────────────────────────────────────────────
# Passthrough
# ───────────────────────────────────────────────────────────────────────────


async def passthrough(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    *,
    upstream_provider: str,
    model: str,
) -> tuple[int, dict[str, str], bytes]:
    """Forward verbatim. Auth header forwarded unchanged from the inbound request."""
    base = _UPSTREAM_URL[upstream_provider]
    url = base + path

    # Only forward auth + content headers; strip hop-by-hop.
    fwd_headers: dict[str, str] = {"Content-Type": "application/json"}
    if upstream_provider == "anthropic":
        fwd_headers["x-api-key"] = headers.get("x-api-key", "")
        # Anthropic requires anthropic-version
        fwd_headers["anthropic-version"] = headers.get("anthropic-version", "2023-06-01")
    elif upstream_provider == "openai":
        fwd_headers["Authorization"] = headers.get("authorization", "")
    elif upstream_provider == "gemini":
        fwd_headers["x-goog-api-key"] = headers.get("x-goog-api-key", "")

    loop = asyncio.get_running_loop()

    def _do_request() -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(url, data=body, headers=fwd_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers or {}), exc.read() or b""
        except urllib.error.URLError as exc:
            err_body = json.dumps({"error": {"message": f"fastpath upstream error: {exc}"}}).encode()
            return 502, {"Content-Type": "application/json"}, err_body

    status, resp_headers, resp_body = await loop.run_in_executor(None, _do_request)
    return status, resp_headers, resp_body


# ───────────────────────────────────────────────────────────────────────────
# Audit
# ───────────────────────────────────────────────────────────────────────────


_AUDIT_LOCK = asyncio.Lock()


async def record_bypass(
    *,
    upstream_provider: str,
    key_hash_short: str,
    model: str,
    body_size: int,
    upstream_status: int,
) -> None:
    audit_file = _audit_path()
    record = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "kind": "proxy.fastpath.bypass",
        "upstream_provider": upstream_provider,
        "key_hash_short": key_hash_short,
        "model": model,
        "body_size": body_size,
        "upstream_status": upstream_status,
    }

    async with _AUDIT_LOCK:
        loop = asyncio.get_running_loop()

        def _write() -> None:
            try:
                audit_file.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with open(audit_file, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
            except OSError as exc:
                logger.warning("fastpath: audit write failed (%s); record dropped.", exc)

        await loop.run_in_executor(None, _write)


def short_key_hash(credential: str | None) -> str:
    if not credential:
        return "no-auth"
    return _hash_key(credential)[:12]
