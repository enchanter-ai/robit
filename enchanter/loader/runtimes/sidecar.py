"""Sidecar runtime — subprocess-backed PluginAdapter speaking JSON-RPC over stdio.

Protocol (newline-framed JSON, one message per line):

  Outgoing requests (parent → sidecar):
    {"jsonrpc":"2.0","id":<int>,"method":"initialize"}
    {"jsonrpc":"2.0","id":<int>,"method":"on_phase","params":{"event":<event-dict>,"context":<ctx-dict>}}

  Outgoing notifications (no reply expected):
    {"jsonrpc":"2.0","method":"shutdown"}

  Incoming responses (sidecar → parent):
    {"jsonrpc":"2.0","id":<int>,"result":<result>}                # success
    {"jsonrpc":"2.0","id":<int>,"error":{"code":<int>,"message":<str>}}  # failure

  initialize.result shape (REQUIRED):
    {
      "name":         "<str>",
      "phases":       ["<phase>", ...],
      "required":     <bool>,
      "budget_tier":  "always" | "med-or-higher" | "high-only",
      "topics":       {"subscribes": ["<topic>", ...], "emits": ["<topic>", ...]}
    }

  on_phase.result shape (mirrors PluginAck):
    {
      "status":         "ack" | "veto" | "error",
      "reason":         "<str-or-null>",
      "derived_events": [<event-dict>, ...],   # optional, default []
      "degraded":       <bool>                  # optional, default false
    }

Lifecycle:
- Lazy spawn on first on_phase (or explicit warm_up()).
- Persistent process per engine (one subprocess per SidecarAdapter instance).
- Hard 5s default timeout per on_phase request → kill + veto-ack.
- Up to 3 auto-restarts on crash; after the 3rd failure the adapter is marked
  failed and future calls short-circuit to a veto.
- 8 MiB per-message body cap matches enchanter.transport.stdio.
- Graceful shutdown sends a `shutdown` notification, waits 2s, then SIGKILL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import asdict
from typing import Any

from enchanter.core.context import RequestContext
from enchanter.core.events import EnchantedEvent, PluginAck
from enchanter.core.plugin import PluginTopics

from ._base import (
    SidecarCrashError,
    SidecarInitError,
    SidecarProtocolError,
    SidecarTimeoutError,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — mirror transport/stdio.py for cross-codebase consistency
# ──────────────────────────────────────────────────────────────────────────────

PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024  # 8 MiB cap
STDERR_RING_CAPACITY: int = 50                     # last 50 lines for crash reports
DEFAULT_TIMEOUT_S: float = 5.0
SHUTDOWN_GRACE_S: float = 2.0
MAX_RESTARTS: int = 3

# Source names reserved for the harness itself. A sidecar MUST NEVER be allowed
# to set raw_event.source to any of these — doing so is an authority forgery
# (the bus recorder & downstream consumers key off source). Even if a sidecar's
# manifest declared name happens to collide with one of these (it shouldn't —
# the loader rejects such manifests upstream — but defence-in-depth), validate
# the event source against this set first and reject before per-adapter match.
_RESERVED_SOURCES: frozenset[str] = frozenset({
    "orchestrator",
    "pipeline",
    "proxy-pipeline",
    "builtin",
    "framework",
})

# Required keys on every derived event dict. Missing any → malformed-event.
_REQUIRED_EVENT_KEYS: tuple[str, ...] = (
    "id",
    "correlation_id",
    "session_id",
    "phase",
    "topic",
    "source",
    "ts",
    "payload",
)


# record_rejection is imported lazily inside the validator (Agent B owns the
# `_audit` module). The module attribute `record_rejection` below is the
# patch point — tests `unittest.mock.patch` it directly. At runtime the helper
# does the import; if the audit module isn't on disk yet, rejection is still
# enforced (the event is dropped) — only the audit-log side-effect is skipped.
record_rejection = None  # set by lazy import; tests patch this name.


# ──────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _event_to_dict(event: EnchantedEvent) -> dict[str, Any]:
    """Serialize an EnchantedEvent to a JSON-safe dict via dataclasses.asdict."""
    return asdict(event)


def _dict_to_event(d: dict[str, Any]) -> EnchantedEvent:
    """Deserialize a dict to an EnchantedEvent. Caller must ensure shape."""
    return EnchantedEvent(
        id=d["id"],
        correlation_id=d["correlation_id"],
        session_id=d["session_id"],
        phase=d["phase"],
        topic=d["topic"],
        source=d["source"],
        budget_tier=d["budget_tier"],
        ts=d["ts"],
        payload=d.get("payload", {}),
    )


def _context_to_dict(ctx: RequestContext) -> dict[str, Any]:
    """Serialize the subset of RequestContext a sidecar can act on.

    We intentionally omit `degraded_findings` (parent-side mutation surface)
    and `started_ms`/`deadline_ms`/`sampling_depth` which the sidecar cannot
    affect. If a future sidecar needs them, extend this dict — never expose
    the whole dataclass.
    """
    return {
        "correlation_id": ctx.correlation_id,
        "session_id": ctx.session_id,
        "phase": ctx.phase,
        "budget_tier": ctx.budget_tier,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SidecarAdapter
# ──────────────────────────────────────────────────────────────────────────────

class SidecarAdapter:
    """PluginAdapter backed by a long-lived subprocess speaking JSON-RPC over stdio.

    Conforms structurally to PluginAdapter Protocol: exposes `name`, `phases`,
    `required`, `topics`, `budget_tier` after the initialize handshake.
    """

    def __init__(
        self,
        command: str,
        args: tuple[str, ...] = (),
        env_allowlist: tuple[str, ...] = ("PATH",),
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._command: str = command
        self._args: tuple[str, ...] = tuple(args)
        self._env_allowlist: tuple[str, ...] = tuple(env_allowlist) or ("PATH",)
        self._timeout_s: float = float(timeout_s)

        # Set by initialize() handshake — required by PluginAdapter Protocol.
        self.name: str = ""
        self.phases: tuple[str, ...] = ()
        self.required: bool = False
        self.topics: PluginTopics = PluginTopics(subscribes=(), emits=())
        self.budget_tier: str = "always"
        # Wave 13.3 — set by load_sidecar_adapter() from the manifest, NOT by
        # the subprocess. The manifest is the single source of truth: a
        # subprocess does not get to override the parallelism contract its
        # own manifest declared.
        self.concurrent_safe: bool = False

        # Subprocess state.
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id: int = 1
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_ring: deque[str] = deque(maxlen=STDERR_RING_CAPACITY)
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._io_lock: asyncio.Lock = asyncio.Lock()
        self._restart_count: int = 0
        self._failed: bool = False  # true after MAX_RESTARTS crashes; no further spawns
        self._initialized: bool = False
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public PluginAdapter Protocol surface
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Eagerly spawn + initialize the subprocess. Optional; on_phase will lazy-init."""
        await self._ensure_initialized()

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        """Send an on_phase request to the sidecar; coerce any error into a veto ack."""
        if self._closed:
            return PluginAck(status="veto", reason="sidecar:closed", degraded=True)
        if self._failed:
            return PluginAck(status="veto", reason="sidecar:failed", degraded=True)

        try:
            await self._ensure_initialized()
        except SidecarBaseError_alias as exc:
            return PluginAck(status="veto", reason=f"sidecar:init:{exc}", degraded=True)

        params = {
            "event": _event_to_dict(event),
            "context": _context_to_dict(ctx),
        }
        try:
            result = await self._request("on_phase", params)
        except SidecarTimeoutError:
            # Subprocess already killed inside _request on timeout.
            return PluginAck(status="veto", reason="sidecar:timeout", degraded=True)
        except SidecarCrashError as exc:
            return PluginAck(status="veto", reason=f"sidecar:crash:{exc}", degraded=True)
        except SidecarProtocolError as exc:
            return PluginAck(status="veto", reason=f"sidecar:protocol:{exc}", degraded=True)

        # Validate derived events BEFORE _parse_ack rebuilds them as
        # EnchantedEvent instances. Rejection drops only the offending event;
        # the sidecar's verdict (ack/veto/error) still flows through. The
        # audit log is the only record of rejected events.
        if isinstance(result, dict):
            raw_derived = result.get("derived_events", []) or []
            if isinstance(raw_derived, list):
                kept: list[dict[str, Any]] = []
                for raw_event in raw_derived:
                    if not isinstance(raw_event, dict):
                        # Pre-existing _parse_ack veto path. Leave as-is so the
                        # ack as a whole is rejected; do not silently filter
                        # non-dict shapes (that's a protocol-level fault, not
                        # a forgery — different escalation).
                        kept.append(raw_event)
                        continue
                    accepted = await self._validate_derived_event(raw_event)
                    if accepted is not None:
                        kept.append(accepted)
                result = dict(result)
                result["derived_events"] = kept

        return _parse_ack(result)

    async def _validate_derived_event(self, raw_event: dict) -> dict | None:
        """Validate one raw derived-event dict from the sidecar.

        Returns the dict on accept. Returns None and records a rejection on
        reject (via the audit log helper). Rules:

        1. SOURCE ALLOWLIST: raw_event["source"] MUST equal self.name (the
           manifest-declared name set during initialize handshake). Reserved
           values that MUST NEVER be accepted from a sidecar:
           {"orchestrator","pipeline","proxy-pipeline","builtin","framework"}.
        2. TOPIC ALLOWLIST: raw_event["topic"] MUST be in self.topics.emits.
        3. PHASE CONSISTENCY: raw_event["phase"] MUST be in self.phases.
        4. Missing required fields → reject with reason="malformed-event".

        On rejection, the event is dropped silently from the ack (caller never
        sees it); the audit log is the only record.
        """
        # Rule 4: required fields. Check first so we know we can index the
        # dict safely for rules 1–3.
        for key in _REQUIRED_EVENT_KEYS:
            if key not in raw_event:
                await self._record_rejection(
                    raw_event, "malformed-event",
                )
                return None

        # Rule 1: source allowlist. Reserved names rejected unconditionally,
        # then the per-adapter manifest-name match.
        source = raw_event["source"]
        if not isinstance(source, str) or source in _RESERVED_SOURCES:
            await self._record_rejection(raw_event, "source-forgery")
            return None
        if source != self.name:
            await self._record_rejection(raw_event, "source-forgery")
            return None

        # Rule 2: topic allowlist.
        topic = raw_event["topic"]
        if not isinstance(topic, str) or topic not in self.topics.emits:
            await self._record_rejection(raw_event, "undeclared-topic")
            return None

        # Rule 3: phase consistency.
        phase = raw_event["phase"]
        if not isinstance(phase, str) or phase not in self.phases:
            await self._record_rejection(raw_event, "phase-out-of-scope")
            return None

        return raw_event

    async def _record_rejection(self, raw_event: dict, reason: str) -> None:
        """Lazy-import the audit-log helper and forward the rejection.

        If Agent B's `_audit` module is not yet on disk, the import fails
        silently — the event is still dropped (enforcement is local), only the
        audit-log side-effect is skipped. Tests patch
        `enchanter.loader.runtimes.sidecar.record_rejection` to capture calls.
        """
        global record_rejection  # noqa: PLW0603
        fn = record_rejection
        if fn is None:
            try:
                from ._audit import record_rejection as _imported  # type: ignore[import-not-found]
                fn = _imported
                record_rejection = _imported
            except Exception:  # noqa: BLE001
                # Audit module unavailable — drop the event but do not raise.
                return
        try:
            await fn(
                adapter_name=self.name,
                rejection_reason=reason,
                raw_event=raw_event,
                expected={
                    "name": self.name,
                    "phases": list(self.phases),
                    "emits": list(self.topics.emits),
                },
            )
        except Exception:  # noqa: BLE001
            # Audit-log failures must never abort the validation path.
            logger.warning(
                "SidecarAdapter: record_rejection raised for adapter=%s reason=%s",
                self.name,
                reason,
            )

    async def shutdown(self) -> None:
        """Send a `shutdown` notification, wait briefly, then SIGKILL if needed."""
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is None or proc.returncode is not None:
            await self._cleanup_stderr_task()
            return

        # Send shutdown as a fire-and-forget notification.
        try:
            payload = (
                json.dumps({"jsonrpc": "2.0", "method": "shutdown"}, ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.write(payload)
                try:
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=SHUTDOWN_GRACE_S)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.error("SidecarAdapter: subprocess could not be killed (pid=%s)", proc.pid)

        await self._cleanup_stderr_task()

    # ------------------------------------------------------------------
    # Internal — subprocess lifecycle
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Spawn + handshake on first call; idempotent after success."""
        if self._initialized and self._proc is not None and self._proc.returncode is None:
            return
        async with self._spawn_lock:
            if self._initialized and self._proc is not None and self._proc.returncode is None:
                return
            await self._spawn()
            await self._do_initialize_handshake()
            self._initialized = True

    async def _spawn(self) -> None:
        env = self._build_env()
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # Start stderr drain.
        self._stderr_ring.clear()
        self._stderr_task = asyncio.get_event_loop().create_task(
            self._drain_stderr(), name="sidecar-stderr-drain"
        )

    def _build_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in self._env_allowlist:
            val = os.environ.get(key)
            if val is not None:
                out[key] = val
        if "PATH" not in out:
            path = os.environ.get("PATH")
            if path is not None:
                out["PATH"] = path
        # Windows-friendly minimums for child to run at all.
        for k in ("PATHEXT", "SYSTEMROOT", "TEMP", "TMP"):
            if k not in out:
                v = os.environ.get(k)
                if v is not None:
                    out[k] = v
        return out

    async def _do_initialize_handshake(self) -> None:
        """Send initialize, mirror returned attributes onto self."""
        try:
            result = await self._request("initialize", None)
        except (SidecarTimeoutError, SidecarCrashError, SidecarProtocolError) as exc:
            raise SidecarInitError(
                f"initialize handshake failed: {exc}",
                stderr_tail=tuple(self._stderr_ring),
            ) from exc

        if not isinstance(result, dict):
            raise SidecarInitError(
                f"initialize result must be a dict, got {type(result).__name__}",
                stderr_tail=tuple(self._stderr_ring),
            )

        try:
            name = result["name"]
            phases = result["phases"]
            required = result["required"]
            budget_tier = result["budget_tier"]
            topics = result["topics"]
        except KeyError as exc:
            raise SidecarInitError(
                f"initialize result missing required key: {exc}",
                stderr_tail=tuple(self._stderr_ring),
            ) from exc

        if not isinstance(name, str) or not isinstance(budget_tier, str):
            raise SidecarInitError("initialize: name/budget_tier must be strings")
        if not isinstance(phases, list) or not all(isinstance(p, str) for p in phases):
            raise SidecarInitError("initialize: phases must be list[str]")
        if not isinstance(required, bool):
            raise SidecarInitError("initialize: required must be bool")
        if (
            not isinstance(topics, dict)
            or not isinstance(topics.get("subscribes"), list)
            or not isinstance(topics.get("emits"), list)
        ):
            raise SidecarInitError(
                "initialize: topics must be {subscribes:[str], emits:[str]}"
            )

        self.name = name
        self.phases = tuple(phases)
        self.required = required
        self.budget_tier = budget_tier
        self.topics = PluginTopics(
            subscribes=tuple(topics["subscribes"]),
            emits=tuple(topics["emits"]),
        )

    # ------------------------------------------------------------------
    # Internal — request/response
    # ------------------------------------------------------------------

    async def _request(self, method: str, params: Any) -> Any:
        """Send one JSON-RPC request, await one response, return its `result`.

        Serializes IO via _io_lock so on_phase calls don't interleave on the
        same subprocess.
        """
        proc = self._proc
        if proc is None or proc.returncode is not None:
            raise SidecarCrashError(
                "subprocess is not running",
                stderr_tail=tuple(self._stderr_ring),
            )

        req_id = self._next_id
        self._next_id += 1
        envelope: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            envelope["params"] = params

        line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        if "\n" in line:
            raise SidecarProtocolError("outgoing message contained embedded newline")
        payload = (line + "\n").encode("utf-8")
        if len(payload) > PER_MESSAGE_BODY_MAX_BYTES:
            raise SidecarProtocolError(
                f"outgoing message exceeds {PER_MESSAGE_BODY_MAX_BYTES} bytes"
            )

        async with self._io_lock:
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(payload)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._handle_crash()
                raise SidecarCrashError(
                    f"write failed: {exc}",
                    stderr_tail=tuple(self._stderr_ring),
                ) from exc

            try:
                line_bytes = await asyncio.wait_for(
                    self._read_line_capped(proc.stdout),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError as exc:
                # Hard kill + restart bookkeeping.
                self._handle_crash(timeout=True)
                raise SidecarTimeoutError(
                    f"no response within {self._timeout_s}s",
                    stderr_tail=tuple(self._stderr_ring),
                ) from exc

            if line_bytes is None:
                self._handle_crash()
                raise SidecarCrashError(
                    "subprocess closed stdout (EOF)",
                    stderr_tail=tuple(self._stderr_ring),
                )

        # Parse response outside io_lock — we already have the line.
        try:
            msg = json.loads(line_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SidecarProtocolError(
                f"malformed response: {exc}",
                stderr_tail=tuple(self._stderr_ring),
            ) from exc

        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            raise SidecarProtocolError(
                "response missing jsonrpc:'2.0' envelope",
                stderr_tail=tuple(self._stderr_ring),
            )
        if msg.get("id") != req_id:
            raise SidecarProtocolError(
                f"response id mismatch: expected {req_id}, got {msg.get('id')!r}",
                stderr_tail=tuple(self._stderr_ring),
            )
        if "error" in msg:
            err = msg["error"]
            raise SidecarProtocolError(
                f"sidecar returned error: {err}",
                stderr_tail=tuple(self._stderr_ring),
            )
        if "result" not in msg:
            raise SidecarProtocolError(
                "response missing both result and error fields",
                stderr_tail=tuple(self._stderr_ring),
            )
        return msg["result"]

    def _handle_crash(self, *, timeout: bool = False) -> None:
        """Kill the subprocess, bump restart counter, possibly mark failed."""
        proc = self._proc
        if proc is not None:
            try:
                if proc.returncode is None:
                    proc.kill()
            except ProcessLookupError:
                pass
        self._initialized = False
        self._proc = None
        self._restart_count += 1
        if self._restart_count >= MAX_RESTARTS:
            self._failed = True

    async def _read_line_capped(self, stream: asyncio.StreamReader) -> bytes | None:
        """Read up to ``\\n``, enforcing PER_MESSAGE_BODY_MAX_BYTES."""
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = await stream.read(65536)
            except Exception:  # noqa: BLE001
                return None
            if not chunk:
                return None
            nl_pos = chunk.find(b"\n")
            if nl_pos != -1:
                before = chunk[:nl_pos]
                remainder = chunk[nl_pos + 1:]
                total += len(before)
                if total > PER_MESSAGE_BODY_MAX_BYTES:
                    raise SidecarProtocolError(
                        f"incoming message exceeds {PER_MESSAGE_BODY_MAX_BYTES} bytes",
                        stderr_tail=tuple(self._stderr_ring),
                    )
                chunks.append(before)
                if remainder:
                    stream.feed_data(remainder)
                line = b"".join(chunks)
                if not line:
                    return await self._read_line_capped(stream)
                return line
            total += len(chunk)
            if total > PER_MESSAGE_BODY_MAX_BYTES:
                raise SidecarProtocolError(
                    f"incoming message exceeds {PER_MESSAGE_BODY_MAX_BYTES} bytes",
                    stderr_tail=tuple(self._stderr_ring),
                )
            chunks.append(chunk)

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            async for raw in proc.stderr:
                text = raw.decode("utf-8", errors="replace").rstrip("\n")
                self._stderr_ring.append(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("SidecarAdapter: stderr drain ended: %r", exc)

    async def _cleanup_stderr_task(self) -> None:
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        self._stderr_task = None


# ──────────────────────────────────────────────────────────────────────────────
# Forward-ref alias for the catch in on_phase (avoids circular import noise)
# ──────────────────────────────────────────────────────────────────────────────

from ._base import SidecarBaseError as SidecarBaseError_alias  # noqa: E402


def _parse_ack(result: Any) -> PluginAck:
    """Translate an on_phase result dict into a PluginAck. Tolerant of missing keys."""
    if not isinstance(result, dict):
        return PluginAck(status="veto", reason="sidecar:malformed-ack", degraded=True)

    status = result.get("status", "ack")
    if status not in ("ack", "veto", "error"):
        return PluginAck(status="veto", reason=f"sidecar:bad-status:{status!r}", degraded=True)

    reason_raw = result.get("reason")
    reason: str | None = reason_raw if isinstance(reason_raw, str) else None

    derived_raw = result.get("derived_events", []) or []
    if not isinstance(derived_raw, list):
        return PluginAck(status="veto", reason="sidecar:bad-derived_events", degraded=True)

    derived: list[EnchantedEvent] = []
    for d in derived_raw:
        if not isinstance(d, dict):
            return PluginAck(status="veto", reason="sidecar:bad-derived_event-shape", degraded=True)
        try:
            derived.append(_dict_to_event(d))
        except KeyError as exc:
            return PluginAck(
                status="veto",
                reason=f"sidecar:derived_event-missing:{exc}",
                degraded=True,
            )

    degraded = bool(result.get("degraded", False))
    return PluginAck(status=status, reason=reason, derived_events=derived, degraded=degraded)


def load_sidecar_adapter(manifest: Any) -> SidecarAdapter:
    """Build a SidecarAdapter from a manifest. The handshake is deferred to
    the first on_phase call (or an explicit warm_up()) so loader-time errors
    are decoupled from runtime errors — a missing binary surfaces at request
    time as a veto, not at registry construction.
    """
    adapter = SidecarAdapter(
        command=manifest.command,
        args=manifest.args,
        env_allowlist=manifest.env_allowlist or ("PATH",),
    )
    # Wave 13.3 — surface from manifest, not from the subprocess handshake.
    adapter.concurrent_safe = bool(getattr(manifest, "concurrent_safe", False))
    return adapter
