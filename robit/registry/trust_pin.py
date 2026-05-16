"""enchanter/registry/trust_pin.py — port of trust-pin.ts v0.3.1.

Implements the Trust-on-First-Use (TOFU) + mismatch-veto layer that closes
FM-10 (MCPoison) at the server-identity level.

On first connect the server's identity digest is pinned (TOFU).  On every
subsequent connect the digest is recomputed and compared; a mismatch raises
TrustPinMismatchError and the agent refuses to connect.

The identity digest is SHA-256 over the canonical JSON of:
    cmd, args (sorted), binary_digest, env_allowlist (sorted),
    url, schema_digests (sorted)

None/missing fields are omitted entirely from the canonical payload so that
stdio and HTTP descriptors produce digests that are stable across transport
types.

Persistence: append-only JSONL store; each line is one operation record.
The latest "pin" op for a label wins on load.  Corrupt tail lines are
tolerated (skipped, no crash).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# TrustPinInputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustPinInputs:
    """All fields that contribute to a server's trust-pin digest.

    Fields
    ------
    cmd:
        stdio command (None for HTTP transports).
    args:
        stdio positional arguments (order preserved in digest).
    binary_digest:
        SHA-256 of the server executable (stdio only; None when unavailable).
    env_allowlist:
        Env-var *names* forwarded to the child process.  Values are
        runtime-bound and intentionally excluded.
    url:
        Streamable-HTTP endpoint URL (None for stdio transports).
    schema_digests:
        Per-tool schema digests from the namespace layer, sorted before hashing.
    """

    cmd: str | None = None
    args: tuple[str, ...] = ()
    binary_digest: str | None = None
    env_allowlist: tuple[str, ...] = ()
    url: str | None = None
    schema_digests: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Canonical JSON + digest
# ---------------------------------------------------------------------------


def _canonicalize(inputs: TrustPinInputs) -> str:
    """Build the canonical JSON payload for digest computation.

    Field order is fixed: cmd, args, binary_digest, env_allowlist, url,
    schema_digests.  Missing (None) fields are omitted entirely so the digest
    is stable across transport types.

    args order is preserved (argv semantics).
    env_allowlist + schema_digests are sorted (set semantics).
    """
    ordered: dict = {}
    if inputs.cmd is not None:
        ordered["cmd"] = inputs.cmd
    if inputs.args:
        ordered["args"] = list(inputs.args)  # order preserved
    if inputs.binary_digest is not None:
        ordered["binaryDigest"] = inputs.binary_digest
    if inputs.env_allowlist:
        ordered["envAllowlist"] = sorted(inputs.env_allowlist)
    if inputs.url is not None:
        ordered["url"] = inputs.url
    # schema_digests always present (may be empty list)
    ordered["schemaDigests"] = sorted(inputs.schema_digests)
    return json.dumps(ordered, sort_keys=False, separators=(",", ":"))


def compute_trust_pin_digest(inputs: TrustPinInputs) -> str:
    """Return the SHA-256 hex digest of the canonical server identity.

    Sorting args, env_allowlist, and schema_digests before hashing means
    reorderings of those tuples produce the same digest.
    """
    payload = _canonicalize(inputs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# TrustPinMismatchError
# ---------------------------------------------------------------------------


class TrustPinMismatchError(Exception):
    """Raised when a server's current identity does not match its pinned digest.

    This is a security veto: the agent refuses to connect until an operator
    explicitly re-consents by calling pin() with the new digest.
    """

    def __init__(self, server_label: str, expected: str, got: str) -> None:
        self.server_label = server_label
        self.expected = expected
        self.got = got
        super().__init__(
            f"trust pin mismatch for server {server_label!r}: "
            f"pinned={expected} current={got} — operator re-consent required"
        )


# ---------------------------------------------------------------------------
# InMemoryTrustPinStore
# ---------------------------------------------------------------------------


class InMemoryTrustPinStore:
    """In-process trust-pin store backed by a plain dict.

    Interface
    ---------
    pin(label, digest)         — record label → digest (TOFU or update)
    verify(label, digest)      — raise TrustPinMismatchError on mismatch;
                                 raise KeyError if label has never been pinned
    has(label) -> bool         — True if label is currently pinned
    clear()                    — wipe all pins (test helper)
    """

    def __init__(self) -> None:
        self._pins: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core ops — subclass hook for persistence layer
    # ------------------------------------------------------------------

    def _on_pin(self, label: str, digest: str) -> None:
        """Called after pin(); override in subclass to persist."""

    def pin(self, label: str, digest: str) -> None:
        """Store *digest* as the accepted identity for *label*."""
        self._pins[label] = digest
        self._on_pin(label, digest)

    def verify(self, label: str, digest: str) -> None:
        """Verify *digest* against the pinned digest for *label*.

        Raises
        ------
        KeyError
            If *label* has never been pinned.  The caller decides whether to
            TOFU-pin on first use.
        TrustPinMismatchError
            If *label* is pinned and the digest does not match.
        """
        if label not in self._pins:
            raise KeyError(label)
        expected = self._pins[label]
        if digest != expected:
            raise TrustPinMismatchError(label, expected, digest)

    def has(self, label: str) -> bool:
        """Return True if *label* has a stored pin."""
        return label in self._pins

    def clear(self) -> None:
        """Remove all pinned entries (test helper)."""
        self._pins.clear()


# ---------------------------------------------------------------------------
# PersistentTrustPinStore — JSONL-backed, restart-survival
# ---------------------------------------------------------------------------
#
# Storage format: one JSON object per line.
# Pin op:   {"op": "pin",   "label": <str>, "digest": <str>, "ts": <int>}
# Unpin op: {"op": "unpin", "label": <str>,                  "ts": <int>}
#
# On load the lines are replayed in order; the last "pin" or "unpin" for a
# label wins.  Corrupt / truncated lines at the tail are skipped silently.
#
# "unpin" is reserved for future operator-consent flows.  Only "pin" is
# written by this module currently; "unpin" is honoured on load.


class PersistentTrustPinStore(InMemoryTrustPinStore):
    """JSONL-backed trust-pin store.

    Parameters
    ----------
    path:
        File path for the JSONL log.  Parent directories are created on
        construction.  The file is created on first write.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Replay the JSONL file into the in-memory dict."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate corrupt / truncated tail line.
                continue
            op = record.get("op")
            label = record.get("label")
            if not isinstance(label, str):
                continue
            if op == "pin":
                digest = record.get("digest")
                if isinstance(digest, str):
                    # Direct dict write — bypass _on_pin to avoid re-appending.
                    self._pins[label] = digest
            elif op == "unpin":
                self._pins.pop(label, None)

    def _append(self, record: dict) -> None:
        """Append *record* as a single JSON line to the JSONL file."""
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def _on_pin(self, label: str, digest: str) -> None:
        self._append({"op": "pin", "label": label, "digest": digest, "ts": int(time.time())})


# ---------------------------------------------------------------------------
# from_transport_descriptor — bridge to the transport layer
# ---------------------------------------------------------------------------


def from_transport_descriptor(
    descriptor,  # robit.transport.descriptor.TransportDescriptor
    schema_digests: tuple[str, ...] = (),
) -> TrustPinInputs:
    """Convert a TransportDescriptor into TrustPinInputs.

    Parameters
    ----------
    descriptor:
        A ``TransportDescriptor`` from ``robit.transport.descriptor``.
    schema_digests:
        Per-tool schema digests from the namespace layer (empty tuple if not
        yet available).
    """
    return TrustPinInputs(
        cmd=descriptor.cmd,
        args=descriptor.args,
        binary_digest=descriptor.binary_digest,
        env_allowlist=descriptor.env_allowlist,
        url=descriptor.url,
        schema_digests=schema_digests,
    )
