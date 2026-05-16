"""enchanter/transport/tls_pin.py — port of tls-pin.ts (FM-6 server-spoofing).

TLS leaf-cert pinning with TOFU and PINNED policies.  Pin store keyed by URL
origin (scheme + host + port).  Each entry stores a SHA-256 hex digest of the
leaf cert's DER bytes.

Two policies
------------
- **TOFU** (trust-on-first-use): first connection for an origin populates the
  pin; later mismatches fail closed.  Default.
- **PINNED** (config-supplied): pins are seeded from config; an unknown origin
  raises :class:`TlsPinUnknownError` (no implicit trust).

Two store backends
------------------
- :class:`InMemoryTlsPinStore` — dict-backed, no persistence.
- :class:`PersistentTlsPinStore` — JSONL append-only log on disk, replayed on
  construction, corrupt-tail tolerant.  Compatible with PersistentReplayStore
  semantics from the TS codebase.

Stdlib only: ``hashlib``, ``json``, ``pathlib``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CertFingerprint = str  # SHA-256 hex lowercase over DER-encoded leaf cert


@dataclass(frozen=True)
class TlsPinEntry:
    """One pinned origin.

    Attributes
    ----------
    origin:
        URL origin — scheme + host + port, e.g. ``"https://mcp.example.com:443"``.
    fingerprint:
        SHA-256 of the leaf cert DER bytes, hex-lowercase.
    pinned_at:
        Unix milliseconds when first pinned.
    source:
        ``"tofu"`` (first-seen) or ``"config"`` (operator-supplied).
    """

    origin: str
    fingerprint: CertFingerprint
    pinned_at: int
    source: Literal["tofu", "config"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TlsPinMismatchError(Exception):
    """Raised when a cert fingerprint does not match the stored pin."""

    def __init__(self, origin: str, expected: CertFingerprint, seen: CertFingerprint) -> None:
        super().__init__(
            f"TLS pin mismatch for {origin}: expected={expected} seen={seen}"
        )
        self.origin = origin
        self.expected = expected
        self.seen = seen


class TlsPinUnknownError(Exception):
    """Raised by PINNED policy when no pin exists for an origin."""

    def __init__(self, origin: str) -> None:
        super().__init__(
            f"TLS pin unknown for {origin}: PINNED policy requires an operator-supplied pin"
        )
        self.origin = origin


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def compute_cert_fingerprint(cert_der: bytes) -> CertFingerprint:
    """Return the SHA-256 hex-lowercase fingerprint of a DER-encoded cert."""
    return hashlib.sha256(cert_der).hexdigest()


def verify_tls_pin(
    store: "TlsPinStore",
    origin: str,
    cert_der: bytes,
    policy: Literal["tofu", "pinned"],
) -> None:
    """Verify *cert_der* against the pin in *store* for *origin*.

    TOFU:   unknown origin → pin and return.
    PINNED: unknown origin → raise :class:`TlsPinUnknownError`.
    Either: fingerprint mismatch → raise :class:`TlsPinMismatchError`.
    """
    seen = compute_cert_fingerprint(cert_der)
    existing = store.get(origin)

    if existing is None:
        if policy == "pinned":
            raise TlsPinUnknownError(origin)
        # TOFU: populate and return.
        store.set(
            TlsPinEntry(
                origin=origin,
                fingerprint=seen,
                pinned_at=int(time.time() * 1000),
                source="tofu",
            )
        )
        return

    if existing.fingerprint != seen:
        raise TlsPinMismatchError(origin, existing.fingerprint, seen)


# ---------------------------------------------------------------------------
# TlsPinStore interface (Protocol-compatible but also an ABC)
# ---------------------------------------------------------------------------


class TlsPinStore:
    """Abstract base — describes the pin store interface.

    Implementations: :class:`InMemoryTlsPinStore`, :class:`PersistentTlsPinStore`.
    """

    def get(self, origin: str) -> TlsPinEntry | None:
        raise NotImplementedError

    def set(self, entry: TlsPinEntry) -> None:
        raise NotImplementedError

    def remove(self, origin: str) -> None:
        raise NotImplementedError

    def list(self) -> list[TlsPinEntry]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InMemoryTlsPinStore
# ---------------------------------------------------------------------------


class InMemoryTlsPinStore(TlsPinStore):
    """Dict-backed pin store with no disk persistence."""

    def __init__(self) -> None:
        self._entries: dict[str, TlsPinEntry] = {}

    # override hooks for the persistent subclass — base is no-op
    def _on_set(self, entry: TlsPinEntry) -> None:  # noqa: B027
        pass

    def _on_remove(self, origin: str) -> None:  # noqa: B027
        pass

    def get(self, origin: str) -> TlsPinEntry | None:
        return self._entries.get(origin)

    def set(self, entry: TlsPinEntry) -> None:
        self._entries[entry.origin] = entry
        self._on_set(entry)

    def remove(self, origin: str) -> None:
        if origin in self._entries:
            del self._entries[origin]
            self._on_remove(origin)

    def list(self) -> list[TlsPinEntry]:
        return list(self._entries.values())


# ---------------------------------------------------------------------------
# PersistentTlsPinStore — JSONL on disk, restart-survival
# ---------------------------------------------------------------------------


class PersistentTlsPinStore(InMemoryTlsPinStore):
    """JSONL append-only pin store that survives process restarts.

    Format: one JSON object per line with ``op`` = ``"set"`` or ``"remove"``.
    Corrupt trailing lines (e.g. from a mid-write crash) are silently skipped.
    The directory is created if it does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._replay_from_disk()

    def _on_set(self, entry: TlsPinEntry) -> None:
        line = json.dumps({"op": "set", "origin": entry.origin, "entry": asdict(entry)})
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _on_remove(self, origin: str) -> None:
        line = json.dumps({"op": "remove", "origin": origin})
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _replay_from_disk(self) -> None:
        if not self._path.exists():
            return
        raw = self._path.read_text(encoding="utf-8")
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # tolerate corrupt tail (crash mid-write)
                continue
            if parsed.get("op") == "set" and "entry" in parsed:
                e = parsed["entry"]
                # direct map mutation — bypasses _on_set to avoid re-appending
                self._entries[e["origin"]] = TlsPinEntry(
                    origin=e["origin"],
                    fingerprint=e["fingerprint"],
                    pinned_at=e["pinned_at"],
                    source=e["source"],
                )
            elif parsed.get("op") == "remove":
                self._entries.pop(parsed["origin"], None)


# ---------------------------------------------------------------------------
# Convenience iterator (audit helper)
# ---------------------------------------------------------------------------


def iter_pins(store: TlsPinStore) -> Iterator[TlsPinEntry]:
    """Yield all pin entries in the store (order unspecified)."""
    yield from store.list()
