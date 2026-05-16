"""tests/transport/test_tls_pin.py — hermetic tests for TlsPinStore.

Test coverage (≥ 4 tests required)
-----------------------------------
1. First-time origin (TOFU) is pinned and accepted
2. Same origin same cert → accepted (no error)
3. Same origin different cert → TlsPinMismatchError raised
4. Pinned store persists across TlsPinStore instances when given a file path

Additional tests:
5. PINNED policy raises TlsPinUnknownError for an unknown origin
6. remove() deletes a pin; subsequent TOFU re-pins successfully
7. PersistentTlsPinStore tolerates a corrupt trailing line in the JSONL file
"""

from __future__ import annotations

import hashlib
import os

import pytest

from robit.transport.tls_pin import (
    InMemoryTlsPinStore,
    PersistentTlsPinStore,
    TlsPinEntry,
    TlsPinMismatchError,
    TlsPinUnknownError,
    compute_cert_fingerprint,
    verify_tls_pin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CERT_A = b"fake-der-bytes-for-cert-A"
_FAKE_CERT_B = b"fake-der-bytes-for-cert-B"
_ORIGIN = "https://mcp.example.com:443"


def _fp(cert_der: bytes) -> str:
    return hashlib.sha256(cert_der).hexdigest()


# ---------------------------------------------------------------------------
# Test 1: First-time origin (TOFU) is pinned and accepted
# ---------------------------------------------------------------------------


def test_tofu_first_time_origin_is_pinned() -> None:
    """TOFU policy: first connection for an unknown origin populates the pin."""
    store = InMemoryTlsPinStore()
    assert store.get(_ORIGIN) is None

    # Should NOT raise — TOFU populates
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "tofu")

    entry = store.get(_ORIGIN)
    assert entry is not None
    assert entry.origin == _ORIGIN
    assert entry.fingerprint == _fp(_FAKE_CERT_A)
    assert entry.source == "tofu"
    assert entry.pinned_at > 0


# ---------------------------------------------------------------------------
# Test 2: Same origin same cert → accepted (no error)
# ---------------------------------------------------------------------------


def test_same_origin_same_cert_accepted() -> None:
    """TOFU policy: presenting the same cert again is accepted silently."""
    store = InMemoryTlsPinStore()
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "tofu")  # first connect — pins
    # Second connect with same cert — must not raise
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "tofu")


# ---------------------------------------------------------------------------
# Test 3: Same origin different cert → TlsPinMismatchError raised
# ---------------------------------------------------------------------------


def test_different_cert_raises_mismatch() -> None:
    """TOFU policy: a different cert for a known origin raises TlsPinMismatchError."""
    store = InMemoryTlsPinStore()
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "tofu")  # pin cert A

    with pytest.raises(TlsPinMismatchError) as exc_info:
        verify_tls_pin(store, _ORIGIN, _FAKE_CERT_B, "tofu")

    err = exc_info.value
    assert err.origin == _ORIGIN
    assert err.expected == _fp(_FAKE_CERT_A)
    assert err.seen == _fp(_FAKE_CERT_B)


# ---------------------------------------------------------------------------
# Test 4: PersistentTlsPinStore persists across instances
# ---------------------------------------------------------------------------


def test_persistent_store_survives_restart(tmp_path) -> None:
    """Pins written by one PersistentTlsPinStore instance are visible to the next."""
    pin_file = tmp_path / "pins.jsonl"

    # First instance: pin cert A
    store1 = PersistentTlsPinStore(str(pin_file))
    verify_tls_pin(store1, _ORIGIN, _FAKE_CERT_A, "tofu")
    assert store1.get(_ORIGIN) is not None

    # Second instance: replays from disk — cert A should still be pinned
    store2 = PersistentTlsPinStore(str(pin_file))
    entry = store2.get(_ORIGIN)
    assert entry is not None, "Pin should have been replayed from disk"
    assert entry.fingerprint == _fp(_FAKE_CERT_A)
    assert entry.source == "tofu"

    # Presenting cert A again must not raise
    verify_tls_pin(store2, _ORIGIN, _FAKE_CERT_A, "tofu")

    # Presenting cert B must raise (mismatch)
    with pytest.raises(TlsPinMismatchError):
        verify_tls_pin(store2, _ORIGIN, _FAKE_CERT_B, "tofu")


# ---------------------------------------------------------------------------
# Test 5: PINNED policy raises TlsPinUnknownError for unknown origin
# ---------------------------------------------------------------------------


def test_pinned_policy_rejects_unknown_origin() -> None:
    """PINNED policy: an origin with no pre-seeded pin raises TlsPinUnknownError."""
    store = InMemoryTlsPinStore()
    # Store is empty — unknown origin

    with pytest.raises(TlsPinUnknownError) as exc_info:
        verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "pinned")

    assert exc_info.value.origin == _ORIGIN
    # PINNED policy must NOT have added a pin (fail closed)
    assert store.get(_ORIGIN) is None


# ---------------------------------------------------------------------------
# Test 6: remove() deletes a pin; TOFU re-pins successfully afterwards
# ---------------------------------------------------------------------------


def test_remove_then_tofu_repins() -> None:
    """After remove(), the next TOFU connection re-pins without error."""
    store = InMemoryTlsPinStore()
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_A, "tofu")
    assert store.get(_ORIGIN) is not None

    store.remove(_ORIGIN)
    assert store.get(_ORIGIN) is None

    # TOFU after remove — should re-pin cert B without error
    verify_tls_pin(store, _ORIGIN, _FAKE_CERT_B, "tofu")
    entry = store.get(_ORIGIN)
    assert entry is not None
    assert entry.fingerprint == _fp(_FAKE_CERT_B)


# ---------------------------------------------------------------------------
# Test 7: PersistentTlsPinStore tolerates a corrupt trailing JSONL line
# ---------------------------------------------------------------------------


def test_persistent_store_tolerates_corrupt_tail(tmp_path) -> None:
    """A corrupt trailing line (mid-write crash simulation) is silently skipped."""
    pin_file = tmp_path / "pins.jsonl"

    # Write a valid entry then append a corrupt line
    store1 = PersistentTlsPinStore(str(pin_file))
    verify_tls_pin(store1, _ORIGIN, _FAKE_CERT_A, "tofu")

    # Simulate a crash that left a partial write
    with open(str(pin_file), "a", encoding="utf-8") as fh:
        fh.write("{corrupt json line without closing brace\n")

    # Replay should succeed and load the valid pin
    store2 = PersistentTlsPinStore(str(pin_file))
    entry = store2.get(_ORIGIN)
    assert entry is not None, "Valid pin before corrupt line should be loaded"
    assert entry.fingerprint == _fp(_FAKE_CERT_A)


# ---------------------------------------------------------------------------
# Bonus: compute_cert_fingerprint is consistent with hashlib.sha256
# ---------------------------------------------------------------------------


def test_compute_cert_fingerprint_matches_hashlib() -> None:
    """compute_cert_fingerprint() output matches a direct hashlib computation."""
    cert_der = b"some-der-bytes"
    expected = hashlib.sha256(cert_der).hexdigest()
    assert compute_cert_fingerprint(cert_der) == expected
    assert all(c in "0123456789abcdef" for c in compute_cert_fingerprint(cert_der))
