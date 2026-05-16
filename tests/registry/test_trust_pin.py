"""tests/registry/test_trust_pin.py — unit tests for robit.registry.trust_pin.

8 tests covering:
  T1  Digest is deterministic and stable across arg/env/schema reorderings.
  T2  Different inputs produce different digests.
  T3  InMemoryStore: verify on unknown label raises KeyError (caller TOFU-pins).
  T4  InMemoryStore: pin then verify with matching digest succeeds.
  T5  InMemoryStore: pin once, verify with mismatched digest raises
      TrustPinMismatchError with correct expected + got.
  T6  PersistentStore: pin survives across instances pointing at the same file.
  T7  PersistentStore: append-only — repinning a label via a new "pin" op;
      the latest entry wins on reload.
  T8  PersistentStore: a corrupt tail line is tolerated (load skips it, no crash).
  T9  End-to-end: TransportDescriptor → from_transport_descriptor() → pin.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robit.registry.trust_pin import (
    InMemoryTrustPinStore,
    PersistentTrustPinStore,
    TrustPinInputs,
    TrustPinMismatchError,
    compute_trust_pin_digest,
    from_transport_descriptor,
)
from robit.transport.descriptor import TransportDescriptor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryTrustPinStore:
    return InMemoryTrustPinStore()


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trust_pins.jsonl"


# ---------------------------------------------------------------------------
# T1 — Digest is deterministic across arg/env/schema reorderings
# ---------------------------------------------------------------------------


def test_digest_deterministic_across_reorderings() -> None:
    """Sorting args, env_allowlist, schema_digests before hashing means
    reorderings of those tuples produce identical digests."""
    base = TrustPinInputs(
        cmd="python",
        args=("server.py", "--port", "8080"),
        env_allowlist=("HOME", "PATH", "DEBUG"),
        schema_digests=("digest_c", "digest_a", "digest_b"),
    )
    reordered_env = TrustPinInputs(
        cmd="python",
        args=("server.py", "--port", "8080"),
        env_allowlist=("DEBUG", "HOME", "PATH"),  # different order
        schema_digests=("digest_b", "digest_c", "digest_a"),  # different order
    )
    assert compute_trust_pin_digest(base) == compute_trust_pin_digest(reordered_env)

    # Call twice — must be identical (determinism check)
    d1 = compute_trust_pin_digest(base)
    d2 = compute_trust_pin_digest(base)
    assert d1 == d2
    assert len(d1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# T2 — Different inputs produce different digests
# ---------------------------------------------------------------------------


def test_digest_different_inputs_produce_different_digests() -> None:
    a = TrustPinInputs(cmd="python", schema_digests=("digest_a",))
    b = TrustPinInputs(cmd="node", schema_digests=("digest_a",))
    c = TrustPinInputs(cmd="python", schema_digests=("digest_b",))
    d = TrustPinInputs(url="https://example.com/mcp", schema_digests=())

    digests = {compute_trust_pin_digest(inp) for inp in (a, b, c, d)}
    assert len(digests) == 4, "each distinct input must produce a unique digest"


# ---------------------------------------------------------------------------
# T3 — InMemoryStore: verify on unknown label raises KeyError
# ---------------------------------------------------------------------------


def test_inmemory_verify_unknown_label_raises_key_error(
    store: InMemoryTrustPinStore,
) -> None:
    """First verify on an unknown label raises KeyError so the caller can
    decide to TOFU-pin."""
    with pytest.raises(KeyError, match="my-server"):
        store.verify("my-server", "abc123")


# ---------------------------------------------------------------------------
# T4 — InMemoryStore: pin then verify with matching digest succeeds
# ---------------------------------------------------------------------------


def test_inmemory_pin_then_verify_match_succeeds(
    store: InMemoryTrustPinStore,
) -> None:
    inputs = TrustPinInputs(cmd="uvx", args=("mcp-server-demo",), schema_digests=())
    digest = compute_trust_pin_digest(inputs)

    assert not store.has("demo-server")
    store.pin("demo-server", digest)
    assert store.has("demo-server")

    # verify must not raise
    store.verify("demo-server", digest)


# ---------------------------------------------------------------------------
# T5 — InMemoryStore: mismatch raises TrustPinMismatchError with expected + got
# ---------------------------------------------------------------------------


def test_inmemory_mismatch_raises_with_expected_and_got(
    store: InMemoryTrustPinStore,
) -> None:
    original = TrustPinInputs(cmd="python", schema_digests=("d1",))
    tampered = TrustPinInputs(cmd="python", schema_digests=("d2",))  # schema changed

    pinned_digest = compute_trust_pin_digest(original)
    current_digest = compute_trust_pin_digest(tampered)

    store.pin("victim-server", pinned_digest)

    with pytest.raises(TrustPinMismatchError) as exc_info:
        store.verify("victim-server", current_digest)

    err = exc_info.value
    assert err.server_label == "victim-server"
    assert err.expected == pinned_digest
    assert err.got == current_digest
    assert pinned_digest in str(err)
    assert current_digest in str(err)


# ---------------------------------------------------------------------------
# T6 — PersistentStore: pin survives across instances
# ---------------------------------------------------------------------------


def test_persistent_pin_survives_reload(tmp_jsonl: Path) -> None:
    inputs = TrustPinInputs(
        url="https://api.example.com/mcp",
        schema_digests=("schema_hash_1", "schema_hash_2"),
    )
    digest = compute_trust_pin_digest(inputs)

    # First instance: pin
    store1 = PersistentTrustPinStore(tmp_jsonl)
    store1.pin("http-server", digest)
    assert store1.has("http-server")

    # Second instance on the same file: must see the pin
    store2 = PersistentTrustPinStore(tmp_jsonl)
    assert store2.has("http-server")
    store2.verify("http-server", digest)  # must not raise


# ---------------------------------------------------------------------------
# T7 — PersistentStore: repinning via a new "pin" op; latest wins on reload
# ---------------------------------------------------------------------------


def test_persistent_repin_latest_wins_on_reload(tmp_jsonl: Path) -> None:
    digest_v1 = compute_trust_pin_digest(TrustPinInputs(cmd="server-v1", schema_digests=()))
    digest_v2 = compute_trust_pin_digest(TrustPinInputs(cmd="server-v2", schema_digests=()))

    store1 = PersistentTrustPinStore(tmp_jsonl)
    store1.pin("evolving-server", digest_v1)
    store1.pin("evolving-server", digest_v2)  # operator approved update

    # Verify the file has TWO "pin" lines (append-only)
    lines = [json.loads(l) for l in tmp_jsonl.read_text().splitlines() if l.strip()]
    pin_lines = [l for l in lines if l["op"] == "pin" and l["label"] == "evolving-server"]
    assert len(pin_lines) == 2, "append-only: both pin ops must be on disk"

    # Reload: latest pin wins
    store2 = PersistentTrustPinStore(tmp_jsonl)
    assert store2.has("evolving-server")
    store2.verify("evolving-server", digest_v2)  # v2 is accepted
    with pytest.raises(TrustPinMismatchError):
        store2.verify("evolving-server", digest_v1)  # v1 is rejected


# ---------------------------------------------------------------------------
# T8 — PersistentStore: corrupt tail line is tolerated
# ---------------------------------------------------------------------------


def test_persistent_corrupt_tail_line_tolerated(tmp_jsonl: Path) -> None:
    inputs = TrustPinInputs(cmd="stable-server", schema_digests=("d1",))
    digest = compute_trust_pin_digest(inputs)

    # Write a good pin, then append a corrupt line (simulates crash mid-write)
    store1 = PersistentTrustPinStore(tmp_jsonl)
    store1.pin("stable-server", digest)

    with tmp_jsonl.open("a", encoding="utf-8") as fh:
        fh.write('{"op": "pin", "label": "stable-server", "digest": "CORRUPT\n')

    # Loading must not raise and the good pin must still be readable
    store2 = PersistentTrustPinStore(tmp_jsonl)
    assert store2.has("stable-server")
    store2.verify("stable-server", digest)


# ---------------------------------------------------------------------------
# T9 — End-to-end: TransportDescriptor → from_transport_descriptor → pin
# ---------------------------------------------------------------------------


def test_end_to_end_from_transport_descriptor(store: InMemoryTrustPinStore) -> None:
    """A TransportDescriptor is converted via from_transport_descriptor(),
    its digest computed, and the server is pinned and verified."""
    td = TransportDescriptor.for_stdio(
        cmd="uvx",
        args=("mcp-server-example", "--port", "9000"),
        env_allowlist=("HOME", "PATH"),
        binary_digest="abcdef1234567890" * 4,  # 64-char fake SHA-256
    )

    schema_digests: tuple[str, ...] = ("tool_hash_1", "tool_hash_2")
    inputs = from_transport_descriptor(td, schema_digests=schema_digests)

    # Verify all fields are correctly mapped
    assert inputs.cmd == "uvx"
    assert inputs.args == ("mcp-server-example", "--port", "9000")
    assert inputs.env_allowlist == ("HOME", "PATH")
    assert inputs.binary_digest == "abcdef1234567890" * 4
    assert inputs.url is None
    assert inputs.schema_digests == schema_digests

    digest = compute_trust_pin_digest(inputs)

    # TOFU-pin flow: not pinned yet → KeyError → caller pins → verify succeeds
    assert not store.has("example-server")
    with pytest.raises(KeyError):
        store.verify("example-server", digest)

    store.pin("example-server", digest)
    store.verify("example-server", digest)  # must not raise

    # Tampered schema → mismatch
    tampered_inputs = from_transport_descriptor(td, schema_digests=("tool_hash_evil",))
    tampered_digest = compute_trust_pin_digest(tampered_inputs)
    with pytest.raises(TrustPinMismatchError):
        store.verify("example-server", tampered_digest)
