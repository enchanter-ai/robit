"""tests/registry/test_namespace.py — NamespaceRegistry test suite.

8 required tests:
  1. Single server's tools registered; bare name resolves to that server.
  2. Two servers register tools with different names; both resolve cleanly.
  3. Two servers register the same bare name → ToolNameCollisionError.
  4. Qualified "server_id.tool_name" always resolves even on collision.
  5. Re-registration with the same schema is idempotent (no error).
  6. Re-registration with changed schema raises SchemaDigestMismatchError.
  7. unregister(server_id) drops tools; bare name resolves again if unambiguous.
  8. Schema digest is canonical: key-order doesn't affect the digest.
"""

import pytest

from enchanter.registry import (
    NamespaceRegistry,
    SchemaDigestMismatchError,
    ToolDescriptor,
    ToolNameCollisionError,
    compute_schema_digest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(name: str, description: str = "A tool", prop: str = "x") -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {prop: {"type": "string"}}},
        output_schema=None,
    )


# ---------------------------------------------------------------------------
# Test 1 — Single server; bare name resolves
# ---------------------------------------------------------------------------

def test_single_server_bare_resolve():
    reg = NamespaceRegistry()
    tool = make_tool("search")
    reg.register("alpha", [tool])

    server_id, tool_name = reg.resolve("search")
    assert server_id == "alpha"
    assert tool_name == "search"


# ---------------------------------------------------------------------------
# Test 2 — Two servers with different bare names both resolve cleanly
# ---------------------------------------------------------------------------

def test_two_servers_different_names():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.register("beta", [make_tool("embed")])

    sid, name = reg.resolve("search")
    assert sid == "alpha" and name == "search"

    sid, name = reg.resolve("embed")
    assert sid == "beta" and name == "embed"


# ---------------------------------------------------------------------------
# Test 3 — Two servers with the same bare name → ToolNameCollisionError
# ---------------------------------------------------------------------------

def test_bare_name_collision_raises():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.register("beta", [make_tool("search")])

    with pytest.raises(ToolNameCollisionError) as exc_info:
        reg.resolve("search")

    err = exc_info.value
    assert err.name_ == "search"
    assert "alpha" in err.server_ids
    assert "beta" in err.server_ids
    assert len(err.server_ids) == 2


# ---------------------------------------------------------------------------
# Test 4 — Qualified name always resolves even when bare name collides
# ---------------------------------------------------------------------------

def test_qualified_name_resolves_on_collision():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.register("beta", [make_tool("search")])

    # Both qualified forms must work
    sid, name = reg.resolve("alpha.search")
    assert sid == "alpha" and name == "search"

    sid, name = reg.resolve("beta.search")
    assert sid == "beta" and name == "search"


# ---------------------------------------------------------------------------
# Test 5 — Re-registration with the same schema is idempotent
# ---------------------------------------------------------------------------

def test_reregister_same_schema_idempotent():
    reg = NamespaceRegistry()
    tool = make_tool("search")
    reg.register("alpha", [tool])
    # Second registration with identical descriptor must not raise
    reg.register("alpha", [tool])

    sid, name = reg.resolve("search")
    assert sid == "alpha" and name == "search"


# ---------------------------------------------------------------------------
# Test 6 — Re-registration with changed schema raises SchemaDigestMismatchError
# ---------------------------------------------------------------------------

def test_reregister_changed_schema_raises():
    reg = NamespaceRegistry()
    tool_v1 = make_tool("search", description="Original description")
    reg.register("alpha", [tool_v1])

    tool_v2 = make_tool("search", description="Changed description")
    with pytest.raises(SchemaDigestMismatchError) as exc_info:
        reg.register("alpha", [tool_v2])

    err = exc_info.value
    assert err.server_id == "alpha"
    assert err.tool_name == "search"
    assert err.expected != err.got


# ---------------------------------------------------------------------------
# Test 7 — unregister(server_id) drops tools; bare name resolvable again
# ---------------------------------------------------------------------------

def test_unregister_clears_server_and_restores_bare():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.register("beta", [make_tool("search")])

    # Currently ambiguous
    with pytest.raises(ToolNameCollisionError):
        reg.resolve("search")

    # Drop alpha
    reg.unregister("alpha")

    # Now unambiguous
    sid, name = reg.resolve("search")
    assert sid == "beta" and name == "search"

    # alpha's qualified name is gone
    with pytest.raises(KeyError):
        reg.resolve("alpha.search")


# ---------------------------------------------------------------------------
# Test 8 — Digest is canonical: key order doesn't matter
# ---------------------------------------------------------------------------

def test_schema_digest_canonical_key_order():
    schema_a = {"b": 2, "a": 1}
    schema_b = {"a": 1, "b": 2}
    assert compute_schema_digest(schema_a) == compute_schema_digest(schema_b)


# ---------------------------------------------------------------------------
# Bonus: tools_for and all_qualified_names
# ---------------------------------------------------------------------------

def test_tools_for_returns_registered_tools():
    reg = NamespaceRegistry()
    tools = [make_tool("search"), make_tool("embed")]
    reg.register("alpha", tools)

    returned = reg.tools_for("alpha")
    assert {t.name for t in returned} == {"search", "embed"}


def test_all_qualified_names():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.register("beta", [make_tool("embed")])

    names = reg.all_qualified_names()
    assert set(names) == {"alpha.search", "beta.embed"}


def test_unregister_unknown_server_is_noop():
    reg = NamespaceRegistry()
    reg.register("alpha", [make_tool("search")])
    reg.unregister("nonexistent")  # must not raise
    sid, _ = reg.resolve("search")
    assert sid == "alpha"
