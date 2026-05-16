"""Tests for robit.loader.manifest — schema parsing and strict validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from robit.loader.manifest import EngineManifest, EngineTopics, parse_manifest
from robit.loader.errors import ManifestSchemaError


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_toml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temp engine.toml and return its path."""
    p = tmp_path / "engine.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_VALID_TOML = """\
    name = "test-engine"
    description = "A test engine for unit testing."
    version = "1.2.3"
    phases = ["trust-gate", "post-response"]
    required = true
    budget_tier = "always"
    adapter = "my.module.path:adapter"
    depends_on = ["other-engine"]
    tags = ["security", "gate"]

    [topics]
    subscribes = ["mcp.tool.call.requested", "lifecycle.trust-gate"]
    emits = ["test-engine.veto"]
"""


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Parsing a valid manifest produces the expected dataclass with all fields
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_valid_manifest_all_fields(tmp_path: Path) -> None:
    p = _write_toml(tmp_path, _VALID_TOML)
    m = parse_manifest(p)

    assert isinstance(m, EngineManifest)
    assert m.name == "test-engine"
    assert m.description == "A test engine for unit testing."
    assert m.version == "1.2.3"
    assert m.phases == ("trust-gate", "post-response")
    assert m.required is True
    assert m.budget_tier == "always"
    assert m.adapter == "my.module.path:adapter"
    assert m.depends_on == ("other-engine",)
    assert m.tags == ("security", "gate")
    assert isinstance(m.topics, EngineTopics)
    assert m.topics.subscribes == ("mcp.tool.call.requested", "lifecycle.trust-gate")
    assert m.topics.emits == ("test-engine.veto",)
    assert str(p) == m.manifest_path


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Manifest missing a required field raises ManifestSchemaError with field name
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing_field", [
    "name",
    "description",
    "version",
    "phases",
    "required",
    "budget_tier",
    "adapter",
])
def test_missing_required_field_raises_with_field_name(
    tmp_path: Path, missing_field: str
) -> None:
    # Build a valid TOML dict and remove the field.
    lines = textwrap.dedent(_VALID_TOML).splitlines()
    filtered = [l for l in lines if not l.startswith(missing_field + " ")]
    content = "\n".join(filtered)

    p = tmp_path / "engine.toml"
    p.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)

    err = exc_info.value
    assert err.field is not None
    # The error mentions the missing field somewhere.
    assert missing_field in str(err) or (err.field and missing_field in err.field)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Manifest with unknown extra fields is rejected (strict mode)
# ──────────────────────────────────────────────────────────────────────────────

def test_unknown_extra_field_rejected(tmp_path: Path) -> None:
    content = _VALID_TOML + '\nphaes = ["trust-gate"]  # typo of "phases"\n'
    p = _write_toml(tmp_path, content)

    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)

    err = exc_info.value
    assert "phaes" in str(err)


def test_unknown_topics_subfield_rejected(tmp_path: Path) -> None:
    content = _VALID_TOML + "\n[topics]\nunknown_key = [\"x\"]\n"
    # Overwrite — use a fresh TOML that puts the unknown key inside [topics].
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
        extra_key = ["z"]
    """)
    p = _write_toml(tmp_path, toml)

    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)

    assert "extra_key" in str(exc_info.value)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: find_engine_manifests globs and returns expected count
# ──────────────────────────────────────────────────────────────────────────────

def test_find_engine_manifests_count(tmp_path: Path) -> None:
    from robit.loader.discovery import find_engine_manifests

    engines_dir = tmp_path / "robit" / "engines"
    for name in ("engine_a", "engine_b", "engine_c"):
        d = engines_dir / name
        d.mkdir(parents=True)
        (d / "engine.toml").write_text(
            textwrap.dedent(f"""\
                name = "{name}"
                description = "test"
                version = "1.0.0"
                phases = ["trust-gate"]
                required = false
                budget_tier = "always"
                adapter = "mod:attr"

                [topics]
                subscribes = ["x"]
                emits = ["y"]
            """),
            encoding="utf-8",
        )

    # Extra dir without engine.toml — should be silently skipped.
    no_toml = engines_dir / "no_manifest_here"
    no_toml.mkdir()

    found = find_engine_manifests(tmp_path)
    assert len(found) == 3
    names = {p.parent.name for p in found}
    assert names == {"engine_a", "engine_b", "engine_c"}


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Topological order — engine with depends_on loads after its dependency
# ──────────────────────────────────────────────────────────────────────────────

def test_topological_order_respects_depends_on(tmp_path: Path) -> None:
    from robit.loader.discovery import find_engine_manifests, load_engine_registry

    engines_dir = tmp_path / "robit" / "engines"

    def _make_engine(name: str, depends_on: list[str] | None = None) -> None:
        d = engines_dir / name
        d.mkdir(parents=True)
        dep_line = (
            f'depends_on = {depends_on!r}\n' if depends_on else ""
        )
        (d / "engine.toml").write_text(
            textwrap.dedent(f"""\
                name = "{name}"
                description = "test"
                version = "1.0.0"
                phases = ["trust-gate"]
                required = false
                budget_tier = "always"
                adapter = "robit.engines.destructive_op_gate.adapter:adapter"
                {dep_line}
                [topics]
                subscribes = ["x"]
                emits = ["y"]
            """),
            encoding="utf-8",
        )

    # "child" depends on "parent"
    _make_engine("parent")
    _make_engine("child", depends_on=["parent"])

    registry = load_engine_registry(tmp_path)
    keys = list(registry.keys())
    assert keys.index("parent") < keys.index("child"), (
        f"'parent' should appear before 'child', got order: {keys}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Dependency cycle raises DependencyCycleError
# ──────────────────────────────────────────────────────────────────────────────

def test_dependency_cycle_raises(tmp_path: Path) -> None:
    from robit.loader.discovery import load_engine_registry
    from robit.loader.errors import DependencyCycleError

    engines_dir = tmp_path / "robit" / "engines"

    def _make_engine(name: str, depends_on: list[str]) -> None:
        d = engines_dir / name
        d.mkdir(parents=True)
        (d / "engine.toml").write_text(
            textwrap.dedent(f"""\
                name = "{name}"
                description = "test"
                version = "1.0.0"
                phases = ["trust-gate"]
                required = false
                budget_tier = "always"
                adapter = "robit.engines.destructive_op_gate.adapter:adapter"
                depends_on = {depends_on!r}

                [topics]
                subscribes = ["x"]
                emits = ["y"]
            """),
            encoding="utf-8",
        )

    # A → B → A cycle
    _make_engine("engine_a", depends_on=["engine_b"])
    _make_engine("engine_b", depends_on=["engine_a"])

    with pytest.raises(DependencyCycleError) as exc_info:
        load_engine_registry(tmp_path)

    err = exc_info.value
    assert len(err.cycle) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Wave 13.1.5 — runtime field & sidecar manifest variants
# ──────────────────────────────────────────────────────────────────────────────


_BASE_HEADER = """\
    name = "test-engine"
    description = "desc"
    version = "1.0.0"
    phases = ["trust-gate"]
    required = false
    budget_tier = "always"

    [topics]
    subscribes = ["x"]
    emits = ["y"]
"""


def test_runtime_absent_defaults_to_python(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.runtime == "python"
    assert m.adapter == "mod:attr"
    assert m.command == ""
    assert m.args == ()


def test_runtime_python_with_adapter_valid(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "python"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.runtime == "python"
    assert m.adapter == "mod:attr"


def test_runtime_python_without_adapter_raises(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "python"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "adapter" in str(exc_info.value)


def test_runtime_sidecar_with_command_valid(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "rust-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "sidecar"
        command = "./bin/engine"
        args = ["--mode", "stdio"]
        env_allowlist = ["PATH", "HOME"]

        [adapter_metadata]
        language = "rust"
        version = "1.2.3"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.runtime == "sidecar"
    assert m.command == "./bin/engine"
    assert m.args == ("--mode", "stdio")
    assert m.env_allowlist == ("PATH", "HOME")
    assert dict(m.adapter_metadata) == {"language": "rust", "version": "1.2.3"}
    assert m.adapter == ""


def test_runtime_sidecar_defaults_env_allowlist_and_args(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "rust-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "sidecar"
        command = "./bin/engine"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.args == ()
    assert m.env_allowlist == ("PATH",)


def test_runtime_sidecar_without_command_raises(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "rust-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "sidecar"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "command" in str(exc_info.value)


def test_runtime_sidecar_with_adapter_forbidden(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "rust-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "sidecar"
        command = "./bin/engine"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "adapter" in str(exc_info.value)


def test_runtime_python_with_command_forbidden(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "python"
        adapter = "mod:attr"
        command = "./bin/engine"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "command" in str(exc_info.value)


# ──────────────────────────────────────────────────────────────────────────────
# Wave 13.3 — concurrent_safe field
# ──────────────────────────────────────────────────────────────────────────────


def test_concurrent_safe_absent_defaults_to_false(tmp_path: Path) -> None:
    """Manifest without ``concurrent_safe`` defaults to False (serial-only)."""
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.concurrent_safe is False


def test_concurrent_safe_true_parses_cleanly(tmp_path: Path) -> None:
    """Manifest with ``concurrent_safe = true`` parses to True."""
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        adapter = "mod:attr"
        concurrent_safe = true

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    m = parse_manifest(p)
    assert m.concurrent_safe is True


def test_concurrent_safe_string_rejected(tmp_path: Path) -> None:
    """Manifest with ``concurrent_safe = "yes"`` raises ManifestSchemaError."""
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        adapter = "mod:attr"
        concurrent_safe = "yes"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "concurrent_safe" in str(exc_info.value) or exc_info.value.field == "concurrent_safe"


def test_unknown_runtime_value_raises(tmp_path: Path) -> None:
    toml = textwrap.dedent("""\
        name = "test-engine"
        description = "desc"
        version = "1.0.0"
        phases = ["trust-gate"]
        required = false
        budget_tier = "always"
        runtime = "wasm"
        adapter = "mod:attr"

        [topics]
        subscribes = ["x"]
        emits = ["y"]
    """)
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ManifestSchemaError) as exc_info:
        parse_manifest(p)
    assert "wasm" in str(exc_info.value) or "runtime" in str(exc_info.value)
