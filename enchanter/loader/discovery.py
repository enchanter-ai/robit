"""Engine discovery — glob manifests, import adapters, return a PluginRegistry.

Public API:

    find_engine_manifests(root: Path) -> list[Path]
        Globs ``engines/*/engine.toml`` under *root*.  Returns paths sorted
        alphabetically so the order is deterministic across runs.

    load_engine_registry(root: Path | None = None) -> PluginRegistry
        Parses every manifest found by ``find_engine_manifests``, imports
        each adapter, applies topological ordering by ``depends_on``, and
        returns a ``dict[name, PluginAdapter]``.

        When *root* is None the package root is inferred from this file's
        location (two levels up: enchanter/loader/ → enchanter/ → repo root).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from enchanter.core.plugin import PluginAdapter, PluginRegistry

from .errors import DependencyCycleError, ManifestSchemaError
from .manifest import EngineManifest, parse_manifest
from .runtimes import load_runtime

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Default root resolution
# ──────────────────────────────────────────────────────────────────────────────

def _default_root() -> Path:
    """Return the repo root: two parents up from enchanter/loader/discovery.py."""
    # __file__ is  <root>/enchanter/loader/discovery.py
    return Path(__file__).parent.parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Manifest discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_engine_manifests(root: Path) -> list[Path]:
    """Return sorted list of engine.toml paths under ``root/engines/*/engine.toml``.

    Directories without an engine.toml are silently skipped — this is expected
    for engine directories under active development.
    """
    engines_dir = root / "enchanter" / "engines"
    if not engines_dir.is_dir():
        return []
    return sorted(engines_dir.glob("*/engine.toml"))


# ──────────────────────────────────────────────────────────────────────────────
# Adapter resolution (delegates to the runtime registry)
# ──────────────────────────────────────────────────────────────────────────────

def _import_adapter(manifest: EngineManifest) -> PluginAdapter:
    """Resolve the manifest to its concrete adapter via the runtime registry.

    For runtime='python' this imports ``module.path:attr`` (historical behavior).
    For runtime='sidecar' this returns a SidecarAdapter wrapping a subprocess
    (lazily spawned on first on_phase call).
    """
    return load_runtime(manifest)


# ──────────────────────────────────────────────────────────────────────────────
# Topological sort (Kahn's algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def _topological_sort(manifests: list[EngineManifest]) -> list[EngineManifest]:
    """Return *manifests* ordered so each engine appears after its ``depends_on``.

    Raises:
        DependencyCycleError: A dependency cycle is detected.
        ManifestSchemaError:  A ``depends_on`` entry names an unknown engine.
    """
    name_to_manifest: dict[str, EngineManifest] = {m.name: m for m in manifests}

    # Validate all depends_on references exist.
    for m in manifests:
        for dep in m.depends_on:
            if dep not in name_to_manifest:
                raise ManifestSchemaError(
                    f"engine {m.name!r} depends_on unknown engine {dep!r}",
                    field="depends_on",
                    manifest_path=m.manifest_path,
                )

    # Build in-degree and adjacency (dep → dependents).
    in_degree: dict[str, int] = {m.name: 0 for m in manifests}
    dependents: dict[str, list[str]] = {m.name: [] for m in manifests}

    for m in manifests:
        for dep in m.depends_on:
            in_degree[m.name] += 1
            dependents[dep].append(m.name)

    # Kahn's BFS.
    queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
    # Sort for determinism when multiple nodes have in-degree 0.
    queue = deque(sorted(queue))

    ordered: list[EngineManifest] = []
    while queue:
        node = queue.popleft()
        ordered.append(name_to_manifest[node])
        for child in sorted(dependents[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(manifests):
        # Cycle detected — find the nodes still in the cycle.
        remaining = [n for n, deg in in_degree.items() if deg > 0]
        raise DependencyCycleError(
            f"dependency cycle detected among engines: {sorted(remaining)}",
            cycle=sorted(remaining),
        )

    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_engine_registry(root: Path | None = None) -> PluginRegistry:
    """Parse all engine manifests, import adapters, return a PluginRegistry.

    Args:
        root: Repository root.  When None, inferred from this file's location.

    Returns:
        A ``dict[name, PluginAdapter]`` keyed by each engine's manifest name,
        ordered by dependency (engines with no deps first).

    Raises:
        ManifestSchemaError:  A manifest fails validation.
        EngineLoadError:      An adapter cannot be imported.
        DependencyCycleError: A circular dependency is detected.
    """
    if root is None:
        root = _default_root()

    manifest_paths = find_engine_manifests(root)

    manifests: list[EngineManifest] = []
    for path in manifest_paths:
        manifests.append(parse_manifest(path))

    ordered = _topological_sort(manifests)

    registry: dict[str, PluginAdapter] = {}
    for manifest in ordered:
        adapter = _import_adapter(manifest)
        registry[manifest.name] = adapter

    return registry
