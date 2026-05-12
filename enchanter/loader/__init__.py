"""enchanter.loader — declarative engine registry construction from engine.toml manifests.

Public surface:

    find_engine_manifests(root: Path) -> list[Path]
        Glob all engine.toml files under ``root/enchanter/engines/*/``.

    load_engine_registry(root: Path | None = None) -> PluginRegistry
        Parse manifests, import adapters, apply topological ordering,
        return a ``dict[name, PluginAdapter]``.

    EngineManifest      — parsed manifest dataclass
    EngineTopics        — topics sub-table dataclass
    ManifestSchemaError — raised on schema violation
    EngineLoadError     — raised when an adapter cannot be imported
    DependencyCycleError — raised when depends_on forms a cycle
"""

from enchanter.loader.discovery import find_engine_manifests, load_engine_registry
from enchanter.loader.errors import DependencyCycleError, EngineLoadError, ManifestSchemaError
from enchanter.loader.manifest import EngineManifest, EngineTopics

__all__ = [
    "find_engine_manifests",
    "load_engine_registry",
    "EngineManifest",
    "EngineTopics",
    "ManifestSchemaError",
    "EngineLoadError",
    "DependencyCycleError",
]
