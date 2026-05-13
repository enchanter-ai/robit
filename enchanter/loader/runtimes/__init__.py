"""Runtime registry — manifest.runtime → PluginAdapter implementation.

Public surface:
    load_runtime(manifest) -> PluginAdapter
        Dispatches on manifest.runtime:
          - "python"  → load_python_adapter (import module:attr)
          - "sidecar" → SidecarAdapter (subprocess JSON-RPC stdio)

    SidecarAdapter, SidecarBaseError, SidecarCrashError, SidecarTimeoutError,
    SidecarProtocolError, SidecarInitError — re-exported for tests/callers.
"""

from __future__ import annotations

from enchanter.core.plugin import PluginAdapter

from ..errors import EngineLoadError
from ..manifest import EngineManifest
from ._base import (
    SidecarBaseError,
    SidecarCrashError,
    SidecarInitError,
    SidecarProtocolError,
    SidecarTimeoutError,
)
from .python import load_python_adapter
from .sidecar import SidecarAdapter, load_sidecar_adapter


def load_runtime(manifest: EngineManifest) -> PluginAdapter:
    """Resolve ``manifest.runtime`` to a concrete PluginAdapter instance."""
    runtime = manifest.runtime or "python"
    if runtime == "python":
        return load_python_adapter(manifest)
    if runtime == "sidecar":
        return load_sidecar_adapter(manifest)  # type: ignore[return-value]
    raise EngineLoadError(
        f"unknown runtime {runtime!r} for engine {manifest.name!r}",
        engine_name=manifest.name,
    )


__all__ = [
    "load_runtime",
    "load_python_adapter",
    "load_sidecar_adapter",
    "SidecarAdapter",
    "SidecarBaseError",
    "SidecarCrashError",
    "SidecarInitError",
    "SidecarProtocolError",
    "SidecarTimeoutError",
]
