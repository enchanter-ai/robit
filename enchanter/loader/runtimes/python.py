"""Python runtime — resolves manifest.adapter into an in-process PluginAdapter.

This is the historical behavior, moved under the runtime registry so the
loader has a single entry point for all runtimes.
"""

from __future__ import annotations

import importlib

from enchanter.core.plugin import PluginAdapter

from ..errors import EngineLoadError
from ..manifest import EngineManifest


def load_python_adapter(manifest: EngineManifest) -> PluginAdapter:
    """Import the module referenced by ``manifest.adapter`` and return the named attribute.

    The adapter string uses Python entry-point notation: ``module.path:attr``.

    Raises:
        EngineLoadError: import fails or the attribute does not exist.
    """
    adapter_str = manifest.adapter
    module_path, _, attr_name = adapter_str.partition(":")

    if not module_path or not attr_name:
        raise EngineLoadError(
            f"adapter string {adapter_str!r} is not valid 'module:attr' notation",
            engine_name=manifest.name,
            adapter_path=adapter_str,
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise EngineLoadError(
            f"cannot import module {module_path!r}: {exc}",
            engine_name=manifest.name,
            adapter_path=adapter_str,
        ) from exc

    if not hasattr(module, attr_name):
        raise EngineLoadError(
            f"module {module_path!r} has no attribute {attr_name!r}",
            engine_name=manifest.name,
            adapter_path=adapter_str,
        )

    adapter = getattr(module, attr_name)

    # Wave 13.3 — surface concurrent_safe from the manifest onto the adapter
    # instance. The engine class itself does not need to declare the
    # attribute; the manifest is the single source of truth. We use a
    # best-effort setattr because some adapters may be frozen dataclass
    # instances or use __slots__ — in that case the dispatcher's
    # ``getattr(..., False)`` fallback keeps them serial.
    try:
        setattr(adapter, "concurrent_safe", bool(manifest.concurrent_safe))
    except (AttributeError, TypeError):
        pass

    return adapter  # type: ignore[return-value]
