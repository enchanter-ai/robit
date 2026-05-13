"""enchanter.proxy.events — pluggable bus-event emitters for the proxy pipeline.

Replaces the hard-coded ``bus.publish(...)`` calls that used to live inline
in :mod:`enchanter.proxy.pipeline`.  Adding a new engine wire-in is now a
matter of dropping a module under this package that exposes a module-level
``emitter`` attribute satisfying the :class:`EventEmitter` protocol — no
edits to ``pipeline.py`` or this ``__init__.py`` are required.

Discovery is intentionally deterministic: emitters are sorted alphabetically
by module name before being returned, so test runs and production logs are
reproducible.  The built-in emitter (``builtin.py``) is therefore always
first; future emitters (``cost_ledger.py``, ``rate_limiter.py``, ...) slot
in by virtue of their filename's alphabetical position.

Logging discipline: emitters that raise during :meth:`emit` MUST NOT crash
the pipeline.  The dispatch path swallows + logs the exception and
continues — emitters are fire-and-forget by contract.  See
:func:`enchanter.proxy.pipeline._run_emitters` for the swallow site.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import List

from ._types import EmitContext, EmitPhase, EventEmitter

_log = logging.getLogger(__name__)


def load_emitters() -> List[EventEmitter]:
    """Discover emitter modules in this package and return them in order.

    Convention: a module is an emitter iff it defines a module-level
    ``emitter`` attribute satisfying the :class:`EventEmitter` protocol.
    The discovery walks ``__path__`` via :func:`pkgutil.iter_modules`, sorts
    module names alphabetically, then imports each one and harvests the
    ``emitter`` attribute if present.

    Modules whose name starts with an underscore (``_types``, ``_internal``,
    ...) are skipped — they are implementation detail, not emitters.

    Returns
    -------
    A deterministic, alphabetised list of :class:`EventEmitter` instances.
    """
    out: list[EventEmitter] = []
    # Sort by module name BEFORE import so the order is stable irrespective
    # of filesystem-iteration order on different OSes.
    module_infos = sorted(
        pkgutil.iter_modules(__path__),
        key=lambda info: info.name,
    )
    for info in module_infos:
        if info.name.startswith("_"):
            continue
        full_name = f"{__name__}.{info.name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            # An emitter module that fails to import is a programming bug,
            # not a runtime failure of the proxy.  Log loudly but keep
            # going — other emitters should still load.
            _log.exception("failed to import emitter module %s", full_name)
            continue
        emitter = getattr(mod, "emitter", None)
        if emitter is None:
            continue
        out.append(emitter)
    return out


__all__ = [
    "EmitContext",
    "EmitPhase",
    "EventEmitter",
    "load_emitters",
]
