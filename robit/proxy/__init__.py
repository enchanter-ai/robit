"""robit.proxy — wire-format-agnostic LLM proxy layer.

Wave layout:

* Wave 0: :mod:`.canonical` shapes, :mod:`.upstream` LiteLLM bridge,
  :mod:`.conduct` injection helper.
* Wave 1: per-provider adapters (Anthropic, OpenAI, Gemini) that parse
  incoming wire bodies into :class:`.canonical.CanonicalRequest` and render
  :class:`.canonical.CanonicalResponse` / chunk streams back out.
* Wave 2: the HTTP server (:mod:`.server`), routing, and lifecycle wiring;
  the pipeline (:mod:`.pipeline`) that runs canonical requests through the
  engine bus + upstream call.

Public re-exports below cover the surfaces external callers (the CLI, tests,
and downstream packages) need.  Pipeline names are re-exported lazily so
that importing this package does not blow up if the pipeline module hasn't
landed yet (parallel-build race tolerance).
"""

from __future__ import annotations

from .canonical import CanonicalRequest, CanonicalResponse
from .server import ProxyServer, serve_proxy

# Pipeline types are re-exported best-effort.  In parallel-build mode the
# pipeline module may not exist yet — importers that need it should fall
# back to ``from robit.proxy import pipeline`` once it lands.
try:  # pragma: no cover — re-export-only branch
    from .pipeline import PipelineOptions, VetoResult  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    PipelineOptions = None  # type: ignore[assignment]
    VetoResult = None  # type: ignore[assignment]

__all__ = [
    "CanonicalRequest",
    "CanonicalResponse",
    "PipelineOptions",
    "ProxyServer",
    "VetoResult",
    "serve_proxy",
]
