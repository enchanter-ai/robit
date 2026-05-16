"""robit.proxy — canonical request/response substrate and pipeline.

Wave 20 trimmed this package down to the substrate the coding agent's
runtime needs: canonical types, conduct injection, the pipeline that wraps
each LLM turn with engines, the upstream LiteLLM bridge, and the streaming
helpers. The HTTP proxy server, wire-format adapters, and fastpath bypass
were moved out to ``enchanter-ai/beholder`` (the TypeScript MCP-client SDK
+ Rust cockpit) and deleted here.

Public re-exports cover the surfaces the agent CLI and tests pull in.
Pipeline names are re-exported lazily to keep import-time cycles tame.
"""

from __future__ import annotations

from .canonical import CanonicalRequest, CanonicalResponse

# Pipeline types are re-exported best-effort.  Importers that need them
# can also do ``from robit.proxy import pipeline`` directly.
try:  # pragma: no cover — re-export-only branch
    from .pipeline import PipelineOptions, VetoResult  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    PipelineOptions = None  # type: ignore[assignment]
    VetoResult = None  # type: ignore[assignment]

__all__ = [
    "CanonicalRequest",
    "CanonicalResponse",
    "PipelineOptions",
    "VetoResult",
]
