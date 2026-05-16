"""inference-substrate engine — advisory post-session evidence accumulation.

Wraps ``robit.inference`` substrate functions as a PluginAdapter that
fires at ``post-session`` (emit buffered artifacts) and ``cross-session``
(reconcile + render briefing).
"""

from .adapter import InferenceSubstrateEngine, adapter

__all__ = ["InferenceSubstrateEngine", "adapter"]
