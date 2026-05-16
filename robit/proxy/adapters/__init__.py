"""robit.proxy.adapters — wire-format adapters per provider API.

Each adapter translates a provider's native HTTP wire format (request body
+ response body / SSE stream) into and out of the provider-neutral
:mod:`robit.proxy.canonical` dataclasses.  Adapters do not call any
upstream SDK — that is :mod:`robit.proxy.upstream`'s job.

The package is grown additively across waves:

* Wave 1A — AnthropicAdapter (this commit)
* Wave 1B — OpenAIAdapter
* Wave 1C — GeminiAdapter
"""

from .errors import AdapterParseError

# Wave 1 — Anthropic.
from .anthropic import AnthropicAdapter

# Wave 1 — OpenAI.
from .openai import OpenAIAdapter

# Wave 1 — Gemini.
from .gemini import GeminiAdapter

# Wave 16.3 — Codex CLI (OpenAI Responses API).
from .codex import CodexAdapter

__all__ = [
    "AdapterParseError",
    "AnthropicAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
]
