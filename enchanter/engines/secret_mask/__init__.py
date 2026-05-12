"""secret-mask — port of `hydra.adapter.ts` `scanResultAtPostResponse` +
`maskSecrets` to the Python enchanter-agent runtime.

Required plugin at `post-response`. Scans tool-result payloads against
the SECRET_PATTERNS table and redacts any matches. On match: returns
PluginAck(status="ack") with a derived `secret-mask.matched` event.
On clean result: returns PluginAck(status="ack") with no derived event.
"""

from .adapter import SecretMask, adapter
from .patterns import SECRET_PATTERNS, SecretPattern

__all__ = [
    "SECRET_PATTERNS",
    "SecretMask",
    "SecretPattern",
    "adapter",
]
