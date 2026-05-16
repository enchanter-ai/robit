"""enchanter/registry/digest.py — canonical-JSON SHA-256 digest for tool schemas.

The canonical form sorts all dict keys recursively, uses no whitespace, and
``separators=(",", ":")``.  This matches the deterministic serialisation the
TypeScript side achieves with ``JSON.stringify(schema, Object.keys(schema).sort())``,
extended to handle nested objects (JSON.stringify's replacer only sorts one level).
"""

from __future__ import annotations

import hashlib
import json


def _canonical(obj: object) -> object:
    """Recursively sort dict keys so serialisation is key-order-independent."""
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_canonical(item) for item in obj]
    return obj


def compute_schema_digest(schema: dict) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *schema*.

    Two schemas with identical content but different key ordering produce
    the same digest.
    """
    canonical_json = json.dumps(_canonical(schema), separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
