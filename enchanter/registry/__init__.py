"""enchanter.registry — tool-name collision guard and schema-digest pinning.

Public surface:
  NamespaceRegistry          — register / resolve / unregister tools
  ToolDescriptor             — descriptor dataclass
  ToolNameCollisionError     — FM-1: bare name maps to >1 server
  SchemaDigestMismatchError  — FM-10: schema changed after pin
  compute_schema_digest      — canonical-JSON SHA-256 helper
"""

from enchanter.registry.digest import compute_schema_digest
from enchanter.registry.errors import SchemaDigestMismatchError, ToolNameCollisionError
from enchanter.registry.namespace import NamespaceRegistry
from enchanter.registry.types import ToolDescriptor

__all__ = [
    "NamespaceRegistry",
    "ToolDescriptor",
    "ToolNameCollisionError",
    "SchemaDigestMismatchError",
    "compute_schema_digest",
]
