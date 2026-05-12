"""enchanter.runtime — tier router and models registry.

Public surface:
  ModelsRegistry     — load and query the 255-model capability registry
  ModelEntry         — typed dataclass for a single registry entry
  TierRouter         — maps TaskClass to a concrete model_id
  TaskClass          — Literal type for tier intent labels
  UnknownModelError  — raised when model_id is not in the registry
  UnknownFamilyError — raised when a family lookup finds nothing
"""

from enchanter.runtime.models_registry import ModelsRegistry, ModelEntry, UnknownModelError, UnknownFamilyError
from enchanter.runtime.tier_router import TierRouter, TaskClass

__all__ = [
    "ModelsRegistry",
    "ModelEntry",
    "TierRouter",
    "TaskClass",
    "UnknownModelError",
    "UnknownFamilyError",
]
