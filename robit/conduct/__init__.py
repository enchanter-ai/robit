"""robit.conduct — conduct module loader and types.

Public surface::

    from robit.conduct import load_conduct, ConductRule, ConductFrontmatterError

Everything else in this sub-package is an implementation detail.
"""

from robit.conduct.loader import load_conduct
from robit.conduct.types import ConductFrontmatterError, ConductRule

__all__ = [
    "load_conduct",
    "ConductRule",
    "ConductFrontmatterError",
]
