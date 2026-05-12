"""Default filesystem paths for the conduct loader.

Keeping the path constant here means tests can patch just this module
rather than monkey-patching ``loader.py`` internals.
"""

from pathlib import Path

# Canonical location of the enchanter-foundations package tree.
# Expected structure: <DEFAULT_FOUNDATIONS_ROOT>/packages/<pkg>/conduct/*.md
DEFAULT_FOUNDATIONS_ROOT: Path = (
    Path(__file__).resolve().parent  # enchanter/conduct/
    .parent                           # enchanter/
    .parent                           # agent/
    .parent                           # enchanter-ai/
    / "foundations"
)
