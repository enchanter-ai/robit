"""robit.agent.tools._paths — shared path-safety helper.

Every file-system tool (file_read, file_write, edit, glob, grep, bash) needs
the same two guarantees:

* Relative paths resolve against the tool context's ``cwd`` — never the
  process's current working directory, which the agent may have drifted from.
* Resolved paths stay *inside* ``cwd`` unless the caller explicitly opts out.
  Symlinks are followed via :meth:`pathlib.Path.resolve`, so a symlink
  pointing outside ``cwd`` is still rejected.

Wave 15.1 sibling agents B (write/edit), C (glob/grep), and D (bash) import
:func:`safe_resolve` and :class:`PathOutsideCwdError` directly — the public
signature here is frozen for the rest of the wave.
"""

from __future__ import annotations

from pathlib import Path


class PathOutsideCwdError(ValueError):
    """Raised when a tool tries to access a path outside its allowed cwd."""


def safe_resolve(
    cwd: Path,
    user_path: str | Path,
    *,
    allow_outside_cwd: bool = False,
) -> Path:
    """Resolve ``user_path`` relative to ``cwd`` and return an absolute :class:`Path`.

    Rules
    -----
    - Absolute paths are returned as-is (after ``.resolve()``).
    - Relative paths are joined to ``cwd`` and ``.resolve()``-d.
    - If ``allow_outside_cwd=False`` (default) AND the resolved path is not
      inside ``cwd``, raise :class:`PathOutsideCwdError`.
    - Symlinks are followed via ``.resolve()``.

    Raises
    ------
    PathOutsideCwdError
        If the resolved path escapes ``cwd`` and ``allow_outside_cwd`` is False.
    TypeError
        If ``user_path`` is not :class:`str` or :class:`pathlib.Path`.
    """
    if not isinstance(user_path, (str, Path)):
        raise TypeError(
            f"user_path must be str or Path, got {type(user_path).__name__}"
        )

    p = Path(user_path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (cwd / p).resolve()

    if not allow_outside_cwd:
        cwd_resolved = cwd.resolve()
        # Path.is_relative_to was added in 3.9 — we target 3.11+.
        if not resolved.is_relative_to(cwd_resolved):
            raise PathOutsideCwdError(
                f"path escapes cwd: {resolved} (cwd={cwd_resolved})"
            )

    return resolved


__all__ = ["PathOutsideCwdError", "safe_resolve"]
