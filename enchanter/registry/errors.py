"""enchanter/registry/errors.py — failure-mode errors for the namespace registry.

FM-1  ToolNameCollisionError  — bare name maps to >1 server; caller must qualify.
FM-10 SchemaDigestMismatchError — re-registration with a changed schema digest.
"""

from __future__ import annotations


class ToolNameCollisionError(Exception):
    """Raised when a bare tool name is exported by more than one server.

    Callers must switch to the qualified form ``<server_id>.<tool_name>``.
    """

    def __init__(self, name: str, server_ids: list[str]) -> None:
        self.name_: str = name  # avoid shadowing Exception.name
        self.server_ids: list[str] = sorted(server_ids)
        super().__init__(
            f'tool name "{name}" exported by multiple servers '
            f"({', '.join(self.server_ids)}); use qualified \"server_id.tool_name\""
        )


class SchemaDigestMismatchError(Exception):
    """Raised when a re-registration presents a different schema digest.

    Indicates potential MCPoison schema mutation — requires re-consent.
    """

    def __init__(
        self,
        server_id: str,
        tool_name: str,
        expected: str,
        got: str,
    ) -> None:
        self.server_id = server_id
        self.tool_name = tool_name
        self.expected = expected
        self.got = got
        qualified = f"{server_id}.{tool_name}"
        super().__init__(
            f"tool {qualified} schema digest changed: "
            f"pinned={expected} seen={got} — requires re-consent"
        )
