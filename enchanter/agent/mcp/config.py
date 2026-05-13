"""enchanter.agent.mcp.config — load MCP server configuration.

File location resolution (same precedence as ``enchanter.agent.session``):

  1. ``$ENCHANTER_HOME/mcp.json``                  (env override)
  2. Windows: ``%APPDATA%\\enchanter\\mcp.json``
  3. POSIX:   ``~/.enchanter/mcp.json``

File shape::

    {
      "servers": {
        "filesystem": {
          "command": "npx",
          "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
          "env_allowlist": ["PATH", "HOME"]
        }
      }
    }

Robustness rules:

* Missing file → ``[]`` (no warning; the absence is the common case).
* Malformed JSON or shape → log a warning and return ``[]``.
* Malformed individual server entry → warn and skip *that* entry only;
  other valid entries still come through.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    """One entry in the MCP server config file.

    Mirrors the relevant slice of
    :class:`enchanter.transport.descriptor.TransportDescriptor` so the
    :class:`~enchanter.agent.mcp.client.MCPClient` can build a descriptor
    without leaking JSON-shape concerns down into the client.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()


def _default_config_path() -> Path:
    override = os.environ.get("ENCHANTER_HOME")
    if override:
        return Path(override) / "mcp.json"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        root = Path(appdata) / "enchanter" if appdata else Path.home() / ".enchanter"
    else:
        root = Path.home() / ".enchanter"
    return root / "mcp.json"


def load_mcp_config(path: Path | None = None) -> list[MCPServerConfig]:
    """Load and parse the MCP server config.

    Parameters
    ----------
    path:
        Optional explicit path. When ``None``, the default resolution rule
        in :func:`_default_config_path` applies.

    Returns
    -------
    list[MCPServerConfig]
        Parsed entries. Empty list on any of: missing file, unreadable file,
        malformed JSON, top-level shape mismatch.
    """
    cfg_path = path if path is not None else _default_config_path()

    if not cfg_path.exists():
        return []

    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("mcp config: cannot read %s: %s", cfg_path, exc)
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("mcp config: malformed JSON in %s: %s", cfg_path, exc)
        return []

    if not isinstance(data, dict):
        logger.warning("mcp config: %s: top-level must be an object", cfg_path)
        return []

    servers = data.get("servers")
    if not isinstance(servers, dict):
        logger.warning(
            "mcp config: %s: 'servers' key missing or not an object", cfg_path,
        )
        return []

    out: list[MCPServerConfig] = []
    for name, entry in servers.items():
        cfg = _parse_entry(name, entry, cfg_path)
        if cfg is not None:
            out.append(cfg)
    return out


def _parse_entry(
    name: str,
    entry: object,
    src: Path,
) -> MCPServerConfig | None:
    """Validate one server entry; warn + return ``None`` on shape mismatch."""
    if not isinstance(name, str) or not name:
        logger.warning("mcp config: %s: server name must be a non-empty string", src)
        return None
    if not isinstance(entry, dict):
        logger.warning(
            "mcp config: %s: server %r entry must be an object", src, name,
        )
        return None
    command = entry.get("command")
    if not isinstance(command, str) or not command:
        logger.warning(
            "mcp config: %s: server %r missing required 'command' string", src, name,
        )
        return None
    args_raw = entry.get("args", [])
    if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
        logger.warning(
            "mcp config: %s: server %r 'args' must be a list of strings", src, name,
        )
        return None
    env_raw = entry.get("env_allowlist", [])
    if not isinstance(env_raw, list) or not all(isinstance(e, str) for e in env_raw):
        logger.warning(
            "mcp config: %s: server %r 'env_allowlist' must be a list of strings",
            src, name,
        )
        return None
    return MCPServerConfig(
        name=name,
        command=command,
        args=tuple(args_raw),
        env_allowlist=tuple(env_raw),
    )


__all__ = ["MCPServerConfig", "load_mcp_config"]
