"""Tests for robit.agent.mcp.config.load_mcp_config."""

from __future__ import annotations

import json
from pathlib import Path

from robit.agent.mcp.config import MCPServerConfig, load_mcp_config


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_mcp_config(tmp_path / "nope.json") == []


def test_malformed_json_returns_empty_with_warning(tmp_path: Path, caplog) -> None:
    p = tmp_path / "mcp.json"
    p.write_text("{ not valid", encoding="utf-8")
    with caplog.at_level("WARNING"):
        result = load_mcp_config(p)
    assert result == []
    assert any("malformed JSON" in r.message for r in caplog.records)


def test_valid_file_returns_parsed_entries(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        json.dumps({
            "servers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env_allowlist": ["PATH", "HOME"],
                },
                "github": {
                    "command": "github-mcp-server",
                },
            }
        }),
        encoding="utf-8",
    )
    cfgs = load_mcp_config(p)
    assert len(cfgs) == 2
    by_name = {c.name: c for c in cfgs}
    fs = by_name["filesystem"]
    assert isinstance(fs, MCPServerConfig)
    assert fs.command == "npx"
    assert fs.args == ("@modelcontextprotocol/server-filesystem", "/tmp")
    assert fs.env_allowlist == ("PATH", "HOME")
    gh = by_name["github"]
    assert gh.command == "github-mcp-server"
    assert gh.args == ()
    assert gh.env_allowlist == ()


def test_malformed_entry_is_skipped_others_pass(tmp_path: Path, caplog) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        json.dumps({
            "servers": {
                "ok": {"command": "x"},
                "bad-missing-command": {"args": []},
                "bad-args-type": {"command": "y", "args": "not a list"},
            }
        }),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        cfgs = load_mcp_config(p)
    names = [c.name for c in cfgs]
    assert names == ["ok"]
    assert any("bad-missing-command" in r.message for r in caplog.records)
    assert any("bad-args-type" in r.message for r in caplog.records)


def test_top_level_must_be_object(tmp_path: Path, caplog) -> None:
    p = tmp_path / "mcp.json"
    p.write_text("[]", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert load_mcp_config(p) == []


def test_servers_key_missing(tmp_path: Path, caplog) -> None:
    p = tmp_path / "mcp.json"
    p.write_text("{}", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert load_mcp_config(p) == []
