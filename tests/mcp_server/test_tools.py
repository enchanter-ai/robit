"""Unit tests for the default tool wrappers."""

from __future__ import annotations

import pytest

from robit.mcp_server.errors import InvalidParamsError
from robit.mcp_server.tools import (
    check_destructive_op_handler,
    scan_secrets_handler,
)


@pytest.mark.asyncio
async def test_scan_secrets_no_match() -> None:
    result = await scan_secrets_handler({"text": "plain old text, nothing sensitive."})
    assert result["matched"] is False
    assert result["matched_patterns"] == []


@pytest.mark.asyncio
async def test_scan_secrets_anthropic_key_match() -> None:
    # Anthropic-style key prefix is in the SECRET_PATTERNS list
    fake = "sk-ant-api03-" + "x" * 80
    result = await scan_secrets_handler({"text": f"my key is {fake}"})
    assert result["matched"] is True
    assert len(result["matched_patterns"]) >= 1


@pytest.mark.asyncio
async def test_scan_secrets_missing_text() -> None:
    with pytest.raises(InvalidParamsError):
        await scan_secrets_handler({})


@pytest.mark.asyncio
async def test_scan_secrets_non_string_text() -> None:
    with pytest.raises(InvalidParamsError):
        await scan_secrets_handler({"text": 12345})


@pytest.mark.asyncio
async def test_check_destructive_op_benign_command() -> None:
    result = await check_destructive_op_handler({"tool": "ls", "args": ["-la"]})
    assert result["vetoed"] is False


@pytest.mark.asyncio
async def test_check_destructive_op_rm_rf() -> None:
    # rm -rf / should hit a destructive pattern
    result = await check_destructive_op_handler({"tool": "rm", "args": ["-rf", "/"]})
    assert result["vetoed"] is True
    assert result["pattern_id"] is not None


@pytest.mark.asyncio
async def test_check_destructive_op_missing_tool() -> None:
    with pytest.raises(InvalidParamsError):
        await check_destructive_op_handler({"args": ["-rf", "/"]})


@pytest.mark.asyncio
async def test_check_destructive_op_args_default_empty() -> None:
    # No args field — should be treated as empty list, not error
    result = await check_destructive_op_handler({"tool": "true"})
    assert "vetoed" in result


@pytest.mark.asyncio
async def test_check_destructive_op_bad_args_type() -> None:
    with pytest.raises(InvalidParamsError):
        await check_destructive_op_handler({"tool": "rm", "args": "not-a-list"})
