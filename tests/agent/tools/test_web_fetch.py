"""Tests for robit.agent.tools.web_fetch.WebFetchTool.

Network is fully mocked. We patch:

* ``socket.gethostbyname`` to control which IP a hostname resolves to (drives
  the SSRF guard).
* The custom opener's ``open`` method to feed synthetic responses without
  touching the network. We do this by replacing ``_make_no_redirect_opener``
  so every test gets a deterministic stub.

The stub responses mimic ``http.client.HTTPResponse`` just enough for the
fetch loop: ``status``, ``headers``, ``geturl()``, ``read(n)``, and
``__enter__/__exit__``.
"""

from __future__ import annotations

import asyncio
import socket
from email.message import Message
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from robit.agent.tools._types import ToolContext
from robit.agent.tools.web_fetch import WebFetchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path, *, timeout_s: float = 5.0, max_out: int = 64 * 1024) -> ToolContext:
    return ToolContext(
        cwd=tmp_path,
        session_id="test-session",
        max_output_bytes=max_out,
        timeout_s=timeout_s,
    )


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    """Minimal HTTPResponse-like stub."""

    def __init__(
        self,
        *,
        status: int,
        body: bytes = b"",
        headers: Optional[dict] = None,
        url: str = "",
    ) -> None:
        self.status = status
        msg = Message()
        for k, v in (headers or {}).items():
            msg[k] = v
        self.headers = msg
        self._body = body
        self._pos = 0
        self._url = url

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ScriptedOpener:
    """Returns successive ``_FakeResponse`` objects per ``open()`` call."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def open(self, req, timeout=None):
        self.calls.append(req.full_url if hasattr(req, "full_url") else str(req))
        if not self._responses:
            raise AssertionError("opener called more times than scripted")
        return self._responses.pop(0)


def _patch_opener(responses: list[_FakeResponse]) -> _ScriptedOpener:
    opener = _ScriptedOpener(responses)
    return opener


# ---------------------------------------------------------------------------
# Static-attribute sanity
# ---------------------------------------------------------------------------


def test_static_attributes():
    t = WebFetchTool()
    assert t.name == "web_fetch"
    assert t.requires_approval is False
    assert t.input_schema["required"] == ["url"]
    assert t.input_schema["additionalProperties"] is False
    assert "url" in t.input_schema["properties"]
    assert "max_bytes" in t.input_schema["properties"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


HTML_DOC = b"""<!doctype html><html>
<head>
  <title>Hello Docs</title>
  <style>body { color: red; }</style>
  <script>alert('xss');</script>
</head>
<body>
  <h1>Welcome</h1>
  <p>This is a paragraph with a <a href="https://example.com/next">link</a>.</p>
  <ul>
    <li>Item one</li>
    <li>Item two</li>
  </ul>
  <pre>verbatim
  whitespace  preserved</pre>
  <noscript>fallback content</noscript>
</body></html>"""


def test_valid_https_html_returns_extracted_text(tmp_path):
    resp = _FakeResponse(
        status=200,
        body=HTML_DOC,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(HTML_DOC)),
        },
        url="https://example.com/docs",
    )
    opener = _patch_opener([resp])

    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/docs"}, _ctx(tmp_path))
        )

    assert result.is_error is False
    assert "Hello Docs" in result.content        # title rendered
    assert "Welcome" in result.content           # h1
    assert "Item one" in result.content
    assert "[link](https://example.com/next)" in result.content
    assert "verbatim" in result.content
    assert "alert('xss')" not in result.content  # script stripped
    assert "color: red" not in result.content    # style stripped
    assert "fallback content" not in result.content
    assert any("text/html" in s for s in result.side_effects)
    assert any(str(len(HTML_DOC)) in s for s in result.side_effects)


# ---------------------------------------------------------------------------
# Scheme rejection
# ---------------------------------------------------------------------------


def test_http_scheme_rejected(tmp_path):
    result = _run(
        WebFetchTool().execute({"url": "http://example.com/"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "https" in result.content.lower()


def test_file_scheme_rejected(tmp_path):
    result = _run(
        WebFetchTool().execute({"url": "file:///etc/passwd"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "https" in result.content.lower()


def test_ftp_scheme_rejected(tmp_path):
    result = _run(
        WebFetchTool().execute({"url": "ftp://example.com/data"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "https" in result.content.lower()


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def test_ip_literal_metadata_rejected(tmp_path):
    # Even when the "resolver" returns the literal, we reject it.
    with patch("socket.gethostbyname", return_value="169.254.169.254"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://169.254.169.254/latest/meta-data/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "metadata" in result.content.lower() or "link-local" in result.content.lower()


def test_loopback_hostname_rejected(tmp_path):
    with patch("socket.gethostbyname", return_value="127.0.0.1"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://localhost.evil.test/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "loopback" in result.content.lower()


def test_rfc1918_10_rejected(tmp_path):
    with patch("socket.gethostbyname", return_value="10.0.0.1"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://internal.corp.example/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "private" in result.content.lower()


def test_rfc1918_192_168_rejected(tmp_path):
    with patch("socket.gethostbyname", return_value="192.168.1.1"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://router.lan/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "private" in result.content.lower()


def test_link_local_ipv4_rejected(tmp_path):
    with patch("socket.gethostbyname", return_value="169.254.10.10"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://link.local.test/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "link-local" in result.content.lower()


def test_dns_failure_rejected(tmp_path):
    def boom(_host):
        raise socket.gaierror("nodename nor servname provided, or not known")

    with patch("socket.gethostbyname", side_effect=boom):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://does-not-exist.invalid/"},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "could not resolve" in result.content.lower()


# ---------------------------------------------------------------------------
# HTTP status handling
# ---------------------------------------------------------------------------


def test_4xx_response_returns_error(tmp_path):
    resp = _FakeResponse(
        status=404,
        body=b"not found",
        headers={"Content-Type": "text/plain", "Content-Length": "9"},
        url="https://example.com/missing",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://example.com/missing"}, _ctx(tmp_path)
            )
        )
    assert result.is_error is True
    assert "404" in result.content


def test_5xx_response_returns_error(tmp_path):
    resp = _FakeResponse(
        status=503,
        body=b"down",
        headers={"Content-Type": "text/plain"},
        url="https://example.com/down",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://example.com/down"}, _ctx(tmp_path)
            )
        )
    assert result.is_error is True
    assert "503" in result.content


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


def test_content_length_exceeds_max_bytes(tmp_path):
    resp = _FakeResponse(
        status=200,
        body=b"x" * 10,
        headers={"Content-Type": "text/plain", "Content-Length": "1000000"},
        url="https://example.com/big",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://example.com/big", "max_bytes": 1024},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is True
    assert "Content-Length" in result.content
    assert "1000000" in result.content


def test_streaming_truncation_without_content_length(tmp_path):
    body = b"a" * 5000
    resp = _FakeResponse(
        status=200,
        body=body,
        headers={"Content-Type": "text/plain"},  # no Content-Length
        url="https://example.com/stream",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://example.com/stream", "max_bytes": 1024},
                _ctx(tmp_path),
            )
        )
    assert result.is_error is False
    # We should only have absorbed 1024 bytes of the original body
    # (output cap is bigger, so content here is the decoded 1024 'a's).
    assert "a" * 1024 in result.content
    # And we should have reported the truncation.
    assert any("truncated" in s.lower() for s in result.side_effects)


# ---------------------------------------------------------------------------
# HTML extraction details
# ---------------------------------------------------------------------------


def test_html_extracts_title_paragraphs_strips_script(tmp_path):
    html = (
        b"<html><head><title>T</title>"
        b"<script>secret()</script></head>"
        b"<body><p>Para A</p><p>Para B</p></body></html>"
    )
    resp = _FakeResponse(
        status=200,
        body=html,
        headers={"Content-Type": "text/html"},
        url="https://example.com/x",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/x"}, _ctx(tmp_path))
        )
    assert result.is_error is False
    assert "T" in result.content
    assert "Para A" in result.content
    assert "Para B" in result.content
    assert "secret()" not in result.content


def test_text_plain_returned_as_is(tmp_path):
    resp = _FakeResponse(
        status=200,
        body=b"hello world\nsecond line",
        headers={"Content-Type": "text/plain"},
        url="https://example.com/txt",
    )
    opener = _patch_opener([resp])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/txt"}, _ctx(tmp_path))
        )
    assert result.is_error is False
    assert result.content == "hello world\nsecond line"


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


def test_https_to_http_redirect_rejected(tmp_path):
    r1 = _FakeResponse(
        status=302,
        body=b"",
        headers={"Location": "http://example.com/insecure"},
        url="https://example.com/start",
    )
    opener = _patch_opener([r1])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/start"}, _ctx(tmp_path))
        )
    assert result.is_error is True
    assert "downgrade" in result.content.lower() or "http" in result.content.lower()


def test_redirect_chain_exceeding_limit_rejected(tmp_path):
    # 7 successive 302s — exceeds the 5-redirect cap.
    responses = [
        _FakeResponse(
            status=302,
            body=b"",
            headers={"Location": f"https://example.com/hop{i+1}"},
            url=f"https://example.com/hop{i}",
        )
        for i in range(7)
    ]
    opener = _patch_opener(responses)
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/hop0"}, _ctx(tmp_path))
        )
    assert result.is_error is True
    assert "redirect" in result.content.lower()


def test_redirect_loop_detected(tmp_path):
    r1 = _FakeResponse(
        status=302,
        body=b"",
        headers={"Location": "https://example.com/start"},
        url="https://example.com/start",
    )
    opener = _patch_opener([r1])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/start"}, _ctx(tmp_path))
        )
    assert result.is_error is True
    assert "loop" in result.content.lower() or "redirect" in result.content.lower()


def test_redirect_ssrf_checked_on_each_hop(tmp_path):
    # First hop resolves to a public IP. Second hop's hostname resolves to
    # 127.0.0.1 — must be rejected even though hop 1 was fine.
    r1 = _FakeResponse(
        status=302,
        body=b"",
        headers={"Location": "https://internal.evil.test/secret"},
        url="https://public.example/start",
    )
    opener = _patch_opener([r1])

    resolutions = {
        "public.example": "93.184.216.34",
        "internal.evil.test": "127.0.0.1",
    }

    def resolve(host):
        return resolutions[host]

    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", side_effect=resolve):
        result = _run(
            WebFetchTool().execute(
                {"url": "https://public.example/start"}, _ctx(tmp_path)
            )
        )
    assert result.is_error is True
    assert "loopback" in result.content.lower()


def test_successful_redirect_followed(tmp_path):
    r1 = _FakeResponse(
        status=302,
        body=b"",
        headers={"Location": "https://example.com/final"},
        url="https://example.com/start",
    )
    r2 = _FakeResponse(
        status=200,
        body=b"final body",
        headers={"Content-Type": "text/plain"},
        url="https://example.com/final",
    )
    opener = _patch_opener([r1, r2])
    with patch(
        "robit.agent.tools.web_fetch._make_no_redirect_opener",
        return_value=opener,
    ), patch("socket.gethostbyname", return_value="93.184.216.34"):
        result = _run(
            WebFetchTool().execute({"url": "https://example.com/start"}, _ctx(tmp_path))
        )
    assert result.is_error is False
    assert "final body" in result.content


# ---------------------------------------------------------------------------
# Arg validation
# ---------------------------------------------------------------------------


def test_missing_url_arg(tmp_path):
    result = _run(WebFetchTool().execute({}, _ctx(tmp_path)))
    assert result.is_error is True


def test_empty_url_arg(tmp_path):
    result = _run(WebFetchTool().execute({"url": ""}, _ctx(tmp_path)))
    assert result.is_error is True


def test_bad_max_bytes(tmp_path):
    result = _run(
        WebFetchTool().execute(
            {"url": "https://example.com/", "max_bytes": 0}, _ctx(tmp_path)
        )
    )
    assert result.is_error is True
