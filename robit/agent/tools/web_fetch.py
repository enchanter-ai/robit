"""robit.agent.tools.web_fetch — fetch an HTTPS URL and return readable text.

Wave 15.1 / Agent E. Stdlib only — no ``requests``, no ``httpx``, no
``beautifulsoup4``. The tool is read-only and idempotent in spirit, but it
talks to the network, which is exactly where SSRF lives, so the execute path
is heavily guarded:

* **HTTPS only.** ``http``, ``file``, ``ftp``, ``gopher`` and friends are
  rejected at parse time. The error message names the offending scheme so the
  LLM can correct itself.
* **SSRF guard at every hop.** Every redirect target's hostname is
  re-resolved and the resulting IP is checked against RFC1918, loopback,
  link-local, and the cloud-metadata literal (169.254.169.254). The check
  fires on the initial URL and on each redirect.
* **HTTPS→HTTP downgrade rejected.** A redirect that steps off TLS is treated
  the same as a non-HTTPS initial URL.
* **Redirect chain capped at 5.** Beyond that the request aborts as an
  infinite-loop suspect.
* **Body bounded by ``max_bytes``.** If ``Content-Length`` already exceeds
  the cap, the download is refused before a single byte is read. Otherwise
  the response is streamed in 8 KB chunks and clipped when the cap is hit;
  truncation is reported in the side-effects line.
* **HTML to text is intentionally simple.** A small ``HTMLParser`` subclass
  pulls out the ``<title>``, paragraph-ish blocks (``<p>``, ``<li>``,
  ``<h1>``–``<h6>``), code (``<pre>``, ``<code>``), and inline link
  references (``<a href>``). ``<script>``, ``<style>``, ``<noscript>`` are
  dropped. This is "good enough to read docs," not a readability clone.

Honesty caveat: DNS rebinding is technically possible — the SSRF guard
resolves the hostname once, then ``urllib`` resolves it again at connect
time. A hostile DNS server could return a public IP to the guard and a
private IP to the connect. The stdlib offers no hook to pin the resolved IP
through the connect, so this race is unfixable here without a third-party
HTTP client. Document it; don't pretend it's solved.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from html.parser import HTMLParser
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from ._types import ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BYTES = 512 * 1024              # 512 KB
_OUTPUT_CAP_BYTES = 32 * 1024                # 32 KB extracted text
_CHUNK_SIZE = 8 * 1024                       # 8 KB streamed reads
_MAX_REDIRECTS = 5
_USER_AGENT = "robit/0.7 (+https://github.com/enchanter-ai/robit)"

# The literal AWS/GCP/Azure metadata IP is also link-local, but we call it out
# by name so the error message tells the LLM exactly what was blocked.
_CLOUD_METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


class _SSRFBlocked(Exception):
    """Raised when a hostname resolves to a forbidden address."""


def _check_ip_blocked(host: str) -> str:
    """Resolve ``host`` and raise :class:`_SSRFBlocked` if the IP is forbidden.

    Returns the resolved IP as a string when the address is acceptable. The
    caller passes the IP into the error path only — ``urllib`` re-resolves on
    its own (see the DNS-rebinding note in the module docstring).
    """
    # If ``host`` is already an IP literal, ``gethostbyname`` returns it
    # unchanged on most platforms; either way we then run the ipaddress
    # checks below.
    try:
        ip_str = socket.gethostbyname(host)
    except socket.gaierror as exc:
        raise _SSRFBlocked(f"could not resolve host: {host} ({exc})") from exc

    if ip_str in _CLOUD_METADATA_IPS or host in _CLOUD_METADATA_IPS:
        raise _SSRFBlocked(
            f"refusing to fetch cloud-metadata address: {ip_str}"
        )

    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise _SSRFBlocked(f"invalid IP {ip_str!r} for host {host}: {exc}") from exc

    if ip_obj.is_loopback:
        raise _SSRFBlocked(f"refusing to fetch loopback address: {ip_str}")
    if ip_obj.is_link_local:
        raise _SSRFBlocked(f"refusing to fetch link-local address: {ip_str}")
    if ip_obj.is_private:
        raise _SSRFBlocked(f"refusing to fetch private address: {ip_str}")
    if ip_obj.is_multicast:
        raise _SSRFBlocked(f"refusing to fetch multicast address: {ip_str}")
    if ip_obj.is_unspecified:
        raise _SSRFBlocked(f"refusing to fetch unspecified address: {ip_str}")
    if ip_obj.is_reserved:
        raise _SSRFBlocked(f"refusing to fetch reserved address: {ip_str}")

    return ip_str


# ---------------------------------------------------------------------------
# HTML to text
# ---------------------------------------------------------------------------


_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "nav", "ul", "ol", "li", "blockquote", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_STRIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
_PRE_TAGS = frozenset({"pre"})
_CODE_TAGS = frozenset({"code"})
_LINK_TAG = "a"
_TITLE_TAG = "title"


class _HtmlToText(HTMLParser):
    """Convert HTML into a flat readable plain-text form.

    Strategy:
    * ``<title>`` content is captured into ``self.title`` (first occurrence wins).
    * ``<script>`` / ``<style>`` / ``<noscript>`` and similar are skipped via a
      depth counter — anything inside contributes nothing to the output.
    * ``<pre>`` content is preserved verbatim (including whitespace and inner
      tag text); a depth counter tracks nested ``<pre>``.
    * ``<code>`` content outside ``<pre>`` is wrapped in backticks inline.
    * Block-level tags emit a newline boundary before and after their content.
    * Headings additionally get a blank line below.
    * ``<a href>`` becomes ``[text](href)``; pure-bareword anchors render as
      plain text if ``href`` is missing.
    * Whitespace runs in normal flow collapse to a single space; ``<pre>``
      preserves them.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self._title_depth: int = 0
        self._strip_depth: int = 0
        self._pre_depth: int = 0
        self._code_depth: int = 0
        self._link_href: list[Optional[str]] = []   # stack: href or None
        self._link_buf: list[list[str]] = []        # stack of buffers per link
        self._out: list[str] = []

    # -- low-level emission ------------------------------------------------

    def _emit(self, s: str) -> None:
        # Inside an <a> tag, capture into the link buffer instead of the
        # main output so we can render `[text](href)` atomically on close.
        if self._link_buf:
            self._link_buf[-1].append(s)
        else:
            self._out.append(s)

    def _emit_block_boundary(self) -> None:
        # Two newlines between blocks; collapse runs of more than two on close.
        self._emit("\n")

    # -- tag handling ------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._strip_depth > 0:
            if tag in _STRIP_TAGS:
                self._strip_depth += 1
            return

        if tag in _STRIP_TAGS:
            self._strip_depth += 1
            return

        if tag == _TITLE_TAG:
            self._title_depth += 1
            return

        if tag in _PRE_TAGS:
            self._pre_depth += 1
            self._emit("\n")
            return

        if tag in _CODE_TAGS:
            self._code_depth += 1
            if self._pre_depth == 0:
                self._emit("`")
            return

        if tag == _LINK_TAG:
            href = None
            for k, v in attrs:
                if k == "href":
                    href = v
                    break
            self._link_href.append(href)
            self._link_buf.append([])
            return

        if tag == "br":
            self._emit("\n")
            return

        if tag in _HEADING_TAGS:
            self._emit_block_boundary()
            self._emit("\n")  # extra leading newline for visual separation
            return

        if tag in _BLOCK_TAGS:
            self._emit_block_boundary()
            return

        # li bullet
        if tag == "li":
            self._emit_block_boundary()
            self._emit("- ")
            return

    def handle_endtag(self, tag: str) -> None:
        if self._strip_depth > 0:
            if tag in _STRIP_TAGS:
                self._strip_depth -= 1
            return

        if tag == _TITLE_TAG:
            if self._title_depth > 0:
                self._title_depth -= 1
            return

        if tag in _PRE_TAGS:
            if self._pre_depth > 0:
                self._pre_depth -= 1
            self._emit("\n")
            return

        if tag in _CODE_TAGS:
            if self._code_depth > 0:
                self._code_depth -= 1
            if self._pre_depth == 0:
                self._emit("`")
            return

        if tag == _LINK_TAG:
            if not self._link_buf:
                return
            inner = "".join(self._link_buf.pop()).strip()
            href = self._link_href.pop() if self._link_href else None
            # Collapse whitespace inside link text.
            inner = re.sub(r"\s+", " ", inner)
            if href:
                rendered = f"[{inner}]({href})" if inner else f"[{href}]({href})"
            else:
                rendered = inner
            # The link rendering itself goes either to the outer link buffer
            # (nested links — rare but legal in HTML5 phrasing) or to the
            # main output stream.
            if self._link_buf:
                self._link_buf[-1].append(rendered)
            else:
                self._out.append(rendered)
            return

        if tag in _HEADING_TAGS:
            self._emit_block_boundary()
            return

        if tag in _BLOCK_TAGS:
            self._emit_block_boundary()
            return

    def handle_data(self, data: str) -> None:
        if self._strip_depth > 0:
            return

        if self._title_depth > 0:
            # First non-empty title wins.
            if not self.title:
                self.title = data.strip()
            else:
                # Append in case of split data calls inside the same title.
                self.title = (self.title + " " + data.strip()).strip()
            return

        if self._pre_depth > 0:
            # Preserve verbatim, but emit into the active context (link buf
            # or main out) so the link-text rendering still works.
            self._emit(data)
            return

        # Normal flow: collapse whitespace runs.
        collapsed = re.sub(r"\s+", " ", data)
        if collapsed:
            self._emit(collapsed)

    # -- finalize ----------------------------------------------------------

    def render(self) -> str:
        text = "".join(self._out)
        # Collapse 3+ newlines down to 2.
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Trim trailing spaces on lines.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = text.strip()
        if self.title:
            text = f"# {self.title}\n\n{text}" if text else f"# {self.title}"
        return text


def _html_to_text(html: str) -> str:
    parser = _HtmlToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # HTMLParser is extremely permissive, but pathological input can still
        # raise. Fall back to whatever was extracted before the failure.
        pass
    return parser.render()


# ---------------------------------------------------------------------------
# Fetch (with manual redirect handling so we can SSRF-check each hop)
# ---------------------------------------------------------------------------


class _FetchError(Exception):
    """Generic non-SSRF fetch failure. ``message`` is LLM-facing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _fetch(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float,
) -> tuple[str, bytes, str, bool]:
    """Fetch ``url`` and return ``(final_url, body, content_type, truncated)``.

    Handles HTTPS-only enforcement, SSRF guarding at every redirect hop, the
    redirect cap, the Content-Length precheck, and bounded streaming reads.
    Raises :class:`_SSRFBlocked` or :class:`_FetchError` on failure.
    """
    current = url
    seen: list[str] = []
    for hop in range(_MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        scheme = (parsed.scheme or "").lower()
        if scheme != "https":
            raise _FetchError(
                f"only https URLs are allowed (got scheme {scheme!r} at {current})"
            )

        host = parsed.hostname
        if not host:
            raise _FetchError(f"URL has no host: {current}")

        # SSRF guard before every connect.
        _check_ip_blocked(host)

        seen.append(current)
        req = Request(
            current,
            method="GET",
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
            },
        )

        try:
            # ``urlopen`` follows redirects on its own; we want manual control,
            # so we set up an opener that does NOT follow them. We do this by
            # installing a redirect handler that raises on 3xx. Simpler: use
            # urlopen, catch HTTPError 3xx-equivalents via the default behaviour.
            #
            # Stdlib doesn't directly let us opt out of redirect-following at the
            # urlopen level, so we use a custom OpenerDirector.
            opener = _make_no_redirect_opener()
            with opener.open(req, timeout=timeout_s) as resp:
                status = resp.status
                headers = resp.headers
                if 300 <= status < 400:
                    loc = headers.get("Location")
                    if not loc:
                        raise _FetchError(
                            f"redirect {status} from {current} had no Location header"
                        )
                    next_url = urljoin(current, loc)
                    next_scheme = (urlparse(next_url).scheme or "").lower()
                    if next_scheme == "http":
                        raise _FetchError(
                            f"refusing https→http downgrade redirect: {current} → {next_url}"
                        )
                    if next_url in seen:
                        raise _FetchError(
                            f"redirect loop detected at {next_url}"
                        )
                    current = next_url
                    continue

                if 400 <= status < 500:
                    raise _FetchError(
                        f"HTTP {status} client error fetching {current}"
                    )
                if 500 <= status < 600:
                    raise _FetchError(
                        f"HTTP {status} server error fetching {current}"
                    )

                content_type = headers.get("Content-Type", "application/octet-stream")
                # Strip parameters (charset etc.) for the type match.
                ct_main = content_type.split(";", 1)[0].strip().lower()

                # Content-Length precheck.
                cl = headers.get("Content-Length")
                if cl is not None:
                    try:
                        cl_int = int(cl)
                    except ValueError:
                        cl_int = -1
                    if cl_int > max_bytes:
                        raise _FetchError(
                            f"Content-Length {cl_int} exceeds max_bytes {max_bytes}"
                        )

                # Bounded streaming read.
                buf = bytearray()
                truncated = False
                while True:
                    remaining = max_bytes - len(buf)
                    if remaining <= 0:
                        # See if there's more data we'd be dropping.
                        peek = resp.read(1)
                        if peek:
                            truncated = True
                        break
                    chunk = resp.read(min(_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    buf.extend(chunk)

                return resp.geturl() or current, bytes(buf), ct_main, truncated

        except _SSRFBlocked:
            raise
        except _FetchError:
            raise
        except HTTPError as exc:
            # urlopen raises HTTPError for 4xx/5xx even with our custom opener,
            # if a handler upstream surfaces them. Reduce to our error type.
            if 300 <= exc.code < 400:
                # Should have been handled by our opener; treat as misbehaviour.
                raise _FetchError(
                    f"unexpected redirect HTTPError {exc.code} at {current}"
                ) from exc
            raise _FetchError(
                f"HTTP {exc.code} {exc.reason} fetching {current}"
            ) from exc
        except ssl.SSLError as exc:
            raise _FetchError(f"TLS error fetching {current}: {exc}") from exc
        except socket.timeout as exc:
            raise _FetchError(
                f"timeout after {timeout_s}s fetching {current}"
            ) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            # Inner socket.timeout shows up as URLError(reason=socket.timeout()).
            if isinstance(reason, socket.timeout):
                raise _FetchError(
                    f"timeout after {timeout_s}s fetching {current}"
                ) from exc
            if isinstance(reason, ssl.SSLError):
                raise _FetchError(f"TLS error fetching {current}: {reason}") from exc
            raise _FetchError(f"network error fetching {current}: {reason}") from exc

    # Fell off the loop without returning → too many redirects.
    raise _FetchError(
        f"too many redirects (> {_MAX_REDIRECTS}): {' → '.join(seen)}"
    )


def _make_no_redirect_opener():
    """Build an opener that surfaces 3xx responses instead of following them.

    The default ``HTTPRedirectHandler`` would follow them transparently and
    skip our per-hop SSRF guard, so we replace it with a handler that returns
    the response unchanged.
    """
    from urllib.request import (
        HTTPHandler,
        HTTPSHandler,
        OpenerDirector,
        HTTPRedirectHandler,
    )

    class _NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            return None  # signals "do not follow"

        def http_error_301(self, req, fp, code, msg, headers):  # type: ignore[override]
            return fp

        def http_error_302(self, req, fp, code, msg, headers):  # type: ignore[override]
            return fp

        def http_error_303(self, req, fp, code, msg, headers):  # type: ignore[override]
            return fp

        def http_error_307(self, req, fp, code, msg, headers):  # type: ignore[override]
            return fp

        def http_error_308(self, req, fp, code, msg, headers):  # type: ignore[override]
            return fp

    opener = OpenerDirector()
    opener.add_handler(HTTPHandler())
    opener.add_handler(HTTPSHandler())
    opener.add_handler(_NoRedirect())
    return opener


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class WebFetchTool:
    """Fetch an HTTPS URL and return its content as plain text."""

    name: str = "web_fetch"
    description: str = (
        "Fetch a URL and return its content as plain text. HTML is converted to a "
        "readable plain-text form (titles + paragraphs + links). Use this to look "
        "up documentation, read GitHub READMEs, etc. Only HTTPS URLs are allowed."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "format": "uri",
                "description": "HTTPS URL to fetch.",
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "default": _DEFAULT_MAX_BYTES,
                "description": "Max bytes to download. Default 512KB.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    requires_approval: bool = False  # Read-only — but SSRF-guarded above.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raw_url = args.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            return ToolResult(
                content="web_fetch: 'url' arg is required and must be a non-empty string",
                is_error=True,
            )

        max_bytes_raw = args.get("max_bytes", _DEFAULT_MAX_BYTES)
        if not isinstance(max_bytes_raw, int) or isinstance(max_bytes_raw, bool) or max_bytes_raw < 1:
            return ToolResult(
                content="web_fetch: 'max_bytes' must be a positive integer",
                is_error=True,
            )
        max_bytes = max_bytes_raw

        # Early scheme check — gives a more specific error than letting the
        # SSRF guard catch e.g. file:// after a hostname lookup.
        parsed = urlparse(raw_url)
        scheme = (parsed.scheme or "").lower()
        if scheme != "https":
            return ToolResult(
                content=(
                    f"only https URLs are allowed (got scheme {scheme!r}): {raw_url}"
                ),
                is_error=True,
            )

        try:
            final_url, body, content_type, truncated = _fetch(
                raw_url,
                max_bytes=max_bytes,
                timeout_s=ctx.timeout_s,
            )
        except _SSRFBlocked as exc:
            return ToolResult(content=str(exc), is_error=True)
        except _FetchError as exc:
            return ToolResult(content=exc.message, is_error=True)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                content=f"web_fetch: unexpected error: {exc}",
                is_error=True,
            )

        n_bytes = len(body)

        # Decode + extract.
        if content_type == "text/html" or content_type.endswith("+html"):
            try:
                html = body.decode("utf-8", errors="replace")
            except Exception:
                html = body.decode("latin-1", errors="replace")
            text = _html_to_text(html)
        elif content_type.startswith("text/"):
            text = body.decode("utf-8", errors="replace")
        else:
            # JSON, octet-stream, etc. — best effort utf-8 decode with replace.
            text = body.decode("utf-8", errors="replace")

        # Apply the smaller of the per-tool output cap and the context cap.
        output_cap = min(_OUTPUT_CAP_BYTES, ctx.max_output_bytes)
        encoded = text.encode("utf-8")
        output_truncated = False
        if len(encoded) > output_cap:
            clipped = encoded[:output_cap]
            text = clipped.decode("utf-8", errors="ignore")
            text += "\n...[truncated]"
            output_truncated = True

        side_effects = [
            f"fetched {final_url} ({n_bytes} bytes, {content_type})",
        ]
        if truncated:
            side_effects.append(
                f"download truncated at max_bytes={max_bytes}"
            )
        if output_truncated:
            side_effects.append("extracted text truncated to output cap")

        return ToolResult(
            content=text,
            is_error=False,
            side_effects=tuple(side_effects),
        )


__all__ = ["WebFetchTool"]
