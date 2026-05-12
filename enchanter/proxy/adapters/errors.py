"""Shared adapter error type.

A single class so `except AdapterParseError` in the server layer catches
parse errors from every adapter family.
"""

from __future__ import annotations


class AdapterParseError(ValueError):
    """Raised by an adapter's parse_request when the input body is malformed.

    The server layer maps this to HTTP 400 with the matching family's error
    envelope shape.
    """
