"""enchanter.llm.mock_client — MockLlmClient for tests.

Does not touch the network.  Accepts either a list of responses (returned in
order) or a dict mapping substring patterns to responses (first key whose
substring appears in the last user-message content wins).

All requests are recorded in ``self.requests`` for test assertions.
"""

from __future__ import annotations

from .types import CompletionRequest, CompletionResponse


class MockLlmClient:
    """Test-only LlmClient with scripted responses.

    Parameters
    ----------
    responses:
        Either a ``list[CompletionResponse]`` (returned in FIFO order) or a
        ``dict[str, CompletionResponse]`` whose keys are substrings matched
        against the content of the *last* user message in the request.

    Raises
    ------
    ValueError
        At construction time if ``responses`` is neither a list nor a dict.
    RuntimeError
        At ``complete()`` time if the scripted responses are exhausted (list
        mode) or no key matches (dict mode).
    """

    def __init__(
        self,
        responses: list[CompletionResponse] | dict[str, CompletionResponse] | None = None,
    ) -> None:
        if responses is None:
            responses = []
        if not isinstance(responses, (list, dict)):
            raise ValueError(
                "responses must be a list[CompletionResponse] or "
                "dict[str, CompletionResponse]."
            )
        self._responses = responses
        self._call_index = 0
        self.requests: list[CompletionRequest] = []

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Return the next scripted response and record the request."""
        # Validate — protocol requires at least one message.
        if not req.messages:
            raise ValueError("CompletionRequest.messages must not be empty.")

        self.requests.append(req)

        if isinstance(self._responses, list):
            if self._call_index >= len(self._responses):
                raise RuntimeError(
                    f"MockLlmClient: no response configured for call #{self._call_index + 1}. "
                    f"Add more entries to the responses list."
                )
            response = self._responses[self._call_index]
            self._call_index += 1
            return response

        # dict mode — match against the last user message's content.
        last_content = ""
        for msg in reversed(req.messages):
            if msg.role == "user":
                last_content = msg.content
                break

        for pattern, response in self._responses.items():
            if pattern in last_content:
                return response

        raise RuntimeError(
            f"MockLlmClient: no pattern key matched the last user message. "
            f"Last user content: {last_content!r}. "
            f"Available keys: {list(self._responses.keys())!r}"
        )
