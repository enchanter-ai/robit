"""Tests for MockLlmClient.

Six mandatory cases:
  1. List-of-responses: returns them in order.
  2. Dict-of-patterns: matches request content against substring keys.
  3. No response configured raises a clear error.
  4. Records every request for assertions.
  5. Empty messages list raises.
  6. System prompt is recorded.
"""

from __future__ import annotations

import pytest

from robit.llm import CompletionRequest, CompletionResponse, Message, MockLlmClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response(text: str = "ok", model: str = "mock") -> CompletionResponse:
    return CompletionResponse(
        text=text,
        model=model,
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
    )


def _req(content: str = "hello", system: str | None = None) -> CompletionRequest:
    return CompletionRequest(
        model="mock-model",
        messages=[Message(role="user", content=content)],
        system=system,
    )


# ---------------------------------------------------------------------------
# Test 1 — list mode: responses returned in FIFO order
# ---------------------------------------------------------------------------

async def test_list_responses_returned_in_order():
    r1 = _response("first")
    r2 = _response("second")
    r3 = _response("third")
    client = MockLlmClient(responses=[r1, r2, r3])

    assert (await client.complete(_req("a"))).text == "first"
    assert (await client.complete(_req("b"))).text == "second"
    assert (await client.complete(_req("c"))).text == "third"


# ---------------------------------------------------------------------------
# Test 2 — dict mode: matches last user message against substring keys
# ---------------------------------------------------------------------------

async def test_dict_pattern_matches_last_user_message():
    r_search = _response("search-result")
    r_write = _response("write-result")
    client = MockLlmClient(responses={"search": r_search, "write": r_write})

    req = CompletionRequest(
        model="mock",
        messages=[
            Message(role="user", content="please search the internet"),
        ],
    )
    resp = await client.complete(req)
    assert resp.text == "search-result"

    req2 = CompletionRequest(
        model="mock",
        messages=[
            Message(role="user", content="please write a file"),
        ],
    )
    resp2 = await client.complete(req2)
    assert resp2.text == "write-result"


# ---------------------------------------------------------------------------
# Test 3 — no response configured raises a clear error
# ---------------------------------------------------------------------------

async def test_exhausted_list_raises_runtime_error():
    client = MockLlmClient(responses=[_response("only")])
    await client.complete(_req("first call"))  # consumes the one response

    with pytest.raises(RuntimeError, match="no response configured for call"):
        await client.complete(_req("second call"))


async def test_no_dict_match_raises_runtime_error():
    client = MockLlmClient(responses={"needle": _response()})

    with pytest.raises(RuntimeError, match="no pattern key matched"):
        await client.complete(_req("haystack with no match"))


async def test_empty_responses_list_raises_on_first_call():
    client = MockLlmClient(responses=[])

    with pytest.raises(RuntimeError, match="no response configured"):
        await client.complete(_req())


# ---------------------------------------------------------------------------
# Test 4 — records every request for later assertions
# ---------------------------------------------------------------------------

async def test_all_requests_are_recorded():
    responses = [_response(str(i)) for i in range(3)]
    client = MockLlmClient(responses=responses)

    contents = ["alpha", "beta", "gamma"]
    for c in contents:
        await client.complete(_req(c))

    assert len(client.requests) == 3
    for i, c in enumerate(contents):
        assert client.requests[i].messages[0].content == c


# ---------------------------------------------------------------------------
# Test 5 — empty messages list raises ValueError (not a network call)
# ---------------------------------------------------------------------------

def test_empty_messages_raises_value_error():
    """CompletionRequest itself rejects empty messages at construction."""
    with pytest.raises(ValueError, match="messages must not be empty"):
        CompletionRequest(model="m", messages=[])


# ---------------------------------------------------------------------------
# Test 6 — system prompt is recorded in the request
# ---------------------------------------------------------------------------

async def test_system_prompt_recorded():
    client = MockLlmClient(responses=[_response()])
    req = _req(content="what time is it", system="You are a helpful clock.")
    await client.complete(req)

    recorded = client.requests[0]
    assert recorded.system == "You are a helpful clock."
    assert recorded.messages[0].content == "what time is it"


# ---------------------------------------------------------------------------
# Bonus — dict mode: last user message wins (assistant turn in middle ignored)
# ---------------------------------------------------------------------------

async def test_dict_mode_uses_last_user_message():
    r_final = _response("final-match")
    client = MockLlmClient(responses={"final": r_final})

    req = CompletionRequest(
        model="mock",
        messages=[
            Message(role="user", content="ignore this unrelated first message"),
            Message(role="assistant", content="some assistant reply"),
            Message(role="user", content="this is the final user message"),
        ],
    )
    resp = await client.complete(req)
    assert resp.text == "final-match"
