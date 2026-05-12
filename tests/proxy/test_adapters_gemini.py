"""Tests for :mod:`enchanter.proxy.adapters.gemini` — wire translation."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from enchanter.proxy.adapters.gemini import AdapterParseError, GeminiAdapter
from enchanter.proxy.canonical import (
    CanonicalChunk,
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
    ToolResultPart,
    ToolUsePart,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


async def _aiter(items):
    for item in items:
        yield item


async def _collect(agen: AsyncIterator[bytes]) -> list[bytes]:
    out: list[bytes] = []
    async for piece in agen:
        out.append(piece)
    return out


def _parse_sse_data(events: list[bytes]) -> list[dict]:
    """Parse a list of ``data: <json>\\n\\n`` chunks back to dicts."""
    payloads: list[dict] = []
    for ev in events:
        decoded = ev.decode("utf-8")
        assert decoded.startswith("data: "), f"missing data: prefix in {decoded!r}"
        assert decoded.endswith("\n\n"), f"missing trailing blank line in {decoded!r}"
        body = decoded[len("data: ") : -2]
        payloads.append(json.loads(body))
    return payloads


# ---------------------------------------------------------------------------
# matches()
# ---------------------------------------------------------------------------


class TestMatches:
    def test_generate_content_path_matches(self):
        assert GeminiAdapter.matches(
            "POST", "/v1beta/models/gemini-1.5-flash:generateContent"
        )

    def test_stream_generate_content_path_matches(self):
        assert GeminiAdapter.matches(
            "POST", "/v1beta/models/gemini-1.5-pro:streamGenerateContent"
        )

    def test_matches_strips_query_string(self):
        assert GeminiAdapter.matches(
            "POST",
            "/v1beta/models/gemini-1.5-flash:generateContent?key=xyz",
        )

    def test_wrong_method_rejected(self):
        assert not GeminiAdapter.matches(
            "GET", "/v1beta/models/gemini-1.5-flash:generateContent"
        )

    def test_wrong_path_rejected(self):
        assert not GeminiAdapter.matches("POST", "/v1/messages")
        assert not GeminiAdapter.matches("POST", "/v1/chat/completions")

    def test_unknown_verb_rejected(self):
        assert not GeminiAdapter.matches(
            "POST", "/v1beta/models/gemini-1.5-flash:countTokens"
        )

    def test_missing_verb_rejected(self):
        assert not GeminiAdapter.matches(
            "POST", "/v1beta/models/gemini-1.5-flash"
        )


# ---------------------------------------------------------------------------
# parse_request()
# ---------------------------------------------------------------------------


class TestParseRequest:
    def test_extracts_model_from_path(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body,
            "/v1beta/models/gemini-1.5-flash:generateContent",
            {},
        )
        assert req.model == "gemini-1.5-flash"

    def test_stream_flag_false_for_generate_content(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.stream is False

    def test_stream_flag_true_for_stream_generate_content(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body,
            "/v1beta/models/gemini-1.5-flash:streamGenerateContent",
            {},
        )
        assert req.stream is True

    def test_path_with_query_string(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body,
            "/v1beta/models/gemini-1.5-flash:generateContent?key=abc",
            {},
        )
        assert req.model == "gemini-1.5-flash"
        assert req.stream is False

    def test_system_instruction_flattened_to_system_field(self):
        body = json.dumps(
            {
                "systemInstruction": {
                    "parts": [
                        {"text": "You are helpful."},
                        {"text": " Be concise."},
                    ]
                },
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.system == "You are helpful. Be concise."

    def test_no_system_instruction(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.system is None

    def test_role_normalisation_model_to_assistant(self):
        body = json.dumps(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "Hi"}]},
                    {"role": "model", "parts": [{"text": "Hello"}]},
                ]
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"

    def test_function_call_and_response_parts(self):
        body = json.dumps(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {"text": "Let me check."},
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "get_weather",
                                    "response": {"content": "Sunny."},
                                }
                            }
                        ],
                    },
                ]
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assistant_msg = req.messages[0]
        assert assistant_msg.role == "assistant"
        assert isinstance(assistant_msg.content[0], TextPart)
        assert isinstance(assistant_msg.content[1], ToolUsePart)
        tu = assistant_msg.content[1]
        assert tu.name == "get_weather"
        assert tu.input == {"city": "NYC"}
        # Synthesised id has the documented prefix.
        assert tu.id.startswith("tool_")

        user_msg = req.messages[1]
        assert user_msg.role == "user"
        assert isinstance(user_msg.content[0], ToolResultPart)
        tr = user_msg.content[0]
        # Gemini doesn't carry tool_use_ids on the wire — documented limitation.
        assert tr.tool_use_id == ""
        # Content is a JSON-stringified {name, response} wrapper.
        decoded = json.loads(tr.content)
        assert decoded["name"] == "get_weather"
        assert decoded["response"] == {"content": "Sunny."}

    def test_generation_config_fields_extracted(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "stopSequences": ["\n\n", "###"],
                },
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.temperature == 0.7
        assert req.top_p == 0.95
        assert req.max_tokens == 1024
        assert req.stop_sequences == ("\n\n", "###")

    def test_tool_config_auto(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.tool_choice == "auto"

    def test_tool_config_any(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.tool_choice == "any"

    def test_tool_config_any_with_single_allowed_function(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather"],
                    }
                },
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.tool_choice == {"type": "tool", "name": "get_weather"}

    def test_tool_config_none(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        assert req.tool_choice == "none"

    def test_tools_flattened_across_blocks(self):
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "a",
                                "description": "first",
                                "parameters": {"type": "object"},
                            },
                            {
                                "name": "b",
                                "description": "second",
                                "parameters": {"type": "object"},
                            },
                        ]
                    },
                    {
                        "functionDeclarations": [
                            {
                                "name": "c",
                                "description": "third",
                                "parameters": {"type": "object"},
                            }
                        ]
                    },
                ],
            }
        ).encode("utf-8")
        req = GeminiAdapter.parse_request(
            body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
        )
        names = [t.name for t in req.tools]
        assert names == ["a", "b", "c"]

    def test_invalid_json_raises_adapter_parse_error(self):
        with pytest.raises(AdapterParseError):
            GeminiAdapter.parse_request(
                b"not json{",
                "/v1beta/models/gemini-1.5-flash:generateContent",
                {},
            )

    def test_non_object_body_raises(self):
        with pytest.raises(AdapterParseError):
            GeminiAdapter.parse_request(
                b'["array body"]',
                "/v1beta/models/gemini-1.5-flash:generateContent",
                {},
            )

    def test_invalid_role_raises(self):
        body = json.dumps(
            {"contents": [{"role": "bogus", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        with pytest.raises(AdapterParseError):
            GeminiAdapter.parse_request(
                body, "/v1beta/models/gemini-1.5-flash:generateContent", {}
            )

    def test_invalid_path_raises(self):
        body = json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        ).encode("utf-8")
        with pytest.raises(AdapterParseError):
            GeminiAdapter.parse_request(body, "/wrong/path", {})


# ---------------------------------------------------------------------------
# render_response()
# ---------------------------------------------------------------------------


class TestRenderResponse:
    def test_plain_text_in_candidate_parts(self):
        resp = CanonicalResponse(
            model="gemini-1.5-flash",
            content=(TextPart(text="Hi there!"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=12, output_tokens=34),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        assert body["candidates"][0]["content"]["parts"] == [{"text": "Hi there!"}]
        assert body["candidates"][0]["content"]["role"] == "model"
        assert body["modelVersion"] == "gemini-1.5-flash"

    def test_function_call_part_emitted(self):
        resp = CanonicalResponse(
            model="gemini-1.5-flash",
            content=(
                TextPart(text="Calling tool."),
                ToolUsePart(id="tool_xx", name="search", input={"q": "foo"}),
            ),
            stop_reason="tool_use",
            usage=CanonicalUsage(input_tokens=1, output_tokens=2),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        parts = body["candidates"][0]["content"]["parts"]
        assert parts[0] == {"text": "Calling tool."}
        assert parts[1] == {
            "functionCall": {"name": "search", "args": {"q": "foo"}}
        }

    def test_finish_reason_mapping_end_turn(self):
        resp = CanonicalResponse(
            model="m",
            content=(TextPart(text="x"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        assert body["candidates"][0]["finishReason"] == "STOP"

    def test_finish_reason_mapping_max_tokens(self):
        resp = CanonicalResponse(
            model="m",
            content=(TextPart(text="x"),),
            stop_reason="max_tokens",
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        assert body["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_finish_reason_mapping_tool_use_becomes_stop(self):
        resp = CanonicalResponse(
            model="m",
            content=(TextPart(text="x"),),
            stop_reason="tool_use",
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        # Gemini has no dedicated tool finish; canonical tool_use → STOP.
        assert body["candidates"][0]["finishReason"] == "STOP"

    def test_finish_reason_mapping_none(self):
        resp = CanonicalResponse(
            model="m",
            content=(TextPart(text="x"),),
            stop_reason=None,
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        assert body["candidates"][0]["finishReason"] == "STOP"

    def test_usage_metadata_emitted(self):
        resp = CanonicalResponse(
            model="m",
            content=(TextPart(text="x"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=12, output_tokens=34),
        )
        body = json.loads(GeminiAdapter.render_response(resp))
        assert body["usageMetadata"] == {
            "promptTokenCount": 12,
            "candidatesTokenCount": 34,
            "totalTokenCount": 46,
        }


# ---------------------------------------------------------------------------
# render_stream()
# ---------------------------------------------------------------------------


class TestRenderStream:
    @pytest.mark.asyncio
    async def test_text_delta_events(self):
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(type="text_delta", index=0, text="Hi"),
            CanonicalChunk(type="text_delta", index=0, text=" there"),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(
                type="message_delta",
                stop_reason="end_turn",
                usage=CanonicalUsage(input_tokens=12, output_tokens=34),
            ),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        payloads = _parse_sse_data(events)
        # We expect at least 2 text events plus a final event with finishReason.
        # First two carry the text deltas.
        assert payloads[0]["candidates"][0]["content"]["parts"] == [{"text": "Hi"}]
        assert payloads[1]["candidates"][0]["content"]["parts"] == [
            {"text": " there"}
        ]

    @pytest.mark.asyncio
    async def test_final_event_has_finish_reason_and_usage(self):
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(type="text_delta", index=0, text="Hi"),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(
                type="message_delta",
                stop_reason="end_turn",
                usage=CanonicalUsage(input_tokens=12, output_tokens=34),
            ),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        payloads = _parse_sse_data(events)
        # The final event must carry both finishReason and usageMetadata.
        last = payloads[-1]
        assert last["candidates"][0].get("finishReason") == "STOP"
        assert last["usageMetadata"] == {
            "promptTokenCount": 12,
            "candidatesTokenCount": 34,
            "totalTokenCount": 46,
        }

    @pytest.mark.asyncio
    async def test_no_done_sentinel_emitted(self):
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(type="text_delta", index=0, text="Hi"),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(
                type="message_delta",
                stop_reason="end_turn",
                usage=CanonicalUsage(input_tokens=1, output_tokens=2),
            ),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        # No event may carry the OpenAI/Anthropic-style [DONE] sentinel.
        for ev in events:
            decoded = ev.decode("utf-8")
            assert "[DONE]" not in decoded

    @pytest.mark.asyncio
    async def test_every_event_is_sse_data_framed(self):
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(type="text_delta", index=0, text="A"),
            CanonicalChunk(type="text_delta", index=0, text="B"),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(type="message_delta", stop_reason="end_turn"),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        assert events  # at least one event emitted
        for ev in events:
            decoded = ev.decode("utf-8")
            assert decoded.startswith("data: ")
            assert decoded.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_final_event_synthesised_without_message_delta(self):
        """Stream that ends with no message_delta still emits a finishReason."""
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(type="text_delta", index=0, text="Hi"),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        payloads = _parse_sse_data(events)
        last = payloads[-1]
        assert last["candidates"][0].get("finishReason") == "STOP"

    @pytest.mark.asyncio
    async def test_tool_use_emitted_as_single_function_call_event(self):
        """A tool_use block emits exactly one functionCall event on stop."""
        tool_args_json = json.dumps(
            {"name": "search", "args": {"q": "foo"}}
        )
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(type="content_block_start", index=0),
            CanonicalChunk(
                type="input_json_delta",
                index=0,
                partial_json=tool_args_json,
            ),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(
                type="message_delta",
                stop_reason="tool_use",
                usage=CanonicalUsage(input_tokens=1, output_tokens=2),
            ),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        payloads = _parse_sse_data(events)
        # Exactly one event carries the functionCall (whole thing in one shot).
        function_call_events = [
            p
            for p in payloads
            if any(
                "functionCall" in part
                for part in p["candidates"][0]["content"]["parts"]
            )
        ]
        assert len(function_call_events) == 1
        fc = function_call_events[0]["candidates"][0]["content"]["parts"][0][
            "functionCall"
        ]
        assert fc == {"name": "search", "args": {"q": "foo"}}

    @pytest.mark.asyncio
    async def test_tool_use_uses_tool_name_from_content_block_start(self):
        """When content_block_start carries block_kind='tool_use' and
        tool_name, the emitted functionCall uses that name and treats
        accumulated input_json_delta fragments as RAW args JSON (not a
        self-describing wrapper) — Wave 1 contract-fix."""
        chunks = [
            CanonicalChunk(type="message_start"),
            CanonicalChunk(
                type="content_block_start",
                index=0,
                block_kind="tool_use",
                tool_id="tool_xyz",
                tool_name="get_weather",
            ),
            CanonicalChunk(
                type="input_json_delta",
                index=0,
                partial_json='{"city":',
            ),
            CanonicalChunk(
                type="input_json_delta",
                index=0,
                partial_json='"Paris"}',
            ),
            CanonicalChunk(type="content_block_stop", index=0),
            CanonicalChunk(
                type="message_delta",
                stop_reason="tool_use",
                usage=CanonicalUsage(input_tokens=1, output_tokens=2),
            ),
            CanonicalChunk(type="message_stop"),
        ]
        events = await _collect(GeminiAdapter.render_stream(_aiter(chunks)))
        payloads = _parse_sse_data(events)
        function_call_events = [
            p
            for p in payloads
            if any(
                "functionCall" in part
                for part in p["candidates"][0]["content"]["parts"]
            )
        ]
        assert len(function_call_events) == 1
        fc = function_call_events[0]["candidates"][0]["content"]["parts"][0][
            "functionCall"
        ]
        assert fc == {"name": "get_weather", "args": {"city": "Paris"}}
