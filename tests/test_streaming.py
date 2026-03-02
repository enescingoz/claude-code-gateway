"""Tests for streaming functionality — stream_adapter and streaming endpoint.

ClaudeRunner is fully mocked. No real CLI invocations.
"""

import json
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from src.services.stream_adapter import _extract_text, adapt_stream


async def _mock_ndjson_stream(lines: list[str]) -> AsyncIterator[str]:
    """Create a mock async iterator yielding NDJSON lines."""
    for line in lines:
        yield line


class TestExtractText:
    """Test _extract_text with all 3 event formats."""

    def test_content_block_delta_format(self):
        """Format 1: content_block_delta with text_delta."""
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        assert _extract_text(data) == "Hello"

    def test_content_block_delta_non_text(self):
        """content_block_delta with non-text delta returns None."""
        data = {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        }
        assert _extract_text(data) is None

    def test_stream_event_format(self):
        """Format 2: stream_event wrapping content_block_delta."""
        data = {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "World"},
            },
        }
        assert _extract_text(data) == "World"

    def test_assistant_message_format(self):
        """Format 3: assistant message with content array."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Final answer"}],
            },
        }
        assert _extract_text(data) == "Final answer"

    def test_assistant_message_empty_content(self):
        """assistant type with empty content returns None."""
        data = {"type": "assistant", "message": {"content": []}}
        assert _extract_text(data) is None

    def test_unknown_type_returns_none(self):
        """Unknown event types return None."""
        data = {"type": "ping", "timestamp": 12345}
        assert _extract_text(data) is None

    def test_empty_dict_returns_none(self):
        assert _extract_text({}) is None

    def test_assistant_message_multiple_content_blocks(self):
        """assistant type with multiple blocks returns last text block."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            },
        }
        assert _extract_text(data) == "second"


class TestAdaptStream:
    """Test adapt_stream converting NDJSON to SSE format."""

    @pytest.mark.asyncio
    async def test_first_chunk_has_role_assistant(self):
        """First SSE chunk should have role: assistant."""
        stream = _mock_ndjson_stream([])
        chunks = []
        async for chunk in adapt_stream(stream, "test-model"):
            chunks.append(chunk)

        # First data chunk (before [DONE])
        assert len(chunks) >= 2  # initial + final + [DONE]
        first_data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_events_start_with_data_prefix(self):
        """All SSE events start with 'data: '."""
        stream = _mock_ndjson_stream([
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hi"},
            })
        ])
        async for chunk in adapt_stream(stream, "test-model"):
            assert chunk.startswith("data: ")

    @pytest.mark.asyncio
    async def test_last_events_finish_reason_and_done(self):
        """Last events should have finish_reason: stop and [DONE]."""
        stream = _mock_ndjson_stream([])
        chunks = []
        async for chunk in adapt_stream(stream, "test-model"):
            chunks.append(chunk)

        # Second-to-last: finish_reason = stop
        final_data = json.loads(chunks[-2].removeprefix("data: ").strip())
        assert final_data["choices"][0]["finish_reason"] == "stop"

        # Last: [DONE]
        assert chunks[-1].strip() == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_content_chunks_from_ndjson(self):
        """Content from NDJSON events appears in SSE chunks."""
        ndjson_lines = [
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            }),
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "World"},
            }),
        ]
        stream = _mock_ndjson_stream(ndjson_lines)
        chunks = []
        async for chunk in adapt_stream(stream, "test-model"):
            chunks.append(chunk)

        # Should have: initial + 2 content + final + [DONE] = 5
        assert len(chunks) == 5

        # Check content chunks (index 1 and 2)
        chunk1 = json.loads(chunks[1].removeprefix("data: ").strip())
        assert chunk1["choices"][0]["delta"]["content"] == "Hello "

        chunk2 = json.loads(chunks[2].removeprefix("data: ").strip())
        assert chunk2["choices"][0]["delta"]["content"] == "World"

    @pytest.mark.asyncio
    async def test_invalid_json_lines_skipped(self):
        """Non-JSON lines in the NDJSON stream should be silently skipped."""
        ndjson_lines = [
            "this is not json",
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "OK"},
            }),
        ]
        stream = _mock_ndjson_stream(ndjson_lines)
        chunks = []
        async for chunk in adapt_stream(stream, "test-model"):
            chunks.append(chunk)

        # initial + 1 content + final + [DONE] = 4
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_non_text_events_skipped(self):
        """Events without extractable text should be skipped."""
        ndjson_lines = [
            json.dumps({"type": "ping"}),
            json.dumps({"type": "content_block_start", "content_block": {}}),
        ]
        stream = _mock_ndjson_stream(ndjson_lines)
        chunks = []
        async for chunk in adapt_stream(stream, "test-model"):
            chunks.append(chunk)

        # initial + final + [DONE] = 3 (no content chunks)
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_model_field_in_chunks(self):
        """All chunks should have the correct model field."""
        stream = _mock_ndjson_stream([])
        async for chunk in adapt_stream(stream, "claude-sonnet-4-20250514"):
            if chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk.removeprefix("data: ").strip())
            assert data["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_chunk_object_type(self):
        """All chunks should have object: chat.completion.chunk."""
        stream = _mock_ndjson_stream([])
        async for chunk in adapt_stream(stream, "test-model"):
            if chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk.removeprefix("data: ").strip())
            assert data["object"] == "chat.completion.chunk"


class TestStreamingEndpoint:
    """Test the streaming endpoint via HTTP client."""

    @pytest.mark.asyncio
    async def test_streaming_returns_event_stream_content_type(self, client):
        """Streaming request returns text/event-stream content type."""

        async def mock_streaming(*args, **kwargs):
            yield json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hi"},
            })

        with patch("src.routes.chat_completions.ClaudeRunner") as MockRunner:
            MockRunner.run_streaming = mock_streaming

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
