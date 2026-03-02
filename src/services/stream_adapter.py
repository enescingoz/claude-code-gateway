from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator


async def adapt_stream(
    ndjson_stream: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    """Convert Claude CLI NDJSON stream to OpenAI SSE format.

    Filters for content_block_delta events with text_delta,
    wraps each in ChatCompletionChunk format, yields as SSE lines.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Send initial chunk with role
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    async for line in ndjson_stream:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = _extract_text(data)
        if text is None:
            continue

        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk with finish_reason=stop
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _extract_text(data: dict) -> str | None:
    """Extract text content from various Claude CLI stream event formats."""
    # Format 1: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
    if data.get("type") == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")

    # Format 2: {"type": "stream_event", "event": {"type": "content_block_delta", ...}}
    if data.get("type") == "stream_event":
        event = data.get("event", {})
        return _extract_text(event)

    # Format 3: {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
    if data.get("type") == "assistant":
        message = data.get("message", {})
        content = message.get("content", [])
        if content and isinstance(content, list):
            last = content[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                return last.get("text")

    return None
