from __future__ import annotations

import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.config import settings
from src.models.openai_types import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from src.services.claude_runner import ClaudeRunner
from src.services.stream_adapter import adapt_stream
from src.utils.message_formatter import format_messages
from src.utils.model_mapper import resolve_model

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint backed by Claude Code CLI."""
    model = resolve_model(request.model)
    prompt, system_prompt = format_messages(request.messages)

    # Merge system prompt sources: extracted from messages + explicit extension field
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if request.append_system_prompt:
        parts.append(request.append_system_prompt)
    effective_system = "\n".join(parts) if parts else None

    if request.stream:
        return await _handle_streaming(request, model, prompt, effective_system)
    return await _handle_blocking(request, model, prompt, effective_system)


async def _handle_blocking(
    request: ChatCompletionRequest,
    model: str,
    prompt: str,
    system_prompt: str | None,
) -> dict:
    """Run blocking claude invocation and return OpenAI-formatted response dict."""
    result = await ClaudeRunner.run_blocking(
        prompt=prompt,
        model=model,
        timeout=settings.claude_cli_timeout,
        max_turns=request.max_turns or settings.default_max_turns,
        working_dir=request.working_dir or settings.working_dir or None,
        session_id=request.session_id,
        permission_mode=request.permission_mode,
        append_system_prompt=system_prompt,
        allowed_tools=request.allowed_tools,
    )

    content = result.get("result", "") if isinstance(result, dict) else str(result)
    session_id = result.get("session_id") if isinstance(result, dict) else None

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )

    resp_dict = response.model_dump()
    if session_id:
        resp_dict["session_id"] = session_id

    return resp_dict


async def _handle_streaming(
    request: ChatCompletionRequest,
    model: str,
    prompt: str,
    system_prompt: str | None,
) -> StreamingResponse:
    """Run streaming claude invocation and return SSE StreamingResponse."""
    ndjson_stream = ClaudeRunner.run_streaming(
        prompt=prompt,
        model=model,
        max_turns=request.max_turns or settings.default_max_turns,
        working_dir=request.working_dir or settings.working_dir or None,
        session_id=request.session_id,
        permission_mode=request.permission_mode,
        append_system_prompt=system_prompt,
        allowed_tools=request.allowed_tools,
    )

    sse_stream = adapt_stream(ndjson_stream, model)

    return StreamingResponse(
        sse_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
