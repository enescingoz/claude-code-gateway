from __future__ import annotations

import json
import re
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import settings
from src.models.openai_types import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from src.services.claude_runner import ClaudeRunner
from src.services.stream_adapter import adapt_stream
from src.utils.message_formatter import format_messages
from src.utils.model_mapper import resolve_model

router = APIRouter()

# Pattern to detect a tool_calls JSON block in Claude's response text.
# Matches a JSON object containing a "tool_calls" key, possibly surrounded
# by whitespace or markdown code fences.
_TOOL_CALLS_RE = re.compile(
    r"(?:```(?:json)?\s*)?"        # optional opening code fence
    r"(\{[^{}]*\"tool_calls\"\s*:\s*\[.*?\]\s*\})"  # the JSON object
    r"(?:\s*```)?"                  # optional closing code fence
    , re.DOTALL
)


def _generate_call_id() -> str:
    """Generate a unique tool call ID like ``call_abc123def456``."""
    return f"call_{uuid.uuid4().hex[:12]}"


def _parse_tool_calls(text: str) -> tuple[list[ToolCall] | None, str | None]:
    """Try to extract tool call JSON from Claude's response text.

    Returns:
        (tool_calls_list, remaining_text)
        - If tool calls found: (list[ToolCall], remaining_text or None)
        - If no tool calls: (None, original_text)
    """
    if not text:
        return None, text

    match = _TOOL_CALLS_RE.search(text)
    if not match:
        return None, text

    json_str = match.group(1)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, text

    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None, text

    tool_calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name", "")
        arguments = raw.get("arguments", {})
        # arguments must be a JSON-encoded string per OpenAI spec
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(
            ToolCall(
                id=_generate_call_id(),
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )

    if not tool_calls:
        return None, text

    # Extract any text outside the tool call block
    remaining = text[:match.start()] + text[match.end():]
    remaining = remaining.strip()
    # Also strip leftover code fence markers
    remaining = re.sub(r"^```(?:json)?\s*", "", remaining)
    remaining = re.sub(r"\s*```$", "", remaining)
    remaining = remaining.strip() or None

    return tool_calls, remaining


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint backed by Claude Code CLI."""
    model = resolve_model(request.model)

    # Convert tools to dicts for the formatter
    tools_dicts = None
    if request.tools:
        tools_dicts = [t.model_dump() for t in request.tools]

    prompt, system_prompt = format_messages(request.messages, tools=tools_dicts)

    # Merge system prompt sources: extracted from messages + explicit extension field
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if request.append_system_prompt:
        parts.append(request.append_system_prompt)
    effective_system = "\n".join(parts) if parts else None

    if request.stream:
        return await _handle_streaming(request, model, prompt, effective_system)
    return await _handle_blocking(request, model, prompt, effective_system, has_tools=bool(request.tools))


async def _handle_blocking(
    request: ChatCompletionRequest,
    model: str,
    prompt: str,
    system_prompt: str | None,
    has_tools: bool = False,
) -> dict:
    """Run blocking claude invocation and return OpenAI-formatted response dict."""
    try:
        if has_tools:
            result = await ClaudeRunner.run_blocking_with_tools(
                prompt=prompt,
                model=model,
                timeout=settings.claude_cli_timeout,
                max_turns=request.max_turns or 1,
                working_dir=request.working_dir or settings.working_dir or None,
                session_id=request.session_id,
                permission_mode=request.permission_mode,
                append_system_prompt=system_prompt,
                allowed_tools=request.allowed_tools,
            )
        else:
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
    except (RuntimeError, TimeoutError) as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "cli_error"}},
        )

    content = result.get("result", "") if isinstance(result, dict) else str(result)
    session_id = result.get("session_id") if isinstance(result, dict) else None

    # Try to parse tool calls from the response when tools were provided
    tool_calls = None
    finish_reason = "stop"

    if has_tools and content:
        tool_calls, remaining_text = _parse_tool_calls(content)
        if tool_calls:
            content = remaining_text  # None or leftover text
            finish_reason = "tool_calls"

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )

    resp_dict = response.model_dump(exclude_none=True)
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
