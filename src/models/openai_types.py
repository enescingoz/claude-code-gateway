from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Tool-related models (OpenAI-compatible) ---


class FunctionParameters(BaseModel):
    """JSON Schema object describing function parameters."""
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Function definition within a tool."""
    name: str
    description: Optional[str] = None
    parameters: Optional[FunctionParameters] = None


class ToolDefinition(BaseModel):
    """A tool the model may call (currently only 'function' type)."""
    type: str = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function call made by the model."""
    name: str
    arguments: str  # JSON-encoded string of arguments


class ToolCall(BaseModel):
    """A tool call in the assistant response."""
    id: str
    type: str = "function"
    function: FunctionCall


# --- Message models ---


class ChatMessage(BaseModel):
    role: str  # system, user, assistant, tool
    content: Optional[str] = None
    # Tool call fields (assistant messages)
    tool_calls: Optional[list[ToolCall]] = None
    # Tool result fields (tool messages)
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[str] = None  # "auto", "none", or specific
    # Claude Code CLI extensions
    max_turns: Optional[int] = None
    working_dir: Optional[str] = None
    allowed_tools: Optional[str] = None
    session_id: Optional[str] = None
    permission_mode: Optional[str] = None
    append_system_prompt: Optional[str] = None
    verbose: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "anthropic"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
