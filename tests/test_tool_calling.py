"""Tests for OpenAI-compatible tool/function calling support.

Covers:
- Tool-related Pydantic models (openai_types.py)
- Tool prompt formatting and tool message handling (message_formatter.py)
- Tool call parsing from Claude responses (chat_completions.py)
"""

import json

import pytest

from src.models.openai_types import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    FunctionParameters,
    ToolCall,
    ToolDefinition,
)
from src.routes.chat_completions import _parse_tool_calls
from src.utils.message_formatter import format_messages, format_tools_prompt


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestToolModels:
    """Pydantic model serialization for tool-related types."""

    def test_chat_message_content_optional(self):
        """ChatMessage.content can be None (required for tool call responses)."""
        msg = ChatMessage(role="assistant", content=None)
        assert msg.content is None

    def test_chat_message_with_tool_calls(self):
        """ChatMessage can carry tool_calls list."""
        tc = ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(name="f", arguments="{}"),
        )
        msg = ChatMessage(role="assistant", content=None, tool_calls=[tc])
        dumped = msg.model_dump(exclude_none=True)
        assert dumped["tool_calls"][0]["function"]["name"] == "f"
        assert "content" not in dumped

    def test_chat_message_with_tool_call_id(self):
        """ChatMessage supports tool_call_id for tool-result messages."""
        msg = ChatMessage(role="tool", content='{"ok": true}', tool_call_id="call_1")
        dumped = msg.model_dump(exclude_none=True)
        assert dumped["role"] == "tool"
        assert dumped["tool_call_id"] == "call_1"

    def test_tool_definition_round_trip(self):
        """ToolDefinition serializes and deserializes correctly."""
        td = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_stock",
                description="Get stock price",
                parameters=FunctionParameters(
                    type="object",
                    properties={"ticker": {"type": "string"}},
                    required=["ticker"],
                ),
            ),
        )
        dumped = td.model_dump()
        assert dumped["function"]["name"] == "get_stock"
        assert "ticker" in dumped["function"]["parameters"]["properties"]

    def test_request_accepts_tools(self):
        """ChatCompletionRequest accepts tools parameter."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[
                ToolDefinition(
                    type="function",
                    function=FunctionDefinition(name="test_fn"),
                )
            ],
        )
        assert len(req.tools) == 1
        assert req.tools[0].function.name == "test_fn"

    def test_request_without_tools_backward_compatible(self):
        """Requests without tools work exactly as before."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.tools is None


# ---------------------------------------------------------------------------
# Tool prompt formatting tests
# ---------------------------------------------------------------------------


class TestFormatToolsPrompt:
    """Tests for format_tools_prompt()."""

    def test_basic_tool_prompt(self):
        """Tool descriptions appear in the prompt."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather forecast",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        result = format_tools_prompt(tools)
        assert "get_weather" in result
        assert "Get weather forecast" in result
        assert "tool_calls" in result  # instruction format

    def test_multiple_tools(self):
        """Multiple tools all appear in the prompt."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A"}},
            {"type": "function", "function": {"name": "tool_b", "description": "B"}},
        ]
        result = format_tools_prompt(tools)
        assert "tool_a" in result
        assert "tool_b" in result

    def test_tool_without_description(self):
        """Tools without descriptions still appear."""
        tools = [{"type": "function", "function": {"name": "bare_tool"}}]
        result = format_tools_prompt(tools)
        assert "bare_tool" in result


class TestFormatMessagesWithTools:
    """Tests for format_messages() when tools are provided."""

    def test_tools_added_to_system_prompt(self):
        """Tool definitions are prepended to the system prompt."""
        tools = [
            {
                "type": "function",
                "function": {"name": "my_tool", "description": "Does stuff"},
            }
        ]
        prompt, system = format_messages(
            [{"role": "user", "content": "Do it"}], tools=tools
        )
        assert system is not None
        assert "my_tool" in system
        assert prompt == "Do it"

    def test_tools_combined_with_system_message(self):
        """Tool prompt is prepended to existing system messages."""
        tools = [
            {"type": "function", "function": {"name": "fn1", "description": "X"}}
        ]
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        prompt, system = format_messages(messages, tools=tools)
        assert "fn1" in system
        assert "Be helpful." in system
        # Tool section should come first
        assert system.index("fn1") < system.index("Be helpful.")

    def test_no_tools_backward_compatible(self):
        """Without tools, behavior is unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        prompt, system = format_messages(messages)
        assert prompt == "Hello"
        assert system is None

    def test_tool_role_message_in_conversation(self):
        """Tool result messages (role=tool) are formatted into conversation."""
        messages = [
            {"role": "user", "content": "Get data"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_data",
                            "arguments": '{"id": 42}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"result": "ok"}',
            },
        ]
        prompt, system = format_messages(messages)
        assert "Tool result" in prompt
        assert "call_1" in prompt
        assert '{"result": "ok"}' in prompt

    def test_assistant_tool_call_in_conversation(self):
        """Assistant messages with tool_calls are reconstructed in conversation."""
        messages = [
            {"role": "user", "content": "Help"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            },
        ]
        prompt, system = format_messages(messages)
        assert "search" in prompt
        assert "Assistant:" in prompt


# ---------------------------------------------------------------------------
# Tool call parsing tests
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for _parse_tool_calls() in chat_completions.py."""

    def test_parse_single_tool_call(self):
        """Single tool call JSON is parsed correctly."""
        text = '{"tool_calls": [{"name": "get_stock", "arguments": {"ticker": "AAPL"}}]}'
        calls, remaining = _parse_tool_calls(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function.name == "get_stock"
        assert json.loads(calls[0].function.arguments) == {"ticker": "AAPL"}
        assert calls[0].id.startswith("call_")
        assert remaining is None

    def test_parse_multiple_tool_calls(self):
        """Multiple tool calls in one response are all parsed."""
        text = json.dumps(
            {
                "tool_calls": [
                    {"name": "fn_a", "arguments": {"x": 1}},
                    {"name": "fn_b", "arguments": {"y": 2}},
                ]
            }
        )
        calls, remaining = _parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].function.name == "fn_a"
        assert calls[1].function.name == "fn_b"

    def test_no_tool_calls_returns_none(self):
        """Regular text without tool calls returns None."""
        text = "This is a regular response with no tool calls."
        calls, remaining = _parse_tool_calls(text)
        assert calls is None
        assert remaining == text

    def test_parse_with_surrounding_text(self):
        """Tool call JSON surrounded by text extracts calls and remaining text."""
        text = 'Let me look that up.\n\n{"tool_calls": [{"name": "search", "arguments": {"q": "test"}}]}\n\nDone.'
        calls, remaining = _parse_tool_calls(text)
        assert calls is not None
        assert calls[0].function.name == "search"
        assert remaining is not None
        assert "Let me look that up." in remaining

    def test_parse_code_fenced_tool_call(self):
        """Tool calls wrapped in markdown code fences are parsed."""
        text = '```json\n{"tool_calls": [{"name": "func", "arguments": {}}]}\n```'
        calls, remaining = _parse_tool_calls(text)
        assert calls is not None
        assert calls[0].function.name == "func"

    def test_parse_empty_string(self):
        """Empty string returns None."""
        calls, remaining = _parse_tool_calls("")
        assert calls is None
        assert remaining == ""

    def test_parse_none_like_content(self):
        """None/falsy content is handled gracefully."""
        calls, remaining = _parse_tool_calls("")
        assert calls is None

    def test_arguments_serialized_as_string(self):
        """Arguments are always JSON-encoded strings per OpenAI spec."""
        text = '{"tool_calls": [{"name": "f", "arguments": {"key": "value"}}]}'
        calls, _ = _parse_tool_calls(text)
        assert isinstance(calls[0].function.arguments, str)
        assert json.loads(calls[0].function.arguments) == {"key": "value"}

    def test_unique_call_ids_generated(self):
        """Each tool call gets a unique ID."""
        text = json.dumps(
            {
                "tool_calls": [
                    {"name": "a", "arguments": {}},
                    {"name": "b", "arguments": {}},
                ]
            }
        )
        calls, _ = _parse_tool_calls(text)
        assert calls[0].id != calls[1].id

    def test_invalid_json_returns_none(self):
        """Malformed JSON is not parsed as tool calls."""
        text = '{"tool_calls": [broken json here]}'
        calls, remaining = _parse_tool_calls(text)
        assert calls is None
        assert remaining == text

    def test_tool_calls_key_but_empty_list(self):
        """tool_calls with empty list returns None."""
        text = '{"tool_calls": []}'
        calls, remaining = _parse_tool_calls(text)
        assert calls is None
        assert remaining == text
