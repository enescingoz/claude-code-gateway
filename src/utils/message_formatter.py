from __future__ import annotations

import json
from typing import Any


def format_tools_prompt(tools: list[dict | Any]) -> str:
    """Convert OpenAI-format tool definitions into a text prompt section.

    Returns a string to append to the system prompt instructing Claude
    how to call tools with structured JSON output.
    """
    lines = [
        "You have access to the following tools. When you need to call a tool, "
        "respond with ONLY a JSON block in this exact format, with no other text "
        "before or after it:",
        "",
        '{"tool_calls": [{"name": "function_name", "arguments": {"arg1": "value1"}}]}',
        "",
        "You may call multiple tools at once by including multiple entries in the "
        "tool_calls array.",
        "",
        "Available tools:",
    ]

    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", {})
        else:
            func = tool.function if hasattr(tool, "function") else {}

        if isinstance(func, dict):
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
        else:
            name = getattr(func, "name", "unknown")
            desc = getattr(func, "description", "") or ""
            params = getattr(func, "parameters", None)
            if params and not isinstance(params, dict):
                params = params.model_dump() if hasattr(params, "model_dump") else {}

        line = f"- {name}"
        if desc:
            line += f": {desc}"
        lines.append(line)

        if params:
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []
            if props:
                lines.append(f"  Parameters: {json.dumps(props)}")
            if required:
                lines.append(f"  Required: {json.dumps(required)}")

    return "\n".join(lines)


def format_messages(
    messages: list[dict | Any],
    tools: list[dict | Any] | None = None,
) -> tuple[str, str | None]:
    """Convert OpenAI messages list to (prompt_text, system_prompt).

    When *tools* are provided, a tool-description block is prepended to the
    system prompt so Claude knows how to format structured tool calls.

    Returns:
        tuple of (prompt_text, system_prompt_or_None)
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        if role == "system":
            if content:
                system_parts.append(content)

        elif role == "user":
            conversation_parts.append(f"Human: {content}")

        elif role == "assistant":
            # Handle assistant messages that contain tool calls
            tool_calls = (
                msg.get("tool_calls") if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None)
            )
            if tool_calls:
                # Reconstruct the tool call JSON so Claude sees what it previously said
                calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                        args_str = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                    else:
                        name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                        args_str = getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = args_str
                    calls.append({"name": name, "arguments": args})
                tool_json = json.dumps({"tool_calls": calls})
                conversation_parts.append(f"Assistant: {tool_json}")
            elif content:
                conversation_parts.append(f"Assistant: {content}")

        elif role == "tool":
            # Tool result message — format as a tool response for Claude
            tool_call_id = (
                msg.get("tool_call_id", "") if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            content_str = content or ""
            conversation_parts.append(
                f"Human: [Tool result for call {tool_call_id}]: {content_str}"
            )

    # Build system prompt with optional tool definitions
    if tools:
        tool_prompt = format_tools_prompt(tools)
        system_parts.insert(0, tool_prompt)

    system_prompt = "\n".join(system_parts) if system_parts else None

    # Single user message with no history -> plain text (no prefix)
    if len(conversation_parts) == 1 and conversation_parts[0].startswith("Human: "):
        prompt_text = conversation_parts[0][len("Human: "):]
    else:
        prompt_text = "\n\n".join(conversation_parts)

    return prompt_text, system_prompt
