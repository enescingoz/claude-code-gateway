from __future__ import annotations


def format_messages(messages: list[dict]) -> tuple[str, str | None]:
    """Convert OpenAI messages list to (prompt_text, system_prompt).

    Returns:
        tuple of (prompt_text, system_prompt_or_None)
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "") if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(f"Human: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    system_prompt = "\n".join(system_parts) if system_parts else None

    # Single user message with no history -> plain text (no prefix)
    if len(conversation_parts) == 1 and conversation_parts[0].startswith("Human: "):
        prompt_text = conversation_parts[0][len("Human: "):]
    else:
        prompt_text = "\n\n".join(conversation_parts)

    return prompt_text, system_prompt
