"""Tests for src.utils.message_formatter — OpenAI messages to prompt conversion."""

from src.utils.message_formatter import format_messages


class TestFormatMessages:
    """Verify messages are correctly split into prompt and system prompt."""

    def test_single_user_message_plain_text(self):
        """Single user message returns plain text without 'Human:' prefix."""
        messages = [{"role": "user", "content": "Hello world"}]
        prompt, system = format_messages(messages)
        assert prompt == "Hello world"
        assert system is None

    def test_multiple_messages_with_prefixes(self):
        """Multiple messages get Human:/Assistant: prefixes."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt, system = format_messages(messages)
        assert "Human: Hi" in prompt
        assert "Assistant: Hello!" in prompt
        assert "Human: How are you?" in prompt
        assert system is None

    def test_system_message_extracted(self):
        """System message should be extracted and returned as system_prompt."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompt, system = format_messages(messages)
        assert prompt == "Hello"
        assert system == "You are helpful."

    def test_mixed_roles_separation(self):
        """System messages separated from user/assistant conversation."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks"},
        ]
        prompt, system = format_messages(messages)
        assert system == "Be concise."
        assert "Human: What is 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "Human: Thanks" in prompt
        # System content should NOT appear in the conversation prompt
        assert "Be concise" not in prompt

    def test_empty_messages_list(self):
        """Empty messages list returns empty prompt and no system."""
        prompt, system = format_messages([])
        assert prompt == ""
        assert system is None

    def test_multiple_system_messages_joined(self):
        """Multiple system messages should be joined with newline."""
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Go"},
        ]
        prompt, system = format_messages(messages)
        assert system == "Rule 1\nRule 2"
        assert prompt == "Go"

    def test_messages_as_objects(self):
        """Messages can be objects with .role and .content attributes."""

        class Msg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            Msg("system", "Be nice"),
            Msg("user", "Hello"),
        ]
        prompt, system = format_messages(messages)
        assert prompt == "Hello"
        assert system == "Be nice"

    def test_messages_as_dicts(self):
        """Messages as dicts use .get() access."""
        messages = [{"role": "user", "content": "Test message"}]
        prompt, system = format_messages(messages)
        assert prompt == "Test message"
        assert system is None

    def test_conversation_parts_joined_with_double_newline(self):
        """Multiple conversation parts are joined with double newline."""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        prompt, system = format_messages(messages)
        assert prompt == "Human: A\n\nAssistant: B"
