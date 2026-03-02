"""Tests for src.routes.chat_completions — OpenAI-compatible chat endpoint.

ClaudeRunner is fully mocked. No real CLI invocations.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestChatCompletionsBlocking:
    """Test non-streaming /v1/chat/completions requests."""

    @pytest.mark.asyncio
    async def test_returns_valid_response_structure(self, client):
        """Response has required OpenAI ChatCompletion fields."""
        mock_result = {
            "type": "result",
            "result": "Mocked response",
            "session_id": None,
        }

        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(return_value=mock_result)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_response_has_correct_model(self, client):
        """Response model field reflects the resolved Claude model."""
        mock_result = {"type": "result", "result": "Hi", "session_id": None}

        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(return_value=mock_result)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        data = resp.json()
        assert data["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_response_has_assistant_message(self, client):
        """Choices contain an assistant message with the CLI output."""
        mock_result = {"type": "result", "result": "Test reply", "session_id": None}

        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(return_value=mock_result)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        data = resp.json()
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Test reply"
        assert choice["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_runner_error_propagates(self, client):
        """RuntimeError from ClaudeRunner propagates as an exception."""
        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(
                side_effect=RuntimeError("CLI crashed")
            )

            with pytest.raises(RuntimeError, match="CLI crashed"):
                await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

    @pytest.mark.asyncio
    async def test_session_id_included_when_present(self, client):
        """Session ID from CLI result is included in the response."""
        mock_result = {
            "type": "result",
            "result": "Hello",
            "session_id": "sess-abc-123",
        }

        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(return_value=mock_result)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        data = resp.json()
        assert data.get("session_id") == "sess-abc-123"

    @pytest.mark.asyncio
    async def test_system_message_extraction(self, client):
        """System messages from the messages array are extracted and passed as system prompt."""
        mock_result = {"type": "result", "result": "OK", "session_id": None}

        with patch(
            "src.routes.chat_completions.ClaudeRunner"
        ) as MockRunner:
            MockRunner.run_blocking = AsyncMock(return_value=mock_result)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [
                        {"role": "system", "content": "Be concise"},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            )

        assert resp.status_code == 200
        # Verify the system prompt was passed to run_blocking
        call_kwargs = MockRunner.run_blocking.call_args.kwargs
        assert "Be concise" in (call_kwargs.get("append_system_prompt") or "")
