"""Tests for src.services.claude_runner — Claude CLI subprocess management.

ALL subprocess calls are mocked. No real CLI invocations.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.claude_runner import ClaudeRunner


class TestBuildEnv:
    """Verify environment construction strips CLAUDECODE."""

    def test_claudecode_stripped(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1", "PATH": "/usr/bin"}):
            env = ClaudeRunner._build_env()
            assert "CLAUDECODE" not in env
            assert "PATH" in env

    def test_other_vars_preserved(self):
        with patch.dict(os.environ, {"HOME": "/home/test", "LANG": "en_US"}, clear=True):
            env = ClaudeRunner._build_env()
            assert env["HOME"] == "/home/test"
            assert env["LANG"] == "en_US"


class TestBuildCmd:
    """Verify CLI command construction."""

    def test_basic_cmd(self):
        cmd = ClaudeRunner._build_cmd(model="claude-sonnet-4-20250514")
        assert cmd[:2] == ["claude", "-p"]
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--model" in cmd
        assert "claude-sonnet-4-20250514" in cmd

    def test_stream_json_adds_verbose(self):
        cmd = ClaudeRunner._build_cmd(
            model="claude-sonnet-4-20250514", output_format="stream-json"
        )
        assert "--verbose" in cmd
        assert "stream-json" in cmd

    def test_max_turns_flag(self):
        cmd = ClaudeRunner._build_cmd(model="test", max_turns=5)
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "5"

    def test_session_id_flag(self):
        cmd = ClaudeRunner._build_cmd(model="test", session_id="abc-123")
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_permission_mode_flag(self):
        cmd = ClaudeRunner._build_cmd(model="test", permission_mode="plan")
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "plan"

    def test_append_system_prompt_flag(self):
        cmd = ClaudeRunner._build_cmd(model="test", append_system_prompt="Be brief")
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Be brief"

    def test_allowed_tools_flag(self):
        cmd = ClaudeRunner._build_cmd(model="test", allowed_tools="Read,Write")
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Write"

    def test_verbose_flag_non_stream(self):
        cmd = ClaudeRunner._build_cmd(model="test", verbose=True)
        assert "--verbose" in cmd

    def test_verbose_not_duplicated_for_stream_json(self):
        """stream-json already adds --verbose; explicit verbose=True should not duplicate."""
        cmd = ClaudeRunner._build_cmd(
            model="test", output_format="stream-json", verbose=True
        )
        assert cmd.count("--verbose") == 1

    def test_all_flags_combined(self):
        cmd = ClaudeRunner._build_cmd(
            model="claude-opus-4-20250514",
            output_format="json",
            max_turns=3,
            session_id="sess-1",
            permission_mode="plan",
            append_system_prompt="system text",
            allowed_tools="Bash",
            verbose=True,
        )
        assert "--max-turns" in cmd
        assert "--resume" in cmd
        assert "--permission-mode" in cmd
        assert "--append-system-prompt" in cmd
        assert "--allowedTools" in cmd
        assert "--verbose" in cmd


class TestRunBlocking:
    """Verify run_blocking parses subprocess output correctly."""

    @staticmethod
    def _make_mock_proc(stdout_data: str, returncode: int = 0, stderr_data: str = ""):
        """Create a mock async subprocess."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(stdout_data.encode(), stderr_data.encode())
        )
        mock_proc.returncode = returncode
        mock_proc.kill = MagicMock()
        return mock_proc

    @pytest.mark.asyncio
    async def test_json_array_with_result_element(self):
        """CLI returns JSON array; extract the result-typed element."""
        result_data = json.dumps([
            {"type": "system", "data": "..."},
            {"type": "result", "result": "Hello from Claude", "session_id": "s1"},
        ])
        mock_proc = self._make_mock_proc(result_data)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

        assert result["type"] == "result"
        assert result["result"] == "Hello from Claude"
        assert result["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_single_json_object(self):
        """CLI returns a single JSON object (not an array)."""
        result_data = json.dumps(
            {"type": "result", "result": "Direct response", "session_id": None}
        )
        mock_proc = self._make_mock_proc(result_data)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

        assert result["type"] == "result"
        assert result["result"] == "Direct response"

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_runtime_error(self):
        """Non-zero exit code should raise RuntimeError."""
        mock_proc = self._make_mock_proc(
            stdout_data="", returncode=1, stderr_data="Some error"
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Some error"):
                await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        """Subprocess timeout should raise TimeoutError."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.returncode = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError, match="timed out"):
                await ClaudeRunner.run_blocking(
                    prompt="Hi", model="test-model", timeout=1
                )
            mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_text_fallback(self):
        """Non-JSON output falls back to plain text result."""
        mock_proc = self._make_mock_proc("This is not JSON at all")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

        assert result["type"] == "result"
        assert result["result"] == "This is not JSON at all"

    @pytest.mark.asyncio
    async def test_json_array_no_result_element(self):
        """JSON array without a result-typed element returns last item."""
        result_data = json.dumps([
            {"type": "system", "data": "init"},
            {"type": "log", "message": "done"},
        ])
        mock_proc = self._make_mock_proc(result_data)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

        assert result["type"] == "log"
        assert result["message"] == "done"

    @pytest.mark.asyncio
    async def test_empty_json_array_fallback(self):
        """Empty JSON array returns fallback dict."""
        mock_proc = self._make_mock_proc("[]")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")

        assert result["type"] == "result"

    @pytest.mark.asyncio
    async def test_nonzero_exit_empty_stderr_uses_exit_code(self):
        """Non-zero exit with empty stderr should mention exit code."""
        mock_proc = self._make_mock_proc(stdout_data="", returncode=42, stderr_data="")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="exited with code 42"):
                await ClaudeRunner.run_blocking(prompt="Hi", model="test-model")
