from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import AsyncIterator


class ClaudeRunner:
    @staticmethod
    def _build_env() -> dict[str, str]:
        """Build environment with CLAUDECODE stripped to prevent nested session errors."""
        return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    @staticmethod
    def _build_cmd(
        model: str,
        output_format: str = "json",
        max_turns: int | None = None,
        session_id: str | None = None,
        permission_mode: str | None = None,
        append_system_prompt: str | None = None,
        allowed_tools: str | None = None,
        verbose: bool = False,
    ) -> list[str]:
        cmd = ["claude", "-p", "--output-format", output_format, "--model", model]
        if output_format == "stream-json":
            cmd.extend(["--verbose"])
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])
        if session_id:
            cmd.extend(["--resume", session_id])
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])
        if append_system_prompt:
            cmd.extend(["--append-system-prompt", append_system_prompt])
        if allowed_tools:
            cmd.extend(["--allowedTools", allowed_tools])
        if verbose and output_format != "stream-json":
            cmd.extend(["--verbose"])
        return cmd

    @staticmethod
    async def run_blocking(
        prompt: str,
        model: str,
        timeout: int = 300,
        max_turns: int | None = None,
        working_dir: str | None = None,
        session_id: str | None = None,
        permission_mode: str | None = None,
        append_system_prompt: str | None = None,
        allowed_tools: str | None = None,
    ) -> dict:
        """Run claude CLI and return parsed result dict."""
        cmd = ClaudeRunner._build_cmd(
            model=model,
            output_format="json",
            max_turns=max_turns,
            session_id=session_id,
            permission_mode=permission_mode,
            append_system_prompt=append_system_prompt,
            allowed_tools=allowed_tools,
        )
        cwd = working_dir or tempfile.mkdtemp(prefix="claude-api-")
        env = ClaudeRunner._build_env()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Claude CLI timed out after {timeout}s")

        if proc.returncode != 0:
            error = stderr.decode().strip() or f"CLI exited with code {proc.returncode}"
            raise RuntimeError(error)

        raw = stdout.decode().strip()
        # Parse JSON output — may be a JSON array; extract the result element
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat raw output as plain text result
            return {"type": "result", "result": raw, "session_id": None}

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "result":
                    return item
            # No result-typed element found; return last item or empty fallback
            return data[-1] if data else {"type": "result", "result": raw, "session_id": None}

        return data

    @staticmethod
    async def run_streaming(
        prompt: str,
        model: str,
        max_turns: int | None = None,
        working_dir: str | None = None,
        session_id: str | None = None,
        permission_mode: str | None = None,
        append_system_prompt: str | None = None,
        allowed_tools: str | None = None,
    ) -> AsyncIterator[str]:
        """Run claude CLI in streaming mode, yielding NDJSON lines."""
        cmd = ClaudeRunner._build_cmd(
            model=model,
            output_format="stream-json",
            max_turns=max_turns,
            session_id=session_id,
            permission_mode=permission_mode,
            append_system_prompt=append_system_prompt,
            allowed_tools=allowed_tools,
        )
        cwd = working_dir or tempfile.mkdtemp(prefix="claude-api-")
        env = ClaudeRunner._build_env()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        proc.stdin.write(prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        async for line in proc.stdout:
            decoded = line.decode().strip()
            if decoded:
                yield decoded

        await proc.wait()
