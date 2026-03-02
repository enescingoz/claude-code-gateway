from __future__ import annotations

import asyncio

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check if claude CLI is available and responding."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        version = stdout.decode().strip()
        return {"status": "ok", "claude_cli_version": version}
    except (FileNotFoundError, asyncio.TimeoutError):
        return {"status": "error", "detail": "claude CLI not found or not responding"}
