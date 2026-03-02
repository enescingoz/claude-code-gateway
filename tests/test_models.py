"""Tests for /v1/models and /health endpoints."""

from unittest.mock import AsyncMock, patch

import pytest


class TestModelsEndpoint:
    """Test GET /v1/models."""

    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_object_is_list(self, client):
        resp = await client.get("/v1/models")
        data = resp.json()
        assert data["object"] == "list"

    @pytest.mark.asyncio
    async def test_response_contains_model_entries(self, client):
        resp = await client.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_model_entry_structure(self, client):
        resp = await client.get("/v1/models")
        data = resp.json()
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "anthropic"

    @pytest.mark.asyncio
    async def test_contains_expected_models(self, client):
        """Should include sonnet, opus, and haiku."""
        resp = await client.get("/v1/models")
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "claude-sonnet-4-20250514" in model_ids
        assert "claude-opus-4-20250514" in model_ids
        assert "claude-haiku-4-5-20251001" in model_ids


class TestHealthEndpoint:
    """Test GET /health with mocked claude CLI."""

    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        """Health check returns ok when claude CLI is available."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"claude-code 1.0.0", b"")
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "claude_cli_version" in data

    @pytest.mark.asyncio
    async def test_health_cli_not_found(self, client):
        """Health check returns error when claude CLI is not available."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "not found" in data["detail"]
