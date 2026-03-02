from __future__ import annotations

import time

from fastapi import APIRouter

from src.models.openai_types import ModelInfo, ModelListResponse
from src.utils.model_mapper import AVAILABLE_MODELS

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available Claude models in OpenAI-compatible format."""
    models = [
        ModelInfo(id=m["id"], created=int(time.time()), owned_by="anthropic")
        for m in AVAILABLE_MODELS
    ]
    return ModelListResponse(data=models)
