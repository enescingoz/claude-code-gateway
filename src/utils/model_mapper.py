from __future__ import annotations

MODEL_MAP: dict[str, str] = {
    "gpt-4": "claude-sonnet-4-20250514",
    "gpt-4o": "claude-sonnet-4-20250514",
    "gpt-4-turbo": "claude-sonnet-4-20250514",
    "gpt-3.5-turbo": "claude-haiku-4-5-20251001",
    "gpt-4o-mini": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "haiku": "claude-haiku-4-5-20251001",
}

AVAILABLE_MODELS: list[dict[str, str]] = [
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
    {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
]


def resolve_model(model: str) -> str:
    """Map model name to Claude model ID. Pass through claude-* names unchanged."""
    if model.startswith("claude-"):
        return model
    return MODEL_MAP.get(model, model)  # fallback: pass through as-is
