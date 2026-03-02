"""Tests for src.utils.model_mapper — model name resolution."""

from src.utils.model_mapper import resolve_model


class TestResolveModel:
    """Verify OpenAI model names map to correct Claude model IDs."""

    def test_gpt4_maps_to_sonnet(self):
        assert resolve_model("gpt-4") == "claude-sonnet-4-20250514"

    def test_gpt4o_maps_to_sonnet(self):
        assert resolve_model("gpt-4o") == "claude-sonnet-4-20250514"

    def test_gpt4_turbo_maps_to_sonnet(self):
        assert resolve_model("gpt-4-turbo") == "claude-sonnet-4-20250514"

    def test_gpt35_turbo_maps_to_haiku(self):
        assert resolve_model("gpt-3.5-turbo") == "claude-haiku-4-5-20251001"

    def test_gpt4o_mini_maps_to_haiku(self):
        assert resolve_model("gpt-4o-mini") == "claude-haiku-4-5-20251001"

    def test_short_alias_sonnet(self):
        assert resolve_model("sonnet") == "claude-sonnet-4-20250514"

    def test_short_alias_opus(self):
        assert resolve_model("opus") == "claude-opus-4-20250514"

    def test_short_alias_haiku(self):
        assert resolve_model("haiku") == "claude-haiku-4-5-20251001"

    def test_claude_name_passthrough(self):
        """claude-* names should pass through unchanged."""
        assert resolve_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_claude_custom_passthrough(self):
        """Any claude-* name should pass through, even if not in the map."""
        assert resolve_model("claude-some-future-model") == "claude-some-future-model"

    def test_unknown_model_passthrough(self):
        """Unknown model names should pass through unchanged."""
        assert resolve_model("llama-3-70b") == "llama-3-70b"

    def test_empty_string_passthrough(self):
        """Empty string should pass through unchanged."""
        assert resolve_model("") == ""
