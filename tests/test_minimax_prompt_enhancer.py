"""Unit tests for MiniMaxPromptEnhancer node."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# conftest.py pre-loads all modules with proper package context.
# Retrieve them from sys.modules directly.
PKG_NAME = "ComfyUI_LTXVideo"
minimax_mod = sys.modules[f"{PKG_NAME}.minimax_prompt_enhancer"]

MINIMAX_API_BASE_URL = minimax_mod.MINIMAX_API_BASE_URL
MINIMAX_MODELS = minimax_mod.MINIMAX_MODELS
MiniMaxPromptEnhancer = minimax_mod.MiniMaxPromptEnhancer
enhance_prompt_via_minimax = minimax_mod.enhance_prompt_via_minimax


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_response(content: str, status_code: int = 200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock.text = content
    return mock


# ---------------------------------------------------------------------------
# Tests for the helper function
# ---------------------------------------------------------------------------


class TestEnhancePromptViaMinimax:
    def test_returns_enhanced_content(self):
        expected = "A cinematic shot of a car driving."
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response(expected)
            result = enhance_prompt_via_minimax(
                prompt="A car drives.",
                system_prompt="Enhance cinematically.",
                api_key="test-key",
            )
        assert result == expected

    def test_uses_correct_default_base_url(self):
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("enhanced")
            enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
            )
        call_url = mock_post.call_args[0][0]
        assert call_url.startswith(MINIMAX_API_BASE_URL)

    def test_uses_custom_base_url(self):
        custom_url = "https://custom.minimax.io/v1"
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("enhanced")
            enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
                base_url=custom_url,
            )
        call_url = mock_post.call_args[0][0]
        assert call_url.startswith(custom_url)

    def test_sends_api_key_in_header(self):
        api_key = "my-secret-key"
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("enhanced")
            enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key=api_key,
            )
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {api_key}"

    def test_temperature_is_not_zero(self):
        """MiniMax requires temperature in (0.0, 1.0]."""
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("enhanced")
            enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] > 0.0
        assert payload["temperature"] <= 1.0

    def test_raises_on_401(self):
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("Unauthorized", 401)
            with pytest.raises(RuntimeError, match="Invalid API key"):
                enhance_prompt_via_minimax(
                    prompt="test",
                    system_prompt="sys",
                    api_key="bad-key",
                )

    def test_raises_on_non_200(self):
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("Server Error", 500)
            with pytest.raises(RuntimeError, match="500"):
                enhance_prompt_via_minimax(
                    prompt="test",
                    system_prompt="sys",
                    api_key="key",
                )

    def test_default_model_is_minimax_m2_7(self):
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response("enhanced")
            enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
            )
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "MiniMax-M2.7"

    def test_strips_whitespace_from_response(self):
        padded = "  enhanced prompt with spaces  "
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response(padded)
            result = enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
            )
        assert result == padded.strip()

    def test_strips_think_tags(self):
        """MiniMax-M2.7 returns <think>…</think> reasoning blocks; strip them."""
        with_think = "<think>\nsome reasoning\n</think>\n\nActual enhanced prompt."
        with patch(f"{minimax_mod.__name__}.requests.post") as mock_post:
            mock_post.return_value = _make_response(with_think)
            result = enhance_prompt_via_minimax(
                prompt="test",
                system_prompt="sys",
                api_key="key",
            )
        assert result == "Actual enhanced prompt."
        assert "<think>" not in result


# ---------------------------------------------------------------------------
# Tests for the ComfyUI node class
# ---------------------------------------------------------------------------


class TestMiniMaxPromptEnhancerNode:
    def test_node_metadata(self):
        assert MiniMaxPromptEnhancer.RETURN_TYPES == ("STRING",)
        assert MiniMaxPromptEnhancer.RETURN_NAMES == ("enhanced_prompt",)
        assert MiniMaxPromptEnhancer.FUNCTION == "enhance"
        assert MiniMaxPromptEnhancer.CATEGORY == "api node/text/Lightricks"

    def test_model_choices_include_m2_7(self):
        input_types = MiniMaxPromptEnhancer.INPUT_TYPES()
        model_choices = input_types["required"]["model"][0]
        assert "MiniMax-M2.7" in model_choices
        assert "MiniMax-M2.7-highspeed" in model_choices

    def test_model_list_matches_constant(self):
        input_types = MiniMaxPromptEnhancer.INPUT_TYPES()
        model_choices = input_types["required"]["model"][0]
        assert model_choices == MINIMAX_MODELS

    def test_enhance_returns_tuple(self):
        node = MiniMaxPromptEnhancer()
        with patch(f"{minimax_mod.__name__}.enhance_prompt_via_minimax") as mock_fn:
            mock_fn.return_value = "A beautiful scene."
            result = node.enhance(
                api_key="test-key",
                prompt="A scene.",
                model="MiniMax-M2.7",
                mode="T2V",
                max_tokens=512,
            )
        assert isinstance(result, tuple)
        assert result[0] == "A beautiful scene."

    def test_enhance_raises_on_empty_api_key(self):
        node = MiniMaxPromptEnhancer()
        with pytest.raises(ValueError, match="API key"):
            node.enhance(api_key="", prompt="test")

    def test_enhance_raises_on_whitespace_api_key(self):
        node = MiniMaxPromptEnhancer()
        with pytest.raises(ValueError, match="API key"):
            node.enhance(api_key="   ", prompt="test")

    def test_enhance_raises_on_empty_prompt(self):
        node = MiniMaxPromptEnhancer()
        with pytest.raises(ValueError, match="Prompt"):
            node.enhance(api_key="key", prompt="")

    def test_t2v_mode_uses_t2v_system_prompt(self):
        node = MiniMaxPromptEnhancer()
        with patch(f"{minimax_mod.__name__}.enhance_prompt_via_minimax") as mock_fn:
            mock_fn.return_value = "enhanced"
            node.enhance(api_key="key", prompt="test", mode="T2V")
        sys_prompt = mock_fn.call_args[1]["system_prompt"]
        assert "cinematic director" in sys_prompt.lower()

    def test_i2v_mode_uses_i2v_system_prompt(self):
        node = MiniMaxPromptEnhancer()
        with patch(f"{minimax_mod.__name__}.enhance_prompt_via_minimax") as mock_fn:
            mock_fn.return_value = "enhanced"
            node.enhance(api_key="key", prompt="test", mode="I2V")
        sys_prompt = mock_fn.call_args[1]["system_prompt"]
        assert "cinematic director" in sys_prompt.lower()

    def test_highspeed_model_forwarded_to_api(self):
        node = MiniMaxPromptEnhancer()
        with patch(f"{minimax_mod.__name__}.enhance_prompt_via_minimax") as mock_fn:
            mock_fn.return_value = "enhanced"
            node.enhance(
                api_key="key",
                prompt="test",
                model="MiniMax-M2.7-highspeed",
            )
        assert mock_fn.call_args[1]["model"] == "MiniMax-M2.7-highspeed"
