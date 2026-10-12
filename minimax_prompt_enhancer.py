"""
API-based prompt enhancement using MiniMax.

Provides an API-based alternative to the local LLM prompt enhancer, allowing
users to enhance their video generation prompts via MiniMax without requiring
local GPU resources.
"""

import logging
import re

import requests

from .nodes_registry import comfy_node
from .prompt_enhancer_utils import I2V_CINEMATIC_PROMPT, T2V_CINEMATIC_PROMPT

logger = logging.getLogger(__name__)

MINIMAX_API_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODELS = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def enhance_prompt_via_minimax(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "MiniMax-M2.7",
    base_url: str = MINIMAX_API_BASE_URL,
    max_tokens: int = 512,
) -> str:
    """Call MiniMax chat completions API to enhance a video generation prompt."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "Invalid API key. Please provide a valid MINIMAX_API_KEY."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"MiniMax API request failed with status {response.status_code}: {response.text}"
        )
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    # Strip <think>…</think> reasoning tokens emitted by MiniMax-M2.7
    content = _THINK_TAG_RE.sub("", content).strip()
    return content


@comfy_node(name="MiniMaxPromptEnhancer")
class MiniMaxPromptEnhancer:
    """
    Enhances video generation prompts using the MiniMax API.

    An API-based alternative to the local LLM prompt enhancer. No local GPU
    resources are required — prompts are sent to the MiniMax chat completions
    endpoint and returned as cinematic descriptions suitable for LTX-Video.

    Inputs:
        - api_key: MiniMax API key (MINIMAX_API_KEY). Get one at https://platform.minimax.io/
        - prompt: Text prompt to enhance
        - model: MiniMax model to use for enhancement
        - mode: Enhancement mode — T2V (text-to-video) or I2V (image-to-video)
        - max_tokens: Maximum number of tokens in the enhanced prompt

    Returns:
        - enhanced_prompt: Cinematically enhanced prompt string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "MINIMAX_API_KEY",
                        "multiline": False,
                        "tooltip": "MiniMax API key. Get one at https://platform.minimax.io/",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to enhance for video generation",
                    },
                ),
                "model": (
                    MINIMAX_MODELS,
                    {
                        "default": "MiniMax-M2.7",
                        "tooltip": "MiniMax model to use for prompt enhancement",
                    },
                ),
                "mode": (
                    ["T2V", "I2V"],
                    {
                        "default": "T2V",
                        "tooltip": (
                            "Enhancement mode: T2V for text-to-video, "
                            "I2V for image-to-video"
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Maximum number of tokens in the enhanced prompt",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "api node/text/Lightricks"
    TITLE = "MiniMax Prompt Enhancer"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Enhances video generation prompts using the MiniMax API. "
        "An API-based alternative to the local LLM enhancer — no local GPU required."
    )

    def enhance(
        self,
        api_key: str,
        prompt: str,
        model: str = "MiniMax-M2.7",
        mode: str = "T2V",
        max_tokens: int = 512,
    ) -> tuple:
        if not api_key.strip():
            raise ValueError(
                "MiniMax API key is required. "
                "Get one at https://platform.minimax.io/"
            )
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        system_prompt = T2V_CINEMATIC_PROMPT if mode == "T2V" else I2V_CINEMATIC_PROMPT

        logger.info(
            "Enhancing prompt via MiniMax API (model=%s, mode=%s): %s...",
            model,
            mode,
            prompt[:60],
        )
        enhanced = enhance_prompt_via_minimax(
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )
        logger.info("Enhanced prompt: %s...", enhanced[:60])
        return (enhanced,)
