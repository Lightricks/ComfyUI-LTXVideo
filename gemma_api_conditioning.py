"""
API-based text encoding that returns CONDITIONING for LTX-2.
Replaces the CLIP encoding step entirely using an external API.
"""

import io
import logging
import pickle

import folder_paths
import requests
from safetensors import safe_open

from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

LTXV_API_BASE_URL = "https://api.ltx.video"
UPDATE_MESSAGE = (
    "Note: If this error persists, the node might be outdated. "
    "Please update ComfyUI-LTXVideo to the latest version."
)
INVALID_API_KEY_MESSAGE = (
    "Invalid API key. Please generate a new API key at: https://console.ltx.video/"
)
MISSING_MODEL_ID_MESSAGE = "Model ID cannot be identified from the provided model file"


def extract_model_id(ckpt_name: str) -> str:
    model_id_key = "encrypted_wandb_properties"
    with safe_open(
        folder_paths.get_full_path_or_raise("checkpoints", ckpt_name),
        framework="pt",
        device="cpu",
    ) as f:
        metadata = f.metadata()
        if not metadata or model_id_key not in metadata:
            raise ValueError(MISSING_MODEL_ID_MESSAGE)
        return metadata[model_id_key]


@comfy_node(name="GemmaAPITextEncode")
class GemmaAPITextEncode:
    """
    Encodes text prompts using the LTX Video API, returning CONDITIONING for LTX-2 models.

    This node replaces the local CLIP encoding step by sending the prompt to an external API
    for processing. It requires an API key and automatically extracts the model ID from the
    checkpoint file metadata.

    Inputs:
        - api_key: Authentication key for the LTX Video API
        - prompt: Text prompt to encode
        - ckpt_name: Checkpoint file containing model metadata

    Returns:
        - CONDITIONING: Encoded prompt conditioning ready for LTX-2 video generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "API_KEY",
                        "multiline": False,
                        "tooltip": "API key for authentication",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to encode",
                    },
                ),
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, the prompt is enhanced using Gemma 3 before encoding",
                    },
                ),
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "api node/text/Lightricks"

    def encode(
        self, api_key: str, prompt: str, ckpt_name: str, enhance_prompt: bool = False
    ):
        if not api_key:
            raise ValueError("API key is required")

        if not prompt.strip():
            raise ValueError("Text prompt cannot be empty")

        if not ckpt_name or not ckpt_name.strip():
            raise ValueError("Model path is required")

        model_id = extract_model_id(ckpt_name)
        payload = {
            "prompt": prompt,
            "model_id": model_id,
            "enhance_prompt": enhance_prompt,
        }
        logger.info(
            f"Calling API to encode prompt: {prompt[:50]}... with model_id: {model_id[:50]}..."
        )
        try:
            response = requests.post(
                f"{LTXV_API_BASE_URL}/v1/prompt-embedding",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )

            if response.status_code == 401:
                raise RuntimeError(INVALID_API_KEY_MESSAGE)

            if response.status_code != 200:
                raise RuntimeError(
                    f"API request failed with status {response.status_code}: {response.text}\n"
                    f"{UPDATE_MESSAGE}"
                )

            conditioning = pickle.load(io.BytesIO(response.content))
            logger.info("Successfully received conditioning from API")

            return (conditioning,)

        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}\n {UPDATE_MESSAGE}")
