"""
API-based text encoding that returns CONDITIONING for LTX-2.
Replaces the CLIP encoding step entirely using an external API.
"""

import io
import json
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
MISSING_MODEL_SELECTOR_MESSAGE = (
    "Cannot identify the text encoder from the provided model file."
)
MODEL_ID_METADATA_KEY = "encrypted_wandb_properties"
GEMMA_SOURCE_CHECKPOINT_METADATA_KEY = "gemma_source_checkpoint"


MODEL_FOLDERS = ("checkpoints", "diffusion_models")


def model_filename_list() -> list[str]:
    """Every model the node can read a text-encoder selector from, de-duplicated.

    A name present in both folders is listed once; resolve_model_path resolves
    it the same way round, so the entry always refers to the same file the
    combo offered.
    """
    names = []
    for folder in MODEL_FOLDERS:
        for name in folder_paths.get_filename_list(folder):
            if name not in names:
                names.append(name)
    return names


def resolve_model_path(ckpt_name: str) -> str:
    """Full path of ckpt_name, looked up in the folder order the combo used."""
    for folder in MODEL_FOLDERS:
        path = folder_paths.get_full_path(folder, ckpt_name)
        if path is not None:
            return path
    raise ValueError(
        f"Model file '{ckpt_name}' was not found in any of: {', '.join(MODEL_FOLDERS)}."
    )


def parse_gemma_source_checkpoint(raw: str) -> dict[str, str]:
    """Parse the ``gemma_source_checkpoint`` safetensors metadata JSON string."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(MISSING_MODEL_SELECTOR_MESSAGE) from exc
    if not isinstance(obj, dict):
        raise ValueError(MISSING_MODEL_SELECTOR_MESSAGE)
    ltx_version, gemma_version = obj.get("ltx_version"), obj.get("gemma_version")
    if not ltx_version or not gemma_version:
        raise ValueError(MISSING_MODEL_SELECTOR_MESSAGE)
    return {"ltx_version": str(ltx_version), "gemma_version": str(gemma_version)}


def prompt_embedding_selector_from_metadata(
    metadata: dict[str, str] | None,
) -> dict[str, str | dict[str, str]]:
    """Build the XOR selector for ``/v1/prompt-embedding``.

    LTX-2.5+ checkpoints identify the text encoder via ``gemma_source_checkpoint``
    (``ltx_version`` + ``gemma_version``). Pre-2.5 checkpoints use
    ``encrypted_wandb_properties`` as ``model_id``. Prefer the 2.5+ selector when
    both are present: OS 2.5 files may still carry an unusable ``model_id``.
    """
    if metadata:
        raw_gemma = metadata.get(GEMMA_SOURCE_CHECKPOINT_METADATA_KEY)
        if raw_gemma:
            return {"model": parse_gemma_source_checkpoint(raw_gemma)}
        model_id = metadata.get(MODEL_ID_METADATA_KEY)
        if model_id:
            return {"model_id": model_id}
    raise ValueError(MISSING_MODEL_SELECTOR_MESSAGE)


def extract_prompt_embedding_selector(
    ckpt_name: str,
) -> dict[str, str | dict[str, str]]:
    with safe_open(
        resolve_model_path(ckpt_name),
        framework="pt",
        device="cpu",
    ) as f:
        return prompt_embedding_selector_from_metadata(f.metadata())


@comfy_node(name="GemmaAPITextEncode")
class GemmaAPITextEncode:
    """
    Encodes text prompts using the LTX Video API, returning CONDITIONING for LTX-2 models.

    This node replaces the local CLIP encoding step by sending the prompt to an external API
    for processing. It requires an API key and selects the text encoder from checkpoint
    metadata: ``model`` (LTX-2.5+) or ``model_id`` (LTX-2.3).

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
                    model_filename_list(),
                    {
                        "tooltip": "The model to read the API text-encoder selector from. "
                        "Either a checkpoint or a diffusion model."
                    },
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

        selector = extract_prompt_embedding_selector(ckpt_name)
        payload = {
            "prompt": prompt,
            "enhance_prompt": enhance_prompt,
            **selector,
        }
        if "model_id" in selector:
            selector_desc = f"model_id: {str(selector['model_id'])[:50]}..."
        else:
            selector_desc = f"model: {selector['model']}"
        logger.info(
            f"Calling API to encode prompt: {prompt[:50]}... with {selector_desc}"
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
