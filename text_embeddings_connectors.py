"""Text Embeddings Pipeline: Feature Extractors + Embeddings Processors.

Provides the 3-block Gemma text encoder pipeline:
  1. Gemma Model (external) -- runs the LLM, gets hidden states
  2. Feature Extractor (V1/V2) -- normalization + projection
  3. Embeddings Processor (Video / AV) -- wraps Embeddings1DConnector(s)
"""

import json
import math
from pathlib import Path

import torch
from comfy.utils import load_torch_file
from einops import rearrange
from torch import nn

from .embeddings_connector import (
    Embeddings1DConnector,
    load_audio_embeddings_connector,
    load_video_embeddings_connector,
)

_PREFIX_BASE = "model.diffusion_model."
_PREFIX_TEXT_PROJ = "text_embedding_projection."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_binary_mask(encoded, encoded_mask):
    """Convert additive mask to binary [B, seq, 1] and zero out padding."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape(encoded.shape[0], encoded.shape[1], 1)
    return encoded * binary_mask, binary_mask


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


def _filter_sd(sd: dict, prefix: str) -> dict:
    """Extract keys with *prefix* and strip the prefix."""
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def _load_aggregate_embed(sd: dict, modality: str, dtype) -> nn.Linear:
    """Load an aggregate_embed Linear from the state dict.

    Args:
        sd: Full checkpoint state dict.
        modality: ``"video"`` or ``"audio"``.
        dtype: Target dtype.

    Returns ``linear`` module.
    """
    name = f"{modality}_aggregate_embed"
    weight_key = f"{_PREFIX_TEXT_PROJ}{name}.weight"
    bias_key = f"{_PREFIX_TEXT_PROJ}{name}.bias"
    weight = sd[weight_key]
    out_features, embedding_dim = weight.shape
    linear = nn.Linear(embedding_dim, out_features, bias=bias_key in sd)
    linear.load_state_dict(_filter_sd(sd, f"{_PREFIX_TEXT_PROJ}{name}."))
    return linear.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Normalization functions (moved from gemma_encoder.py)
# ---------------------------------------------------------------------------


def norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """
    Normalize a 4D tensor [B, T, D, L] per sample and per layer, using sequence_lengths to mask.
    Returns [B, T,  D * L] tensor with original padding preserved.

    Args:
        encoded_text: 4D tensor [B, T, D, L]
        sequence_lengths: 1D tensor [B] with actual sequence lengths
        padding_side: "left" or "right" to indicate which side has padding
    """
    B, T, D, L = encoded_text.shape
    device = encoded_text.device

    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(T, device=device)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = T - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    # denom = mask.sum(dim=(1, 2), keepdim=True)  # [B, 1, 1, 1]
    denom = (sequence_lengths * D).view(B, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + 1e-6)

    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )
    range_ = x_max - x_min

    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + 1e-6)

    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(B, T, -1)  # [B, T, D * L]

    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, D * L)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization.

    Each token is normalized independently using RMSNorm over its D
    dimension.  This is naturally causal and supports packing since each
    token's normalization is self-contained.

    Args:
        encoded_text: 4D tensor [B, T, D, L]
        attention_mask: [B, T] binary mask (1=real, 0=pad)

    Returns:
        [B, T, D*L] normalized and flattened tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B,T,1,L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)  # [B,T,D,L]
    normed = normed.reshape(B, T, D * L)
    # Zero out padding
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    normed = torch.where(mask_3d, normed, torch.zeros_like(normed))
    return normed


# ---------------------------------------------------------------------------
# Feature Extractors (Block 2)
# ---------------------------------------------------------------------------


class FeatureExtractorV1(nn.Module):
    """19B: per-segment norm -> aggregate_embed -> 3840"""

    def __init__(self, aggregate_embed: nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(self, all_layer_hiddens, attention_mask, padding_side="left"):
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = norm_and_concat_padded_batch(
            all_layer_hiddens, sequence_lengths, padding_side
        )
        normed = normed.to(all_layer_hiddens.dtype)
        features = self.aggregate_embed(normed)
        if self.is_av:
            return {"video": features, "audio": features}
        return {"video": features}


class FeatureExtractorV2(nn.Module):
    """22B: per-token RMS norm -> rescale -> aggregate_embed(s)"""

    def __init__(
        self,
        video_aggregate_embed: nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(self, all_layer_hiddens, attention_mask, padding_side="left"):
        normed = norm_and_concat_per_token_rms(all_layer_hiddens, attention_mask)
        normed = normed.to(all_layer_hiddens.dtype)
        v_dim = self.video_aggregate_embed.out_features
        result = {
            "video": self.video_aggregate_embed(
                _rescale_norm(normed, v_dim, self.embedding_dim)
            )
        }
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            result["audio"] = self.audio_aggregate_embed(
                _rescale_norm(normed, a_dim, self.embedding_dim)
            )
        return result


# ---------------------------------------------------------------------------
# Embeddings Processors (Block 3)
# ---------------------------------------------------------------------------


class VideoEmbeddingsProcessor(nn.Module):
    """Video-only embeddings processor: single connector."""

    def __init__(self, video_connector: Embeddings1DConnector):
        super().__init__()
        self.video_connector = video_connector

    def create_embeddings(self, features, attention_mask):
        encoded, mask = self.video_connector(features["video"], attention_mask)
        encoded, binary_mask = _to_binary_mask(encoded, mask)
        return encoded, binary_mask.squeeze(-1)


class AVEmbeddingsProcessor(nn.Module):
    """Audio-video embeddings processor: dual connectors + concat."""

    def __init__(
        self,
        video_connector: Embeddings1DConnector,
        audio_connector: Embeddings1DConnector,
    ):
        super().__init__()
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def create_embeddings(self, features, attention_mask):
        encoded, mask = self.video_connector(features["video"], attention_mask)
        encoded, binary_mask = _to_binary_mask(encoded, mask)
        audio_encoded, _ = self.audio_connector(features["audio"], attention_mask)
        encoded = torch.cat([encoded, audio_encoded], dim=-1)
        return encoded, binary_mask.squeeze(-1)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_single_aggregate_embed(sd, dtype):
    """Load the single aggregate_embed (V1 models).

    Returns:
        nn.Linear or None if key not found.
    """
    key = f"{_PREFIX_TEXT_PROJ}aggregate_embed.weight"
    if key not in sd:
        return None
    weight = sd[key]
    in_features, out_features = weight.shape[1], weight.shape[0]
    # Check for bias
    bias_key = f"{_PREFIX_TEXT_PROJ}aggregate_embed.bias"
    has_bias = bias_key in sd
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    linear.load_state_dict(_filter_sd(sd, f"{_PREFIX_TEXT_PROJ}aggregate_embed."))
    return linear.to(dtype=dtype)


def _load_single_aggregate_embed_from_file(path, dtype):
    """Load aggregate_embed from a standalone checkpoint file (legacy fallback).

    Args:
        path: Path to safetensors file containing aggregate_embed weights.
        dtype: Target dtype.

    Returns:
        nn.Linear or None if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    loaded_sd = load_torch_file(str(path), return_metadata=False)
    if "aggregate_embed.weight" not in loaded_sd:
        raise ValueError(
            f"Checkpoint {path} does not contain 'aggregate_embed.weight'."
        )
    weight = loaded_sd["aggregate_embed.weight"]
    in_features, out_features = weight.shape[1], weight.shape[0]
    bias_key = "aggregate_embed.bias"
    has_bias = bias_key in loaded_sd
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    linear.load_state_dict(
        {
            k.removeprefix("aggregate_embed."): v
            for k, v in loaded_sd.items()
            if k.startswith("aggregate_embed.")
        }
    )
    return linear.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------


def load_text_embeddings_pipeline(
    ltxv_path, dtype=torch.bfloat16, fallback_proj_path=None
):
    """Load feature extractor + embeddings processor from LTX-V checkpoint.

    Auto-detects model variant (19B/22B, video-only/AV).

    Args:
        ltxv_path: Path to the LTX-V checkpoint.
        dtype: Target dtype for loaded modules.
        fallback_proj_path: Optional path to a standalone ``proj_linear.safetensors``
            file.  Used as a legacy fallback when the aggregate_embed is not
            stored inside the LTX-V checkpoint (V1 models).

    Returns:
        (feature_extractor, embeddings_processor)
    """
    sd, metadata = load_torch_file(str(ltxv_path), return_metadata=True)
    config = json.loads(metadata.get("config", "{}"))
    transformer_config = config.get("transformer", {})

    is_av = f"{_PREFIX_BASE}audio_adaln_single.linear.weight" in sd
    has_dual_aggregate = f"{_PREFIX_TEXT_PROJ}video_aggregate_embed.weight" in sd

    # Load connectors (always needed)
    video_connector = load_video_embeddings_connector(sd, transformer_config, dtype)

    # Build embeddings processor (Block 3)
    if is_av:
        audio_connector = load_audio_embeddings_connector(sd, transformer_config, dtype)
        processor = AVEmbeddingsProcessor(video_connector, audio_connector)
    else:
        processor = VideoEmbeddingsProcessor(video_connector)

    # Build feature extractor (Block 2)
    if has_dual_aggregate:
        # V2 (22B): validate config matches expected settings
        _expected = {
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": False,
            "caption_projection_second_linear": False,
            "caption_proj_before_connector": True,
            "text_encoder_norm_type": "per_token_rms",
        }
        for key, expected_val in _expected.items():
            actual = transformer_config.get(key)
            assert actual == expected_val, (
                f"Unexpected config for dual-aggregate model: "
                f"{key}={actual!r}, expected {expected_val!r}"
            )
        video_agg = _load_aggregate_embed(sd, "video", dtype)
        audio_agg = _load_aggregate_embed(sd, "audio", dtype) if is_av else None
        embedding_dim = transformer_config.get("prompt_embedding_dim", 3840)
        return (FeatureExtractorV2(video_agg, embedding_dim, audio_agg), processor)
    # V1 (19B)
    aggregate_embed = _load_single_aggregate_embed(sd, dtype)
    if aggregate_embed is None and fallback_proj_path is not None:
        aggregate_embed = _load_single_aggregate_embed_from_file(
            fallback_proj_path, dtype
        )
    return (FeatureExtractorV1(aggregate_embed, is_av=is_av), processor)
