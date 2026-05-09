"""
Monkey-patch comfy.model_detection.detect_unet_config to add missing LTXAV
detection logic for:

- cross_attention_adaln (9 vs 6 ada params per transformer block)
- audio cross-attention dimension
- audio attention head dimension
- embeddings connector dimensions (num_heads, head_dim)

The core ComfyUI detection code sets basic params but doesn't auto-detect
cross_attention_adaln or per-connector dimensions from the checkpoint,
which causes "size mismatch" errors when loading LTXAV GGUF checkpoints
that were trained with cross_attention_adaln=True and different connector
dimensions than the hardcoded defaults.
"""
import logging
import comfy.model_detection

_original_detect_unet_config = comfy.model_detection.detect_unet_config


def _patched_detect_unet_config(state_dict, key_prefix, metadata=None):
    config = _original_detect_unet_config(state_dict, key_prefix, metadata)
    if config is None or config.get("image_model") != "ltxav":
        return config

    state_dict_keys = state_dict.keys()
    prefix = key_prefix

    # Detect cross_attention_adaln: prompt_scale_shift_table only exists
    # when cross_attention_adaln=True (it handles the separate prompt
    # timestep modulation for cross-attention layers).
    prompt_ss_key = f"{prefix}transformer_blocks.0.prompt_scale_shift_table"
    if prompt_ss_key in state_dict_keys:
        config["cross_attention_adaln"] = True
        logging.info("LTXAV patch: detected cross_attention_adaln=True")

    # Detect audio cross-attention dimension and head dim.
    # audio_attn2.to_k.weight shape: [audio_inner_dim, audio_cross_attention_dim]
    # audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
    audio_attn_key = f"{prefix}transformer_blocks.0.audio_attn2.to_k.weight"
    if audio_attn_key in state_dict_keys:
        audio_inner_dim, audio_cross_dim = state_dict[audio_attn_key].shape
        config["audio_cross_attention_dim"] = audio_cross_dim
        # Standard LTX architecture uses 32 attention heads
        config["audio_attention_head_dim"] = audio_inner_dim // 32
        logging.info(
            "LTXAV patch: audio_cross_attention_dim=%d, audio_attention_head_dim=%d",
            audio_cross_dim,
            audio_inner_dim // 32,
        )

    # Detect embeddings connector dimensions.
    # learnable_registers shape: [num_registers, inner_dim]
    # inner_dim = num_attention_heads * attention_head_dim
    # We default num_attention_heads=32 (standard LTX head count) and
    # derive attention_head_dim from inner_dim.
    video_reg_key = f"{prefix}video_embeddings_connector.learnable_registers"
    if video_reg_key in state_dict_keys:
        connector_inner_dim = state_dict[video_reg_key].shape[1]
        config["connector_num_attention_heads"] = 32
        config["connector_attention_head_dim"] = connector_inner_dim // 32
        logging.info(
            "LTXAV patch: video connector inner_dim=%d, num_heads=32, head_dim=%d",
            connector_inner_dim,
            connector_inner_dim // 32,
        )

    audio_reg_key = f"{prefix}audio_embeddings_connector.learnable_registers"
    if audio_reg_key in state_dict_keys:
        audio_connector_inner_dim = state_dict[audio_reg_key].shape[1]
        config["audio_connector_num_attention_heads"] = 32
        config["audio_connector_attention_head_dim"] = audio_connector_inner_dim // 32
        logging.info(
            "LTXAV patch: audio connector inner_dim=%d, num_heads=32, head_dim=%d",
            audio_connector_inner_dim,
            audio_connector_inner_dim // 32,
        )

    return config


# Apply the patch
comfy.model_detection.detect_unet_config = _patched_detect_unet_config
logging.info("LTXAV model detection patch applied.")
