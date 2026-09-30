import hashlib
from pathlib import Path
from typing import Any

import folder_paths
import safetensors
import torch
from comfy_api.latest import io

from .nodes_registry import comfy_node

# Pipeline HDR IC-LoRA scene embeddings (`hdr_ic_lora._load_video_context`).
_PIPELINE_VIDEO_KEYS = ("video_context", "video_prompt_embeds")
_PIPELINE_AUDIO_KEYS = ("audio_context", "audio_prompt_embeds")
# LTX-AV text encode layout: video 4096 + audio 2048 on the last dim.
_VIDEO_CONTEXT_DIM = 4096
_AUDIO_CONTEXT_DIM = 2048
_AV_CONTEXT_DIM = _VIDEO_CONTEXT_DIM + _AUDIO_CONTEXT_DIM


def _unsqueeze_batch(t: torch.Tensor, *, source: str, kind: str) -> torch.Tensor:
    """Ensure conditioning tensors are ``[B, T, D]`` (pipeline files are often ``[T, D]``)."""
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    raise ValueError(
        f"{source}: {kind} embedding must be 2D [T, D] or 3D [B, T, D]; "
        f"got shape {tuple(t.shape)}"
    )


def _load_pipeline_scene_embedding(f: Any, source: str) -> list[list[Any]]:
    """Load ``video_context`` (+ optional ``audio_context``) into Comfy CONDITIONING.

    Matches ``ltx_pipelines.hdr_ic_lora``: the safetensors replaces text encoding.
    For AV Comfy models, video (4096) and audio (2048) contexts are concatenated on
    the last dim — same layout as a fully processed LTX-AV text encode.
    """
    keys = list(f.keys())
    video_key = next((k for k in _PIPELINE_VIDEO_KEYS if k in keys), None)
    if video_key is None:
        return []

    video = _unsqueeze_batch(f.get_tensor(video_key), source=source, kind=video_key)
    video_dim = int(video.shape[-1])
    audio_key = next((k for k in _PIPELINE_AUDIO_KEYS if k in keys), None)

    if video_dim == _AV_CONTEXT_DIM:
        # Already video∥audio on the last dim (rare export); use as-is.
        tensor = video
    elif video_dim == _VIDEO_CONTEXT_DIM:
        if audio_key is None:
            raise ValueError(
                f"{source}: {video_key} last dim is {_VIDEO_CONTEXT_DIM} but no "
                f"audio_context / audio_prompt_embeds found. AV conditioning needs "
                f"{_VIDEO_CONTEXT_DIM}+{_AUDIO_CONTEXT_DIM}="
                f"{_AV_CONTEXT_DIM} on the last dim."
            )
        audio = _unsqueeze_batch(f.get_tensor(audio_key), source=source, kind=audio_key)
        audio_dim = int(audio.shape[-1])
        if audio_dim != _AUDIO_CONTEXT_DIM:
            raise ValueError(
                f"{source}: {audio_key} last dim must be {_AUDIO_CONTEXT_DIM}; "
                f"got {audio_dim} with shape {tuple(audio.shape)}"
            )
        if audio.shape[:-1] != video.shape[:-1]:
            raise ValueError(
                f"{source}: audio/video context shape mismatch "
                f"(excluding last dim): {tuple(audio.shape)} vs {tuple(video.shape)}"
            )
        tensor = torch.cat([video, audio], dim=-1)
    else:
        raise ValueError(
            f"{source}: {video_key} last dim must be {_VIDEO_CONTEXT_DIM} "
            f"(or pre-concatenated {_AV_CONTEXT_DIM}); got {video_dim} with shape "
            f"{tuple(video.shape)}. Check that video/audio keys are not swapped."
        )

    return [[tensor, {}]]


@comfy_node(name="LTXVLoadConditioning")
class LTXVLoadConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = folder_paths.get_filename_list("embeddings")
        if not files:
            files = [""]
        return io.Schema(
            node_id="LTXVLoadConditioning",
            display_name="🅛🅣🅧 LTXV Load Conditioning",
            category="lightricks/LTXV",
            description=(
                "Load precomputed text conditioning from models/embeddings. "
                "Supports Comfy conditioning_data_* files and pipeline scene "
                "embeddings (video_context / audio_context), e.g. HDR IC-LoRA."
            ),
            inputs=[
                io.Combo.Input("file_name", options=sorted(files)),
                io.Combo.Input("device", options=["cpu", "gpu"]),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, file_name: str, device: str) -> io.NodeOutput:
        file_path = folder_paths.get_full_path("embeddings", file_name)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Conditioning file not found: {file_path}")

        target_device = "cpu"
        if device == "gpu":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        conditioning: list[list[Any]] = []

        with safetensors.safe_open(
            file_path, framework="pt", device=target_device
        ) as f:
            tensor_keys = [k for k in f.keys() if k.startswith("conditioning_data_")]

            if tensor_keys:
                for tensor_key in sorted(tensor_keys):
                    idx = tensor_key.replace("conditioning_data_", "")
                    tensor = f.get_tensor(tensor_key)

                    options: dict[str, Any] = {}
                    mask_key = f"attention_mask_{idx}"
                    if mask_key in f.keys():
                        options["attention_mask"] = f.get_tensor(mask_key)

                    conditioning.append([tensor, options])
            else:
                conditioning = _load_pipeline_scene_embedding(f, source=file_name)

        if not conditioning:
            raise ValueError(
                f"No conditioning data found in file: {file_name}. "
                "Expected conditioning_data_* or video_context / video_prompt_embeds."
            )

        return io.NodeOutput(conditioning)

    @classmethod
    def fingerprint_inputs(cls, file_name: str, device: str) -> str:
        file_path = folder_paths.get_full_path("embeddings", file_name)
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def validate_inputs(cls, file_name: str, device: str) -> bool | str:
        if not file_name:
            return "No files found. Please save a conditioning first."
        try:
            file_path = folder_paths.get_full_path("embeddings", file_name)
            if not Path(file_path).exists():
                return f"File not found: {file_name}"
        except Exception:
            return f"Invalid file: {file_name}"
        return True
