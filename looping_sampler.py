import copy
from dataclasses import dataclass
from typing import Optional

import comfy
import torch
from comfy.nested_tensor import NestedTensor

from .easy_samplers import LTXVBaseSampler, LTXVExtendSampler, LTXVInContextSampler, _get_raw_conds_from_guider
from .latents import LTXVDilateLatent, LTXVSelectLatents
from .nodes_registry import comfy_node


@dataclass
class TileConfig:
    """Configuration for spatial tile processing."""

    tile_latents: dict
    tile_guiding_latents: dict
    tile_negative_index_latents: dict
    tile_keyframes: torch.Tensor
    keyframe_per_tile_indices: list
    tile_normalizing_latents: dict
    tile_height: int
    tile_width: int
    v: int
    h: int
    vertical_tiles: int
    horizontal_tiles: int
    first_seed: int


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""

    temporal_tile_size: int
    temporal_overlap: int
    temporal_overlap_cond_strength: float
    cond_image_strength: float
    guiding_strength: float
    adain_factor: float
    optional_negative_index: int
    optional_negative_index_strength: float
    optional_positive_conditionings: list
    time_scale_factor: int
    width_scale_factor: int
    height_scale_factor: int
    per_tile_seed_offsets: list
    guiding_start_step: int
    guiding_end_step: int


@dataclass
class ModelConfig:
    """Configuration for model components."""

    model: object
    vae: object
    noise: object
    sampler: object
    sigmas: object
    guider: object


@comfy_node(
    name="LTXVLoopingSampler",
)
class LTXVLoopingSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to use."}),
                "vae": ("VAE", {"tooltip": "The VAE to use."}),
                "noise": ("NOISE", {"tooltip": "The noise to use."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler to use."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigmas to use."}),
                "guider": (
                    "GUIDER",
                    {"tooltip": "The guider to use, must be a STGGuiderAdvanced."},
                ),
                "latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for creating the long video, they can be guiding latents or empty latents when no guidance is used."
                    },
                ),
                "temporal_tile_size": (
                    "INT",
                    {
                        "default": 80,
                        "min": 24,
                        "max": 1000,
                        "step": 8,
                        "tooltip": "The size of the temporal tile to use for the sampling, in pixel frames, in addition to the overlapping region.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 24,
                        "min": 16,
                        "max": 80,
                        "step": 8,
                        "tooltip": "The overlap between the temporal tiles, in pixel frames.",
                    },
                ),
                "guiding_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the conditioning on guiding latents, when optional_guiding_latents are provided.",
                    },
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the conditioning on the latents from the previous temporal tile.",
                    },
                ),
                "cond_image_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the conditioning on the optional_cond_images.",
                    },
                ),
                "horizontal_tiles": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "Number of horizontal spatial tiles.",
                    },
                ),
                "vertical_tiles": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 6,
                        "tooltip": "Number of vertical spatial tiles.",
                    },
                ),
                "spatial_overlap": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "tooltip": "Overlap between spatial tiles.",
                    },
                ),
            },
            "optional": {
                "optional_cond_images": (
                    "IMAGE",
                    {
                        "tooltip": "The image to use for conditioning the first frame in the video (i2v setup). If not provided, the first frame will be unconditioned (t2v setup). The image will be resized to the size of the first frame."
                    },
                ),
                "optional_guiding_latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for guiding the sampling, typically with an IC-LoRA."
                    },
                ),
                "adain_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the AdaIn operation used to fix the statistics of each new generated temporal tile, to prevent accumulated oversaturation.",
                    },
                ),
                "optional_positive_conditionings": (
                    "CONDITIONING",
                    {
                        "tooltip": "Optional way to provide changing positive prompts, one per temporal tile, using the MultiPromptProvider node."
                    },
                ),
                "optional_negative_index_latents": (
                    "LATENT",
                    {
                        "tooltip": "Special optional latents to condition on a negative index before each new temporal tile as a way to provide long term context during video generation."
                    },
                ),
                "guiding_start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "The step at which the guiding latents start to be used.",
                    },
                ),
                "guiding_end_step": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "The step at which the guiding latents stop to be used.",
                    },
                ),
                "optional_cond_image_indices": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "The indices of the keyframes to use for the sampling, separated by commas. The indices are counted from the end of the video.",
                    },
                ),
                "optional_normalizing_latents": (
                    "LATENT",
                    {
                        "tooltip": "The latents to use for normalizing the output latents, they will be used to normalize the output latents to the same statistics as the input latents."
                    },
                ),
                "optional_negative_index_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The strength of the negative-index latent conditioning. Lower values reduce the influence of the reference image(s) provided via optional_negative_index_latents.",
                    },
                ),
                "save_checkpoints": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, after each temporal tile writes the accumulated latent as ComfyUI .latent files into the input folder (ltxv_looping_ckpt_v{v}_h{h}_video.latent, and _audio.latent for AV), so a mid-run crash leaves a decodable partial result. Reload with the stock LoadLatent node (+ LTXVConcatAVLatent for AV). Overwritten each tile (the latent is cumulative).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_output",)

    FUNCTION = "sample"
    CATEGORY = "sampling"

    def _extract_latent_spatial_tile(self, latent_dict, v_start, v_end, h_start, h_end):
        """Extract spatial tile from a latent dictionary."""
        if latent_dict is None:
            return None
        tile_samples = latent_dict["samples"][:, :, :, v_start:v_end, h_start:h_end]
        if "noise_mask" in latent_dict and latent_dict["noise_mask"] is not None:
            noise_mask = latent_dict["noise_mask"]
            # If the noise mask has broadcast spatial dims (1x1), keep them
            # as-is rather than slicing (which would produce zero-size dims
            # for tiles starting past index 0).
            if noise_mask.ndim == 5 and noise_mask.shape[3] <= 1 and noise_mask.shape[4] <= 1:
                tile_masks = noise_mask
            else:
                tile_masks = noise_mask[:, :, :, v_start:v_end, h_start:h_end]
            return {"samples": tile_samples, "noise_mask": tile_masks}
        else:
            return {"samples": tile_samples}

    def _extract_spatial_tile(
        self,
        latents,
        optional_guiding_latents,
        optional_negative_index_latents,
        optional_normalizing_latents,
        optional_keyframes,
        v_start,
        v_end,
        h_start,
        h_end,
        height_scale_factor,
        width_scale_factor,
    ):
        """Extract spatial tiles from all inputs for a given spatial region."""
        # Extract spatial tile from latents
        tile_latents = self._extract_latent_spatial_tile(
            latents, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from guiding latents if provided
        tile_guiding_latents = self._extract_latent_spatial_tile(
            optional_guiding_latents, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from negative index latents if provided
        tile_negative_index_latents = self._extract_latent_spatial_tile(
            optional_negative_index_latents, v_start, v_end, h_start, h_end
        )

        # Extract spatial tile from normalizing latents if provided
        tile_normalizing_latents = self._extract_latent_spatial_tile(
            optional_normalizing_latents, v_start, v_end, h_start, h_end
        )

        if optional_keyframes is not None:
            # Scale coordinates for image
            img_h_start = v_start * height_scale_factor
            img_h_end = v_end * height_scale_factor
            img_w_start = h_start * width_scale_factor
            img_w_end = h_end * width_scale_factor

            tile_keyframes = optional_keyframes[
                :, img_h_start:img_h_end, img_w_start:img_w_end, :
            ]
        else:
            tile_keyframes = None

        return (
            tile_latents,
            tile_guiding_latents,
            tile_negative_index_latents,
            tile_keyframes,
            tile_normalizing_latents,
        )

    def _process_temporal_chunks(
        self,
        tile_config: TileConfig,
        sampling_config: SamplingConfig,
        model_config: ModelConfig,
        audio_info: Optional[dict] = None,
        save_checkpoints: bool = False,
    ):
        """Process all temporal chunks for a single spatial tile."""
        chunk_index = 0
        tile_out_latents = None
        first_tile_out_latents = None
        accumulated_audio = None

        for i_temporal_tile, (start_index, end_index) in enumerate(
            zip(
                range(
                    0,
                    tile_config.tile_latents["samples"].shape[2]
                    + sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                    sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                ),
                range(
                    sampling_config.temporal_tile_size,
                    tile_config.tile_latents["samples"].shape[2]
                    + sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                    sampling_config.temporal_tile_size
                    - sampling_config.temporal_overlap,
                ),
            )
        ):
            if tile_config.tile_guiding_latents is not None:
                guiding_latent_chunk = LTXVSelectLatents().select_latents(
                    tile_config.tile_guiding_latents,
                    start_index,
                    min(
                        end_index - 1,
                        tile_config.tile_guiding_latents["samples"].shape[2] - 1,
                    ),
                )[0]
            else:
                guiding_latent_chunk = None

            latent_chunk = LTXVSelectLatents().select_latents(
                tile_config.tile_latents,
                start_index,
                min(end_index - 1, tile_config.tile_latents["samples"].shape[2] - 1),
            )[0]

            print(
                "Processing temporal chunk at index",
                start_index,
                "to",
                min(end_index - 1, tile_config.tile_latents["samples"].shape[2] - 1),
            )

            seed_offset = self._get_per_tile_value(
                sampling_config.per_tile_seed_offsets, i_temporal_tile
            )

            if tile_config.tile_normalizing_latents is not None:
                normalizing_latent_chunk = LTXVSelectLatents().select_latents(
                    tile_config.tile_normalizing_latents,
                    start_index,
                    min(
                        end_index - 1,
                        tile_config.tile_normalizing_latents["samples"].shape[2] - 1,
                    ),
                )[0]
                normalize_per_frame = True
                print(
                    "Normalizing latents provided, normalizing per frame and channel with factor",
                    sampling_config.adain_factor,
                )
            else:
                normalizing_latent_chunk = first_tile_out_latents
                normalize_per_frame = False
                print(
                    "No normalizing latents provided, normalizing per channel using first chunk with factor",
                    sampling_config.adain_factor,
                )

            model_config.noise.seed = self._calculate_tile_seed(
                tile_config.first_seed,
                start_index,
                tile_config.vertical_tiles,
                tile_config.horizontal_tiles,
                tile_config.v,
                tile_config.h,
                seed_offset,
            )

            # Handle optional positive conditionings
            new_guider = self._prepare_guider_for_chunk(
                model_config.guider,
                sampling_config.optional_positive_conditionings,
                chunk_index,
            )

            this_chunk_keyframe_indices = [
                in_tile_index
                for (tile_index, in_tile_index) in tile_config.keyframe_per_tile_indices
                if tile_index == i_temporal_tile
            ]
            if this_chunk_keyframe_indices and tile_config.tile_keyframes is not None:
                this_chunk_keyframes = torch.cat(
                    [
                        tile_config.tile_keyframes[i_keyframe].unsqueeze(0)
                        for (i_keyframe, (tile_index, in_tile_index)) in enumerate(
                            tile_config.keyframe_per_tile_indices
                        )
                        if tile_index == i_temporal_tile
                    ]
                )
                print(
                    f"Chunk {i_temporal_tile} keyframe indices: {this_chunk_keyframe_indices}"
                )
            else:
                this_chunk_keyframes = None
                print(f"Chunk {i_temporal_tile} has no keyframes")
            this_chunk_keyframe_indices = ",".join(
                [str(i) for i in this_chunk_keyframe_indices]
            )
            if start_index == 0:
                # Create audio tile for the base tile.
                # If input audio data is available (stage-2 refinement),
                # use the corresponding slice; otherwise create zeros
                # (stage-1 generation from scratch).
                audio_tile = None
                if audio_info is not None:
                    video_tile_frames = min(
                        sampling_config.temporal_tile_size,
                        tile_config.tile_latents["samples"].shape[2],
                    )
                    audio_tile_frames = max(
                        1,
                        round(
                            video_tile_frames
                            * audio_info["total_audio_frames"]
                            / max(audio_info["total_video_frames"], 1)
                        ),
                    )
                    src_audio = audio_info.get("tensor")
                    if src_audio is not None:
                        # Refinement: use input audio slice
                        available = min(audio_tile_frames, src_audio.shape[2])
                        audio_tile = src_audio[:, :, :available].clone()
                        if available < audio_tile_frames:
                            pad = torch.zeros(
                                1, audio_info["channels"],
                                audio_tile_frames - available,
                                audio_info["freq_bins"],
                                device=audio_info["device"],
                                dtype=audio_info["dtype"],
                            )
                            audio_tile = torch.cat([audio_tile, pad], dim=2)
                        print(
                            f"[LoopingSampler] Base tile audio (from input): {audio_tile.shape}"
                        )
                    else:
                        # Generation: start from zeros
                        audio_tile = torch.zeros(
                            1,
                            audio_info["channels"],
                            audio_tile_frames,
                            audio_info["freq_bins"],
                            device=audio_info["device"],
                            dtype=audio_info["dtype"],
                        )
                        print(
                            f"[LoopingSampler] Base tile audio (zeros): {audio_tile.shape}"
                        )

                if tile_config.tile_guiding_latents is not None:
                    tile_out_latents = LTXVInContextSampler().sample(
                        vae=model_config.vae,
                        guider=new_guider,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        noise=model_config.noise,
                        guiding_latents=guiding_latent_chunk,
                        optional_cond_images=this_chunk_keyframes,
                        optional_cond_indices=this_chunk_keyframe_indices,
                        num_frames=-1,
                        optional_negative_index_latents=tile_config.tile_negative_index_latents,
                        optional_negative_index=sampling_config.optional_negative_index,
                        optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                        optional_initialization_latents=latent_chunk,
                        cond_image_strength=sampling_config.cond_image_strength,
                        guiding_strength=sampling_config.guiding_strength,
                        guiding_start_step=sampling_config.guiding_start_step,
                        guiding_end_step=sampling_config.guiding_end_step,
                        _audio_tile=audio_tile,
                    )[0]
                else:
                    tile_out_latents = LTXVBaseSampler().sample(
                        model=model_config.model,
                        vae=model_config.vae,
                        noise=model_config.noise,
                        sampler=model_config.sampler,
                        sigmas=model_config.sigmas,
                        guider=new_guider,
                        num_frames=(
                            min(
                                sampling_config.temporal_tile_size,
                                tile_config.tile_latents["samples"].shape[2],
                            )
                            - 1
                        )
                        * sampling_config.time_scale_factor
                        + 1,
                        width=tile_config.tile_width
                        * sampling_config.width_scale_factor,
                        height=tile_config.tile_height
                        * sampling_config.height_scale_factor,
                        optional_cond_images=this_chunk_keyframes,
                        optional_cond_indices=this_chunk_keyframe_indices,
                        crop="center",
                        crf=30,
                        strength=sampling_config.cond_image_strength,
                        optional_negative_index_latents=tile_config.tile_negative_index_latents,
                        optional_negative_index=sampling_config.optional_negative_index,
                        optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                        optional_initialization_latents=latent_chunk,
                        guiding_start_step=sampling_config.guiding_start_step,
                        guiding_end_step=sampling_config.guiding_end_step,
                        _audio_tile=audio_tile,
                    )[0]

                # Extract denoised audio from base tile
                accumulated_audio = tile_out_latents.pop("_audio", None)
                first_tile_out_latents = copy.deepcopy(tile_out_latents)
            else:
                # Compute audio init data for the "new frames" portion of
                # this extend tile (for stage-2 refinement).
                _audio_new_init = None
                src_audio = audio_info.get("tensor") if audio_info else None
                if src_audio is not None and accumulated_audio is not None:
                    # The extend tile adds new video frames after the overlap.
                    # Map the video new-frame region to audio frames.
                    acc_audio_T = accumulated_audio.shape[2]
                    audio_ratio = (
                        audio_info["total_audio_frames"]
                        / max(audio_info["total_video_frames"], 1)
                    )
                    video_new_latent = (
                        latent_chunk["samples"].shape[2]
                        - sampling_config.temporal_overlap
                    )
                    audio_new_frames = max(
                        1, round(video_new_latent * audio_ratio)
                    )
                    # The new audio starts where accumulated audio ends
                    audio_start = acc_audio_T
                    audio_end = min(
                        audio_start + audio_new_frames,
                        src_audio.shape[2],
                    )
                    if audio_start < src_audio.shape[2]:
                        _audio_new_init = src_audio[
                            :, :, audio_start:audio_end
                        ]

                tile_out_latents = LTXVExtendSampler().sample(
                    model=model_config.model,
                    vae=model_config.vae,
                    sampler=model_config.sampler,
                    sigmas=model_config.sigmas,
                    noise=model_config.noise,
                    latents=tile_out_latents,
                    num_new_frames=(
                        latent_chunk["samples"].shape[2]
                        - sampling_config.temporal_overlap
                    )
                    * sampling_config.time_scale_factor,
                    frame_overlap=sampling_config.temporal_overlap
                    * sampling_config.time_scale_factor,
                    guider=new_guider,
                    strength=sampling_config.temporal_overlap_cond_strength,
                    guiding_strength=sampling_config.guiding_strength,
                    cond_image_strength=sampling_config.cond_image_strength,
                    optional_guiding_latents=guiding_latent_chunk,
                    optional_cond_images=this_chunk_keyframes,
                    optional_cond_indices=this_chunk_keyframe_indices,
                    optional_reference_latents=normalizing_latent_chunk,
                    normalize_per_frame=normalize_per_frame,
                    adain_factor=sampling_config.adain_factor,
                    optional_negative_index_latents=tile_config.tile_negative_index_latents,
                    optional_negative_index=sampling_config.optional_negative_index,
                    optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                    optional_initialization_latents=latent_chunk,
                    guiding_start_step=sampling_config.guiding_start_step,
                    guiding_end_step=sampling_config.guiding_end_step,
                    _audio_tile=accumulated_audio,
                    _audio_new_init=_audio_new_init,
                )[0]

                # Update accumulated audio from extend tile
                accumulated_audio = tile_out_latents.pop("_audio", accumulated_audio)

            if save_checkpoints:
                self._save_chunk_checkpoint(
                    tile_out_latents, accumulated_audio, tile_config, chunk_index
                )

            chunk_index += 1

        # Store accumulated audio in the output for the caller
        if accumulated_audio is not None:
            tile_out_latents["_audio"] = accumulated_audio

        return tile_out_latents

    def _save_chunk_checkpoint(
        self, tile_out_latents, accumulated_audio, tile_config, chunk_index
    ):
        """Salvage checkpoint: persist the accumulated latent after each temporal
        chunk so a mid-run crash leaves a decodable partial result on disk.

        Writes ComfyUI-native ``.latent`` files (one for video, one for audio if
        present) into the ``input`` directory, so recovery needs no custom node:
        reload with the stock ``LoadLatent`` node(s) and, for AV, recombine with
        ``LTXVConcatAVLatent``. The files carry the ``latent_format_version_0``
        marker, so ``LoadLatent`` round-trips them with multiplier 1.0.

        The latent is cumulative, so each write supersedes the previous one; we
        overwrite a single per-spatial-tile file per stream and rename atomically
        (.tmp -> final) to avoid a corrupt file if the process dies mid-write.
        Best-effort — a checkpoint failure must never abort generation. See
        CLAUDE.md ("save_checkpoints salvage toggle") for the recovery workflow.
        """
        try:
            import os

            import comfy.utils
            import folder_paths

            samples = tile_out_latents["samples"]
            if isinstance(samples, NestedTensor) and len(samples.tensors) == 2:
                video, audio = samples.tensors[0], samples.tensors[1]
            else:
                video, audio = samples, accumulated_audio

            in_dir = folder_paths.get_input_directory()
            base = f"ltxv_looping_ckpt_v{tile_config.v}_h{tile_config.h}"

            def _write_latent(tensor, suffix):
                # ComfyUI .latent format: LoadLatent reads "latent_tensor" and,
                # when "latent_format_version_0" is present, uses multiplier 1.0.
                payload = {
                    "latent_tensor": tensor.detach().to("cpu", torch.float32).contiguous(),
                    "latent_format_version_0": torch.tensor([]),
                }
                path = os.path.join(in_dir, f"{base}_{suffix}.latent")
                tmp = path + ".tmp"
                comfy.utils.save_torch_file(payload, tmp)
                os.replace(tmp, path)
                return os.path.basename(path)

            written = [_write_latent(video, "video")]
            if audio is not None:
                written.append(_write_latent(audio, "audio"))

            print(
                f"[LoopingSampler] Saved salvage checkpoint (chunk {chunk_index}, "
                f"video={list(video.shape)}"
                + (f", audio={list(audio.shape)}" if audio is not None else "")
                + f") -> {', '.join(written)} in input/"
            )
        except Exception as e:
            print(
                f"[LoopingSampler] WARNING: failed to write salvage checkpoint: {e}"
            )

    def _create_spatial_weights(
        self,
        tile_shape,
        v,
        h,
        horizontal_tiles,
        vertical_tiles,
        spatial_overlap,
        device,
        dtype,
    ):
        """Create blending weights for spatial tiles."""
        tile_weights = torch.ones(tile_shape, device=device, dtype=dtype)

        # Apply horizontal blending weights
        if h > 0:  # Left overlap
            h_blend = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
            tile_weights[:, :, :, :, :spatial_overlap] *= h_blend.view(1, 1, 1, 1, -1)
        if h < horizontal_tiles - 1:  # Right overlap
            h_blend = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
            tile_weights[:, :, :, :, -spatial_overlap:] *= h_blend.view(1, 1, 1, 1, -1)

        # Apply vertical blending weights
        if v > 0:  # Top overlap
            v_blend = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
            tile_weights[:, :, :, :spatial_overlap, :] *= v_blend.view(1, 1, 1, -1, 1)
        if v < vertical_tiles - 1:  # Bottom overlap
            v_blend = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
            tile_weights[:, :, :, -spatial_overlap:, :] *= v_blend.view(1, 1, 1, -1, 1)

        return tile_weights

    def _calculate_tile_seed(
        self,
        first_seed,
        start_index,
        vertical_tiles,
        horizontal_tiles,
        v,
        h,
        seed_offset,
    ):
        """Calculate the seed value for a specific temporal and spatial tile."""
        return (
            first_seed
            + start_index * (vertical_tiles * horizontal_tiles)
            + v * horizontal_tiles
            + h
            + seed_offset
        )

    def _get_per_tile_value(self, value_list, tile_index):
        """Get a value from a per-tile configuration list, using the last value if the list is shorter."""
        return value_list[min(tile_index, len(value_list) - 1)]

    def _parse_comma_separated_string(
        self, config_string, default_value, converter_func, optional_total_size=None
    ):
        """Parse a comma-separated configuration string into a list with type conversion."""
        if config_string == "":
            config_string = default_value
        values = [converter_func(item.strip()) for item in config_string.split(",")]

        def handle_negative_index(value):
            if value < 0:
                return value + optional_total_size
            return value

        if optional_total_size is not None:
            values = [handle_negative_index(value) for value in values]
        return values

    def _prepare_guider_for_chunk(
        self, guider, optional_positive_conditionings, chunk_index
    ):
        """Prepare the guider for a specific chunk, handling optional positive conditionings."""
        if optional_positive_conditionings is not None:
            new_guider = copy.copy(guider)
            positive, negative = _get_raw_conds_from_guider(guider)
            # Use the conditioning at chunk_index, or the last one if we've run out
            conditioning_index = min(
                chunk_index, len(optional_positive_conditionings) - 1
            )
            new_cond = optional_positive_conditionings[conditioning_index]
            print(
                f"[LoopingSampler] Chunk {chunk_index}: using prompt {conditioning_index} "
                f"(of {len(optional_positive_conditionings)}), "
                f"cond shape={new_cond[0][0].shape if new_cond and len(new_cond[0]) > 0 else 'N/A'}, "
                f"has frame_rate={'frame_rate' in new_cond[0][1] if new_cond and len(new_cond[0]) > 1 else 'N/A'}"
            )
            new_guider.set_conds(
                new_cond,
                negative,
            )
            new_guider.raw_conds = (
                new_cond,
                negative,
            )
            return new_guider
        else:
            return guider

    def _calculate_keyframe_per_tile_indices(
        self, keyframe_indices, temporal_tile_size, temporal_overlap, num_frames
    ):
        """
        Calculate which temporal tile each keyframe falls into.

        Returns a list of tuples (temporal_tile_index, in_tile_index) for each keyframe.
        - temporal_tile_index: which temporal tile the keyframe falls in
        - in_tile_index: index within that specific temporal tile

        First tile: frames [0, temporal_tile_size - 8] (size = temporal_tile_size - 8 + 1)
        Subsequent tiles follow the pattern from _process_temporal_chunks:
        - Tile n starts at: n * (temporal_tile_size - temporal_overlap)
        - Tile n ends at: temporal_tile_size + n * (temporal_tile_size - temporal_overlap) - 1

        For subsequent tiles, keyframes with in_tile_index < temporal_overlap
        don't count as falling in that tile.
        """
        result = []

        for keyframe_index in keyframe_indices:
            if keyframe_index >= num_frames:
                print(
                    f"Keyframe index {keyframe_index} is greater than num_frames {num_frames}, skipping"
                )
                continue
            # First tile (tile 0): covers frames [0, temporal_tile_size - 8]
            if keyframe_index < temporal_tile_size - 7:
                result.append((0, keyframe_index))
                continue

            # Find which subsequent tile this keyframe could fall into
            # Tile n starts at: n * (temporal_tile_size - temporal_overlap) - 7
            # Tile n ends at: temporal_tile_size + n * (temporal_tile_size - temporal_overlap) - 1 - 7

            tile_step = temporal_tile_size - temporal_overlap
            tile_index = 1

            while True:
                tile_start = tile_index * tile_step - 7
                tile_end = temporal_tile_size + tile_index * tile_step - 1 - 7
                print(
                    f"Tile {tile_index} starts at {tile_start} and ends at {tile_end}"
                )

                # Check if keyframe falls within this tile's range
                if keyframe_index <= tile_end:
                    in_tile_index = (
                        keyframe_index - tile_start - 7
                    )  # this extra  -7 is needed because the first latent is re-interpreted inside the temporal tile as 1 pixel frame

                    # For tiles > 0, if in_tile_index < temporal_overlap,
                    # the keyframe doesn't "fall" in this tile - assign to previous tile
                    if in_tile_index < temporal_overlap:
                        tile_index -= 1
                        if tile_index == 0:
                            # Previous tile is the first tile
                            in_tile_index = keyframe_index
                        else:
                            # Previous tile starts at (tile_index) * tile_step
                            prev_start = tile_start - tile_step
                            in_tile_index = (
                                keyframe_index - prev_start - 7
                            )  # same reason as above

                    result.append((tile_index, in_tile_index))
                    break

                tile_index += 1

        return result

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        guiding_strength,
        adain_factor,
        temporal_tile_size,
        temporal_overlap,
        temporal_overlap_cond_strength,
        horizontal_tiles,
        vertical_tiles,
        spatial_overlap,
        optional_cond_images=None,
        cond_image_strength=1.0,
        optional_guiding_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index_strength=1.0,
        optional_positive_conditionings=None,
        guiding_start_step=0,
        guiding_end_step=1000,
        optional_cond_image_indices="0",
        optional_normalizing_latents=None,
        save_checkpoints=False,
        per_tile_seed_offsets="0",  # hidden interface
    ):
        # Get dimensions and prepare for spatial tiling
        samples = latents["samples"]

        # Handle AV latents: separate video and audio, process video through
        # the tile loop, then reassemble AV output at the end.
        audio_info = None
        if (
            isinstance(samples, NestedTensor)
            and len(samples.tensors) == 2
        ):
            video_tensor = samples.tensors[0]
            audio_tensor = samples.tensors[1]
            audio_info = {
                "channels": audio_tensor.shape[1],
                "freq_bins": audio_tensor.shape[3],
                "total_video_frames": video_tensor.shape[2],
                "total_audio_frames": audio_tensor.shape[2],
                "device": audio_tensor.device,
                "dtype": audio_tensor.dtype,
                "tensor": audio_tensor,  # preserve for stage-2 refinement
            }
            # Switch to video-only for existing tiling logic
            latents = latents.copy()
            latents["samples"] = video_tensor
            if "noise_mask" in latents and isinstance(latents["noise_mask"], NestedTensor):
                latents["noise_mask"] = latents["noise_mask"].tensors[0]
            samples = video_tensor
            print(
                f"[LoopingSampler] AV latent detected: video={video_tensor.shape}, "
                f"audio={audio_tensor.shape}. Audio will be generated jointly."
            )

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        temporal_tile_size = temporal_tile_size // time_scale_factor
        temporal_overlap = temporal_overlap // time_scale_factor
        first_seed = noise.seed

        per_tile_seed_offsets = self._parse_comma_separated_string(
            per_tile_seed_offsets, "0", int
        )

        keyframe_indices = self._parse_comma_separated_string(
            optional_cond_image_indices,
            "0",
            int,
            optional_total_size=frames * time_scale_factor - 7,
        )
        keyframe_per_tile_indices = self._calculate_keyframe_per_tile_indices(
            keyframe_indices,
            temporal_tile_size * time_scale_factor,
            temporal_overlap * time_scale_factor,
            frames * time_scale_factor - 7,
        )
        print(f"Keyframe per tile indices: {keyframe_per_tile_indices}")
        if optional_cond_images is not None:
            optional_keyframes = (
                comfy.utils.common_upscale(
                    optional_cond_images.movedim(-1, 1),
                    width * width_scale_factor,
                    height * height_scale_factor,
                    "bilinear",
                    crop="center",
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
        else:
            optional_keyframes = None

        if optional_guiding_latents is not None:
            guide = optional_guiding_latents["samples"]
            assert (
                samples.shape[2] == guide.shape[2]
            ), "The number of frames in the latents and optional_guiding_latents must be the same"
            assert (
                samples.shape[3] % guide.shape[3] == 0
                and samples.shape[4] % guide.shape[4] == 0
            ), "The ratio of the height and width of the latents and optional_guiding_latents must be an integer"
            grid_size_h = samples.shape[3] // guide.shape[3]
            grid_size_w = samples.shape[4] // guide.shape[4]
            optional_guiding_latents = LTXVDilateLatent().dilate_latent(
                optional_guiding_latents, grid_size_w, grid_size_h
            )[0]

        # Calculate tile sizes with overlap
        base_tile_height = (
            height + (vertical_tiles - 1) * spatial_overlap
        ) // vertical_tiles
        base_tile_width = (
            width + (horizontal_tiles - 1) * spatial_overlap
        ) // horizontal_tiles

        # Output tensors will be initialized after first tile to get correct temporal dimension
        final_output = None
        weights = None

        # Process each spatial tile
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - spatial_overlap)
                v_start = v * (base_tile_height - spatial_overlap)

                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )

                # Calculate actual tile dimensions
                tile_height = v_end - v_start
                tile_width = h_end - h_start

                print(f"Processing spatial tile at row {v}, col {h}:")
                print(f"  Position: ({v_start}:{v_end}, {h_start}:{h_end})")
                print(f"  Size: {tile_height}x{tile_width}")

                # Extract spatial tiles from all inputs
                (
                    tile_latents,
                    tile_guiding_latents,
                    tile_negative_index_latents,
                    tile_keyframes,
                    tile_normalizing_latents,
                ) = self._extract_spatial_tile(
                    latents,
                    optional_guiding_latents,
                    optional_negative_index_latents,
                    optional_normalizing_latents,
                    optional_keyframes,
                    v_start,
                    v_end,
                    h_start,
                    h_end,
                    height_scale_factor,
                    width_scale_factor,
                )

                # Process all temporal chunks for this spatial tile
                tile_config = TileConfig(
                    tile_latents=tile_latents,
                    tile_guiding_latents=tile_guiding_latents,
                    tile_negative_index_latents=tile_negative_index_latents,
                    tile_keyframes=tile_keyframes,
                    keyframe_per_tile_indices=keyframe_per_tile_indices,
                    tile_normalizing_latents=tile_normalizing_latents,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    v=v,
                    h=h,
                    vertical_tiles=vertical_tiles,
                    horizontal_tiles=horizontal_tiles,
                    first_seed=first_seed,
                )

                sampling_config = SamplingConfig(
                    temporal_tile_size=temporal_tile_size,
                    temporal_overlap=temporal_overlap,
                    temporal_overlap_cond_strength=temporal_overlap_cond_strength,
                    cond_image_strength=cond_image_strength,
                    guiding_strength=guiding_strength,
                    adain_factor=adain_factor,
                    optional_negative_index=(
                        -1 * tile_negative_index_latents["samples"].shape[2]
                        if tile_negative_index_latents is not None
                        else -1
                    ),
                    optional_negative_index_strength=optional_negative_index_strength,
                    optional_positive_conditionings=optional_positive_conditionings,
                    time_scale_factor=time_scale_factor,
                    width_scale_factor=width_scale_factor,
                    height_scale_factor=height_scale_factor,
                    per_tile_seed_offsets=per_tile_seed_offsets,
                    guiding_start_step=guiding_start_step,
                    guiding_end_step=guiding_end_step,
                )

                model_config = ModelConfig(
                    model=model,
                    vae=vae,
                    noise=noise,
                    sampler=sampler,
                    sigmas=sigmas,
                    guider=guider,
                )

                # Only process audio for the first spatial tile (audio has no spatial dim)
                tile_audio_info = audio_info if (v == 0 and h == 0) else None
                tile_out_latents = self._process_temporal_chunks(
                    tile_config,
                    sampling_config,
                    model_config,
                    audio_info=tile_audio_info,
                    save_checkpoints=save_checkpoints,
                )

                # Extract accumulated audio from first spatial tile
                if v == 0 and h == 0 and audio_info is not None:
                    accumulated_audio = tile_out_latents.pop("_audio", None)

                # Initialize output tensors on first tile (to get correct temporal dimension)
                if final_output is None:
                    out_temporal = tile_out_latents["samples"].shape[2]
                    final_output = torch.zeros(
                        batch,
                        channels,
                        out_temporal,
                        height,
                        width,
                        device=samples.device,
                        dtype=samples.dtype,
                    )
                    weights = torch.zeros_like(final_output)

                # Move tile samples to output device and create weight mask
                tile_samples = tile_out_latents["samples"].to(final_output.device)
                tile_weights = self._create_spatial_weights(
                    tile_samples.shape,
                    v,
                    h,
                    horizontal_tiles,
                    vertical_tiles,
                    spatial_overlap,
                    final_output.device,
                    final_output.dtype,
                )

                # Add weighted tile to final output
                final_output[:, :, :, v_start:v_end, h_start:h_end] += (
                    tile_samples * tile_weights
                )
                weights[:, :, :, v_start:v_end, h_start:h_end] += tile_weights

        # Normalize by weights
        final_output = final_output / (weights + 1e-8)

        # Reassemble AV output if audio was processed
        if audio_info is not None and accumulated_audio is not None:
            out_latents = {"samples": NestedTensor([final_output, accumulated_audio])}
            print(
                f"[LoopingSampler] AV output: video={final_output.shape}, "
                f"audio={accumulated_audio.shape}"
            )
        else:
            out_latents = {"samples": final_output}

        noise.seed = first_seed
        return (out_latents,)


@comfy_node(
    name="MultiPromptProvider",
)
class MultiPromptProvider:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompts to encode, separated by |. Each prompt will be encoded separately. Each prompt will be used in one temporal_tile in LTXVLoopingSampler.",
                    },
                ),
                "clip": ("CLIP", {"tooltip": "CLIP model to encode the prompts."}),
            },
            "optional": {
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Frame rate to embed in the conditioning (same as LTXVConditioning). Required for proper temporal and audio generation.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditionings",)

    FUNCTION = "get_prompt_list"
    CATEGORY = "prompt"

    def get_prompt_list(self, prompts, clip, frame_rate=24.0):
        import node_helpers

        prompt_list = prompts.split("|")
        prompt_list = [prompt.strip() for prompt in prompt_list]
        encoded_prompt_list = [
            node_helpers.conditioning_set_values(
                clip.encode_from_tokens_scheduled(clip.tokenize(prompt)),
                {"frame_rate": frame_rate},
            )
            for prompt in prompt_list
        ]
        return (encoded_prompt_list,)
